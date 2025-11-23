import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.secret_key = "trovista_secret_key"  # for flash messages
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----- Paths (adjust if you changed filenames) -----
CSV_PATH = os.path.join("dataset", "Myntra Fasion Clothing.csv")  # your CSV exact name
MODEL_DIR = "model"
TEXT_KNN_PATH = os.path.join(MODEL_DIR, "text_recommender.pkl")
TEXT_VEC_PATH = os.path.join(MODEL_DIR, "text_vectorizer.pkl")
IMAGE_FEATURES_PATH = os.path.join(MODEL_DIR, "image_features.npy")
IMAGE_MAP_PATH = os.path.join(MODEL_DIR, "image_map.csv")

# ----- Load dataset -----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, low_memory=False)
else:
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please place your dataset there.")

# Keep original df for lookups
df_original = df.copy()

# ----- Load text models (if present) -----
text_knn = None
text_vectorizer = None
if os.path.exists(TEXT_KNN_PATH) and os.path.exists(TEXT_VEC_PATH):
    try:
        text_knn = joblib.load(TEXT_KNN_PATH)
        text_vectorizer = joblib.load(TEXT_VEC_PATH)
        print("Loaded text KNN and vectorizer.")
    except Exception as e:
        print("Error loading text models:", e)
else:
    print("Text models not found. Please run train_model.py to create text_recommender and text_vectorizer.")

# ----- Load image features & build KNN if available -----
img_knn = None
img_features = None
img_map = None  # pandas dataframe with mapping info (image url / index)
if os.path.exists(IMAGE_FEATURES_PATH) and os.path.exists(IMAGE_MAP_PATH):
    try:
        img_features = np.load(IMAGE_FEATURES_PATH)
        img_map = pd.read_csv(IMAGE_MAP_PATH, low_memory=False)
        # fit a simple NearestNeighbors on the loaded embeddings
        img_knn = NearestNeighbors(n_neighbors=6, metric="cosine", n_jobs=-1)
        img_knn.fit(img_features)
        print("Loaded image features and built image KNN.")
    except Exception as e:
        print("Error loading image features/map:", e)
else:
    print("Image features or map not found. If you want image recommendations, run train_image_model.py first.")


# ----- Helper: safe get product fields -----
def product_record_from_index(idx):
    """Return a dict of useful fields for templating for the dataset row at dataframe index idx."""
    try:
        row = df_original.iloc[idx]
    except Exception:
        return {}
    # adapt these fields to whatever your CSV contains
    rec = {
        "index": int(idx),
        "productDisplayName": row.get("productDisplayName", "") or row.get("productName", "") or "Item",
        "masterCategory": row.get("masterCategory", ""),
        "subCategory": row.get("subCategory", ""),
        "articleType": row.get("articleType", ""),
        "mrp": row.get("mrp", ""),
        "discountedPrice": row.get("discountedPrice", ""),
        "image": row.get("imageURL", "") or row.get("image", "") or row.get("productImage", "")
    }
    return rec


# ----- ROUTES -----
@app.route("/")
def index():
    """Home page with body-measurement form."""
    # Optionally pass some sample categories to show in UI
    body_shapes = ["Hourglass", "Pear", "Apple", "Rectangle"]
    genders = ["Women", "Men"]
    return render_template("index.html", body_shapes=body_shapes, genders=genders)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Text-based recommendation using TF-IDF + KNN."""
    if text_knn is None or text_vectorizer is None:
        flash("Text models not found. Please run training script (train_model.py). Text recommendations unavailable.", "error")
        return redirect(url_for("index"))

    height = request.form.get("height", "").strip()
    weight = request.form.get("weight", "").strip()
    body = request.form.get("body", "").strip()
    gender = request.form.get("gender", "").strip()
    # build a textual query that the vectorizer understands
    query = f"{gender} {body} {height}cm {weight}kg outfit"
    # transform and query
    qv = text_vectorizer.transform([query])
    dists, indices = text_knn.kneighbors(qv)
    indices = indices[0].tolist()
    # Convert indices to product dicts
    results = [product_record_from_index(int(i)) for i in indices]
    return render_template("recommend.html", results=results, source="text")


@app.route("/image", methods=["GET", "POST"])
def image_route():
    """Image upload page (GET) and prediction (POST)."""
    if request.method == "GET":
        return render_template("upload_image.html")

    # POST: handle uploaded file
    f = request.files.get("file")
    if not f:
        flash("No file uploaded.", "error")
        return redirect(url_for("image_route"))

    # Save the uploaded image
    filename = f.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    # If no image model/embeddings available, show helpful message
    if img_knn is None or img_features is None or img_map is None:
        flash("Image model not ready (train_image_model.py not run or failed). Try text recommendations or run image training.", "error")
        return redirect(url_for("index"))

    # compute embedding for uploaded image with MobileNetV2
    try:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
    except Exception as e:
        flash(f"TensorFlow/Keras import failed: {e}", "error")
        return redirect(url_for("index"))

    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    img = keras_image.load_img(save_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    emb = base_model.predict(arr, verbose=0)
    emb = emb.reshape(1, -1)

    # query image knn
    dists, idxs = img_knn.kneighbors(emb)
    idxs = idxs[0].tolist()

    # map embedding indices to original dataframe indices using img_map
    # the train script stored a mapping file with columns 'imageURL' and 'index' (index points to df index)
    mapped_df = []
    try:
        for emb_pos in idxs:
            # img_map row corresponds to embedding position
            row = img_map.iloc[emb_pos]
            df_idx = int(row.get("index", emb_pos))
            mapped_df.append(product_record_from_index(df_idx))
    except Exception:
        # fallback: treat emb_pos as direct df index
        mapped_df = [product_record_from_index(int(i)) for i in idxs]

    return render_template("recommend.html", results=mapped_df, source="image", image=filename)


@app.route("/product/<int:idx>")
def product_detail(idx):
    """Simple product details page if you want to view a single product."""
    rec = product_record_from_index(idx)
    if not rec:
        flash("Product not found.", "error")
        return redirect(url_for("index"))
    return render_template("product.html", product=rec)


# ----- Helpful debug route (optional) -----
@app.route("/status")
def status():
    s = {
        "csv_loaded": CSV_PATH,
        "n_products": len(df_original),
        "text_model": os.path.exists(TEXT_KNN_PATH) and os.path.exists(TEXT_VEC_PATH),
        "image_features": os.path.exists(IMAGE_FEATURES_PATH) and os.path.exists(IMAGE_MAP_PATH)
    }
    return s


if __name__ == "__main__":
    # run the app
    app.run(debug=True)