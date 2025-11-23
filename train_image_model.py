import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import joblib
import os
from tqdm import tqdm
import requests

# Load CSV
df = pd.read_csv("dataset/Myntra Fasion Clothing.csv")

# Image column is usually named 'imageURL'
IMAGE_COL = "imageURL"  # If different, tell me

# Create folders
os.makedirs("static/downloads", exist_ok=True)

# Pretrained model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

features = []
valid_images = []
indexes = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row[IMAGE_COL]

    try:
        img_data = requests.get(url).content
        with open(f"static/downloads/{idx}.jpg", "wb") as f:
            f.write(img_data)

        img = image.load_img(f"static/downloads/{idx}.jpg", target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array)[0]
        
        features.append(feature)
        valid_images.append(url)
        indexes.append(idx)

    except:
        continue

features = np.array(features)

# Save
np.save("model/image_features.npy", features)
pd.DataFrame({"imageURL": valid_images, "index": indexes}).to_csv("model/image_map.csv", index=False)

print("Image Model Training Completed!")