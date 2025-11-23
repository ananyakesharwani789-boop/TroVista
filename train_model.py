import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

CSV_PATH = "dataset/Myntra Fasion Clothing.csv"
MODEL_DIR = "model"

df = pd.read_csv(CSV_PATH, low_memory=False)

# Make sure text columns exist
df["BrandName"] = df["BrandName"].fillna("")
df["Category"] = df["Category"].fillna("")
df["Individual_category"] = df["Individual_category"].fillna("")
df["Description"] = df["Description"].fillna("")

# Create combined column for text model
df["combined"] = (
    df["BrandName"] + " " +
    df["Category"] + " " +
    df["Individual_category"] + " " +
    df["Description"]
).str.lower()

print("Creating vectorizer...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=7000)
tfidf_matrix = vectorizer.fit_transform(df["combined"])

print("Training KNN model...")
knn = NearestNeighbors(metric="cosine", n_neighbors=6)
knn.fit(tfidf_matrix)

# Save models
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(vectorizer, f"{MODEL_DIR}/text_vectorizer.pkl")
joblib.dump(knn, f"{MODEL_DIR}/text_recommender.pkl")

print("TEXT MODEL TRAINING SUCCESSFULLY COMPLETED!")