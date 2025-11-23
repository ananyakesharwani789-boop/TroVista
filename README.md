# TroVista
A sensible application which decides your outfits according to the trends.
# Trovista - Fashion Clothing Recommendation System

A Flask-based web application that provides personalized fashion recommendations using both text-based and image-based machine learning models.

## Features

- **Text-Based Recommendations**: Get outfit suggestions based on body measurements, shape, and gender
- **Image-Based Recommendations**: Upload a fashion image to find similar items
- **Product Details**: View detailed information about recommended products
- **Dual Recommendation Engine**: Powered by TF-IDF vectorization and MobileNetV2 image embeddings

## Prerequisites

- Python 3.7+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trovista
```

2. Install required dependencies:
```bash
pip install flask pandas numpy scikit-learn joblib tensorflow
```

3. Create the necessary directories:
```bash
mkdir -p dataset model static/uploads
```

4. Place your dataset:
   - Add `Myntra Fasion Clothing.csv` to the `dataset/` folder
   - Ensure the CSV contains columns: `productDisplayName`, `masterCategory`, `subCategory`, `articleType`, `mrp`, `discountedPrice`, `imageURL`

## Setup and Training

### Text Model Training

Run the text model training script (you'll need to create `train_model.py`):
```bash
python train_model.py
```

This generates:
- `model/text_recommender.pkl` - KNN model for text recommendations
- `model/text_vectorizer.pkl` - TF-IDF vectorizer

### Image Model Training

Run the image model training script (you'll need to create `train_image_model.py`):
```bash
python train_image_model.py
```

This generates:
- `model/image_features.npy` - Pre-computed image embeddings
- `model/image_map.csv` - Mapping between embeddings and products

## Running the Application

Start the Flask development server:
```bash
python app.py
```

Access the application at `http://127.0.0.1:5000`

## Usage

### Text-Based Recommendations

1. Navigate to the home page
2. Enter your measurements:
   - Height (cm)
   - Weight (kg)
   - Body shape (Hourglass, Pear, Apple, Rectangle)
   - Gender (Women, Men)
3. Submit to receive personalized recommendations

### Image-Based Recommendations

1. Click on the image upload option
2. Upload a fashion image
3. Receive similar product recommendations based on visual similarity

## Project Structure

```
trovista/
├── app.py                      # Main Flask application
├── train_model.py             # Text model training script (to be created)
├── train_image_model.py       # Image model training script (to be created)
├── dataset/
│   └── Myntra Fasion Clothing.csv
├── model/
│   ├── text_recommender.pkl
│   ├── text_vectorizer.pkl
│   ├── image_features.npy
│   └── image_map.csv
├── static/
│   └── uploads/               # Uploaded images
└── templates/
    ├── index.html             # Home page
    ├── upload_image.html      # Image upload page
    ├── recommend.html         # Recommendations display
    └── product.html           # Product details
```

## API Endpoints

- `GET /` - Home page with body measurement form
- `POST /recommend` - Text-based recommendations
- `GET /image` - Image upload page
- `POST /image` - Image-based recommendations
- `GET /product/<idx>` - Product detail page
- `GET /status` - System status and model availability check

## Technical Details

### Text Recommendations
- Uses TF-IDF vectorization for text feature extraction
- K-Nearest Neighbors (KNN) algorithm for similarity matching
- Query format: `{gender} {body_shape} {height}cm {weight}kg outfit`

### Image Recommendations
- MobileNetV2 pre-trained on ImageNet for feature extraction
- Cosine similarity metric for finding similar images
- Returns top 6 similar products

## Troubleshooting

**Models not loading:**
- Ensure you've run both training scripts (`train_model.py` and `train_image_model.py`)
- Check that model files exist in the `model/` directory

**Dataset not found:**
- Verify `Myntra Fasion Clothing.csv` is in the `dataset/` folder
- Check the file name spelling matches exactly

**Image upload fails:**
- Ensure TensorFlow and Keras are properly installed
- Verify image model has been trained

## Future Enhancements

- User authentication and saved preferences
- Shopping cart functionality
- Advanced filtering options
- Real-time inventory integration
- Enhanced recommendation algorithms

Contributions are welcome! Please feel free to submit a Pull Request.

