# Breast Cancer Diagnosis Prediction

This project uses machine learning to predict whether a breast cancer tumor is malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Project Overview

The application provides:

1. **Data Exploration**: Visualize and understand the breast cancer dataset
2. **Model Training**: Train an SVM model with customizable parameters
3. **Prediction**: Input feature values to get a diagnosis prediction

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Features describe characteristics of the cell nuclei present in the image.

Features include:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter^2 / area - 1.0)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension ("coastline approximation" - 1)

For each feature, the mean, standard error, and "worst" (mean of the three largest values) are computed, resulting in 30 features.

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Application Features

### Data Exploration
- View dataset overview
- Visualize class distribution
- Explore feature correlations
- Analyze feature distributions by diagnosis

### Model Training
- Customize SVM model parameters
- Train the model with selected parameters
- View model evaluation metrics and visualizations

### Prediction
- Input feature values through an intuitive interface
- Get prediction results with probability scores
- Visualize prediction confidence

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn