# Bengaluru House Price Predictor

A web application to predict house prices in Bengaluru and whether a house is buyable, using machine learning.

## Features
- Predict house prices based on location, BHK, bath, sqft, and balcony.
- Predict whether a house should be bought or not (buyable model).
- Uses trained models (`RidgeModel.pkl` for price, `rf_model.pkl` for buyable decision) and cleaned data (`Cleaned_data.csv`).
- Simple web interface built with Flask.

## Input Features
- Square Footage (sqft)
- Number of Balconies
- Number of Bathrooms
- Area Location
- BHK (Bedrooms, Hall, Kitchen)

## Dataset
- Original dataset of ~130,000 entries
- After feature engineering and data cleaning, reduced to ~75,000 high-quality data points

## Machine Learning Models Used
- Ridge Regression (for price prediction)
- Random Forest (for buyable prediction)

## Getting Started
**Prerequisites**
Make sure you have the following Python libraries installed:
- Flask
- Pandas
- Scikit-learn

## License
This project is licensed under the MIT License.

-Numpy

-Pickle

-PickleMixin

**Project Structure**
app.py - Flask application script

model.pkl - Saved trained machine learning model

data/ - Folder containing dataset files

templates/ - HTML templates for Flask frontend

**INSTALLATIONS:**
pip install flask pandas scikit-learn numpy pickle-mixin
