from click.core import batch
from flask import Flask ,render_template,request
import pandas as pd
import pickle
import numpy as np
app =Flask(__name__)
data  = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))
buyable_model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location', '').strip()
    bhk = request.form.get('bhk', '').strip()
    bath = request.form.get('bath', '').strip()
    sqft = request.form.get('total_sqft', '').strip()
    balcony = request.form.get('balcony', '').strip()

    # Better validation to catch empty or non-numeric fields
    if not location or not bhk or not bath or not sqft or not balcony:
        return "Error: All fields are required and must be non-empty."

    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
        balcony = int(balcony)
    except ValueError:
        return "Error: BHK, bath, sqft, and balcony must be valid numbers."

    # Prediction input
    input_df = pd.DataFrame([[location, sqft, bath, balcony, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'balcony', 'bhk'])

    # Predict price
    predicted_price_lakhs = pipe.predict(input_df)[0]  # Keep in lakhs for buyability model
    predicted_price_rupees = predicted_price_lakhs * 1e5  # Convert to rupees for display

    # Predict buyability — add predicted price in lakhs
    input_df_buy = input_df.copy()
    input_df_buy['price'] = predicted_price_lakhs  # Use lakhs, not rupees
    print("Buyable Input:\n", input_df_buy)
    print("Buyable Model Prediction:", buyable_model.predict(input_df_buy))

    buyable = buyable_model.predict(input_df_buy)[0]
    buyable_text = "Yes" if buyable == 1 else "No"

    # Format price with commas
    formatted_price = f"{predicted_price_rupees:,.2f}"

    return f"{formatted_price}|{buyable_text}"




if __name__ == "__main__":
    app.run(debug=True,port=5001)