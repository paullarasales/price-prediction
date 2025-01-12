from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        square_feet = float(request.form.get('square_feet', 0))
        bedrooms = int(request.form.get('bedrooms', 0))
        bathrooms = int(request.form.get('bathrooms', 0))
        location = request.form.get('location', '')

        if not location:
            return render_template('index.html', error="Location is required.")

        # Encode the location
        all_locations = label_encoder.classes_
        location_encoded = [1 if loc == location else 0 for loc in all_locations]

        # Prepare the feature vector
        features = [[square_feet, bedrooms, bathrooms] + location_encoded]

        # Predict price
        predicted_price = model.predict(features)[0]

        return render_template('index.html', predicted_price=round(predicted_price, 2))
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', error="An unexpected error occurred.")

if __name__ == '__main__':
    app.run(debug=True)
