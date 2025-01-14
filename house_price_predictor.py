import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

class HousePricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()


    def preprocess_data(self, data):
        """Encodes categorical features and scales numerical features."""
        data['location'] = self.label_encoder.fit_transform(data['location'])
        features = data[['square_feet', 'bedrooms', 'bathrooms', 'location']]
        target = data['price']
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, target
    
    def train(self, data):
        """Trains the Random Forest model."""
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Training Complete")
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

    def save_model(self, model_path='model.pkl', scaler_path='scaler.pkl', encoder_path='label_encder.pkl'):
        """Loads the model, scaler, and label encoder."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Model, Scaler, and Label Encoder loaded!")

    def predict(self, square_feet, bedrooms, bathrooms, location):
        """Predicts the house price based on input features."""
        location_encoded = self.label_encoder.transform([location])[0]
        features = [[square_feet, bedrooms, bathrooms, location_encoded]]
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)[0]
