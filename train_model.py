from house_price_predictor import HousePricePredictor
import pandas as pd 

if __name__ == '__main__':
    data = pd.read_csv('house_data.csv')

    predictor = HousePricePredictor()

    predictor.train(data)

    predictor.save_model()