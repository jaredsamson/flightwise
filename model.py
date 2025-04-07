from lightgbm import LGBMRegressor
import pandas as pd

cat_features = ['segmentsAirlineName', 'startingAirport', 'destinationAirport', 'segmentsCabinCode']
num_features = ['daysUntilFlight']
model_features = cat_features + num_features

def train_model(df):
    for col in cat_features:
        df[col] = df[col].astype('category')

    X = df[model_features].copy()
    y = df['totalFare']

    model = LGBMRegressor(n_estimators=300, random_state=42)
    model.fit(X, y, categorical_feature=cat_features)

    return model, X
