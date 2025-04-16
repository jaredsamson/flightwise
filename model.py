from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

num_features = ['daysUntilFlight', 'dayOfWeek']
cat_features = ['segmentsAirlineName', 'startingAirport', 'destinationAirport', 'segmentsCabinCode']

model_features = cat_features + num_features

def train_model(df):
    for col in cat_features:
        df[col] = df[col].astype('category')

    X = df[model_features].copy()
    y = np.log1p(df['totalFare'])  # this compresses the price range


    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        random_state=42
    )



    #model = LGBMRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train, categorical_feature=cat_features)
    # 🔁 Predict on log scale, then revert back to dollars
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  # this is the actual fare prediction

    # Convert true values back too
    y_test_dollars = np.expm1(y_test)
    preds = np.clip(preds, 50, 1000)  # ensure predictions are within fare limits
    mae = mean_absolute_error(y_test_dollars, preds)
    rmse = mean_squared_error(y_test_dollars, preds) ** 0.5
    r2 = r2_score(y_test_dollars, preds)


    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    return model, X
