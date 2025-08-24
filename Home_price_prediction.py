import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# dataset load kar
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# features ani target split
x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression pipeline
lin = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
lin.fit(x_train, y_train)
y_pred_lin = lin.predict(x_test)

# evaluation - linear regression
mae_lin = mean_absolute_error(y_test, y_pred_lin)
rmse_lin = mean_squared_error(y_test, y_pred_lin) ** 0.5
r2_lin = r2_score(y_test, y_pred_lin)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# evaluation - random forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

# best model निवड
best = lin if rmse_lin <= rmse_rf else rf
best_name = 'Linear Regression' if best is lin else 'Random Forest'

# model save कर
joblib.dump(best, 'house_price_model.joblib')

print('Linear Regression:', round(mae_lin,4), round(rmse_lin,4), round(r2_lin,4))
print('Random Forest:', round(mae_rf,4), round(rmse_rf,4), round(r2_rf,4))
print('Best Model:', best_name)

# एक sample prediction
def predict_price(model, dct, cols):
    row = pd.DataFrame([dct], columns=cols)
    return float(model.predict(row)[0])

sample = {
    'MedInc': 5.0,
    'HouseAge': 30.0,
    'AveRooms': 6.0,
    'AveBedrms': 1.0,
    'Population': 1000.0,
    'AveOccup': 3.0,
    'Latitude': 34.0,
    'Longitude': -118.0
}

print('Sample Prediction:', predict_price(best, sample, list(x.columns)))
