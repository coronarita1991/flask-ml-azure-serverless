# model_retrain.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 예시 데이터
X = np.random.rand(100, 6)
y = np.random.rand(100)

model = GradientBoostingRegressor()
model.fit(X, y)

joblib.dump(model, "boston_housing_prediction.joblib")
