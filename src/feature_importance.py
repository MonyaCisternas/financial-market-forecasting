import pandas as pd
import numpy as np
from xgboost import XGBClassifier

print("Loading data set...")
data = pd.read_csv("data/regime_data.csv")
data["Date"] = pd.to_datetime(data["Date"], errors = "coerce")

stocks = ["Naspers", "Standard_Bank", "Sasol", "Anglo_American", "Shoprite"]

features = [
    "Return_Z",
    "Relative_Return",
    "Return_1d",
    "Return_5d",
    "Volatility",
    "Volatility_60",
    "Volatility_Ratio",
    "MA_ratio",
    "Momentum_10",
    "Rolling_Sharpe",
    "ARIMA_Forecast",
    "GARCH_Volatility",
    "regime"
]

feature_scores = np.zeros(len(features))

for stock in stocks:
    print(f"Analyzing {stock}")
    df = data[data["Asset"] == stock].dropna()
    X = df[features]
    y = (df["Log_Return"].shift(-1) > 0).astype(int)
    model = XGBClassifier(n_estimators = 400, max_depth = 5, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, random_state = 42, eval_metric = "logloss")
    model.fit(X, y)
    feature_scores += model.feature_importances_

feature_scores = feature_scores / len(stocks)
importance_df = pd.DataFrame({"Features": features, "Importance": feature_scores})
importance_df = importance_df.sort_values("Importance", ascending = False)


print("\nFeature Importance Ranking")
print("---------------------------------")
print(importance_df)
