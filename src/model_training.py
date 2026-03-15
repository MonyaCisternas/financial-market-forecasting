import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

print("Loading regime dataset...")
data = pd.read_csv("data/regime_data.csv")

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

os.makedirs("models", exist_ok = True)

for stock in stocks:
    print(f"\n==============================")
    print(f"Training model for {stock}")
    print(f"==============================")
    df = data[data["Asset"] == stock].copy()
    df["Target"] = (df["Log_Return"].shift(-1) > 0).astype(int)
    df = df.dropna()
    X = df[features].astype(float)
    y = df["Target"]
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    model = XGBClassifier(
        n_estimators = 400,
        max_depth = 5,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        eval_metric = "logloss"
    )
    print("Training model...")
    model.fit(X_train, y_train)
    print("Evaluating model...")
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs > 0.5).astype(int)
    print(classification_report(y_test, preds))
    model_path = f"models/{stock.lower()}_model.pkl"
    joblib.dump(model, model_path)
    print(f"{stock} model saved to {model_path}")

print("\nAll models trained successfully.")
