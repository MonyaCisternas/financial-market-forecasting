import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

print("Loading regime dataset...")
data = pd.read_csv("data/regime_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Asset", "Date"])

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

print("Preparing latest snapshot (avoiding lookahead bias)...")
latest_rows = []

for stock in stocks:
    df = data[data["Asset"] == stock].copy()
    df = df.dropna(subset = features + ["Log_Return", "Volatility_60"])
    if len(df) < 2:
        print(f"Skipping {stock} (not enough data)")
        continue
    latest_rows.append(df.iloc[-2])

latest_df = pd.DataFrame(latest_rows)

print("Generating expected returns (ML + ARIMA + regime + confidence)...")
expected_returns = []
confidences = []
valid_assets = []

for _, row in latest_df.iterrows():
    stock = row["Asset"]
    try:
        model = joblib.load(f"models/{stock.lower()}_model.pkl")
    except:
        print(f"Model not found for {stock}, skipping...")
        continue

    X = row[features].values.reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    ml_alpha = (prob - 0.5) * 2
    arima_alpha = row["ARIMA_Forecast"]

    if row["regime"] == 1:
        ml_weight = 0.8
    else:
        ml_weight = 0.6

    combined_alpha = ml_weight * ml_alpha + (1 - ml_weight) * arima_alpha

    vol = row["Volatility_60"]
    risk_scaled_alpha = combined_alpha * vol
    confidence = abs(ml_alpha)

    if confidence < 0.1:
        risk_scaled_alpha = 0

    regime_multiplier = 1.0 if row ["regime"] == 1 else 0.5
    expected_return = risk_scaled_alpha * confidence * regime_multiplier * 5
    expected_returns.append(expected_return)
    confidences.append(confidence)
    valid_assets.append(stock)

latest_df = latest_df[latest_df["Asset"].isin(valid_assets)].copy()
latest_df["Expected_Return"] = expected_returns
latest_df["Confidence"] = confidences
threshold = 0.0005
latest_df["Active"] = latest_df["Expected_Return"] > threshold
active_df = latest_df[latest_df["Active"]].copy()

if len(active_df) == 0:
    raise ValueError("No assets passed the threshold.")

valid_assets = active_df["Asset"].tolist()

print("Building covariance matrix (last 60 days)...")
returns_matrix = data.pivot(index = "Date", columns = "Asset", values = "Log_Return")
returns_matrix = returns_matrix[valid_assets]
returns_matrix = returns_matrix.tail(60).dropna()
cov_matrix = returns_matrix.cov() * 5

print("Running portfolio optimization...")
def objective(weights, mu, cov, risk_aversion, penalty):
    return -(weights @ mu - risk_aversion * weights @ cov @ weights - penalty * np.sum(weights ** 2))

def optimize(mu, cov, risk_aversion = 3, penalty = 0.1):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, 0.5)] * n 
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    results = minimize(
        objective,
        w0,
        args = (mu, cov, risk_aversion, penalty),
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints
    )
    if not results.success:
        print("Optimization failed, using equal weights")
        return w0

    return results.x

avg_regime = latest_df["regime"].mean()
risk_aversion = 5 if avg_regime < 0.5 else 2
mu = active_df["Expected_Return"].values
weights = optimize(mu, cov_matrix.values, risk_aversion)
latest_df["Weight"] = 0.0
latest_df.loc[latest_df["Active"], "Weight"] = weights
latest_df["Rebalance_Date"] = pd.Timestamp.today().date()

print("Formatting output...")
output_df = latest_df[["Asset", "Weight", "Expected_Return", "Confidence", "Rebalance_Date"]].copy()
output_df["Weight (%)"] = output_df["Weight"] * 100
output_df["Score"] = output_df["Expected_Return"] * output_df["Confidence"]
output_df = output_df.sort_values("Weight", ascending = False)

print("Saving weekly portfolio...")
output_df.to_csv("data/weekly_portfolio.csv", index = False)

print("\nWeekly Portfolio Generated Successfully")
print("----------------------------------------")
print(output_df)
