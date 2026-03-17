import pandas as pd
import numpy as np
import joblib

print("Loading data...")
data = pd.read_csv("data/regime_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

stocks = ["Naspers", "Standard_Bank", "Sasol", "Anglo_American", "Shoprite"]

features = [
    "Return_Z","Relative_Return","Return_1d","Return_5d",
    "Volatility","Volatility_60","Volatility_Ratio",
    "MA_ratio","Momentum_10","Rolling_Sharpe",
    "ARIMA_Forecast","GARCH_Volatility","regime"
]

print("Generating signals...")
portfolio_returns = []

for stock in stocks:
    df = data[data["Asset"] == stock].dropna().copy()
    model = joblib.load(f"models/{stock.lower()}_model.pkl")
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    df["prob"] = probs
    df["signal"] = np.where(probs > 0.65, 1, np.where(probs < 0.35, -1, 0))
    df["position"] = df["signal"]
    df["strategy_return"] = df["position"].shift(1) * df["Log_Return"]
    portfolio_returns.append(df[["Date", "strategy_return"]])

portfolio_df = pd.concat(portfolio_returns)
portfolio_df = portfolio_df.groupby("Date")["strategy_return"].mean()
initial_capital = 100000
capital_curve = (1 + portfolio_df).cumprod() * initial_capital
sharpe = np.sqrt(252) * portfolio_df.mean() / portfolio_df.std()
max_drawdown = (capital_curve / capital_curve.cummax() - 1).min()
total_return = capital_curve.iloc[-1] / initial_capital - 1

print("Saving performance data...")
performance_df = pd.DataFrame({"Date": capital_curve.index, "Portfolio_Value": capital_curve.values})
performance_df.to_csv("data/portfolio_performance.csv", index=False)
metrics = pd.DataFrame({"Sharpe": [sharpe], "Max_Drawdown": [max_drawdown], "Total_Return": [total_return]})
metrics.to_csv("data/performance_metrics.csv", index=False)
print("Performance tracking complete.")
