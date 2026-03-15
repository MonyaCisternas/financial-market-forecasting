import pandas as pd
import joblib
import numpy as np

print("Loading regime dataset...")
data = pd.read_csv("data/regime_data.csv")

stocks = [
    "Naspers",
    "Standard_Bank",
    "Sasol",
    "Anglo_American",
    "Shoprite"
]

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

initial_capital = 100000
transaction_cost = 0.001
portfolio_returns = []

for stock in stocks:
    print(f"\nBacktesting {stock}")
    model = joblib.load(f"models/{stock.lower()}_model.pkl")
    df = data[data["Asset"] == stock].copy()
    df = df.dropna()
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    df["probability"] = probs
    df["signal"] = 0
    df.loc[df["probability"] > 0.65, "signal"] = 1
    df.loc[df["probability"] < 0.35, "signal"] = -1
    df.loc[df["regime"] == 0, "signal"] = 0
    vol_target = 0.02
    df["position"] = df["signal"] * (vol_target / df["Volatility_60"])
    df["position"] = df["position"].clip(-0.2, 0.2)
    df["strategy_return"] = df["position"] * df["Log_Return"]
    df["strategy_return"] -= transaction_cost * df["position"].diff().abs().fillna(0)
    portfolio_returns.append(df[["Date", "strategy_return"]])

portfolio_df = pd.concat(portfolio_returns)
portfolio_df = portfolio_df.groupby("Date")["strategy_return"].mean()
capital_curve = (1 + portfolio_df).cumprod() * initial_capital
sharpe = np.sqrt(252) * portfolio_df.mean() / portfolio_df.std()
max_drawdown = (capital_curve / capital_curve.cummax() - 1).min()

print("\nStrategy Results")
print("--------------------------")
print(f"Final Capital: {capital_curve.iloc[-1]:.2f}")
print(f"Total Return: {(capital_curve.iloc[-1] / initial_capital - 1) * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
