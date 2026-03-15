import pandas as pd
import numpy as np

print("Loading regime dataset...")
data = pd.read_csv("data/regime_data.csv")

stocks = ["Naspers", "Standard_Bank", "Sasol", "Anglo_American", "Shoprite"]

results = []

for stock in stocks:
    df = data[data["Asset"] == stock].copy()
    df = df.dropna()
    returns = df["Log_Return"]
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    results.append({"Stock" : stock, "Annual Return": annual_return, "Annual Volatility": annual_vol, "Sharpe Ratio": sharpe})

results_df = pd.DataFrame(results)

print("\nPortfolio Diagnostics")
print("-------------------------")
print(results_df)

print("\nRisk Contribution")
print("-------------------------")

risk_weights = results_df["Annual Volatility"] / results_df["Annual Volatility"].sum()
results_df["Risk Weight"] = risk_weights

print(results_df[["Stock", "Risk Weight"]])
