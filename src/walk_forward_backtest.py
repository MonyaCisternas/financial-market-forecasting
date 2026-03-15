import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

print("Loading regime dataset...")
data = pd.read_csv("data/regime_data.csv")
data["Date"] = pd.to_datetime(data["Date"], errors = "coerce")
data = data.sort_values("Date")


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

target_vol = 0.10
trading_days = 252
initial_capital = 100000
weights = {}

for stock in stocks:
    df = data[data["Asset"] == stock]
    vol = df["Volatility_60"].mean()
    if vol > 0:
        weights[stock] = 1 / vol
    else:
        weights[stock] = 0

total = sum(weights.values())

for k in weights:
    weights[k] /= total

print("\nPortfolio Weights")

years = sorted(data["Date"].dt.year.unique())
portfolio_returns = []

for year in years[4:]:
    print(f"\nWalk-forward step: {year}")
    train = data[data["Date"].dt.year < year]
    test = data[data["Date"].dt.year == year]
    yearly_df = []
    for stock in stocks:
        train_stock = train[train["Asset"] == stock].dropna()
        test_stock = test[test["Asset"] == stock].dropna()
        if len(test_stock) == 0:
            continue
        X_train = train_stock[features]
        y_train = (train_stock["Log_Return"].shift(-1) > 0).astype(int)
        X_test = test_stock[features]
        xgb = XGBClassifier(
            n_estimators = 400,
            max_depth = 5,
            learning_rate = 0.05,
            subsample = 0.8,
            colsample_bytree = 0.8,
            random_state = 42,
            eval_metric = "logloss"
        )
        rf = RandomForestClassifier(
            n_estimators = 300,
            max_depth = 6,
            random_state = 42
        )
        xgb.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        prob_xgb = xgb.predict_proba(X_test)[:, 1]
        prob_rf = rf.predict_proba(X_test)[:, 1]
        probs = 0.7 * prob_xgb + 0.3 * prob_rf
        signals = np.where(probs > 0.65, 1, np.where(probs < 0.35, -1, 0))
        test_stock = test_stock.copy()
        test_stock["signal"] = signals
        vol = test_stock["Volatility_60"]
        annual_vol = vol * np.sqrt(trading_days)
        position_size = target_vol / annual_vol
        position_size = position_size.clip(0, 1.5)
        test_stock["strategy_return"] = (test_stock["signal"].shift(1) * position_size * test_stock["Log_Return"])
        test_stock["strategy_return"] *= weights[stock]
        yearly_df.append(test_stock[["Date", "strategy_return"]])
    if yearly_df:
        yearly_df = pd.concat(yearly_df)
        yearly_df = yearly_df.groupby("Date")["strategy_return"].sum()
        portfolio_returns.append(yearly_df)

portfolio_returns = pd.concat(portfolio_returns).fillna(0)
capital_curve = (1 + portfolio_returns).cumprod() * initial_capital
sharpe = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
max_drawdown = (capital_curve / capital_curve.cummax() - 1).min()

print("\nWalk-Forward Results")
print("-----------------------")
print(f"Final Capital: {capital_curve.iloc[-1]:.2f}")
print(f"Total Return: {(capital_curve.iloc[-1] / initial_capital - 1) * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe: .2f}")
print(f"Max Drawdown: {max_drawdown * 100: .2f}%")
