import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

print("Loading engineered dataset...")
data = pd.read_csv("data/feature_engineered_data.csv")

assets = data["Asset"].unique()
results = []

for asset in assets:
    print(f"Processing {asset}")
    df = data[data["Asset"] == asset].copy()
    returns = df["Log_Return"].dropna()
    window = 500
    returns_recent = returns.tail(window)
    try:
        arima = ARIMA(returns_recent, order = (1, 0, 1)).fit()
        arima_forecast = arima.predict(start = 0, end = len(df) - 1)
    except:
        arima_forecast = pd.Series([0] * len(df))

    try:
        garch = arch_model(returns_recent, vol = "Garch", p = 1, q = 1).fit(disp = "off")
        garch_vol = garch.conditional_volatility
        garch_forecast = pd.Series(garch_vol.values).reindex(range(len(df))).fillna(method = "ffill")
    except:
        garch_forecast = pd.Series([0] * len(df))

    df["ARIMA_Forecast"] = arima_forecast.values[:len(df)]
    df["GARCH_Volatility"] = garch_forecast.values[:len(df)]
    results.append(df)

final = pd.concat(results)

print("Saving statistical features...")
final.to_csv("data/statistical_features.csv", index = False)

print("Statistical modeling complete.")
