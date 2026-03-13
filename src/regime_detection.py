import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

print("Loading engineered dataset...")
data = pd.read_csv("data/feature_engineered_data.csv")

macro_assets = ["Top40", "Gold", "USDZAR"]
macro_data = data[data["Asset"].isin(macro_assets)].copy()

print("Preparing macro features...")
features = macro_data[["Log_Return", "Volatility"]].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Training Hidden Markov Model...")
model = GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 1000, random_state = 42)
model.fit(scaled_features)
regimes = model.predict(scaled_features)
macro_data = macro_data.loc[features.index]
macro_data["regime"] = regimes
regime_by_date = macro_data.groupby("Date")["regime"].first().reset_index()

print("Merging regimes back into dataset...")
data = data.merge(regime_by_date, on = "Date", how = "left")
data["regime"] = data["regime"].ffill()
data.to_csv("data/regime_data.csv", index = False)

print("Regime detection completed.")
print(data["regime"].value_counts())

