import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/feature_engineerd_data.csv")
features = data[["Log_Return", "Volatility", "Momentum_10", "MA_50", "MA_200"]].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 1000, random_state = 42)
model.fit(scaled_features)
regimes = model.predict(scaled_features)
data.loc[features.index, "regime"] = regimes
data.to_csv("data/regime_data.csv", index = False)

print("Market regimes detected and saved.")
print(data["regime"].value_counts())

