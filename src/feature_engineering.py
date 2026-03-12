import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/raw_market_data.csv", index_col = 0)
    df.reset_index(inplace = True)
    df.rename(columns = {"index": "Date"}, inplace = True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Asset", "Date"])
    return df

def calculate_returns(df):
    df["Log_Return"] = np.log(df["Close"] / df.groupby("Asset")["Close"].shift(1))
    return df

def calculate_volatility(df, window = 30):
    df["Volatility"] = df.groupby("Asset")["Log_Return"].rolling(window).std().reset_index(level = 0, drop = True)
    return df

def moving_averages(df):
    df["MA_50"] = df.groupby("Asset")["Close"].transform(lambda x: x.rolling(50).mean())
    df["MA_200"] = df.groupby("Asset")["Close"].transform(lambda x: x.rolling(200).mean())
    return df

def momentum(df):
    df["Momentum_10"] = df.groupby("Asset")["Close"].transform(lambda x: x / x.shift(10) - 1)
    return df

def missing_values(df):
    df = df.groupby("Asset").ffill()
    df = df.dropna(subset = ["Log_Return", "Volatility"])
    return df

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_data()
    print("Calculating returns...")
    df = calculate_returns(df)
    print("Calculating volatility...")
    df = calculate_volatility(df)
    print("Calculating moving averages...")
    df = moving_averages(df)
    print("Calculating momentum...")
    df = momentum(df)
    print("Dropping missing values...")
    df = missing_values(df)
    print("Saving engineerd dataset...")
    df.to_csv("data/feature_engineerd_data.csv", index = False)
    print("Feature engineering completed.")

