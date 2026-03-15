import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/raw_market_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Asset", "Date"])
    return df

def calculate_returns(df):
    df["Log_Return"] = df.groupby("Asset")["Close"].transform(lambda x: np.log(x / x.shift(1)))
    df["Return_1d"] = df.groupby("Asset")["Log_Return"].shift(1)
    df["Return_5d"] = df.groupby("Asset")["Log_Return"].shift(5)
    df["Return_Z"] = df.groupby("Asset")["Log_Return"].transform(lambda x: (x - x.mean()) / x.std())
    return df

def calculate_volatility(df, window = 30):
    df["Volatility"] = df.groupby("Asset")["Log_Return"].transform(lambda x: x.rolling(window).std())
    df["Volatility_60"] = df.groupby("Asset")["Log_Return"].transform(lambda x: x.rolling(60).std())
    df["Volatility_Ratio"] = df["Volatility"] / df["Volatility_60"]
    return df

def moving_averages(df):
    df["MA_50"] = df.groupby("Asset")["Close"].transform(lambda x: x.rolling(50).mean())
    df["MA_200"] = df.groupby("Asset")["Close"].transform(lambda x: x.rolling(200).mean())
    df["MA_ratio"] = df["MA_50"] / df["MA_200"]
    return df

def momentum(df):
    df["Momentum_10"] = df.groupby("Asset")["Close"].transform(lambda x: x / x.shift(10) - 1)
    return df

def missing_values(df):
    df = df.sort_values(["Asset", "Date"])
    cols = ["Open", "High", "Low", "Close", "Volume", "Log_Return", "Volatility", "MA_50", "MA_200", "Momentum_10"]
    df[cols] = df.groupby("Asset")[cols].transform(lambda x: x.ffill())
    df = df.dropna(subset = ["Log_Return", "Volatility"])
    return df

def relative_market_return(df):
    market = df[df["Asset"] == "Top40"][["Date", "Log_Return"]]
    market = market.rename(columns = {"Log_Return": "Market_Return"})
    df = df.merge(market, on = "Date", how = "left")
    df["Relative_Return"] = df["Log_Return"] - df["Market_Return"]
    return df

def rolling_sharpe(df):
    df["Rolling_Sharpe"] = df.groupby("Asset")["Log_Return"].transform(lambda x: x.rolling(30).mean() / x.rolling(30).std())
    return df

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_data()
    print("Calculating returns...")
    df = calculate_returns(df)
    print("Calculating relative market return...")
    df = relative_market_return(df)
    print("Calculating volatility...")
    df = calculate_volatility(df)
    print("Calculating moving averages...")
    df = moving_averages(df)
    print("Calculating momentum...")
    df = momentum(df)
    print("Calculating rolling Sharpe...")
    df = rolling_sharpe(df)
    print("Dropping missing values...")
    df = missing_values(df)
    print("Saving engineerd dataset...")
    df.to_csv("data/feature_engineered_data.csv", index = False)
    print("Feature engineering completed.")

