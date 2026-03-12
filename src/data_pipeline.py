import yfinance as yf
import pandas as pd


stocks = {"Naspers": "NPN.JO", "Standard_Bank": "SBK.JO", "Sasol": "SOL.JO", "Anglo_American": "AGL.JO", "Shoprite": "SHP.JO"}

macro_indicators = {"Top40": "STX40.JO", "Gold": "GC=F", "USDZAR": "ZAR=X"}

def download_data(tickers, start_date="2015-01-01"):
    all_data = []
    for name, ticker in tickers.items():
        print(f"Downloading {name}...")
        df = yf.download(ticker, start = start_date)
        df["Asset"] = name
        all_data.append(df)
    combined = pd.concat(all_data)
    return combined

if __name__ == "__main__":
    print("Downloading stock data...")
    stock_data = download_data(stocks)
    print("Downloading macro data...")
    macro_data = download_data(macro_indicators)
    print("Combining datasets...")
    full_data = pd.concat([stock_data, macro_data])
    print("Saving dataset...")
    full_data.to_csv("data/raw_market_data.csv")
    print("Data pipeline completed successfully.")
