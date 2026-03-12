import yfinance as yf
import pandas as pd


stocks = {"Naspers": "NPN.JO", "Standard_Bank": "SBK.JO", "Sasol": "SOL.JO", "Anglo_American": "AGL.JO", "Shoprite": "SHP.JO"}

macro = {"Top40": "STX40.JO", "Gold": "GC=F", "USDZAR": "ZAR=X"}

def download_assets(asset_dict, start_date = "2015-01-01"):
    data_frames = []
    for asset_name, ticker in asset_dict.items():
        print(f"Downloading {asset_name} ({ticker})")
        df = yf.download(ticker, start = start_date)
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Asset"] = asset_name
        df = df[["Date","Open","High","Low","Close","Volume","Asset"]]
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index = True)

if __name__ == "__main__":
    print("Downloading stock data...")
    stock_data = download_assets(stocks)
    print("Downloading macro data...")
    macro_data = download_assets(macro)
    print("Combining datasets...")
    full_data = pd.concat([stock_data, macro_data], ignore_index = True)
    print("Saving dataset...")
    full_data.to_csv("data/raw_market_data.csv", index = False)
    print("Pipeline finished successfully.")
