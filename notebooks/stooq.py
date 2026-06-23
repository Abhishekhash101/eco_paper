import pandas as pd
import requests
from io import StringIO
import time

START = "2005-01-01"
END   = "2024-01-01"

def get_stooq_local(symbol, label):
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&d1={START.replace('-','')}&d2={END.replace('-','')}&i=d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or "Date" not in r.text:
            return pd.Series(name=label)
            
        df = pd.read_csv(StringIO(r.text), index_col="Date", parse_dates=True)
        if "Close" in df.columns:
            print(f"✅ Downloaded: {symbol}")
            return df["Close"].rename(label)
    except Exception:
        pass
    return pd.Series(name=label)

# List of tickers to pull
tickers = {
    "us2yt.b": "US_2Y_Stooq", "gb2yt.b": "UK_2Y_Stooq", "de2yt.b": "DE_2Y_Stooq",
    "fr2yt.b": "FR_2Y_Stooq", "it2yt.b": "IT_2Y_Stooq", "es2yt.b": "ES_2Y_Stooq",
    "jp2yt.b": "JP_2Y_Stooq", "au2yt.b": "AU_2Y_Stooq", "ca2yt.b": "CA_2Y_Stooq",
    "ch2yt.b": "CH_2Y_Stooq", "se2yt.b": "SE_2Y_Stooq", "no2yt.b": "NO_2Y_Stooq",
    "dk2yt.b": "DK_2Y_Stooq", "nz2yt.b": "NZ_2Y_Stooq", "pl2yt.b": "PL_2Y_Stooq",
    "cz2yt.b": "CZ_2Y_Stooq", "hu2yt.b": "HU_2Y_Stooq", "br2yt.b": "BR_2Y_Stooq",
    "mx2yt.b": "MX_2Y_Stooq", "za2yt.b": "ZA_2Y_Stooq", "tr2yt.b": "TR_2Y_Stooq",
    "in2yt.b": "IN_2Y_Stooq", "kr2yt.b": "KR_2Y_Stooq", "cl2yt.b": "CL_2Y_Stooq",
    "us10yt.b": "US_10Y_Stooq", "gb10yt.b": "UK_10Y_Stooq", "de10yt.b": "DE_10Y_Stooq", "jp10yt.b": "JP_10Y_Stooq"
}

frames = []
for sym, lbl in tickers.items():
    frames.append(get_stooq_local(sym, lbl))
    time.sleep(1.5)

# Combine and save locally
local_stooq_df = pd.concat(frames, axis=1)
local_stooq_df.to_csv("stooq.csv")
print("All done. File saved as stooq_backup.csv")
