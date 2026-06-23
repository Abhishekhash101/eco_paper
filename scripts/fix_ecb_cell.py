"""Fix the ECB SDW cell to add retry logic and longer timeout."""
import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "Ticker_Fetch.ipynb"
)

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = [
    '# ## Cell 6 \u2014 ECB Statistical Data Warehouse (Daily Bond Yields)\n',
    '# ECB SDW provides **daily** Eurozone member bond yields.\n',
    '# API docs: https://data-api.ecb.europa.eu/help/\n',
    '# Key dataset: IRS (Interest Rate Statistics) or YC (Yield Curve).\n',
    '\n',
    'import requests\n',
    'import time as _time\n',
    '\n',
    'def get_ecb_sdw(flow, key, label, max_retries=3, timeout=90):\n',
    '    """\n',
    '    Fetch daily data from ECB Statistical Data Warehouse (free, no key).\n',
    '    flow: e.g. \'IRS\' | key: e.g. \'M.DE.L.L40.CI.0.EUR.N.Z\'\n',
    '    Returns a DataFrame with DatetimeIndex.\n',
    '    Includes retry logic for timeouts (ECB can be slow).\n',
    '    """\n',
    '    url = (\n',
    '        f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"\n',
    '        f"?format=csvdata&startPeriod={START[:10]}&endPeriod={END[:10]}"\n',
    '    )\n',
    '    \n',
    '    for attempt in range(1, max_retries + 1):\n',
    '        try:\n',
    '            r = requests.get(url, timeout=timeout)\n',
    '            if r.status_code != 200:\n',
    '                print(f"  \u274c ECB SDW {key}: HTTP {r.status_code}")\n',
    '                return pd.DataFrame()\n',
    '            from io import StringIO\n',
    '            df = pd.read_csv(StringIO(r.text))\n',
    '            if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:\n',
    '                print(f"  \u274c ECB SDW {key}: unexpected columns {df.columns.tolist()}")\n',
    '                return pd.DataFrame()\n',
    '            df["Date"] = pd.to_datetime(df["TIME_PERIOD"])\n',
    '            df = df.set_index("Date")[["OBS_VALUE"]].rename(columns={"OBS_VALUE": label})\n',
    '            df = df.dropna()\n',
    '            print(f"  \u2705 ECB SDW {key}: {len(df)} obs | {df.index[0].date()} \u2192 {df.index[-1].date()}")\n',
    '            return df\n',
    '        except requests.exceptions.Timeout:\n',
    '            if attempt < max_retries:\n',
    '                wait = attempt * 15\n',
    '                print(f"  \u26a0\ufe0f  ECB SDW {key}: Timeout (attempt {attempt}/{max_retries}), retrying in {wait}s...")\n',
    '                _time.sleep(wait)\n',
    '            else:\n',
    '                print(f"  \u274c ECB SDW {key}: Timed out after {max_retries} attempts (server may be down)")\n',
    '                return pd.DataFrame()\n',
    '        except Exception as e:\n',
    '            print(f"  \u274c ECB SDW {key}: {e}")\n',
    '            return pd.DataFrame()\n',
    '\n',
    '# ECB Yield Curve \u2014 AAA Euro area government bonds\n',
    '# Dataset: YC | Series: B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y (2Y spot rate)\n',
    '# and B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y (10Y spot rate)\n',
    'print("\u2500\u2500 Euro Area AAA Government Bond Yield Curve \u2500\u2500")\n',
    'ecb_2y  = get_ecb_sdw("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",  "EA_2Y_ECB")\n',
    'ecb_10y = get_ecb_sdw("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "EA_10Y_ECB")\n',
    '\n',
    '# NOTE: IRS.M.DE.L.L40.CI.0.EUR.N.Z was removed \u2014 returns HTTP 404.\n',
    '# Germany 10Y is already available from FRED (IRLTLT01DEM156N) above.\n',
    '\n',
    'if not ecb_2y.empty:\n',
    '    print(ecb_2y.tail(3))\n',
]

nb["cells"][6]["source"] = new_source
nb["cells"][6]["outputs"] = []
nb["cells"][6]["execution_count"] = None

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\u2705 ECB cell updated: timeout=90s, max_retries=3, with exponential backoff")
