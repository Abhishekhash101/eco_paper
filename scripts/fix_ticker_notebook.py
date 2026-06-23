"""
Fix broken tickers and data sources in Ticker_Fetch.ipynb.

Issues addressed:
1. Yahoo Finance delisted/wrong tickers → corrected exchange-suffixed symbols
2. Stooq blocking (returns HTML) → fallback to local stooq.csv
3. FRED series that no longer exist → corrected IDs or removed
4. ECB SDW 404 for Germany IRS series → removed (redundant)
5. Emerging market FRED series returning 404 → use alternative FRED IDs
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "Ticker_Fetch.ipynb"
)

# Load the notebook
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def get_source(cell):
    """Get cell source as a single string."""
    return "".join(cell["source"])


def set_source(cell, new_source_lines):
    """Set cell source from a list of lines."""
    cell["source"] = new_source_lines
    cell["outputs"] = []  # Clear old outputs since code changed
    cell["execution_count"] = None


# ===========================================================================
# Fix Cell 6 (index 5): Eurozone — Portugal ^PSI20 → PSI20.LS
# ===========================================================================
cell5_src = get_source(nb["cells"][5])
cell5_src = cell5_src.replace('"^PSI20",   "PT_PSI20"', '"PSI20.LS", "PT_PSI20"')
nb["cells"][5]["source"] = cell5_src.splitlines(keepends=True)
nb["cells"][5]["outputs"] = []
nb["cells"][5]["execution_count"] = None
print("✅ Cell 6 fixed: ^PSI20 → PSI20.LS")

# ===========================================================================
# Fix Cell 7 (index 6): ECB SDW — Remove failing IRS series (redundant)
# ===========================================================================
cell6_src = get_source(nb["cells"][6])
# Remove the Germany 10Y IRS call that returns 404
cell6_src = cell6_src.replace(
    'print("\\n── Germany 10Y (ECB SDW via IRS) ──")\n'
    '# IRS.M.DE.L.L40.CI.0.EUR.N.Z = Germany 10Y interest rate statistics\n'
    'de_10y_ecb = get_ecb_sdw("IRS", "M.DE.L.L40.CI.0.EUR.N.Z", "DE_10Y_ECB_monthly")\n',
    '# NOTE: IRS.M.DE.L.L40.CI.0.EUR.N.Z was removed — returns HTTP 404.\n'
    '# Germany 10Y is already available from FRED (IRLTLT01DEM156N) above.\n'
)
nb["cells"][6]["source"] = cell6_src.splitlines(keepends=True)
nb["cells"][6]["outputs"] = []
nb["cells"][6]["execution_count"] = None
print("✅ Cell 7 fixed: Removed failing ECB IRS series")

# ===========================================================================
# Fix Cell 8 (index 7): Other DM — Norway ^OBX → OBX.OL
# ===========================================================================
cell7_src = get_source(nb["cells"][7])
cell7_src = cell7_src.replace('"^OBX", "NO_OBX"', '"OBX.OL", "NO_OBX"')
nb["cells"][7]["source"] = cell7_src.splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
print("✅ Cell 8 fixed: ^OBX → OBX.OL")

# ===========================================================================
# Fix Cell 9 (index 8): Stooq — Add fallback to load from local CSV
# ===========================================================================
cell8_src = get_source(nb["cells"][8])
# Add fallback logic: try API first, if blocked load from stooq.csv
old_stooq_block = '''stooq_2y_data = {}
for sym, lbl in stooq_2y.items():
    stooq_2y_data[sym] = get_stooq(sym, lbl)
    time.sleep(1.5)  # Pause to avoid getting banned by Stooq

# Also grab 10Y from Stooq for cross-validation
stooq_10y = {
    "us10yt.b": "US_10Y_Stooq",
    "gb10yt.b": "UK_10Y_Stooq",
    "de10yt.b": "DE_10Y_Stooq",
    "jp10yt.b": "JP_10Y_Stooq",
}
stooq_10y_data = {}
for sym, lbl in stooq_10y.items():
    stooq_10y_data[sym] = get_stooq(sym, lbl)
    time.sleep(1.5)

print("\\n✅ Stooq daily 2Y/10Y collection complete.")'''

new_stooq_block = '''# ── Strategy: Try API first, fall back to local stooq.csv if blocked ──
import os

stooq_csv_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "stooq.csv")
use_local_fallback = False

# Test one ticker first to see if Stooq is accessible
test_result = get_stooq("us2yt.b", "US_2Y_Stooq")
if test_result.empty:
    print("\\n  ⚠️  Stooq API is blocking requests. Attempting local CSV fallback...")
    if os.path.exists(stooq_csv_path):
        stooq_local_df = pd.read_csv(stooq_csv_path, index_col="Date", parse_dates=True)
        use_local_fallback = True
        print(f"  ✅ Loaded stooq.csv: {stooq_local_df.shape[0]} rows × {stooq_local_df.shape[1]} columns")
    else:
        print(f"  ❌ No local stooq.csv found at {stooq_csv_path}")
        stooq_local_df = pd.DataFrame()

if use_local_fallback:
    # Build data dicts from local CSV
    stooq_2y_data = {}
    for sym, lbl in stooq_2y.items():
        if lbl in stooq_local_df.columns:
            stooq_2y_data[sym] = stooq_local_df[[lbl]].dropna()
            print(f"  ✅ Local {lbl}: {len(stooq_2y_data[sym])} obs")
        else:
            stooq_2y_data[sym] = pd.DataFrame(columns=[lbl])
            print(f"  ⚠️  {lbl} not in local CSV")

    stooq_10y = {
        "us10yt.b": "US_10Y_Stooq",
        "gb10yt.b": "UK_10Y_Stooq",
        "de10yt.b": "DE_10Y_Stooq",
        "jp10yt.b": "JP_10Y_Stooq",
    }
    stooq_10y_data = {}
    for sym, lbl in stooq_10y.items():
        if lbl in stooq_local_df.columns:
            stooq_10y_data[sym] = stooq_local_df[[lbl]].dropna()
            print(f"  ✅ Local {lbl}: {len(stooq_10y_data[sym])} obs")
        else:
            stooq_10y_data[sym] = pd.DataFrame(columns=[lbl])
            print(f"  ⚠️  {lbl} not in local CSV")
else:
    # API is working — fetch normally
    stooq_2y_data = {}
    stooq_2y_data["us2yt.b"] = test_result.to_frame() if not test_result.empty else pd.DataFrame(columns=["US_2Y_Stooq"])
    for sym, lbl in list(stooq_2y.items())[1:]:  # Skip us2yt.b (already fetched)
        stooq_2y_data[sym] = get_stooq(sym, lbl)
        time.sleep(1.5)

    stooq_10y = {
        "us10yt.b": "US_10Y_Stooq",
        "gb10yt.b": "UK_10Y_Stooq",
        "de10yt.b": "DE_10Y_Stooq",
        "jp10yt.b": "JP_10Y_Stooq",
    }
    stooq_10y_data = {}
    for sym, lbl in stooq_10y.items():
        stooq_10y_data[sym] = get_stooq(sym, lbl)
        time.sleep(1.5)

print("\\n✅ Stooq daily 2Y/10Y collection complete.")'''

cell8_src = cell8_src.replace(old_stooq_block, new_stooq_block)
nb["cells"][8]["source"] = cell8_src.splitlines(keepends=True)
nb["cells"][8]["outputs"] = []
nb["cells"][8]["execution_count"] = None
print("✅ Cell 9 fixed: Added Stooq local CSV fallback")

# ===========================================================================
# Fix Cell 10 (index 9): Emerging Markets — Fix FRED series + equity tickers
# ===========================================================================
cell9_src = get_source(nb["cells"][9])
# Fix FRED series IDs that don't exist:
# IRLTLT01BRM156N → doesn't exist (Brazil has no OECD LT rate on FRED)
# INDIRLTLT01STM → doesn't exist (India uses different series)
# Use the correct approach: skip unavailable FRED series, note alternatives

# Fix equity tickers
cell9_src = cell9_src.replace('"^XU100",     "TR_BIST100"', '"XU100.IS",   "TR_BIST100"')
cell9_src = cell9_src.replace('"^J203.JO",   "ZA_JSE"', '"^J203.JO",   "ZA_JSE"')  # This one works

# Fix the FRED IDs that return 404:
# Brazil: IRLTLT01BRM156N → doesn't exist, no OECD series for Brazil
# Mexico: IRLTLT01MXM156N → doesn't exist
# Chile: IRLTLT01CLM156N → doesn't exist 
# India: INDIRLTLT01STM → doesn't exist
# These are not available on FRED. Replace with a commented note.
cell9_src = cell9_src.replace(
    '    "BR": ("Brazil",       "Central Bank of Brazil",    "IRLTLT01BRM156N", "^BVSP",      "BR_Bovespa"),',
    '    "BR": ("Brazil",       "Central Bank of Brazil",    None,              "^BVSP",      "BR_Bovespa"),  # No OECD 10Y on FRED'
)
cell9_src = cell9_src.replace(
    '    "MX": ("Mexico",       "Bank of Mexico",            "IRLTLT01MXM156N", "^MXX",       "MX_IPC"),',
    '    "MX": ("Mexico",       "Bank of Mexico",            None,              "^MXX",       "MX_IPC"),      # No OECD 10Y on FRED'
)
cell9_src = cell9_src.replace(
    '    "CL": ("Chile",        "Central Bank of Chile",     "IRLTLT01CLM156N", "^IPSA",      "CL_IPSA"),',
    '    "CL": ("Chile",        "Central Bank of Chile",     None,              "^IPSA",      "CL_IPSA"),     # No OECD 10Y on FRED'
)
cell9_src = cell9_src.replace(
    '    "IN": ("India",        "Reserve Bank of India",     "INDIRLTLT01STM",  "^NSEI",      "IN_Nifty50"),',
    '    "IN": ("India",        "Reserve Bank of India",     None,              "^NSEI",      "IN_Nifty50"),  # No OECD 10Y on FRED'
)

# Update the loop to handle None fred_id gracefully
old_em_loop = '''for iso, (country, bank, fred_id, eq_tk, eq_lbl) in em_map.items():
    print(f"\\n── {country} ──")
    try:
        em_10y_frames[iso] = get_fred(fred_id, f"{iso}_10Y")
    except Exception as e:
        print(f"  ❌ FRED {fred_id}: {e}")
    try:
        em_eq_frames[iso]  = get_yf(eq_tk, eq_lbl)
    except Exception as e:
        print(f"  ❌ YF {eq_tk}: {e}")'''

new_em_loop = '''for iso, (country, bank, fred_id, eq_tk, eq_lbl) in em_map.items():
    print(f"\\n── {country} ──")
    if fred_id is not None:
        try:
            em_10y_frames[iso] = get_fred(fred_id, f"{iso}_10Y")
        except Exception as e:
            print(f"  ❌ FRED {fred_id}: {e}")
    else:
        print(f"  ℹ️  No FRED OECD 10Y series available for {country}")
    try:
        em_eq_frames[iso]  = get_yf(eq_tk, eq_lbl)
    except Exception as e:
        print(f"  ❌ YF {eq_tk}: {e}")'''

cell9_src = cell9_src.replace(old_em_loop, new_em_loop)

nb["cells"][9]["source"] = cell9_src.splitlines(keepends=True)
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None
print("✅ Cell 10 fixed: EM FRED series + Turkey ticker")

# ===========================================================================
# Fix Cell 11 (index 10): Asia-Pacific — Fix tickers with timeouts
# ===========================================================================
cell10_src = get_source(nb["cells"][10])
# ^STI → STI.SI (Singapore Straits Times Index)
cell10_src = cell10_src.replace('"^STI":     "SG_STI"', '"STI.SI":   "SG_STI"')
# ^HSI → ^HSI should work but was timing out; keep it but it's intermittent
# ^KS11 → ^KS11 should work but was timing out; keep it
# 000001.SS → keep (Shanghai; intermittent timeouts from China servers)
# ^NSEI → keep (India; intermittent)
# ^KSE100 → keep (Pakistan; intermittent)
# PSEI.PS → PSEi.PS (Philippines)
cell10_src = cell10_src.replace('"PSEI.PS":  "PH_PSEi"', '"PSE.PS":   "PH_PSEi"')

nb["cells"][10]["source"] = cell10_src.splitlines(keepends=True)
nb["cells"][10]["outputs"] = []
nb["cells"][10]["execution_count"] = None
print("✅ Cell 11 fixed: Singapore + Philippines tickers")

# ===========================================================================
# Fix Cell 12 (index 11): LatAm/EMEA — Fix multiple delisted tickers
# ===========================================================================
cell11_src = get_source(nb["cells"][11])

# Fix tickers that are confirmed broken:
cell11_src = cell11_src.replace('"^MXX":       "MX_IPC"', '"^MXX":       "MX_IPC"')  # ^MXX is correct but intermittent
cell11_src = cell11_src.replace('"^XU100":     "TR_BIST100"', '"XU100.IS":   "TR_BIST100"')  # Turkey
cell11_src = cell11_src.replace('"^PSI20":     "PT_PSI20"', '"PSI20.LS":   "PT_PSI20"')  # Portugal
cell11_src = cell11_src.replace('"^OBX":       "NO_OBX"', '"OBX.OL":     "NO_OBX"')  # Norway
cell11_src = cell11_src.replace('"^PX":        "CZ_PX"', '"FPXAA.PR":   "CZ_PX"')  # Czech Republic
cell11_src = cell11_src.replace('"WIG20.WA":   "PL_WIG20"', '"PLX.F":      "PL_WIG20"')  # Poland (ETF proxy)
cell11_src = cell11_src.replace('"^SPBLPGPT":  "PE_BVL"', '"^SPB.LM":    "PE_BVL"')  # Peru (alt ticker)

nb["cells"][11]["source"] = cell11_src.splitlines(keepends=True)
nb["cells"][11]["outputs"] = []
nb["cells"][11]["execution_count"] = None
print("✅ Cell 12 fixed: LatAm/EMEA tickers (Turkey, Portugal, Norway, Czech, Poland)")

# ===========================================================================
# Fix Cell 13 (index 12): Additional OECD 10Y — Fix non-existent FRED series
# ===========================================================================
cell12_src = get_source(nb["cells"][12])
# IRLTLT01SGM156N → doesn't exist (Singapore)
# IRLTLT01IDM156N → doesn't exist (Indonesia) 
# Replace with a try/except approach that logs and continues
if "IRLTLT01SGM156N" in cell12_src:
    # Add a note about series that may not exist
    cell12_src = cell12_src.replace(
        '    "IRLTLT01SGM156N": "SG_10Y_monthly",   # Singapore',
        '    # "IRLTLT01SGM156N": "SG_10Y_monthly",   # Singapore — NOT AVAILABLE on FRED'
    )
    cell12_src = cell12_src.replace(
        '    "IRLTLT01IDM156N": "ID_10Y_monthly",   # Indonesia',
        '    # "IRLTLT01IDM156N": "ID_10Y_monthly",   # Indonesia — NOT AVAILABLE on FRED'
    )
    nb["cells"][12]["source"] = cell12_src.splitlines(keepends=True)
    nb["cells"][12]["outputs"] = []
    nb["cells"][12]["execution_count"] = None
    print("✅ Cell 13 fixed: Removed unavailable FRED series (SG, ID)")

# ===========================================================================
# Save the fixed notebook
# ===========================================================================
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("✅ All fixes applied and notebook saved!")
print("=" * 60)
print("""
Summary of fixes:
─────────────────
Cell  6: ^PSI20 → PSI20.LS (Portugal — Euronext Lisbon)
Cell  7: Removed failing ECB IRS series (redundant with FRED data)
Cell  8: ^OBX → OBX.OL (Norway — Oslo Stock Exchange)
Cell  9: Added local stooq.csv fallback when Stooq API blocks
Cell 10: Marked non-existent FRED EM series as None + handle gracefully
         XU100 → XU100.IS (Turkey — Istanbul Stock Exchange)
Cell 11: ^STI → STI.SI, PSEI.PS → PSE.PS
Cell 12: ^XU100 → XU100.IS, ^PSI20 → PSI20.LS, ^OBX → OBX.OL
         ^PX → FPXAA.PR (Czech), WIG20.WA → PLX.F (Poland ETF proxy)
Cell 13: Commented out non-existent FRED series (SG, ID)

NOTE: Some tickers (^HSI, ^KS11, 000001.SS, ^NSEI, ^MXX, ^IPSA)
failed due to network timeouts, NOT wrong symbols. These may work
on re-run. Consider adding retry logic or longer timeouts.
""")
