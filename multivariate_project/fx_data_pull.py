"""
Standalone Bloomberg pull: daily FX closes for a currency basket.
No modelling -- just fetches and saves. Built for xbbg >= 1.0.

Output
------
  fx_raw_quotes.csv   : raw PX_LAST per Bloomberg ticker (unmodified)
  fx_usd_panel.csv    : same data converted to USD per 1 unit of each currency
                        (USDxxx pairs inverted) so log levels are comparable

Rows are NOT dropped -- NaNs from unequal histories / EM gaps are preserved
for you to clean downstream.

Requirements
------------
    pip install xbbg pandas narwhals
    A logged-in Bloomberg Terminal on the same machine.
"""

from __future__ import annotations

import pandas as pd

# ------------------------------- Config ------------------------------------
# label -> (Bloomberg ticker, invert?)  -> all become USD per 1 unit of ccy.
# Commodity / risk bloc (coherent theme). Swap wholesale for another theme
# rather than mixing -- e.g. the European set below.
SERIES = {
    "AUD": ("AUDUSD Curncy", False),
    "NZD": ("NZDUSD Curncy", False),
    "CAD": ("USDCAD Curncy", True),
    "NOK": ("USDNOK Curncy", True),
    "SEK": ("USDSEK Curncy", True),
    "ZAR": ("USDZAR Curncy", True),
    "MXN": ("USDMXN Curncy", True),
}

# Alternative European bloc (uncomment to use instead):
# SERIES = {
#     "EUR": ("EURUSD Curncy", False),
#     "GBP": ("GBPUSD Curncy", False),
#     "CHF": ("USDCHF Curncy", True),
#     "SEK": ("USDSEK Curncy", True),
#     "NOK": ("USDNOK Curncy", True),
#     "PLN": ("USDPLN Curncy", True),
#     "HUF": ("USDHUF Curncy", True),
# }

FIELD = "PX_LAST"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")

RAW_CSV = "fx_raw_quotes.csv"
USD_CSV = "fx_usd_panel.csv"
# ---------------------------------------------------------------------------


def to_pandas(obj) -> pd.DataFrame:
    """Coerce xbbg v1 output (Narwhals / Polars / PyArrow / pandas) to pandas."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    import narwhals as nw
    return nw.from_native(obj, eager_only=True).to_pandas()


def _find_col(df, *names, contains=None):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if contains:
        for c in df.columns:
            if contains.lower() in str(c).lower():
                return c
    return None


def _bdh_series(ticker: str, field: str = FIELD) -> pd.Series:
    """Daily close for one ticker -> a date-indexed Series (handles v1 long)."""
    from xbbg import blp
    df = to_pandas(blp.bdh(ticker, field, START_DATE, END_DATE))
    if df.empty:
        return pd.Series(dtype="float64")
    val = _find_col(df, "value", "values")
    if val is not None:                                   # v1 long
        dcol = _find_col(df, "date", "time", "index")
        if dcol is None:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    dcol = c
                    break
        s = df.set_index(dcol)[val]
    else:                                                 # already wide
        s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index().astype(float)


def main():
    print(f"Fetching {FIELD} {START_DATE}..{END_DATE} for {list(SERIES)} ...")
    raw, usd = {}, {}
    for label, (ticker, invert) in SERIES.items():
        s = _bdh_series(ticker)
        raw[ticker] = s
        usd[label] = (1.0 / s) if invert else s
        print(f"  {label:4s} {ticker:16s}{' (inverted)' if invert else ''}"
              f"  {int(s.notna().sum())} obs")

    raw_df = pd.DataFrame(raw).sort_index()        # outer join, NaNs preserved
    usd_df = pd.DataFrame(usd).sort_index()

    raw_df.to_csv(RAW_CSV)
    usd_df.to_csv(USD_CSV)
    print(f"\nSaved raw quotes  -> {RAW_CSV}  ({raw_df.shape[0]} rows x {raw_df.shape[1]} cols)")
    print(f"Saved USD panel   -> {USD_CSV}  ({usd_df.shape[0]} rows x {usd_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
