"""
FX cointegration / VECM relative-value model.

Idea
----
If a basket of exchange rates (in log levels) is cointegrated, the
error-correction term beta' y is stationary. When it deviates from its
long-run mean we lean against the deviation: hold the cheap legs, short the
rich legs, weighted by the cointegrating vector beta. That answers "which
currencies to hold" at each date.

Honest limitations (read before trusting any P&L)
-------------------------------------------------
* Cointegration in FX is UNSTABLE -- it breaks at regime shifts. A fixed model
  decays out-of-sample. Use rolling re-estimation for anything real.
* This models PRICES and ignores CARRY (rollover / rate differentials), which
  often dominates FX P&L. A complete model must add it.
* Transaction / financing costs for a retail trader are large and eat thin
  edges first. The backtest applies a simple per-turnover cost; tune it up.
* Do NOT include a triangular-linked set (e.g. EURUSD + USDJPY + EURJPY): that
  "cointegration" is just the no-arb identity and is untradeable.

This is research code, not financial advice.

Requirements
------------
    pip install xbbg pandas numpy statsmodels narwhals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# ------------------------------- Config ------------------------------------
# label -> (Bloomberg ticker, invert?)   All converted to USD per 1 unit of ccy,
# so log levels are comparable across currencies.
SERIES = {
    "AUD": ("AUDUSD Curncy", False),   # already USD per AUD
    "NZD": ("NZDUSD Curncy", False),
    "CAD": ("USDCAD Curncy", True),    # CAD per USD -> invert
    "NOK": ("USDNOK Curncy", True),
}

START_DATE = "2015-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

TRAIN_FRAC = 0.70      # fit beta/mu/sigma on first 70%, evaluate on the rest
K_AR_DIFF = 1          # lags in first differences for Johansen / VECM
DET_ORDER = 0          # Johansen deterministic term: -1 none, 0 const, 1 trend
COST_BPS = 1.0         # round-trip cost per unit turnover, in basis points
OUT_CSV = "fx_vecm_signal.csv"
# ---------------------------------------------------------------------------


# --------------------------- Bloomberg (v1) --------------------------------
def to_pandas(obj) -> pd.DataFrame:
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


def _bdh_series(ticker: str, field: str = "PX_LAST") -> pd.Series:
    from xbbg import blp
    df = to_pandas(blp.bdh(ticker, field, START_DATE, END_DATE))
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


def fetch_fx() -> pd.DataFrame:
    """Daily USD value (USD per 1 unit) of each currency, one column per label."""
    cols = {}
    for label, (ticker, invert) in SERIES.items():
        s = _bdh_series(ticker)
        cols[label] = (1.0 / s) if invert else s
        print(f"  {label}: {ticker}{' (inverted)' if invert else ''} "
              f"-> {s.notna().sum()} obs")
    return pd.DataFrame(cols).dropna()
# ---------------------------------------------------------------------------


# ------------------------------- Modelling ---------------------------------
def johansen_rank(logpx: pd.DataFrame, det_order=DET_ORDER, k_ar_diff=K_AR_DIFF):
    """Cointegration rank from the Johansen trace test at 95%."""
    jres = coint_johansen(logpx.values, det_order, k_ar_diff)
    rank = 0
    for i in range(len(jres.lr1)):
        if jres.lr1[i] > jres.cvt[i, 1]:   # trace stat vs 95% critical value
            rank += 1
        else:
            break
    return rank, jres


def fit_vecm(logpx: pd.DataFrame, rank: int, k_ar_diff=K_AR_DIFF):
    model = VECM(logpx, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci")
    return model.fit()


def build_signal(logpx: pd.DataFrame, beta: np.ndarray,
                 mu: float, sigma: float) -> pd.DataFrame:
    """
    Convert the error-correction spread into per-currency weights.
    spread = beta . logpx ; z = (spread - mu)/sigma (train stats -> no look-ahead).
    Lean against the deviation: w = -z * beta, gross-normalised to 1.
    """
    spread = logpx.values @ beta
    z = (spread - mu) / sigma
    raw = -np.outer(z, beta)                       # (T, n)
    gross = np.abs(raw).sum(axis=1, keepdims=True)
    gross[gross == 0] = 1.0
    w = raw / gross
    return pd.DataFrame(w, index=logpx.index, columns=logpx.columns)


def backtest(weights: pd.DataFrame, logpx: pd.DataFrame,
             cost_bps=COST_BPS) -> tuple[pd.Series, dict]:
    """Positions set at t are held into t+1. Costs charged on turnover."""
    rets = logpx.diff().shift(-1)                  # next-day log returns
    gross_pnl = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    pnl = (gross_pnl - cost).dropna()
    ann = np.sqrt(252)
    sharpe = ann * pnl.mean() / pnl.std() if pnl.std() > 0 else np.nan
    stats = {
        "days": int(pnl.shape[0]),
        "ann_return": float(pnl.mean() * 252),
        "ann_vol": float(pnl.std() * ann),
        "sharpe_net": float(sharpe),
        "cum_return": float(pnl.sum()),
        "avg_turnover": float(turnover.mean()),
    }
    return pnl.cumsum(), stats


def run_model(logpx: pd.DataFrame) -> dict:
    """Full pipeline on a log-price panel; train/test split avoids look-ahead."""
    split = int(len(logpx) * TRAIN_FRAC)
    train, test = logpx.iloc[:split], logpx.iloc[split:]

    rank, jres = johansen_rank(train)
    print(f"Johansen trace test -> cointegration rank = {rank} (of {logpx.shape[1]})")
    if rank < 1:
        print("No cointegration in-sample. This basket won't support the model.")
        return {"rank": 0}

    vecm_res = fit_vecm(train, rank)
    beta = np.asarray(vecm_res.beta)[:, 0]         # dominant cointegrating vector
    spread_train = train.values @ beta
    mu, sigma = spread_train.mean(), spread_train.std()
    print(f"Cointegrating vector (beta): "
          f"{dict(zip(logpx.columns, np.round(beta, 3)))}")

    # Apply train-estimated beta/mu/sigma to the held-out test set
    weights = build_signal(test, beta, mu, sigma)
    curve, stats = backtest(weights, test)
    print("\nOut-of-sample backtest (costs applied, CARRY IGNORED):")
    for k, v in stats.items():
        print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")
    print("  NB: ignores carry/rollover and assumes fills at close. Treat the")
    print("      Sharpe as an optimistic upper bound, not an achievable number.")

    return {"rank": rank, "beta": beta, "weights": weights,
            "curve": curve, "stats": stats}


def main():
    print(f"Fetching daily FX {START_DATE}..{END_DATE} for {list(SERIES)} ...")
    px = fetch_fx()
    if px.shape[0] < 250:
        print("Too few observations.")
        return
    logpx = np.log(px)
    res = run_model(logpx)
    if res.get("rank", 0) >= 1:
        out = res["weights"].copy()
        out["cum_pnl"] = res["curve"]
        out.to_csv(OUT_CSV)
        print(f"\nSaved test-period weights + equity curve -> {OUT_CSV}")


if __name__ == "__main__":
    main()
