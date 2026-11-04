"""Proof-of-concept demo: the equity-selection chain, end-to-end.

    data download -> features -> correlation graph -> GCN -> ranked selection

Purpose: show the MACHINERY works for the proposal meeting. The GCN is
UNTRAINED, so the scores are not predictions - making them good is the
project itself (weeks 5-6 of the plan). Data is cached under data/ after
the first pull, so reruns do not need the network.
"""

from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import torch

# torch emits a harmless FutureWarning on import under Python 3.14; hide it
# so the live-demo console stays clean
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parent.parent
CLOSE_CACHE = ROOT / "data" / "sp100_close.csv"

# ~100 largest US caps (approximates the S&P 100; the script validates and
# drops any ticker without enough history).
TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMT", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
    "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TSLA", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]
MIN_ROWS = 300      # a ticker needs this many daily closes to stay in the universe
CORR_WINDOW = 120   # trading days used to estimate the correlation graph
K_NEIGHBORS = 3     # each stock connects to its K most-correlated peers
TOP_DECILE_N = 10   # selected portfolio size


def load_close() -> pd.DataFrame:
    if CLOSE_CACHE.exists():
        print(f"[1/5] data: loaded from cache ({CLOSE_CACHE.name})")
        return pd.read_csv(CLOSE_CACHE, index_col=0, parse_dates=True)

    import yfinance as yf

    print(f"[1/5] data: downloading {len(TICKERS)} tickers from Yahoo Finance...")
    raw = yf.download(TICKERS, start="2025-01-01", auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    CLOSE_CACHE.parent.mkdir(exist_ok=True)
    close.to_csv(CLOSE_CACHE)
    print(f"      cached to {CLOSE_CACHE}")
    return close


def validate(close: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers without enough history so features and graph share one universe."""
    keep = close.columns[close.notna().sum() >= MIN_ROWS]
    dropped = [t for t in close.columns if t not in keep]
    if dropped:
        print(f"      dropped {len(dropped)} tickers (delisted or short history): {', '.join(dropped)}")
    return close[keep].dropna()


def build_features(close: pd.DataFrame) -> pd.DataFrame:
    rets = close.pct_change()
    feats = pd.DataFrame({
        "mom_20": close.iloc[-1] / close.iloc[-21] - 1,            # 1-month momentum
        "mom_60": close.iloc[-1] / close.iloc[-61] - 1,            # 3-month momentum
        "vol_60": rets.tail(60).std() * np.sqrt(252),              # annualized volatility
    })
    print(f"[2/5] features: momentum(20d,60d) + volatility(60d) for {len(feats)} stocks "
          f"({close.index[0].date()} .. {close.index[-1].date()})")
    return feats


def build_graph(close: pd.DataFrame) -> torch.Tensor:
    corr = close.pct_change().tail(CORR_WINDOW).corr()
    edges = []
    for t in corr.columns:
        peers = corr[t].drop(t).nlargest(K_NEIGHBORS).index
        edges += [[corr.columns.get_loc(t), corr.columns.get_loc(p)] for p in peers]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    n_edges = edge_index.shape[1]
    print(f"[3/5] graph: {corr.shape[0]} nodes, {n_edges} edges "
          f"(correlation kNN, k={K_NEIGHBORS}; avg degree {n_edges / corr.shape[0]:.1f})")
    return edge_index


def score_with_gcn(feats: pd.DataFrame, edge_index: torch.Tensor) -> pd.Series:
    torch.manual_seed(0)  # fixed random weights -> reproducible demo
    x = (feats - feats.mean()) / feats.std()          # standardize across stocks
    x = torch.tensor(x.values, dtype=torch.float32)

    gcn1, gcn2 = GCNConv(x.shape[1], 16), GCNConv(16, 1)
    h = torch.relu(gcn1(x, edge_index))
    scores = gcn2(h, edge_index).squeeze(-1).detach().numpy()
    print(f"[4/5] model: untrained 2-layer GCN forward pass -> 1 score per stock")
    return pd.Series(scores, index=feats.index, name="score")


def main() -> None:
    close = validate(load_close())
    feats = build_features(close)
    edge_index = build_graph(close)
    scores = score_with_gcn(feats, edge_index)

    ranked = pd.concat([scores, feats], axis=1).sort_values("score", ascending=False)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    top = ranked.head(TOP_DECILE_N)
    print(f"[5/5] selection: ranked all {len(ranked)}; top decile ({TOP_DECILE_N} stocks) selected\n")

    fmt = top[["rank", "score", "mom_20", "mom_60", "vol_60"]].copy()
    for c in fmt.columns[1:]:
        fmt[c] = fmt[c].map("{:+.3f}".format)
    print(fmt.to_string(index=True, justify="right"))

    out = ROOT / "docs" / "poc_demo_output.md"
    body = [
        "# PoC demo output - equity-selection chain, end-to-end",
        "",
        f"Generated {pd.Timestamp.now().date()} by `scripts/poc_selection_demo.py`.",
        "GCN is UNTRAINED: scores demonstrate the machinery, not prediction skill.",
        "",
        f"- Universe: {len(ranked)} stocks (approx. S&P 100), {close.index[0].date()} .. {close.index[-1].date()}",
        f"- Graph: correlation kNN over last {CORR_WINDOW} days, k={K_NEIGHBORS}",
        f"- Features: mom_20, mom_60, vol_60 (subset of the planned feature set)",
        "",
        "## Top decile (the 'selected portfolio')",
        "",
        top.to_markdown(),
        "",
        "## Full ranking",
        "",
        ranked.to_markdown(),
    ]
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"\nFull ranking written to {out}")


if __name__ == "__main__":
    main()
