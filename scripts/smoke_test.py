"""Smoke test: verify venv, PyG convolutions, and yfinance download."""

import sys

import numpy as np
import pandas as pd
import torch

print(f"Python : {sys.version.split()[0]}")
print(f"torch  : {torch.__version__}")
print(f"numpy  : {np.__version__}")
print(f"pandas : {pd.__version__}")

# --- PyG GCN/GAT forward pass (native scatter, no compiled extensions) ---
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv

print(f"pyg    : {torch_geometric.__version__}")

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 8)

out = GCNConv(8, 4)(x, edge_index)
assert out.shape == (3, 4), "GCNConv shape mismatch"
out = GATConv(8, 4, heads=2)(x, edge_index)
assert out.shape == (3, 8), "GATConv shape mismatch"
print("PyG convolutions OK")

# --- yfinance download ---
import yfinance as yf

data = yf.download(
    ["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
    auto_adjust=True, progress=False,
)
assert not data.empty, "yfinance returned empty frame"
print(f"yfinance OK: {data['Close'].shape[0]} rows, columns={list(data.columns.levels[0])}")

print("\nALL CHECKS PASSED")
