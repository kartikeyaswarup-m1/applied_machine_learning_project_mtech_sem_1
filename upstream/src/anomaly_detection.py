"""
Anomaly detection — unsupervised outlier flagging with Isolation Forest.

Flags unusual invoice-level transactions (e.g. abnormally large baskets or
prices) that may warrant review. Visualised on a 2D PCA projection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import config


def run(sales: pd.DataFrame):
    print("\n=== Anomaly Detection (Isolation Forest) ===")
    config.set_style()

    inv = sales.groupby("Invoice").agg(
        Items=("StockCode", "nunique"),
        Quantity=("Quantity", "sum"),
        Value=("TotalPrice", "sum"),
        AvgUnitPrice=("UnitPrice", "mean"),
    )
    inv = inv[inv["Value"] > 0]

    X = np.log1p(inv.clip(lower=0))
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=300)
    inv["Anomaly"] = iso.fit_predict(Xs)  # -1 = anomaly, 1 = normal
    n_anom = (inv["Anomaly"] == -1).sum()
    print(f"  flagged {n_anom:,} anomalous invoices "
          f"({n_anom/len(inv):.1%} of {len(inv):,})")

    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
    normal = inv["Anomaly"] == 1
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[normal.values, 0], coords[normal.values, 1],
               s=10, alpha=0.3, color=config.PALETTE[0], label="Normal")
    ax.scatter(coords[~normal.values, 0], coords[~normal.values, 1],
               s=40, alpha=0.8, color=config.PALETTE[6], marker="x",
               label="Anomaly")
    ax.set_title("Transaction Anomaly Detection — PCA View")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend()
    config.save_fig(fig, "19_anomaly_detection.png")

    top = inv[inv["Anomaly"] == -1].sort_values("Value", ascending=False).head(5)
    print("  largest flagged invoices by value:")
    for idx, r in top.iterrows():
        print(f"    invoice {idx}: £{r['Value']:,.0f}, "
              f"{int(r['Quantity'])} units, {int(r['Items'])} items")
    return inv
