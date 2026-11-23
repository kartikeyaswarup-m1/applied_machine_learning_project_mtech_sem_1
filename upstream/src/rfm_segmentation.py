"""
RFM (Recency, Frequency, Monetary) analysis + KMeans customer segmentation.

Unsupervised ML: StandardScaler -> KMeans (k chosen via elbow + silhouette),
plus a PCA 2D projection to visualise the clusters.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from . import config


def build_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = sales.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm = rfm[rfm["Monetary"] > 0]
    return rfm


def run(sales: pd.DataFrame) -> pd.DataFrame:
    print("\n=== RFM Segmentation (KMeans) ===")
    config.set_style()
    rfm = build_rfm(sales)

    # Log-transform to tame skew, then standardise
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    Xs = StandardScaler().fit_transform(X)

    # Choose k via elbow (inertia) + silhouette ------------------------------
    ks = range(2, 9)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(ks), inertia, marker="o", color=config.PALETTE[0])
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k (clusters)"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(ks), sil, marker="o", color=config.PALETTE[3])
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("k (clusters)"); axes[1].set_ylabel("Silhouette")
    # Silhouette often peaks at k=2, but for actionable RFM segmentation we
    # select the best score within an interpretable range (3-6 segments).
    ks_list = list(ks)
    sil_arr = np.array(sil)
    cand = [i for i, k in enumerate(ks_list) if 3 <= k <= 6]
    best_k = ks_list[cand[int(np.argmax(sil_arr[cand]))]]
    axes[1].axvline(best_k, color=config.PALETTE[6], ls="--",
                    label=f"chosen k = {best_k}")
    axes[1].legend()
    config.save_fig(fig, "06_kmeans_k_selection.png")
    print(f"  chosen k = {best_k} (silhouette = {sil_arr[ks_list.index(best_k)]:.3f}; "
          f"global max at k={ks_list[int(sil_arr.argmax())]})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)
    rfm["Cluster"] = km.labels_

    # Label clusters by monetary value for readability
    rank = rfm.groupby("Cluster")["Monetary"].mean().rank(ascending=False).astype(int)
    names = {c: f"Tier {rank[c]}" for c in rank.index}
    rfm["Segment"] = rfm["Cluster"].map(names)

    # PCA 2D projection of clusters ------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(rfm["Cluster"].unique()):
        m = rfm["Cluster"] == c
        ax.scatter(coords[m.values, 0], coords[m.values, 1], s=18, alpha=0.5,
                   color=config.PALETTE[c % len(config.PALETTE)],
                   label=f"{names[c]} (n={m.sum()})")
    ax.set_title("Customer Segments — PCA Projection of RFM")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.legend(loc="best", fontsize=9)
    config.save_fig(fig, "07_customer_segments_pca.png")

    # Segment profile heatmap-ish bar chart ----------------------------------
    profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    profile = profile.sort_values("Monetary", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                              [config.PALETTE[6], config.PALETTE[5], config.PALETTE[0]]):
        ax.bar(profile.index, profile[col], color=color)
        ax.set_title(f"Avg {col} by Segment")
        ax.tick_params(axis="x", rotation=30)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
    config.save_fig(fig, "08_segment_profiles.png")

    print("  segment sizes:\n", rfm["Segment"].value_counts().to_string())
    return rfm
