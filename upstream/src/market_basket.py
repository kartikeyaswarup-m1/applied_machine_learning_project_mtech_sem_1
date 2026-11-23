"""
Market Basket Analysis — association rule mining (Apriori).

Builds a basket matrix (invoice x product) for UK product sales, mines frequent
itemsets, derives association rules, and visualises support/confidence/lift.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from . import config
from .data_loader import product_transactions


def run(sales: pd.DataFrame):
    print("\n=== Market Basket Analysis (Apriori) ===")
    config.set_style()
    prod = product_transactions(sales)

    # Focus on UK (largest market) to keep the basket matrix tractable
    uk = prod[prod["Country"] == "United Kingdom"]

    # Keep the most popular products so Apriori stays fast & meaningful
    top_items = uk["Description"].value_counts().head(300).index
    uk = uk[uk["Description"].isin(top_items)]

    basket = (uk.groupby(["Invoice", "Description"])["Quantity"]
              .sum().unstack().fillna(0))
    basket = (basket > 0)  # boolean: was the item in the basket?
    print(f"  basket matrix: {basket.shape[0]:,} invoices x {basket.shape[1]} products")

    itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.5)
    rules = rules.sort_values("lift", ascending=False)
    print(f"  frequent itemsets: {len(itemsets):,} | rules (lift>1.5): {len(rules):,}")

    if rules.empty:
        print("  No rules at this threshold — skipping rule plots.")
        return rules

    # Scatter: support vs confidence, sized/coloured by lift -----------------
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(rules["support"], rules["confidence"],
                    s=rules["lift"] * 12, c=rules["lift"],
                    cmap="viridis", alpha=0.6, edgecolor="white")
    ax.set_title("Association Rules — Support vs Confidence")
    ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
    fig.colorbar(sc, ax=ax, label="Lift")
    config.save_fig(fig, "17_association_rules_scatter.png")

    # Top 12 rules by lift ----------------------------------------------------
    top = rules.head(12).copy()

    def fmt(s):
        items = list(s)
        return " + ".join(items[:2])[:35] + ("…" if len(items) > 2 else "")

    top["label"] = top["antecedents"].apply(fmt) + "  ->  " + \
        top["consequents"].apply(fmt)
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["label"], top["lift"], color=config.PALETTE[5])
    ax.set_title("Top Product Association Rules by Lift")
    ax.set_xlabel("Lift")
    ax.tick_params(axis="y", labelsize=8)
    config.save_fig(fig, "18_top_association_rules.png")

    best = rules.iloc[0]
    print(f"  strongest rule: {set(best['antecedents'])} -> "
          f"{set(best['consequents'])} (lift={best['lift']:.1f})")
    return rules
