"""
Exploratory Data Analysis — generates the core descriptive visuals.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import config
from .data_loader import product_transactions


def run(sales: pd.DataFrame):
    print("\n=== EDA ===")
    config.set_style()
    prod = product_transactions(sales)

    # 1) Monthly revenue trend ------------------------------------------------
    monthly = sales.groupby("Month")["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color=config.PALETTE[0], lw=2)
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color=config.PALETTE[0])
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue (£)")
    ax.set_xticks(range(0, len(monthly), 2))
    ax.set_xticklabels(monthly.index[::2], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M"))
    config.save_fig(fig, "01_monthly_revenue.png")

    # 2) Top 10 products by revenue ------------------------------------------
    top_prod = (prod.groupby("Description")["TotalPrice"].sum()
                .sort_values(ascending=False).head(10).iloc[::-1])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top_prod.index, top_prod.values, color=config.PALETTE[1])
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue (£)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
    config.save_fig(fig, "02_top_products.png")

    # 3) Top countries by revenue (excluding UK, which dominates) ------------
    by_country = sales.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    ex_uk = by_country.drop("United Kingdom", errors="ignore").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(ex_uk.index, ex_uk.values, color=config.PALETTE[2])
    ax.set_title("Top 10 Export Markets by Revenue (excl. United Kingdom)")
    ax.set_xlabel("Revenue (£)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
    config.save_fig(fig, "03_top_countries.png")

    # 4) Sales by day of week & hour of day ----------------------------------
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = sales.groupby("DayOfWeek")["TotalPrice"].sum().reindex(order).fillna(0)
    hour = sales.groupby("Hour")["TotalPrice"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(dow.index, dow.values, color=config.PALETTE[3])
    axes[0].set_title("Revenue by Day of Week")
    axes[0].set_xticklabels(dow.index, rotation=45, ha="right")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M"))
    axes[1].bar(hour.index, hour.values, color=config.PALETTE[5])
    axes[1].set_title("Revenue by Hour of Day")
    axes[1].set_xlabel("Hour")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M"))
    config.save_fig(fig, "04_time_patterns.png")

    # 5) Distribution of invoice (order) value -------------------------------
    invoice_val = sales.groupby("Invoice")["TotalPrice"].sum()
    invoice_val = invoice_val[invoice_val > 0]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(np.log10(invoice_val), bins=50, color=config.PALETTE[4], edgecolor="white")
    ax.set_title("Distribution of Order Value (log scale)")
    ax.set_xlabel("Order value (£, log10)")
    ax.set_ylabel("Number of orders")
    median = invoice_val.median()
    ax.axvline(np.log10(median), color=config.PALETTE[6], ls="--", lw=2,
               label=f"median £{median:,.0f}")
    ax.legend()
    config.save_fig(fig, "05_order_value_dist.png")

    print(f"  median order value: £{median:,.2f} | "
          f"mean: £{invoice_val.mean():,.2f}")
    return monthly
