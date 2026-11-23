"""
Customer Lifetime Value (next-period spend) — supervised regression.

Features come from the observation window; the target is each customer's total
spend in the future window. Compares Linear, Ridge, RandomForest and XGBoost
regressors and visualises predicted vs actual plus feature importance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from . import config


def build_dataset(sales: pd.DataFrame):
    start, end = sales["InvoiceDate"].min(), sales["InvoiceDate"].max()
    cutoff = start + (end - start) * 0.7
    obs = sales[sales["InvoiceDate"] <= cutoff]
    future = sales[sales["InvoiceDate"] > cutoff]

    feat = obs.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
        AvgBasket=("TotalPrice", "mean"),
        Quantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
    )
    target = future.groupby("CustomerID")["TotalPrice"].sum()
    feat["FutureSpend"] = target.reindex(feat.index).fillna(0.0).clip(lower=0).values
    feat["Monetary"] = feat["Monetary"].clip(lower=0)
    return feat


def run(sales: pd.DataFrame):
    print("\n=== CLV / Future-Spend Prediction (regression) ===")
    config.set_style()
    data = build_dataset(sales)
    X = data.drop(columns="FutureSpend")
    y = np.log1p(data["FutureSpend"])  # log target — heavy right skew
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.9, random_state=42, n_jobs=-1),
    }

    scores = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        mae = mean_absolute_error(yte, pred)
        scores[name] = dict(model=model, r2=r2, rmse=rmse, mae=mae, pred=pred)
        print(f"  {name:<20} R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  (log-£)")

    # Model comparison bar chart ---------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(scores)
    ax.bar(names, [scores[n]["r2"] for n in names], color=config.PALETTE[:len(names)])
    ax.set_title("Regression Model Comparison (R²)")
    ax.set_ylabel("R²  (test set)")
    ax.tick_params(axis="x", rotation=15)
    for i, n in enumerate(names):
        ax.text(i, scores[n]["r2"] + 0.005, f"{scores[n]['r2']:.3f}",
                ha="center", fontsize=10)
    config.save_fig(fig, "12_clv_model_comparison.png")

    # Predicted vs actual for best model -------------------------------------
    best = max(scores, key=lambda k: scores[k]["r2"])
    pred = scores[best]["pred"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.expm1(yte), np.expm1(pred), s=14, alpha=0.4, color=config.PALETTE[0])
    lim = max(np.expm1(yte).max(), np.expm1(pred).max())
    ax.plot([0, lim], [0, lim], ls="--", color=config.PALETTE[6], lw=2)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_title(f"Predicted vs Actual Future Spend — {best}")
    ax.set_xlabel("Actual (£)"); ax.set_ylabel("Predicted (£)")
    config.save_fig(fig, "13_clv_pred_vs_actual.png")

    # Feature importance ------------------------------------------------------
    bm = scores[best]["model"]
    if hasattr(bm, "feature_importances_"):
        imp = pd.Series(bm.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(imp.index, imp.values, color=config.PALETTE[2])
        ax.set_title(f"Feature Importance — {best}")
        config.save_fig(fig, "14_clv_feature_importance.png")

    print(f"  best regressor: {best}")
    return scores
