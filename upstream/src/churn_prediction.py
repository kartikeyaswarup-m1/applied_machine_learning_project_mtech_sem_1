"""
Customer churn prediction — supervised classification.

We split the timeline: features are engineered from the first ~70% of the
period; the label is whether a customer purchased again in the final window.
Four models are trained and compared (LogReg, RandomForest, GradientBoosting,
XGBoost), with ROC curves, a confusion matrix, and feature importances.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from . import config


def build_features(sales: pd.DataFrame):
    start, end = sales["InvoiceDate"].min(), sales["InvoiceDate"].max()
    cutoff = start + (end - start) * 0.7  # 70% observation / 30% outcome window

    obs = sales[sales["InvoiceDate"] <= cutoff]
    future = sales[sales["InvoiceDate"] > cutoff]

    feat = obs.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
        AvgBasket=("TotalPrice", "mean"),
        Quantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Tenure=("InvoiceDate", lambda x: (x.max() - x.min()).days),
    )
    feat["Monetary"] = feat["Monetary"].clip(lower=0)
    returned = set(future["CustomerID"].unique())
    feat["Churned"] = (~feat.index.isin(returned)).astype(int)  # 1 = did NOT return
    return feat


def run(sales: pd.DataFrame):
    print("\n=== Churn Prediction (classification) ===")
    config.set_style()
    feat = build_features(sales)
    print(f"  customers in observation window: {len(feat):,} | "
          f"churn rate: {feat['Churned'].mean():.1%}")

    X = feat.drop(columns="Churned")
    y = feat["Churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                 subsample=0.9, eval_metric="logloss",
                                 random_state=42, n_jobs=-1),
    }

    results = {}
    fig, ax = plt.subplots(figsize=(9, 8))
    for (name, model), color in zip(models.items(), config.PALETTE):
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        results[name] = (model, auc, proba)
        print(f"  {name:<22} AUC = {auc:.3f}")

    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_title("Churn Models — ROC Curves")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    config.save_fig(fig, "09_churn_roc_curves.png")

    # Best model: confusion matrix + feature importance ----------------------
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc, best_proba = results[best_name]
    print(f"  best model: {best_name} (AUC={best_auc:.3f})")
    ypred = (best_proba >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(yte, ypred),
                           display_labels=["Retained", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    config.save_fig(fig, "10_churn_confusion_matrix.png")

    # Feature importance (tree models have it; LogReg uses |coef|)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X.columns)
    else:
        imp = pd.Series(np.abs(best_model[-1].coef_[0]), index=X.columns)
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.index, imp.values, color=config.PALETTE[5])
    ax.set_title(f"Feature Importance — {best_name}")
    config.save_fig(fig, "11_churn_feature_importance.png")

    print("\n", classification_report(yte, ypred,
                                       target_names=["Retained", "Churned"]))
    return results
