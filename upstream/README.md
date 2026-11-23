# 🛍️ E-Commerce Data Analysis

An end-to-end machine-learning analysis of online retail transactions, covering
**descriptive analytics, customer segmentation, churn prediction, lifetime-value
regression, demand forecasting, market-basket mining, and anomaly detection** —
all in one reproducible Python pipeline.

The pipeline runs on the **[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
dataset (UCI / Kaggle): ~1M transactions from a UK-based online retailer between
December 2009 and December 2011.

| | |
|---|---|
| **Sales rows (cleaned)** | 805,549 |
| **Customers** | 5,878 |
| **Products** | 4,631 |
| **Countries** | 41 |
| **Total revenue** | £17.7M |
| **Period** | Dec 2009 → Dec 2011 |

---

## 🤖 Machine-learning techniques

This project deliberately spans the major families of ML tooling:

| Area | Technique | Library |
|------|-----------|---------|
| **Clustering** | RFM + KMeans (elbow & silhouette), PCA projection | scikit-learn |
| **Classification** | Churn prediction — Logistic Regression, Random Forest, Gradient Boosting, **XGBoost** | scikit-learn, xgboost |
| **Regression** | Customer lifetime / future-spend — Linear, Ridge, Random Forest, **XGBoost** | scikit-learn, xgboost |
| **Time series** | Seasonal decomposition + Holt-Winters forecasting (with hold-out backtest) | statsmodels |
| **Association rules** | Market-basket analysis — Apriori | mlxtend |
| **Anomaly detection** | Isolation Forest on invoice-level features | scikit-learn |
| **Dimensionality reduction** | PCA (cluster & anomaly visualisation) | scikit-learn |

---

## 📂 Project structure

```
ecommerce-data-analysis/
├── main.py                  # runs the full pipeline end-to-end
├── download_data.py         # fetches the dataset (UCI; Kaggle alternative documented)
├── requirements.txt
├── data/
│   └── sample_data.csv      # 2,000-row sample (full data is downloaded, not committed)
├── images/                  # all generated visualisations (19 figures)
└── src/
    ├── config.py            # paths + shared plotting style
    ├── data_loader.py       # load, clean, cache
    ├── eda.py               # exploratory analysis
    ├── rfm_segmentation.py  # KMeans customer segmentation
    ├── churn_prediction.py  # classification models
    ├── clv_regression.py    # future-spend regression
    ├── sales_forecasting.py # time-series forecast
    ├── market_basket.py     # Apriori association rules
    └── anomaly_detection.py # Isolation Forest
```

---

## 🚀 Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download the dataset (~45 MB, from UCI — no credentials needed)
python download_data.py

# 3. run the full analysis (writes all figures to images/)
python main.py
```

> **Prefer Kaggle?** Drop your `kaggle.json` in `~/.kaggle/`, then
> `kaggle datasets download -d mashlyn/online-retail-ii-uci -p data --unzip`.
> See `download_data.py` for details.

The whole pipeline runs in ~30 seconds on a laptop.

---

## 📊 Selected results

### Customer segmentation (RFM + KMeans → PCA)
Four data-driven customer tiers emerge from Recency / Frequency / Monetary
features, cleanly separated in PCA space (95% of variance in 2 components).

![Customer segments](images/07_customer_segments_pca.png)

### Churn prediction
Four classifiers compared on a forward-looking churn label (did the customer
return in the final 30% of the period?). Gradient Boosting led at **AUC ≈ 0.81**.

![Churn ROC curves](images/09_churn_roc_curves.png)

### Revenue forecasting (Holt-Winters)
Daily revenue is decomposed and forecast 60 days forward, capturing the strong
weekly seasonality (backtest MAPE ≈ 24%).

![Revenue forecast](images/16_revenue_forecast.png)

### Market-basket analysis (Apriori)
Association rules surface natural product affinities — e.g. the *Green Regency
Teacup* strongly predicts the *Roses Regency Teacup* (**lift ≈ 26**).

![Association rules](images/18_top_association_rules.png)

### Anomaly detection (Isolation Forest)
Flags ~1% of invoices as unusual (bulk orders, pricing outliers) for review.

![Anomaly detection](images/19_anomaly_detection.png)

All 19 figures are in [`images/`](images/).

---

## 📝 Notes

- **Data cleaning**: cancellations (invoices starting with `C`), non-positive
  quantities/prices, and missing customer IDs are removed for the sales analysis;
  fees/postage/adjustment stock codes are excluded from product-level analysis.
- The full raw dataset is **not committed** (it's 45 MB and regenerable). A
  2,000-row sample lives in `data/sample_data.csv`.

## 📜 License

MIT — see `LICENSE`. Dataset © UCI Machine Learning Repository
(Chen, D. 2019, *Online Retail II*), CC BY 4.0.
