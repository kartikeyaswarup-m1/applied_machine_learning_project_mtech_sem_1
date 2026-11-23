# LedgerLens — Retail Intelligence Dashboard

Local Python + Streamlit project for an M.Tech machine learning elective. Runs on Apple Silicon CPU, with no API key, paid service or cloud infrastructure. Five ML tasks, temporal customer evaluation, baseline comparisons, a retention targeting simulator and Apriori product suggestions.

## Setup and exact macOS commands

Use Python 3.11 (the tested version). From Terminal:

```bash
cd "/Users/kartikeyaswarupsharma/Documents/ChatGPT/Applied Machine Learning Project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python download_data.py
python train.py --mode full
python -m pytest -q
python -m streamlit run app.py
```

Open the localhost URL printed by Streamlit. `requirements-lock.txt` captures the versions actually installed for verification; `requirements.txt` records the direct dependency ranges. No GPU or XGBoost is required. Training limits common numerical thread pools to two threads; Isolation Forest uses two jobs.

For a fast model smoke test after the first Excel parse:

```bash
source .venv/bin/activate
python train.py --mode smoke
python -m streamlit run app.py
```

Select **smoke** in the sidebar. This uses 600 deterministically sampled customers across the entire observation period, preserving their baskets and chronology. `artifacts/smoke/` and `artifacts/full/` are separate. Smoke results must not be presented as full-data estimates. The first run in either mode still downloads and parses both official workbook sheets; subsequent runs read the cached Parquet data. A SHA-256 plus cleaning-version check invalidates stale caches. To rebuild cleaning deliberately, remove `data/clean.parquet` and `data/cleaning.json`, or increment the cleaning version after a logic change.

Training also performs evaluation and writes `metrics.json`; no second evaluation command is needed. `python -m pytest -q` runs focused checks plus app integration when full artifacts exist. App integration is explicitly skipped when they are absent. App startup only reads precomputed data and displays an actionable message if artifacts are missing.

## Data and cleaning policy

Official source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Both sheets are concatenated. Required columns: Invoice, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country. Original `Customer ID` and `Price` aliases are normalized. Invalid dates/numbers or missing required fields are removed. Exact duplicate rows across the combined sheets are removed. Customer IDs become stable strings.

Cancellation invoices (case-insensitive C prefix), nonpositive quantities (including returns), and nonpositive prices are excluded. Missing customer IDs are excluded from **all** analyses for a consistent customer-linked sales population. Non-product charges use the upstream exclusion list plus a stock-code policy requiring five digits and optional letter suffixes. This conservative policy can omit legitimate unusual codes. Missing descriptions and countries are excluded. Cleaning reason counts overlap and are reported in the dashboard and `data/cleaning.json`.

Revenue is quantity × unit price in **GBP**, positive product sales after those exclusions. It is **not net sales, profit, or spend after refunds**. Returns are not matched to prior purchases. Anonymous sales and fees do not contribute to the forecasts. The last observed date is omitted everywhere because it is only a partial trading day. Missing calendar days are filled with zero sales, which assumes no recorded transactions means zero rather than a reporting outage.

## Modeling and evaluation

- **Classification:** HistGradientBoostingClassifier predicts no qualifying purchase in the following 30 days. It is compared with descending days since last purchase using ROC AUC, average precision, and precision/recall at campaign budgets. Non-return is a window-specific outcome, not permanent churn.
- **Regression:** HistGradientBoostingRegressor predicts the following 30-day product spending, including zeros for non-buyers. Predictions are clipped at zero. MAE in £/customer is compared with repeating the previous 30-day spend.
- **Forecasting:** additive Holt–Winters, weekly seasonality, versus seasonal naive repeating the last seven days. Four expanding-history backtests each forecast seven untouched days; reported MAE aggregates those 28 daily predictions (£/day). Refit on all complete days for the seven-day future forecast. The future is in 2011, not today. Both methods are clipped at zero.
- **Anomaly detection:** Isolation Forest on standardized log1p invoice item counts, quantities, total value and mean line unit price. Scores rank unusual invoices; configured contamination flags approximately 1%. Original invoice values and underlying product rows are inspectable. No fraud labels or anomaly accuracy claims.
- **Clustering:** K-means with four clusters on standardized log1p recency/frequency/monetary values at observation end. Reports silhouette on at most 3,000 customers and profiles with counts, mean days since purchase, invoices and cumulative £ spend. This is descriptive, without held-out cluster labels; four segments are a fixed demonstration choice.
- **Frequently bought together:** adapted upstream Apriori on UK invoices and top 300 descriptions. Minimum support 2%, minimum lift 1.5, pair itemsets only to bound CPU/memory. User selects one product and sees suggestions, support, confidence and lift. Denominator is invoices containing one or more top products. Rules describe historical association and have no prospective recommendation evaluation.

Customer snapshots include only customers with at least one purchase strictly **before** cutoff. Features are recency (fractional days), historical distinct invoice count, cumulative spending, tenure in days and last-30-day spending. Labels use `[cutoff, cutoff+30 days)`. Features never use future rows. Six earlier cutoffs form training; the seventh is held out. Each cutoff is 30 days apart, so outcome windows do not overlap within or across splits. All scaler fitting and model fitting use training data only. There is no tuning on test results. Existing customers may appear at multiple cutoffs; this measures predicting known customers, not generalizing to entirely unseen people.

The **Retention Campaign Simulator** is the original contribution. Budgets of 25, 50 and 100 contact slots compare model risk ranking with descending recency on the same held-out snapshot. Precision@K = identified eventual non-returners / actual contacted; recall@K = identified / all eventual non-returners. Ties use customer ID. **This evaluates targeting quality. The data cannot establish whether contact changes behavior or estimate campaign uplift.**

## Structure and generated outputs

```text
app.py                  Streamlit pages; no training at startup
train.py                Fit, evaluate and persist all tasks
retail/data.py          Schema checks, cleaning, content-hash cache
retail/customer.py      Historical snapshots, splits, budget metrics
download_data.py        Adapted upstream official UCI downloader
tests/                  Focused semantics and Streamlit integration
upstream/               Unmodified reference source and README
ATTRIBUTION.md          Module-by-module origin and changes
LICENSE                 Preserved upstream MIT license
requirements-lock.txt   Tested environment versions
```

Ignored local outputs: `data/` (archive, workbook, clean Parquet and audit); `artifacts/{full,smoke}/` (metrics, models, held-out customers, segment profiles/membership, daily sales, forecast/backtest, ranked invoices, product transactions, association rules); `reports/` for generated reports. Do not commit these or load untrusted pickle/joblib artifacts. Streamlit does not need to unpickle trained models.

## Attribution and limitations

Based on [mervees/E-commerce-Data-Analysis](https://github.com/mervees/E-commerce-Data-Analysis), MIT, Copyright (c) 2026 Merve Çavaş, commit `8b12a5fc1fc9888b152a4b55edb389d9931d5704`. Original source is preserved under `upstream/`; see [ATTRIBUTION.md](ATTRIBUTION.md) for reused and added modules. Upstream published metrics are not reused as our evaluation results.

Dataset citation: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D). Dataset licensed CC BY 4.0; source code MIT.

Historical, single-retailer data includes wholesale purchases and strong seasonality. Exclusions change the population; late-year evaluation is not representative of every season. Only one customer holdout and four weekly forecast folds are used. Uncalibrated model scores, extreme spending, repeated customers and distribution drift limit generalization. Models are demonstrations, not validated operational decision systems. Baselines can outperform learned models; measured results are displayed without assuming improvement. See `reports/verification.md` for actual runs and results.
