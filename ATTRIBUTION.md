# Upstream attribution and changes

Base: [mervees/E-commerce-Data-Analysis](https://github.com/mervees/E-commerce-Data-Analysis), commit `8b12a5fc1fc9888b152a4b55edb389d9931d5704`, MIT, Copyright (c) 2026 Merve Çavaş. Root LICENSE preserves the complete upstream notice and permission.

- `upstream/src/` and `upstream/README.md`: unmodified reference copies. These are not imported at app startup or training time. The original exploratory workflow is retained for comparison; its dependency list and evaluation results are not this project's results.
- `download_data.py`: upstream downloader, using the official UCI archive; subsequently hardened to extract the expected workbook only and download atomically.
- `retail/data.py`: adapted upstream loader, aliases and non-product exclusion list. Added schema validation, duplicate removal, product-code validation, cleaning audit, content-hash cache invalidation and Parquet cache. Unlike upstream, missing IDs and non-product lines are excluded consistently across all tasks.
- `train.py`: adapts upstream RFM/K-means, invoice feature engineering/Isolation Forest, daily aggregation/Holt–Winters and UK/top-300 Apriori patterns. Adds CPU bounds, persisted artifacts, rolling forecast evaluation, baselines and bounded pair mining. Customer classifier and regressor are newly implemented using temporal snapshots rather than upstream proxy labels.
- `retail/customer.py`, `app.py`, `tests/`, README and verification documentation: new work. Original contribution: historical Retention Campaign Simulator comparing model targeting and recency at equal contact budgets, with precision/recall and outcome counts.

Dataset: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Dataset license: CC BY 4.0. Data and trained outputs are excluded from Git. Dataset license is separate from the source-code MIT license.
