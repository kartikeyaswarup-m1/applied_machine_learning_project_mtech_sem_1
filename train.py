"""CPU-only training; app reads outputs and never trains."""
import argparse
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, roc_auc_score, silhouette_score, average_precision_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules
from retail.data import load
from retail.customer import FEATURES, snapshot, split_cutoffs, budget_metrics

def run(mode):
    sales, audit = load()
    # Last date is a partial trading day. Exclude it from every analysis.
    end = sales.InvoiceDate.max().normalize()
    sales = sales[sales.InvoiceDate < end].copy()
    if mode == 'smoke':
        ids = sales.CustomerID.drop_duplicates().sample(n=min(600,sales.CustomerID.nunique()),random_state=42)
        sales = sales[sales.CustomerID.isin(ids)].copy()
    out = Path('artifacts')/mode
    out.mkdir(parents=True,exist_ok=True)
    train_cuts, test_cut = split_cutoffs(end)
    tr = pd.concat([snapshot(sales,c) for c in train_cuts],ignore_index=True)
    te = snapshot(sales,test_cut)
    print(f'{mode}: {len(sales):,} rows; {len(tr):,} training snapshots; {len(te):,} held-out customers',flush=True)
    clf = make_pipeline(StandardScaler(),HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=5,random_state=42))
    reg = make_pipeline(StandardScaler(),HistGradientBoostingRegressor(loss='squared_error',max_iter=100,max_leaf_nodes=15,l2_regularization=5,random_state=42))
    clf.fit(tr[FEATURES],tr.nonreturn)
    reg.fit(tr[FEATURES],tr.spending)
    te['risk'] = clf.predict_proba(te[FEATURES])[:,1]
    te['predicted_spend'] = np.maximum(0,reg.predict(te[FEATURES]))
    te['baseline_spend'] = te.spend30
    metrics = {'mode':mode,'seed':42,'cleaning':audit,'rows_used':len(sales),'customers':sales.CustomerID.nunique(),
      'start':str(sales.InvoiceDate.min()),'end_exclusive':str(end),'train_cutoffs':[str(c.date()) for c in train_cuts],
      'test_cutoff':str(test_cut.date()),'outcome_end_exclusive':str(end.date()),
      'classification':{'model_auc':roc_auc_score(te.nonreturn,te.risk),'recency_auc':roc_auc_score(te.nonreturn,te.recency),
       'model_average_precision':average_precision_score(te.nonreturn,te.risk),'recency_average_precision':average_precision_score(te.nonreturn,te.recency),'nonreturn_prevalence':float(te.nonreturn.mean())},
      'regression':{'model_mae_gbp':mean_absolute_error(te.spending,te.predicted_spend),'last30_mae_gbp':mean_absolute_error(te.spending,te.baseline_spend)},
      'campaign':[{**budget_metrics(te,s,k),'method':s} for k in (25,50,100) for s in ('risk','recency')]}
    te.to_parquet(out/'customers.parquet',index=False)
    # RFM descriptive analysis at end of observation (not an input to held-out models).
    rfm = snapshot(sales,end)
    z = StandardScaler().fit_transform(np.log1p(rfm[['recency','frequency','monetary']]))
    km = KMeans(n_clusters=4,n_init=10,random_state=42)
    rfm['segment'] = km.fit_predict(z)
    metrics['silhouette'] = silhouette_score(z,rfm.segment,sample_size=min(3000,len(z)),random_state=42)
    profiles = rfm.groupby('segment').agg(customers=('CustomerID','size'),recency_days=('recency','mean'),invoices=('frequency','mean'),spend_gbp=('monetary','mean'))
    profiles['profile'] = [f'Segment {i}: {r.recency_days:.0f} days since purchase, {r.invoices:.1f} invoices, £{r.spend_gbp:.0f} spend' for i,r in profiles.iterrows()]
    profiles.to_csv(out/'segments.csv')
    rfm[['CustomerID','segment']].to_parquet(out/'membership.parquet',index=False)
    daily = sales.set_index('InvoiceDate').Revenue.resample('D').sum().asfreq('D',fill_value=0)
    # Four independent seven-day blocks, with refits using only preceding dates.
    backtests=[]
    for n in (28,21,14,7):
        history=daily.iloc[:-n]; actual=daily.iloc[-n:][:7]
        hw=ExponentialSmoothing(history,trend='add',seasonal='add',seasonal_periods=7).fit().forecast(7).clip(lower=0)
        backtests.append(pd.DataFrame({'date':actual.index,'actual_gbp':actual.values,'holt_winters_gbp':hw.values,'seasonal_naive_gbp':history.iloc[-7:].values}))
    bt=pd.concat(backtests,ignore_index=True)
    metrics['forecast']={'start':str(bt.date.min().date()),'end':str(bt.date.max().date()),'hw_mae_gbp':mean_absolute_error(bt.actual_gbp,bt.holt_winters_gbp),'naive_mae_gbp':mean_absolute_error(bt.actual_gbp,bt.seasonal_naive_gbp)}
    future=ExponentialSmoothing(daily,trend='add',seasonal='add',seasonal_periods=7).fit().forecast(7).clip(lower=0)
    pd.DataFrame({'date':future.index,'holt_winters_gbp':future.values,'seasonal_naive_gbp':daily.iloc[-7:].values}).to_csv(out/'forecast.csv',index=False)
    bt.to_csv(out/'backtest.csv',index=False)
    daily.rename('revenue_gbp').to_csv(out/'daily.csv')
    inv=sales.groupby('Invoice').agg(date=('InvoiceDate','min'),customer=('CustomerID','first'),items=('StockCode','nunique'),quantity=('Quantity','sum'),value_gbp=('Revenue','sum'),average_unit_price_gbp=('UnitPrice','mean'))
    cols=['items','quantity','value_gbp','average_unit_price_gbp']
    iso=make_pipeline(StandardScaler(),IsolationForest(n_estimators=150,contamination=.01,random_state=42,n_jobs=2))
    ix=np.log1p(inv[cols]); iso.fit(ix)
    inv['unusual_score']=-iso.score_samples(ix); inv['flagged']=iso.predict(ix)==-1
    inv.sort_values('unusual_score',ascending=False).to_csv(out/'invoices.csv')
    sales.to_parquet(out/'transactions.parquet',index=False)
    # Adapted upstream Apriori: UK, top 300 descriptions; bounded pair rules for CPU memory.
    uk=sales[sales.Country.eq('United Kingdom')]
    top=uk.Description.value_counts().head(300).index
    basket=pd.crosstab(uk[uk.Description.isin(top)].Invoice,uk[uk.Description.isin(top)].Description).gt(0)
    itemsets=apriori(basket,min_support=.02,use_colnames=True,max_len=2,low_memory=True)
    rules=association_rules(itemsets,metric='lift',min_threshold=1.5) if not itemsets.empty else pd.DataFrame()
    records=[]
    for _,r in rules.iterrows():
        records.append({'item':next(iter(r.antecedents)),'suggestion':next(iter(r.consequents)),'support':r.support,'confidence':r.confidence,'lift':r.lift})
    pd.DataFrame(records,columns=['item','suggestion','support','confidence','lift']).to_csv(out/'rules.csv',index=False)
    joblib.dump({'classifier':clf,'regressor':reg,'kmeans':km,'isolation_forest':iso},out/'models.joblib')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,default=lambda x:x.item()))
    print(json.dumps(metrics,indent=2,default=lambda x:x.item()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['full','smoke'],default='full'); run(p.parse_args().mode)
