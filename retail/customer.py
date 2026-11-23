import numpy as np
import pandas as pd
FEATURES = ['recency','frequency','monetary','tenure','spend30']

def snapshot(sales, cutoff):
    """Features strictly before cutoff; labels in [cutoff, cutoff+30d)."""
    cutoff = pd.Timestamp(cutoff)
    past = sales[sales.InvoiceDate < cutoff]
    g = past.groupby('CustomerID')
    x = g.agg(last=('InvoiceDate','max'), first=('InvoiceDate','min'),
              frequency=('Invoice','nunique'), monetary=('Revenue','sum'))
    x['recency'] = (cutoff-x.pop('last')).dt.total_seconds()/86400
    x['tenure'] = (cutoff-x.pop('first')).dt.total_seconds()/86400
    x['spend30'] = past[past.InvoiceDate >= cutoff-pd.Timedelta(days=30)].groupby('CustomerID').Revenue.sum().reindex(x.index, fill_value=0)
    future = sales[(sales.InvoiceDate >= cutoff) & (sales.InvoiceDate < cutoff+pd.Timedelta(days=30))]
    x['spending'] = future.groupby('CustomerID').Revenue.sum().reindex(x.index, fill_value=0)
    x['nonreturn'] = (~x.index.isin(future.CustomerID)).astype(int)
    x['cutoff'] = cutoff
    return x.reset_index()

def budget_metrics(frame, score, k):
    if k <= 0: raise ValueError('Budget must be positive')
    ranked = frame.sort_values([score,'CustomerID'], ascending=[False,True]).head(k)
    found = int(ranked.nonreturn.sum())
    total = int(frame.nonreturn.sum())
    return {'budget':k,'contacted':len(ranked),'identified':found,
            'precision':found/len(ranked) if len(ranked) else 0,
            'recall':found/total if total else 0}

def split_cutoffs(end):
    # Disjoint 30-day outcome windows; latest complete day is excluded separately.
    end = pd.Timestamp(end)
    cuts = [end-pd.Timedelta(days=d) for d in (210,180,150,120,90,60,30)]
    assert cuts[-2]+pd.Timedelta(days=30) <= cuts[-1]
    return cuts[:-1], cuts[-1]
