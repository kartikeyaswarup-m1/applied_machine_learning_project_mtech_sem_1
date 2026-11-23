"""Adapted from upstream/src/data_loader.py; see ATTRIBUTION.md."""
from pathlib import Path
import hashlib
import json
import pandas as pd
NON_PRODUCT_CODES = {'BANKCHARGES', 'B', 'D', 'C2', 'GIFT', 'POST', 'TEST002', 'S', 'M', 'CRUK', 'gift_0001', 'DOT', 'BANK CHARGES', 'ADJUST', 'AMAZONFEE', 'TEST001', 'ADJUST2', 'PADS'}

VERSION = '3'
def load():
    path = Path('data/online_retail_II.xlsx')
    if not path.exists():
        raise FileNotFoundError('Run python download_data.py first')
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = Path('data/clean.parquet')
    audit = Path('data/cleaning.json')
    if cache.exists() and audit.exists():
        meta = json.loads(audit.read_text())
        if meta.get('sha256') == signature and meta.get('version') == VERSION:
            return pd.read_parquet(cache), meta
    print('Parsing all Excel sheets (cached after this run)', flush=True)
    raw = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    raw = raw.rename(columns={'Customer ID':'CustomerID','Price':'UnitPrice','InvoiceNo':'Invoice'})
    required = {'Invoice','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'}
    if required - set(raw):
        raise ValueError(f'Missing columns: {required-set(raw)}')
    df = raw.copy()
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    for col in ['Quantity','UnitPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Invoice','StockCode']:
        df[col] = df[col].astype('string').str.strip().str.upper()
    df['Description'] = df.Description.astype('string').str.strip()
    reasons = {
        'duplicate': df.duplicated(),
        'invalid_required': df[list(required)].isna().any(axis=1),
        'cancellation': df.Invoice.str.startswith('C').fillna(False),
        'nonpositive_quantity': df.Quantity.le(0),
        'nonpositive_price': df.UnitPrice.le(0),
        'nonproduct': df.StockCode.isin({s.upper() for s in NON_PRODUCT_CODES}) | ~df.StockCode.str.match(r'^\d{5}[A-Z]*$', na=False),
    }
    excluded = pd.Series(False, index=df.index)
    for mask in reasons.values(): excluded |= mask
    sales = df.loc[~excluded].copy()
    sales['CustomerID'] = sales.CustomerID.astype(int).astype(str)
    sales['Revenue'] = sales.Quantity * sales.UnitPrice
    sales = sales.sort_values('InvoiceDate').reset_index(drop=True)
    meta = {'version':VERSION,'sha256':signature,'raw_rows':len(raw),'clean_rows':len(sales),
            'excluded_counts_overlapping':{k:int(v.sum()) for k,v in reasons.items()}}
    sales.to_parquet(cache, index=False)
    audit.write_text(json.dumps(meta, indent=2))
    return sales, meta
