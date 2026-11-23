"""Presentation layer: reads precomputed results without retraining."""
from pathlib import Path
import json
import html
import altair as alt
import pandas as pd
import streamlit as st
from retail.customer import budget_metrics

st.set_page_config(page_title='LedgerLens | Retail Intelligence', page_icon='◧', layout='wide')
dark_mode = st.sidebar.toggle('Dark mode', key='dark_mode', help='Switch between light and dark appearance. Your choice stays active while you explore.')
colors = ({'bg':'#11161D','panel':'#19212B','card':'#161D26','border':'#303B49','text':'#E6EBF1','muted':'#A7B2C1','accent':'#C3CEDB'} if dark_mode else
          {'bg':'#FFFFFF','panel':'#F5F6F8','card':'#FAFBFC','border':'#E4E7EC','text':'#17212F','muted':'#657080','accent':'#334155'})
style = '''<style>
.block-container {max-width:1320px; padding:2.6rem 3rem 4rem;}
html, body, [class*="css"] {font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
h1 {font-weight:650!important; font-size:2.25rem!important; letter-spacing:-.055em; line-height:1.2!important;}
h2 {font-weight:600!important; font-size:1.3rem!important; letter-spacing:-.025em; margin-top:1.25rem!important;}
h3 {font-weight:600!important; font-size:1.04rem!important; letter-spacing:-.015em;}
p {line-height:1.65;}
[data-testid="stCaptionContainer"] {color:#657080;}
[data-testid="stSidebar"] {border-right:1px solid #E4E7EC; background:#F5F6F8;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:7px 0;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:3px;}
[data-testid="stMetric"] {background:#FAFBFC; border:1px solid #E4E7EC; border-radius:8px; padding:20px 22px;}
[data-testid="stMetricLabel"] {color:#657080; font-size:.8rem; font-weight:500;}
[data-testid="stMetricValue"] {font-size:1.95rem; font-weight:600; letter-spacing:-.045em;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:8px!important; border-color:#E4E7EC!important;}
[data-testid="stAlert"] {background:#F5F6F8; color:#334155; border:1px solid #E4E7EC; border-radius:6px;}
[data-testid="stAlert"] svg {color:#64748B;}
[data-testid="stDataFrame"] {border:1px solid #E4E7EC; border-radius:6px;}
.stButton button {border-color:#D7DCE3; border-radius:5px; color:#334155; font-size:.82rem;}
.stButton button:hover {border-color:#64748B; background:#F1F3F5; color:#17212F;}
button[data-baseweb="tab"] {font-size:.9rem;}
.brand-name {font-size:1.7rem; font-weight:700; letter-spacing:-.06em; color:#17212F; line-height:1.2;}
.brand-subtitle {font-size:.65rem; color:#64748B; letter-spacing:.16em; text-transform:uppercase; margin:8px 0 24px;}
.page-kicker {font-size:.66rem; letter-spacing:.15em; font-weight:600; color:#64748B; text-transform:uppercase; margin-bottom:14px;}
.context-strip {display:flex; gap:24px; flex-wrap:wrap; padding:12px 0; border-top:1px solid #E4E7EC; border-bottom:1px solid #E4E7EC; color:#657080; font-size:.75rem; margin:0 0 28px;}
.context-strip strong {font-weight:600; color:#334155;}
@media (max-width:700px) {.block-container {padding:1.5rem 1rem;} h1 {font-size:1.9rem!important;} .context-strip {gap:8px 18px;}}
</style>'''
for original, replacement in {'#FFFFFF':colors['bg'], '#F5F6F8':colors['panel'], '#FAFBFC':colors['card'], '#E4E7EC':colors['border'], '#D7DCE3':colors['border'], '#17212F':colors['text'], '#657080':colors['muted'], '#64748B':colors['muted'], '#334155':colors['accent'], '#F1F3F5':colors['panel']}.items():
    style = style.replace(original, replacement)
st.markdown(style, unsafe_allow_html=True)
if dark_mode:
    st.markdown('''<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#11161D; color:#E6EBF1; color-scheme:dark;}
    h1,h2,h3,h4,p,label,[data-testid="stMarkdownContainer"] {color:inherit;}
    [data-testid="stSidebar"] {color:#E6EBF1;}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] p {color:#E6EBF1!important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#A7B2C1!important;}
    [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], input, textarea {background:#19212B!important; color:#E6EBF1!important; border-color:#303B49!important;}
    [data-baseweb="popover"], [role="listbox"], [role="option"] {background:#19212B!important; color:#E6EBF1!important;}
    button, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {color:#E6EBF1;}
    .stButton button {background:#19212B;}
    [data-testid="stExpander"] {background:#161D26; border-color:#303B49;}
    [data-testid="stTabs"] [role="tab"] {color:#A7B2C1;}
    [data-testid="stTabs"] [aria-selected="true"] {color:#E6EBF1;}
    [data-testid="stSidebar"] hr, hr {border-color:#303B49;}
    </style>''', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-name">LedgerLens<span style="color:#94A3B8">.</span></div><div class="brand-subtitle">Retail intelligence</div>', unsafe_allow_html=True)
page = st.sidebar.radio('WORKSPACE', [
    'Overview', 'Customer predictions & segments', 'Sales forecast', 'Invoice review',
    'Retention Campaign Simulator', 'Frequently bought together'],
    format_func=lambda x: {'Customer predictions & segments':'Customers & segments',
        'Retention Campaign Simulator':'Retention simulator','Frequently bought together':'Product suggestions'}.get(x,x), key='navigation')
st.sidebar.divider()
mode = st.sidebar.selectbox('Data to explore', ['full', 'smoke'],
    format_func=lambda x: 'Full dataset' if x == 'full' else 'Small sample · smoke test',
    help='The sample uses 600 customers to verify the workflow quickly. Use the full dataset for your presentation.')
out = Path('artifacts') / mode
required = ['metrics.json', 'customers.parquet', 'segments.csv', 'membership.parquet',
    'forecast.csv', 'backtest.csv', 'daily.csv', 'invoices.csv', 'transactions.parquet', 'rules.csv']
if any(not (out / f).exists() for f in required):
    st.title('Let’s prepare your dashboard')
    st.warning('The selected dataset has not been prepared yet.')
    st.write('Run these commands in your project terminal, then refresh this page:')
    st.code(f'python download_data.py\npython train.py --mode {mode}', language='bash')
    st.stop()

@st.cache_data
def read_table(path, stamp):
    return pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)

def table(name):
    p = out / name
    return read_table(str(p), p.stat().st_mtime_ns)

def date(value):
    return pd.Timestamp(value).strftime('%d %b %Y')

def money(value):
    return f'£{value:,.2f}'

def show(frame, **kwargs):
    display = frame.style.set_properties(**{'background-color':colors['card'], 'color':colors['text']}) if dark_mode else frame
    st.dataframe(display, hide_index=True, width='stretch', **kwargs)

def heading(title, description, task=None):
    st.markdown('<div class="page-kicker">LedgerLens / '+html.escape(page.replace('Customer predictions & segments','Customers').replace('Retention Campaign Simulator','Retention'))+'</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)
    if task:
        st.caption(f'MACHINE LEARNING TASK · {task}')

def navigate(destination):
    st.session_state.navigation = destination


def line_chart(frame, key, unit='Revenue (£/day)'):
    """Date-only navigation preserves the baseline and revenue scale."""
    frame = frame.copy().sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.apply(pd.to_numeric, errors='coerce')
    data = frame.rename_axis('Date').reset_index().melt('Date', var_name='Series', value_name='Revenue').dropna()
    names = list(frame.columns)
    span = (frame.index.max() - frame.index.min()).days
    compact = span <= 35
    palette = ['#E0E7F0','#94A3B8','#64748B'] if dark_mode else ['#25364B','#718198','#A4AEBB']
    zoom = alt.selection_interval(name='date_zoom', bind='scales', encodings=['x'], clear='dblclick')
    chart = alt.Chart(data).mark_line(
        strokeWidth=2.2, clip=True, point=alt.OverlayMarkDef(size=40, filled=True) if compact else False
    ).encode(
        x=alt.X('Date:T', title=None, scale=alt.Scale(nice=False),
            axis=alt.Axis(format='%d %b' if compact else '%b %Y', tickCount=min(7,len(frame)) if compact else 7,
                          labelOverlap=True, labelPadding=10, grid=False)),
        y=alt.Y('Revenue:Q', title=unit, scale=alt.Scale(zero=True),
            axis=alt.Axis(format='~s', tickCount=5, titlePadding=16, gridColor=colors['border'], gridOpacity=.55)),
        color=alt.Color('Series:N', scale=alt.Scale(domain=names, range=palette[:len(names)]),
            legend=alt.Legend(title=None, orient='bottom', labelLimit=300, columns=1, symbolStrokeWidth=2.5) if len(names)>1 else None),
        strokeDash=alt.StrokeDash('Series:N', scale=alt.Scale(domain=names,range=[[1,0],[6,4],[2,3]][:len(names)]),legend=None),
        order=alt.Order('Date:T'),
        tooltip=[alt.Tooltip('Date:T',format='%d %b %Y',title='Date'),
                 alt.Tooltip('Series:N',title='Series'), alt.Tooltip('Revenue:Q',format=',.2f',title=unit)]
    ).add_params(zoom).properties(height=340, background=colors['bg']).configure_legend(
        labelColor=colors['muted'],padding=16
    ).configure_view(stroke=None).configure_axis(
        labelColor=colors['muted'],titleColor=colors['muted'],domain=False,
        tickColor=colors['border'],labelFontSize=11,titleFontSize=11
    )
    if st.button('Reset chart view',key=key+'_reset'):
        st.session_state[key+'_version'] = st.session_state.get(key+'_version',0)+1
    st.altair_chart(chart, width='stretch', theme=None, key=f"{key}_{st.session_state.get(key+'_version',0)}")
    st.caption('Scroll over the plot to zoom dates · Drag to move through time · Double-click or use Reset chart view to restore. Revenue starts at £0; zoom does not distort the value axis.')

m = json.loads((out / 'metrics.json').read_text())
last_day = pd.Timestamp(m['end_exclusive']) - pd.Timedelta(days=1)
st.sidebar.caption(f"Sales history\n\n{date(m['start'])} – {date(last_day)}")
st.sidebar.caption('All amounts in British pounds (£). Historical data, not live trading.')
st.markdown(f'<div class="context-strip"><span><strong>PERIOD</strong> &nbsp; {date(m["start"])} — {date(last_day)}</span><span><strong>CURRENCY</strong> &nbsp; GBP (£)</span><span><strong>DATA</strong> &nbsp; {"Full dataset" if mode == "full" else "Sample · 600 customers"}</span></div>', unsafe_allow_html=True)
if mode == 'smoke':
    st.warning('Sample view · Results use 600 customers and are separate from the full-data results.')

if page == 'Overview':
    heading('Retail, in perspective.', 'A clear view of customer behavior, sales performance and the decisions ahead.')
    daily = table('daily.csv')
    a,b,c = st.columns(3)
    a.metric('Product sales revenue', f"£{daily.iloc[:,1].sum():,.0f}")
    b.metric('Customers', f"{m['customers']:,}")
    c.metric('Sales lines analyzed', f"{m['rows_used']:,}")
    st.header('Analysis workspace')
    cards = [
        ('Customers & segments', 'Who might not return? How much might they spend? Which customers behave similarly?', 'Classification · Regression · Clustering'),
        ('Sales forecast', 'What could the next seven days of revenue look like?', 'Time-series forecasting'),
        ('Invoice review', 'Which unusual orders deserve a closer look?', 'Anomaly detection'),
        ('Retention simulator', 'Who would we identify with a limited contact budget?', 'Model versus recency baseline'),
        ('Product suggestions', 'Which products were often bought together?', 'Apriori association rules')]
    destinations = dict(zip([c[0] for c in cards], ['Customer predictions & segments','Sales forecast','Invoice review','Retention Campaign Simulator','Frequently bought together']))
    for row in (cards[:3],cards[3:]):
        for col,(title,description,tag) in zip(st.columns(len(row)),row):
            with col.container(border=True):
                st.subheader(title)
                st.write(description)
                st.caption(tag)
                st.button('Open analysis →', key='open_'+title, on_click=navigate, args=(destinations[title],))
    st.header('Sales over time')
    history = daily.rename(columns={daily.columns[0]:'Date',daily.columns[1]:'Product revenue (£)'})
    history['Date'] = pd.to_datetime(history.Date)
    left,right = st.columns(2)
    detail = left.selectbox('Sales detail', ['Daily','Weekly','Monthly'], index=1)
    period = right.selectbox('Time range', ['All history','Last year','Last 90 days'])
    series = history.set_index('Date')
    if period != 'All history':
        days = 365 if period == 'Last year' else 90
        series = series.loc[series.index >= series.index.max()-pd.Timedelta(days=days-1)]
    if detail != 'Daily':
        series = series.resample('W-MON' if detail == 'Weekly' else 'MS',label='left',closed='left').sum()
    unit = {'Daily':'day','Weekly':'week','Monthly':'month'}[detail]
    st.caption(f'{detail} product revenue · {date(history.Date.min()) if period == "All history" else date(series.index.min())} – {date(history.Date.max())}. '+
               ('Totals are grouped by period start; edge periods may be incomplete.' if detail != 'Daily' else 'Daily values include zero-sale days.'))
    line_chart(series, key='sales_history_'+detail+'_'+period, unit=f'Revenue (£/{unit})')
    st.info('Reading this dashboard: sales are positive product purchases by known customers. Refunds, cancellations, duplicates and non-product charges are excluded. Revenue is not profit or revenue after refunds.')
    with st.expander('Data preparation & source'):
        audit=m['cleaning']
        show(pd.DataFrame({'Step':['Original workbook rows','Cleaned sales lines'], 'Rows':[audit['raw_rows'],audit['clean_rows']]}))
        show(pd.DataFrame([{'Reason':k.replace('_',' ').capitalize(),'Rows':v} for k,v in audit['excluded_counts_overlapping'].items()]))
        st.caption('Exclusion counts overlap. The final, partial trading day is also omitted from analysis.')
        st.write('Chen, D. (2012). [Online Retail II · UCI](https://doi.org/10.24432/C5CG6D), CC BY 4.0. Based on [mervees/E-commerce-Data-Analysis](https://github.com/mervees/E-commerce-Data-Analysis), MIT. See ATTRIBUTION.md for module origins.')

elif page == 'Customer predictions & segments':
    heading('Understand your customers', 'Explore who may not return, expected spending and customer groups.', 'Classification · Regression · Clustering')
    st.info(f"Historical evaluation: predictions made on {date(m['test_cutoff'])}, checked against purchases through {date(last_day)}. Only information available before the prediction date was used.")
    customers = table('customers.parquet')
    retention, spending, segments = st.tabs(['Who may not return?', 'Expected spending', 'Customer segments'])
    with retention:
        st.header('Who may make no purchase in the next 30 days?')
        st.write('The model ranks existing customers by their risk of not buying. The simple comparison ranks customers by how long it has been since their last purchase.')
        cm=m['classification']
        a,b=st.columns(2)
        a.metric('Model ranking score · AUC',f"{cm['model_auc']:.3f}")
        b.metric('Recency baseline · AUC',f"{cm['recency_auc']:.3f}")
        st.caption('AUC measures ranking quality: 0.5 is chance; 1.0 is perfect. It does not tell us whether a retention campaign will work.')
        st.subheader('Customers with the highest model scores')
        query=st.text_input('Find a customer by ID',placeholder='Enter all or part of a customer ID')
        ranked=customers.sort_values('risk',ascending=False)
        if query: ranked=ranked[ranked.CustomerID.astype(str).str.contains(query,regex=False)]
        ranked = ranked.copy()
        ranked['nonreturn'] = ranked.nonreturn.map({1:'Yes',0:'No'})
        show(ranked[['CustomerID','risk','recency','nonreturn']].rename(columns={
            'CustomerID':'Customer ID','risk':'Risk score','recency':'Days since last purchase', 'nonreturn':'Actually made no purchase'}),
            column_config={'Risk score':st.column_config.NumberColumn(format='%.3f'),
                'Days since last purchase':st.column_config.NumberColumn(format='%.1f')})
        st.caption('Higher risk scores mean higher priority for review. These are uncalibrated model scores, not guaranteed probabilities. Actual outcomes are shown for this historical evaluation only.')
        with st.expander('More classification results & evaluation method'):
            show(pd.DataFrame({'Measure':['Average precision','Non-return prevalence'],
                'Model':[cm['model_average_precision'],cm['nonreturn_prevalence']],
                'Recency baseline':[cm['recency_average_precision'],cm['nonreturn_prevalence']]}))
            st.write('Average precision summarizes ranking across different selection sizes. Compare it with the share of customers who did not return (prevalence).')
            st.write('Training snapshot dates: '+', '.join(date(x) for x in m['train_cutoffs'])+'. Each label uses the next 30 days. Training and evaluation outcome windows do not overlap; preprocessing is fitted on training data only. The same known customer may appear at several snapshots.')
    with spending:
        st.header('How much might each customer spend?')
        st.write('Predicts product spending over the next 30 days, including £0 for customers who do not buy. The baseline repeats each customer’s previous 30-day spending.')
        a,b=st.columns(2)
        a.metric('Model average error',money(m['regression']['model_mae_gbp']))
        b.metric('Baseline average error',money(m['regression']['last30_mae_gbp']))
        st.caption('Mean absolute error (MAE): the average size of a prediction error, in £ per customer. Lower is better.')
        show(customers[['CustomerID','predicted_spend','baseline_spend','spending']].rename(columns={
            'CustomerID':'Customer ID','predicted_spend':'Model prediction (£)','baseline_spend':'Baseline prediction (£)','spending':'Actual spending (£)'}),
            column_config={c:st.column_config.NumberColumn(format='£%.2f') for c in ['Model prediction (£)','Baseline prediction (£)','Actual spending (£)']})
    with segments:
        st.header('Which customers behave similarly?')
        st.write('K-means groups customers using time since last purchase, number of orders and total spending. These groups describe the complete observed history.')
        profiles=table('segments.csv')
        friendly=profiles.rename(columns={'segment':'Group','customers':'Customers','recency_days':'Days since purchase','invoices':'Orders per customer','spend_gbp':'Total spend per customer (£)'})
        show(friendly.drop(columns=['profile'],errors='ignore').round(1))
        st.bar_chart(friendly.set_index('Group')[['Customers']], color='#A7B8CE' if dark_mode else '#475569')
        st.metric('Group separation · silhouette score',f"{m['silhouette']:.3f}")
        st.caption('Silhouette ranges from −1 to 1; higher values mean more distinct groups. Computed on up to 3,000 customers. Group numbers are arbitrary, not a value ranking.')
        st.caption(f"Profiles as of {date(m['end_exclusive'])}. Days, orders and spending are group averages. K-means uses four groups and log-transformed, standardized features.")
        with st.expander('Look up customer group membership'):
            show(table('membership.parquet').rename(columns={'CustomerID':'Customer ID','segment':'Group'}))

elif page == 'Sales forecast':
    heading('Plan the next seven days', 'Compare the sales forecast with a simple weekly pattern.', 'Time-series forecasting')
    future=table('forecast.csv')
    future['date']=pd.to_datetime(future.date)
    fm=m['forecast']
    st.header(f"Forecast · {date(future.date.min())} – {date(future.date.max())}")
    st.caption('These dates follow the historical dataset in 2011. This is not a forecast for today.')
    st.metric('Forecast product revenue · 7 days',f"£{future.holt_winters_gbp.sum():,.0f}")
    labels={'holt_winters_gbp':'Holt–Winters forecast (£)','seasonal_naive_gbp':'Repeat last week (£)','actual_gbp':'Actual sales (£)','date':'Date'}
    line_chart(future.rename(columns=labels).set_index('Date'), key='future_forecast')
    st.header('How accurate was it on unseen dates?')
    st.write(f"Tested over {date(fm['start'])} – {date(fm['end'])}, using four separate seven-day forecasts. Each forecast uses only the earlier sales history.")
    a,b=st.columns(2)
    a.metric('Holt–Winters · average daily error',money(fm['hw_mae_gbp']))
    b.metric('Repeat last week · average daily error',money(fm['naive_mae_gbp']))
    st.caption('MAE is the average prediction error in £ per day. Lower is better. Holt–Winters learns trend and weekly seasonality; the baseline repeats the previous seven days.')
    backtest=table('backtest.csv'); backtest['date']=pd.to_datetime(backtest.date)
    line_chart(backtest.rename(columns=labels).set_index('Date'), key='forecast_backtest')
    with st.expander('View forecast values & held-out results'):
        show(future.rename(columns=labels))
        show(backtest.rename(columns=labels))
    st.caption('Both forecasts are clipped at £0. Revenue excludes returns, cancellations, fees and customers without IDs.')

elif page == 'Invoice review':
    heading('Review unusual orders', 'Start with the most unusual invoices, then inspect the products in an order.', 'Anomaly detection · Isolation Forest')
    st.info('Unusual does not mean fraudulent. There are no fraud labels in this dataset; these scores help prioritize human review.')
    invoices=table('invoices.csv')
    a,b=st.columns(2)
    a.metric('Invoices analyzed',f'{len(invoices):,}')
    b.metric('Flagged for review',f'{int(invoices.flagged.sum()):,}')
    flagged_only=st.checkbox('Show flagged invoices only',value=True)
    view=invoices[invoices.flagged] if flagged_only else invoices
    st.header('Review queue')
    st.caption('Highest unusual score first. The model compares product count, quantity, invoice value and average unit price; approximately 1% are flagged by the configured threshold.')
    show(view.head(200).rename(columns={'Invoice':'Invoice ID','date':'Order date','customer':'Customer ID','items':'Distinct products',
        'quantity':'Units','value_gbp':'Order value (£)','average_unit_price_gbp':'Average line price (£)','unusual_score':'Unusual score','flagged':'Flagged'}).round(3))
    selected=st.selectbox('Choose an invoice to inspect',view.Invoice.astype(str).tolist())
    detail=invoices[invoices.Invoice.astype(str).eq(selected)].iloc[0]
    a,b,c=st.columns(3)
    a.metric('Order value',money(detail.value_gbp)); b.metric('Units',f'{detail.quantity:,.0f}'); c.metric('Distinct products',int(detail['items']))
    sales=table('transactions.parquet')
    lines=sales[sales.Invoice.astype(str).eq(selected)]
    show(lines[['Description','StockCode','Quantity','UnitPrice','Revenue']].rename(columns={
        'Description':'Product','StockCode':'Product code','Quantity':'Units','UnitPrice':'Unit price (£)','Revenue':'Line value (£)'}))

elif page == 'Retention Campaign Simulator':
    heading('Try a retention contact budget', 'Compare two ways to choose customers to contact: model risk or time since their last purchase.')
    st.caption(f"Historical predictions: {date(m['test_cutoff'])} · Purchases checked through {date(last_day)}")
    st.header('1. Choose how many customers to contact')
    k=st.select_slider('Contact budget (customers)',options=[25,50,100],value=50)
    customers=table('customers.parquet')
    results=pd.DataFrame([{**budget_metrics(customers,s,k),'method':label} for s,label in [('risk','Model'),('recency','Recency baseline')]])
    st.header('2. Compare who each method identified')
    a,b=st.columns(2)
    for col,(_,r) in zip([a,b],results.iterrows()):
        with col.container(border=True):
            st.subheader(r['method'])
            st.metric('Customers who did not return',f"{r.identified} of {r.contacted}")
            st.write(f"**{r.precision:.1%} precision** · Share of selected customers who did not return.")
            st.write(f"**{r.recall:.2%} recall** · Share of all non-returners identified.")
    difference=int(results.iloc[0].identified-results.iloc[1].identified)
    if difference == 0:
        st.write('Both methods identified the same number of non-returners.')
    else:
        st.write(f'The model identified {abs(difference)} '+('more non-returners.' if difference>0 else 'fewer non-returners.'))
    show(results.rename(columns={'precision':'precision@K','recall':'recall@K','identified':'Non-returners identified','method':'Selection method'}).round(4))
    st.info('This measures targeting quality, not the effect of contacting people. The dataset cannot show whether a message would bring customers back or increase sales.')
    with st.expander('How to read these results'):
        st.write('A non-returner made no qualifying purchase in the next 30 days. Precision@K = identified non-returners ÷ contacted customers. Recall@K = identified non-returners ÷ all non-returners in the snapshot.')
        st.write('Both methods use the same customers and contact budget. Higher scores rank first, with customer ID breaking ties. If fewer than K customers are available, the actual contacted count is used.')

else:
    heading('Discover products bought together', 'Select a product to find items that often appeared in the same shopping basket.', 'Association rules · Apriori')
    rules=table('rules.csv')
    st.header('Find a product suggestion')
    if rules.empty:
        st.info('No product pairs met the thresholds for this dataset. Try the full dataset.')
    else:
        item=st.selectbox('Which product are you interested in?',sorted(rules.item.unique()))
        selected=rules[rules.item.eq(item)].sort_values('lift',ascending=False)
        show(selected[['suggestion','support','confidence','lift']].rename(columns={
            'suggestion':'Suggested product','support':'Baskets with both','confidence':'Buyers also choosing suggestion','lift':'Association strength'}),
            column_config={'Baskets with both':st.column_config.NumberColumn(format='percent'),
                'Buyers also choosing suggestion':st.column_config.NumberColumn(format='percent'),
                'Association strength':st.column_config.NumberColumn(format='%.2f×')})
    with st.expander('What do the numbers mean?',expanded=True):
        st.write('**Support**: share of analyzed baskets containing both products. **Confidence**: among baskets with your selected product, the share also containing the suggestion. **Lift**: how much more often the pair occurs than expected if the products were chosen independently; above 1 means a positive association.')
        st.caption('UK orders, 300 most frequent descriptions, product pairs only. Support ≥2%, lift ≥1.5. Baskets include orders with at least one of these products. These historical associations are not a personalized or prospectively tested recommender.')
