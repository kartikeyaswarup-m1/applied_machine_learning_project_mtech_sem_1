import pandas as pd
import pytest
from retail.customer import snapshot, split_cutoffs, budget_metrics

def fixture():
    return pd.DataFrame({'CustomerID':['a','a','a','b','c','a'],
      'InvoiceDate':pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-01-10','2020-02-10','2020-03-02']),
      'Invoice':['1','2','3','4','5','6'],'Revenue':[10,20,30,40,50,60]})

def test_label_boundaries_and_zero_spend():
    x=snapshot(fixture(),'2020-02-01').set_index('CustomerID')
    assert set(x.index)=={'a','b'} # future-only customer excluded
    assert x.loc['a','monetary']==10
    assert x.loc['a','spending']==50 # Feb 1 and Mar 1 included; Mar 2 excluded
    assert x.loc['a','nonreturn']==0
    assert x.loc['b','spending']==0 and x.loc['b','nonreturn']==1

def test_future_changes_cannot_change_features():
    a=fixture(); b=a.copy(); b.loc[b.InvoiceDate.ge('2020-02-01'),'Revenue']=999
    cols=['CustomerID','recency','frequency','monetary','tenure','spend30']
    pd.testing.assert_frame_equal(snapshot(a,'2020-02-01')[cols],snapshot(b,'2020-02-01')[cols])

def test_nonoverlapping_outcomes():
    train,test=split_cutoffs('2011-12-09')
    for left,right in zip(train+[test],(train+[test])[1:]):
        assert left+pd.Timedelta(days=30)<=right
    assert max(train)+pd.Timedelta(days=30)<=test
    assert test+pd.Timedelta(days=30)==pd.Timestamp('2011-12-09')

def test_budget_metrics():
    x=pd.DataFrame({'CustomerID':['a','b','c'],'score':[3,2,1],'nonreturn':[1,0,1]})
    r=budget_metrics(x,'score',2)
    assert r['identified']==1 and r['precision']==.5 and r['recall']==.5
    assert budget_metrics(x,'score',100)['contacted']==3
    assert budget_metrics(x.assign(nonreturn=0),'score',2)['recall']==0
    with pytest.raises(ValueError): budget_metrics(x,'score',0)
