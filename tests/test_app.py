"""Integration checks use locally generated artifacts; no training in app."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.parametrize('dark_mode',[False,True])
@pytest.mark.parametrize('mode',['full','smoke'])
def test_every_dashboard_page(mode, dark_mode):
    if not Path(f'artifacts/{mode}/metrics.json').exists():
        pytest.skip(f'Run {mode} training first')
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    app.sidebar.toggle[0].set_value(dark_mode).run()
    app.sidebar.selectbox[0].set_value(mode).run()
    assert not app.exception
    pages=['Overview','Customer predictions & segments','Sales forecast','Invoice review','Retention Campaign Simulator','Frequently bought together']
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert len(app.header)>0
        assert len(app.dataframe)>0, page
    app.sidebar.radio[0].set_value('Retention Campaign Simulator').run()
    for budget in (25,50,100):
        app.select_slider[0].set_value(budget).run()
        assert not app.exception
        assert app.dataframe[0].value.budget.tolist()==[budget,budget]
