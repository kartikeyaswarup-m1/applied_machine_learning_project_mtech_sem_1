"""
Sales forecasting — time series.

Daily revenue is resampled to a regular frequency, decomposed (trend /
seasonal / residual), then forecast with Holt-Winters exponential smoothing.
A hold-out backtest reports RMSE / MAPE before forecasting the future.
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from . import config

warnings.filterwarnings("ignore")


def daily_series(sales: pd.DataFrame) -> pd.Series:
    s = (sales.set_index("InvoiceDate")["TotalPrice"]
         .resample("D").sum().asfreq("D").fillna(0.0))
    return s


def run(sales: pd.DataFrame):
    print("\n=== Sales Forecasting (time series) ===")
    config.set_style()
    daily = daily_series(sales)

    # Seasonal decomposition (weekly period) ---------------------------------
    decomp = seasonal_decompose(daily, model="additive", period=7)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, comp, title, color in zip(
        axes,
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal (weekly)", "Residual"],
        [config.PALETTE[0], config.PALETTE[2], config.PALETTE[5], config.PALETTE[6]],
    ):
        ax.plot(comp.index, comp.values, color=color, lw=1.2)
        ax.set_ylabel(title, fontsize=10)
    axes[0].set_title("Daily Revenue — Seasonal Decomposition")
    config.save_fig(fig, "15_timeseries_decomposition.png")

    # Backtest: hold out last 45 days ----------------------------------------
    horizon = 60
    test_n = 45
    train, test = daily.iloc[:-test_n], daily.iloc[-test_n:]
    hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                              seasonal_periods=7).fit()
    bt = hw.forecast(test_n)
    rmse = np.sqrt(mean_squared_error(test, bt))
    nz = test > 0
    mape = (np.abs((test[nz] - bt[nz]) / test[nz])).mean() * 100
    print(f"  backtest RMSE = £{rmse:,.0f} | MAPE = {mape:.1f}%")

    # Refit on all data and forecast forward ---------------------------------
    model = ExponentialSmoothing(daily, trend="add", seasonal="add",
                                 seasonal_periods=7).fit()
    fc = model.forecast(horizon)
    future_idx = pd.date_range(daily.index[-1] + pd.Timedelta(days=1),
                               periods=horizon, freq="D")

    fig, ax = plt.subplots(figsize=(13, 6))
    recent = daily.iloc[-180:]
    ax.plot(recent.index, recent.values, color=config.PALETTE[0],
            lw=1.3, label="Actual (last 180 days)")
    ax.plot(future_idx, fc.values, color=config.PALETTE[3], lw=2,
            label=f"Forecast (+{horizon} days)")
    ax.axvline(daily.index[-1], color="grey", ls="--", lw=1)
    ax.set_title("Daily Revenue Forecast — Holt-Winters")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
    ax.legend()
    config.save_fig(fig, "16_revenue_forecast.png")

    print(f"  forecast horizon: {horizon} days | "
          f"projected revenue: £{fc.sum():,.0f}")
    return fc
