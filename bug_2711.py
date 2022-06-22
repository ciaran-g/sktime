# -*- coding: utf-8 -*-
"""Test bug 2711."""
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import Reconciler
from sktime.transformations.series.date import DateTimeFeatures
from sktime.utils._testing.hierarchical import _make_hierarchical

y_train_hier = _make_hierarchical()
X_train_hier = _make_hierarchical()

FH = [1]

steps = [
    ("daily_season", DateTimeFeatures(ts_freq="D")),
    (
        "daily_season2",
        DateTimeFeatures(manual_selection=["week_of_month", "day_of_quarter"]),
    ),
]

# forecaster_hier = NaiveForecaster()
forecaster_hier = ARIMA(trend="n")

# Version 1A: Local Vectorized (Hierarchical) Forecasts (without reconciliation)
pipe_hier_norecon = ForecastingPipeline(steps=steps + [("forecaster", forecaster_hier)])

# Version 1B: Local Vectorized (Hierarchical) Forecasts (with reconciliation)
pipe_hier_recon = pipe_hier_norecon * Reconciler(method="ols")

# Version 1A: Local Vectorized (Hierarchical) Forecasts (without reconciliation)
# This works
_ = pipe_hier_norecon.fit(y_train_hier, X_train_hier, fh=FH)

# Version 1B: Local Vectorized (Hierarchical) Forecasts (with reconciliation)
# Does not work with X_train (is it because of empty X dataframe?)
# If 1A works with X_train, shouldn't 1B also work?
pipe_hier_recon.fit(
    Aggregator().fit_transform(y_train_hier),
    Aggregator().fit_transform(X_train_hier),
    fh=FH,
)


# This works (without passing X)
# _ = pipe_hier_recon.fit(y_train_hier, fh=FH)


# steps_recon=[
#         ("aggregator", Aggregator()),
#         ("daily_season", DateTimeFeatures(ts_freq="D")),
#         ("daily_season2", DateTimeFeatures(manual_selection=["week_of_month",
# day_of_quarter"])),
# ]


# # Version 1A: Local Vectorized (Hierarchical) Forecasts (without reconciliation)
# pipe_hier_norecon = ForecastingPipeline(steps= steps_recon + [("forecaster",
# forecaster_hier)])

# # Version 1B: Local Vectorized (Hierarchical) Forecasts (with reconciliation)
# pipe_hier_recon = Aggregator() * pipe_hier_norecon * Reconciler(method="ols")

# _ = pipe_hier_recon.fit(y_train_hier, X_train_hier, fh=FH)


# steps_recon=[
#         ("aggregator", Aggregator()),
#         ("daily_season", DateTimeFeatures(ts_freq="D")),
#         ("daily_season2", DateTimeFeatures(
# manual_selection=["week_of_month", "day_of_quarter"])),
# ]

# # # Version 1A: Local Vectorized (Hierarchical) Forecasts (without reconciliation)
# pipe_hier_norecon = ForecastingPipeline(steps= steps_recon + [
# ("forecaster",  Aggregator() * forecaster_hier)
# ])

# # # Version 1B: Local Vectorized (Hierarchical) Forecasts (with reconciliation)
# pipe_hier_recon = pipe_hier_norecon * Reconciler(method="ols")

# _ = pipe_hier_recon.fit(y_train_hier, X_train_hier, fh=FH)


# # Version 1A: Local Vectorized (Hierarchical) Forecasts (without reconciliation)
# pipe_hier_norecon =   ForecastingPipeline([Aggregator(), forecaster_hier])

# _ = pipe_hier_norecon.fit(y_train_hier, X_train_hier, fh=FH)
