# -*- coding: utf-8 -*-
"""Test bug."""
from sktime.datatypes import get_examples
from sktime.datatypes._utilities import get_window
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.hierarchical.aggregate import Aggregator

X = get_examples("pd_multiindex_hier")[0]
y = get_examples("pd_multiindex_hier")[1]

X_train = get_window(X, lag=1)
y_train = get_window(y, lag=1)
X_test = get_window(X, window_length=1)

f = Aggregator() * ForecastingPipeline([Aggregator(), ARIMA()])
# f = ForecastingPipeline([Aggregator(), Aggregator() * ARIMA()])

f.fit(y=y_train, X=X_train, fh=1)
f.predict(X=X_test)
