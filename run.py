import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

from portfolio_builder import PortfolioBuilder

portfolio_2017 = PortfolioBuilder().init_data().fit_portfolio(2014)

print(portfolio_2017.portfolio_sharpe_ratio)
print(portfolio_2017.benchmark_sharpe_ratio)
