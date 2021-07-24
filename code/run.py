import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

from portfolio_builder import PortfolioBuilder

for year in range(2014, 2018):
    y_portfolio = portfolio_2017 = PortfolioBuilder().init_data().fit_portfolio(year)
    
    y_portfolio.print_sharpe_ratio()
