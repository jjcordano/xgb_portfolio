from code.helpers import *
from code.portfolio_builder import PortfolioBuilder

pd.options.mode.chained_assignment = None  # default='warn'

for year in range(2014, 2018):
    y_portfolio = portfolio_2017 = PortfolioBuilder().init_data().fit_portfolio(year)
    
    y_portfolio.print_sharpe_ratio()
