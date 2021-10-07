from helpers import *
from portfolio_builder import PortfolioBuilder

pd.options.mode.chained_assignment = None  # default='warn'

## Prints Sharpe ratios for 4 years of interest
for year in range(2014, 2018):
    y_portfolio = PortfolioBuilder().init_data().fit_portfolio(year)
    
    #print(y_portfolio.get_model_accuracy())
    
    y_portfolio.print_sharpe_ratio()
