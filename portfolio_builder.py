
import numpy as np
import pandas as pd
import xgboost

from helpers import *

class PortfolioBuilder:
    def __init__(self,
                 benchmark = 'eurostoxx_600',
                 probability_weighted = True,
                 bucket_delimiters = (-.15,.18),
                 short_limit = 10):
        
        assert benchmark in ['eurostoxx_600','russell_3000'], "Benchmark argument not valid. Must either be 'eurostoxx_600' or 'russell_3000'."

        self.benchmark = benchmark
        self.proba_weighted = probability_weighted
        self.bucket_delimiters = bucket_delimiters
        self.short_limit = short_limit

        self.tickers = None
        self.data = None
        self.prices = None
        self.daily_benchmark_levels = None
        
        self.dict_predictions = None
        self.dict_weights = None
        
        self.portfolio_return = None
        self.portfolio_stdev = None
        self.portfolio_sharpe_ratio = None
        self.benchmark_sharpe_ratio = None
    
    def init_data(self,
                  esg_metrics_path = PATH_DATAFOLDER + PATH_ESG_METRICS, 
                  prices_path = PATH_DATAFOLDER + PATH_PRICES):
        '''
        Function that collects and groups required data to build the desired portfolio 
        along the lines of the 2020 paper 'Seeking Signals from ESG Data' by Bloomberg Quant Research.

        Arguments:

        - esg_metrics_path: path to CSV file containing Bloomberg ESG metrics for selected companies.  
        Default is path to ESG data for 46 european companies between 2007 and 2017.

        - prices_path: path to CSV file containing year-end prices for selected companies.
        Default is path to prices for 46 european companies between 2007 and 2017.
        '''

        benchmark_path_dict = {
            'eurostoxx_600' : PATH_DATAFOLDER + PATH_EUROSTOXX600,
            'russell_3000' : PATH_DATAFOLDER + PATH_RUSSELL3000
            }

        ## Import Bloomberg ESG metrics for selected companies from Excel file
        data = pd.read_csv(esg_metrics_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Ticker'] = [e.split(" ")[0] for e in data['Entreprises']]
        data['Year'] = pd.DatetimeIndex(data['Date']).year

        ## Import year-end stock prices for selected companies from CSV file
        prices = pd.read_csv(prices_path)
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices.drop('Unnamed: 0', axis = 1, inplace = True)
        prices['Year'] = [d.year for d in prices['Date']]
        prices = prices.set_index('Date')

        ## Get year-end prices and annual returns for selected stocks
        data = get_annual_returns_col(data, prices)

        # Obtain dependant variable
        data, self.daily_benchmark_levels = get_benchmark_returns(data, benchmark_path_dict[self.benchmark])

        data['Label'] = data['Excess Returns Y+1'].map(lambda x:get_label(x,self.bucket_delimiters[0],self.bucket_delimiters[1]))          
            
        self.tickers = list(data['Ticker'].unique())
        self.data = data
        self.prices = prices
        
        return self
        
        
    def get_benchmark_annual_return(self,
                                    year):
        
        benchmark_daily_levels_y = self.daily_benchmark_levels[self.daily_benchmark_levels['Year']==year]
        benchmark_daily_levels_y = benchmark_daily_levels_y.sort_index(ascending = True)
        
        level_t0 = benchmark_daily_levels_y.iloc[0]['Adj Close']
        level_t1 = benchmark_daily_levels_y.iloc[-1]['Adj Close']
        
        return (level_t1 - level_t0) / level_t0
                
    
    def fit_portfolio(self,
                        year = 2017,
                        xgb_params = {'n_estimators':200,
                                      'learning_rate':0.1,
                                      'max_depth':5,
                                      'min_child_weight':5,
                                      'subsample':0.8,
                                      'colsample_bytree':0.8,
                                     },
                        print_accuracy = False):
        
        assert year in [2014, 2015, 2016, 2017], "Invalid y argument.  Must be 2014, 2015, 2016 or 2017."
        
        year_dataset, X_train, X_test, Y_train, Y_test = get_train_test(self.data,
                                                                        year,
                                                                        X_columns,
                                                                        Y_label)
        
        predictions, pred_proba_short_list, pred_proba_long_list = xgb_predict(xgb_params,
                                                                     X_train,
                                                                     X_test,
                                                                     Y_train,
                                                                     Y_test,
                                                                     print_accuracy)
        
        year_dataset['Predictions'] = predictions
        
        ## Computing Sharpe Ratios for each model
        ### Obtaining weights for each stock
        year_dataset = get_weight_col(year_dataset, 
                                      self.short_limit, 
                                      pred_proba_short_list, 
                                      pred_proba_long_list, 
                                      self.proba_weighted)

        weights = year_dataset['Weight']
        stocks = year_dataset['Ticker']
        
        ## Portfolio returns
        returns_y1 = np.array(year_dataset['Return Y+1'])
        self.portfolio_return = np.dot(weights,returns_y1)

        ## Portfolio variance
        annual_cov_matrix = self.prices.pct_change()[self.tickers].cov() * DAYS_IN_YEAR
        self.portfolio_stdev = np.sqrt(np.dot(weights.T,np.dot(annual_cov_matrix,weights)))
        
        ## Annual Risk-Free Rate
        rf_rate = get_rf_rate(year)

        ## Portfolio Sharpe ratio 
        self.portfolio_sharpe_ratio = (self.portfolio_return - rf_rate) / self.portfolio_stdev

        ##  Benchmark Sharpe ratio
        benchmark_ret = self.get_benchmark_annual_return(year)
        benchmark_std = self.daily_benchmark_levels['Adj Close'].std() * np.sqrt(DAYS_IN_YEAR)
        
        self.benchmark_sharpe_ratio = (benchmark_ret - rf_rate) / benchmark_std
        
        return self