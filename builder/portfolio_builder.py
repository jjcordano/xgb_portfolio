import os
#from os import dirname, join
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import accuracy_score
from builder.helpers import *

class PortfolioBuilder:
    
    """
    XGBoost classifier object that uses selected stocks' ESG metrics to select stocks in a portfolio.
    
    Arguments:
    - benchmark: String, selected stock index on which to compute excess returns.  Benchmark should be either 'eurostoxx_600' or 'russell_3000'.
    Defaults to 'eurostoxx_600' as selected stocks are of large european capitalizations.
    - probability_weighted: Boolean, if True, weight of stock in portfolio weighted by probability that XGBoost prediction is correct.
    - bucket_delimiters: Tuple, gives limits between short, ignore and long buckets for stock returns in excess of benchmark.
    - short_limit: Float, stock value under which stocks will not be shorted.
    """    

    
    def __init__(self,
                 benchmark = 'eurostoxx_600',
                 probability_weighted = True,
                 bucket_delimiters = (-0.15,0.18),
                 short_limit = 10.0):
        
        assert benchmark in ['eurostoxx_600','russell_3000'], "Benchmark argument not valid. Must either be 'eurostoxx_600' or 'russell_3000'."

        self.benchmark = benchmark
        self.proba_weighted = probability_weighted
        self.bucket_delimiters = bucket_delimiters
        self.short_limit = short_limit

        self.year = None
        self.tickers = None
        self.data = None
        self.prices = None
        self.daily_benchmark_levels = None
        
        self.__dict_predictions = None
        self.__dict_weights = None
        
        self.__accuracy_score = None
        
        self.__portfolio_return = None
        self.__portfolio_stdev = None
        self.__portfolio_sharpe_ratio = None
        
        self.__benchmark_return = None
        self.__benchmark_sharpe_ratio = None
    
    def init_data(self,
                  esg_metrics_ = FILENAME_ESG_METRICS, 
                  prices_path = FILENAME_PRICES,
                  data_folder_name = DATAFOLDER):
        '''
        Function that collects and groups required data to build the desired portfolio 
        along the lines of the 2020 paper 'Seeking Signals from ESG Data' by Bloomberg Quant Research.

        Arguments:
        - esg_metrics_path: path to CSV file containing Bloomberg ESG metrics for selected companies.  
        Default is path to ESG data for 46 european companies between 2007 and 2017.

        - prices_path: path to CSV file containing year-end prices for selected companies.
        Default is path to prices for 46 european companies between 2007 and 2017.
        '''
        
        parent_path = os.path.dirname(os.path.dirname(__file__))
        

        ## Import Bloomberg ESG metrics for selected companies from Excel file
        data = pd.read_csv(os.path.join(parent_path, DATAFOLDER, FILENAME_ESG_METRICS))
        data['Date'] = pd.to_datetime(data['Date'])
        data['Ticker'] = [e.split(" ")[0] for e in data['Entreprises']]
        data['Year'] = pd.DatetimeIndex(data['Date']).year

        ## Import year-end stock prices for selected companies from CSV file
        prices = pd.read_csv(os.path.join(parent_path, DATAFOLDER, FILENAME_PRICES))
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices.drop('Unnamed: 0', axis = 1, inplace = True)
        prices['Year'] = [d.year for d in prices['Date']]
        prices = prices.set_index('Date')

        ## Get year-end prices and annual returns for selected stocks
        data = get_annual_returns_col(data, prices)

        # Obtain dependant variable
        path_benchmark_data = os.path.join(parent_path, DATAFOLDER, benchmark_path_dict[self.benchmark])
        data, self.daily_benchmark_levels = get_benchmark_returns(data, path_benchmark_data)

        data['Label'] = data['Excess Returns Y+1'].map(lambda x:get_label(x,self.bucket_delimiters[0],self.bucket_delimiters[1]))          
            
        self.tickers = list(data['Ticker'].unique())
        self.data = data
        self.prices = prices
        
        return self
        
        
    def get_benchmark_annual_return(self):
        
        """Method that computes annual return for the selected test year.

        Returns:
            Float: annual benchmark return.
        """        
        
        benchmark_daily_levels_y = self.daily_benchmark_levels[self.daily_benchmark_levels['Year']==self.year]
        benchmark_daily_levels_y = benchmark_daily_levels_y.sort_index(ascending = True)
        
        level_t0 = benchmark_daily_levels_y.iloc[0]['Adj Close']
        level_t1 = benchmark_daily_levels_y.iloc[-1]['Adj Close']
        
        return (level_t1 - level_t0) / level_t0
                
    
    def fit_portfolio(self,
                      year = 2017,
                      xgb_params = DEFAULT_XGB_PARAMS,
                      print_accuracy = False):
        
        """Creates portfolio based on XGBoost classfications of short, ignore or long for a chosen year between 2014 and 2017.
        
        Arguments:
        - year: Integer, year chosen for annually held portfolio.  Model will be trained on 7 previous years.  Defaults to 2017.
        - xgb_params: Dict, parameters of XGBoost classifier.  Defaults to parameters used in 'Seeking Signals from ESG Data' paper by Bloomberg Quant Research.
        - print_accuracy: Boolean, if True, accuracy of XGBoost classifier is printed. Defaults to False.

        Returns:
            PortfolioBuilder: self.
        """        
        
        assert year in [2014, 2015, 2016, 2017], "Invalid y argument.  Must be 2014, 2015, 2016 or 2017."
        
        self.year = year
        
        year_dataset, self.__X_train, X_test, Y_train, Y_test = get_train_test(self.data,
                                                                        self.year,
                                                                        X_columns,
                                                                        Y_label)
        
        self.__model, predictions, pred_proba_short_list, pred_proba_long_list, self.__accuracy_score = xgb_predict(xgb_params,
                                                                                                    self.__X_train,
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
        
        #print(year_dataset)

        weights = year_dataset['Weight']
        stocks = year_dataset['Ticker']
        
        ## Portfolio returns
        returns_y1 = np.array(year_dataset['Return Y+1'])
        self.__portfolio_return = np.dot(weights,returns_y1)

        ## Portfolio variance
        annual_cov_matrix = self.prices.pct_change()[self.tickers].cov() * DAYS_IN_YEAR
        self.__portfolio_stdev = np.sqrt(np.dot(weights.T,np.dot(annual_cov_matrix,weights)))
        
        ## Annual Risk-Free Rate
        rf_rate = get_rf_rate(self.year)

        ## Portfolio Sharpe ratio 
        self.__portfolio_sharpe_ratio = (self.__portfolio_return - rf_rate) / self.__portfolio_stdev

        ##  Benchmark Sharpe ratio
        self.__benchmark_return = self.get_benchmark_annual_return()
        benchmark_std = self.daily_benchmark_levels['Adj Close'].std() * np.sqrt(DAYS_IN_YEAR)
        
        self.__benchmark_sharpe_ratio = (self.__benchmark_return - rf_rate) / benchmark_std
        
        ## Create Predictions and Weights dictionaries
        self.__dict_predictions = dict(zip([list(companies_dict.keys())[list(companies_dict.values()).index(stock)]
                                         for stock in year_dataset['Ticker']],
                                         [round(w, 4) for w in year_dataset['Predictions']]
                                         )
                                       )
        
        self.__dict_weights = dict(zip([list(companies_dict.keys())[list(companies_dict.values()).index(stock)]
                                         for stock in year_dataset['Ticker']],
                                         [round(w, 4) for w in year_dataset['Weight']]
                                         )
                                       )
                
        return self
    
    
    def print_sharpe_ratio(self):
        
        """
        Prints portfolio Sharpe ratio and compares it to benchmark's Sharpe ratio.
        """        
        
        print('**********************************************')
        print("\n")
        print('Sharpe ratio comparison for the year {}:'.format(self.year))
        print('Portfolio Sharpe Ratio: {}'.format(round(self.__benchmark_return,5)))
        print('{} Sharpe Ratio:{}'.format(benchmark_dict[self.benchmark], round(self.__benchmark_sharpe_ratio,5)))
        print("\n")
        print('**********************************************')
        
    
    def get_portfolio_return(self):
        
        """
        Method that returns the prediction (0: short, 1: ignore, 2: long) for each stock.

        Returns:
            Dictionary: Prediction for each stock.
        """    
        
        return self.__portfolio_return
    
    
    def get_dict_predictions(self):
        
        """
        Method that returns the prediction (0: short, 1: ignore, 2: long) for each stock.

        Returns:
            Dictionary: Prediction for each stock.
        """        
        
        return self.__dict_predictions
    
    
    def get_dict_weights(self):
        
        """
        Method that returns the weights associated to each stock.

        Returns:
            Dictionary: Weights associated to each stock.
        """        
    
        return self.__dict_weights
    
    
    def get_model_accuracy(self):
        
        return self.__accuracy_score
    

    def get_portfolio_stdev(self):
        
        return self.__portfolio_stdev 
    
    
    def get_portfolio_sharpe_ratio(self):
        
        return self.__portfolio_sharpe_ratio


    def get_benchmark_return(self):
        
        return self.__benchmark_return


    def get_benchmark_sharpe_ratio(self):
        
        return self.__benchmark_sharpe_ratio 
    
    
    def get_X_Train(self):
        
        return self.__X_train 
    
    
    def get_model(self):
        
        return self.__model 



    def create_kpi_dataframe(list_years = [2014,2015,2016,2017]):
        
        kpi_df = pd.DataFrame(columns=['Year','Portfolio','Annual Return','Sharpe Ratio'])

        for year in list_years:
            y_portfolio = PortfolioBuilder().init_data().fit_portfolio(year)
        
            kpi_df.concat([year,'Benchmark',y_portfolio.get_benchmark_return(),y_portfolio.get_benchmark_sharpe_ratio()])
            kpi_df.concat([year,'XGB Portfolio',y_portfolio.get_portfolio_return(),y_portfolio.get_portfolio_sharpe_ratio()])
            
        return kpi_df