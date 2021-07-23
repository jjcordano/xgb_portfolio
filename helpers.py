import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

## Constants
DAYS_IN_YEAR = 252

## Paths
PATH_DATAFOLDER = r'data/'
PATH_ESG_METRICS = r'data_fi_v2.csv'
PATH_PRICES = r'prices.csv'
PATH_EUROSTOXX600 = r'STOXX600.csv'
PATH_RUSSELL3000 = r'RUSSELL.csv'
PATH_RF_RATE = r'FR_TBill_10y.csv'

X_columns = ['NUMBER_EMPLOYEES_CSR','AUDIT_COMMITTEE_MEETINGS', 'SAY_PAY_SUPPORT_LEVEL',
             'TOT_OTHER_COMP_AW_TO_CEO_&_EQUIV', 'TOTAL_EXEC_PAY_AS_PCT_OPEX',
             'TOT_SALARIES_PAID_TO_CEO_&_EQUIV', 'TOT_SALARIES_&_BNS_PD_TO_EXECS',
             'TOT_N_EQTY_INCENT_GIVEN_TO_EXECS', 'SAY_PAY_NUMBER_OF_VOTES_FOR',
             'TOT_EXEC_PAY_AS_PCT_SG&A_NET_R&D', 'TOT_OPTION_AWARDS_GIVEN_TO_EXECS',
             'TOT_EXEC_PAY_AS_PCT_TOT_PSNL_EXP', 'TOT_N_EQT_INCENT_GVN_TO_CEO_&_EQ',
             'PCT_BOD_COMP_PD_IN_STK_AWD', 'NUM_EXECUTIVE_CHANGES',
             'AVERAGE_BOD_TOTAL_COMPENSATION', 'ESG_DISCLOSURE_SCORE',
             'CFO_TENURE_AS_OF_FY_END', 'CHG_OF_CTRL_BFIT_GOLD_CHUTE_AGR',
             'CLAWBACK_PROVISION_FOR_EXEC_COMP', 'GENDER_PAY_GAP_BREAKOUT',
             'BLANK_CHECK_PREFERRED_AUTHORIZED']

Y_label = 'Label'


## Helper functions
def get_annual_returns_col(data,
                           daily_prices):
    
    prices = daily_prices
    
    data['Price Y'] = np.nan
    data['Price Y+1'] = np.nan
    data['Return Y+1'] = np.nan

    for row in data.index:
        s = data.loc[row, 'Ticker']
        y0 = data.loc[row, 'Year']
        y1 = data.loc[row,'Year']+1

        prices0_temp=prices[prices["Year"] == y0]
        prices1_temp=prices[prices["Year"] == y1]
        prices_y0_s=list(prices0_temp[s])
        prices_y1_s=list(prices1_temp[s])

        try:
            p0 = 0
            p1 = 0
            i = -1
            while pd.isnull(prices_y0_s[i]):
                i=i-1
            p0=prices_y0_s[i]

            i = -1
            while pd.isnull(prices_y1_s[i]):
                i=i-1
            p1=prices_y1_s[i]

            data.loc[row,'Price Y'] = p0
            data.loc[row,'Price Y+1'] = p1
            data.loc[row,'Return Y+1'] = (p1-p0)/p0

        except IndexError:
            pass
        
        
    data = data[['Date','Entreprises'] + X_columns + ['Ticker','Year','Price Y','Price Y+1','Return Y+1']]
    
    ## Delete rows where Return Y+1 is null
    return data[data['Return Y+1'].notna()]


def get_benchmark_returns(data,
                          daily_benchmark_levels_path):
    
    ## Import and merge chosen index year-end levels
    index = pd.read_csv(daily_benchmark_levels_path)
    index['Date'] = pd.to_datetime(index['Date'])
    index['Year'] = pd.DatetimeIndex(index['Date']).year
    index.set_index('Date', inplace=True)

    index_y = pd.DataFrame(index = list(range(2000,2021)))
    index_y['levelYearEnd'] = np.nan
    
    for row in index_y.index:
        y = row

        index_temp=index[index["Year"] == y]
        level_y_temp=list(index_temp['Adj Close'])

        try:
            l = 0
            i = -1
            while pd.isnull(level_y_temp[i]):
                i = i-1
            l = level_y_temp[i]
            index_y.loc[row,'levelYearEnd'] = l

        except IndexError:
            pass

    index_y['Benchmark Returns'] = index_y['levelYearEnd'].pct_change()
    index_y.index = index_y.index - 1
    index_y.rename({"Benchmark Returns" : "Benchmark Returns Y+1"}, inplace = True, axis = 1)
    
    data = data.merge(index_y, left_on = 'Year', right_index = True)
    
    ### Dependant variable : RUSSELL Excess Returns Y+1
    data['Excess Returns Y+1'] = data['Return Y+1'] - data['Benchmark Returns Y+1']
    
    data = data[['Date','Entreprises'] + X_columns + ['Ticker','Year','Price Y','Price Y+1','Return Y+1','Excess Returns Y+1']]
    
    return data, index



def get_russell3000_returns(data, 
                            russell3000_path):
    
    ## Import and merge RUSSELL 3000 year-end levels
    russell = pd.read_csv(russell3000_path)
    russell['Date'] = pd.to_datetime(russell['Date'])
    russell['Year'] = [d.year for d in russell['Date']]

    russell_y = pd.DataFrame(index = list(range(2000,2021)))
    russell_y['levelYearEnd'] = np.nan

    for row in russell_y.index:
        y = row

        russell_temp=russell[russell["Year"] == y]
        level_y_temp=list(russell_temp['Adj Close'])

        try:
            l = 0
            i = -1
            while pd.isnull(level_y_temp[i]):
                i = i-1
            l = level_y_temp[i]
            russell_y.loc[row,'levelYearEnd'] = l

        except IndexError:
            pass

    russell_y['RUSSELL 3000 Returns'] = russell_y['levelYearEnd'].pct_change()
    russell_y.index = russell_y.index - 1
    russell_y.rename({"RUSSELL 3000 Returns":"RUSSELL 3000 Returns Y+1"},inplace = True,axis = 1)

    data = pd.merge(data,russell_y,left_on = 'Year',right_index=True)

    ### Dependant variable : RUSSELL Excess Returns Y+1
    data['Excess Returns Y+1 (RUSSELL)'] = data['Return Y+1'] - data['RUSSELL 3000 Returns Y+1']
    
    return data.drop(['levelYearEnd','Unnamed: 24','RUSSELL 3000 Returns Y+1'],axis = 1,inplace = True)


def get_eurostoxx600_returns(data,
                             eurostoxx600_path):
    
    ## Import and merge EUROSTOXX 600 year-end levels
    eurostoxx600 = pd.read_csv(eurostoxx600_path)
    eurostoxx600['Date'] = pd.to_datetime(eurostoxx600['Date'])
    eurostoxx600['Year'] = [d.year for d in eurostoxx600['Date']]

    eurostoxx600_y = pd.DataFrame(index = list(range(2000,2021)))
    eurostoxx600_y['levelYearEnd'] = np.nan

    for row in eurostoxx600_y.index:
        y = row

        eurostoxx600_temp = eurostoxx600[eurostoxx600["Year"] == y]
        level_y_temp = list(eurostoxx600_temp['Adj Close'])

        try:
            l = 0
            i = -1

            while pd.isnull(level_y_temp[i]):
                i = i-1
            l = level_y_temp[i]
            eurostoxx600_y.loc[row,'levelYearEnd'] = l

        except IndexError:
            pass

    eurostoxx600_y['EUROSTOXX 600 Returns'] = eurostoxx600_y['levelYearEnd'].pct_change()
    eurostoxx600_y.index = eurostoxx600_y.index-1
    eurostoxx600_y.rename({"EUROSTOXX 600 Returns":"EUROSTOXX 600 Returns Y+1"},inplace = True,axis = 1)

    data = pd.merge(data,eurostoxx600_y,left_on = 'Year',right_index=True)
    data.drop('levelYearEnd',axis = 1,inplace = True)

    ### Dependant variable 2 : EUROSTOXX Excess Returns Y+1
    data['Excess Returns Y+1 (EUROSTOXX)'] = data['Return Y+1'] - data['EUROSTOXX 600 Returns Y+1']
    
    return data.drop(['levelYearEnd','Unnamed: 24','EUROSTOXX 3000 Returns Y+1'], axis = 1, inplace = True)


def get_label(excess_returns, 
              short_lim, 
              long_lim):
    label = 0
    if excess_returns < short_lim:
        label = 0
    elif excess_returns > long_lim:
        label = 2
    else:
        label = 1
    return label


def get_train_test(data,
                   y,
                   X_columns,
                   Y_label):
    
    years = list(range(y-7,y))
    dataset = data[data['Year'].isin([y])]

    X_train = data[data['Year'].isin(years)][X_columns]
    X_test = data[data['Year'].isin([y])][X_columns]

    Y_train = data[data['Year'].isin(years)][Y_label]
    Y_test = data[data['Year'].isin([y])][Y_label]
    
    return dataset, X_train, X_test, Y_train, Y_test


def xgb_predict(xgb_params,
                X_train,
                X_test,
                Y_train,
                Y_test,
                print_accuracy):
    
    model = xgboost.XGBClassifier(**xgb_params)
    model.fit(X_train,Y_train)

    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    pred_proba_short_list = model.predict_proba(X_test)[:,0]
    pred_proba_long_list = model.predict_proba(X_test)[:,2]

    accuracy = accuracy_score(Y_test, predictions)
    
    if print_accuracy:
        print("Model Accuracy : %.2f%%" % (accuracy * 100.0))
        print("------------------------------------")
    
    return predictions, pred_proba_short_list, pred_proba_long_list


def get_portfolio_weights(pred, 
                          price0, 
                          nb_short, 
                          nb_long, 
                          short_limit=10):
    
    if pred==0:
        if price0 < short_limit: # Stock not shorted if price at beginning of year below limit
            return 0
        
        else:
            return (-1/nb_short)
    
    elif pred==1:
        return 0
    
    else:
        return (1/nb_long)
    
    
def get_portfolio_weights_wProba(pred, 
                                 price0,
                                 sum_proba_short,
                                 sum_proba_long,
                                 val_proba_short, 
                                 val_proba_long, 
                                 short_limit=10):

    if pred == 0:
        if price0 < short_limit: # Stock not shorted if price at beginning of year below limit
            return 0
        
        else:
            return (- val_proba_short / sum_proba_short)
    
    elif pred == 1:
        return 0
    
    else:
        return (val_proba_long / sum_proba_long)
    
    
def get_weight_col(year_dataset,
                   short_limit,
                   pred_proba_short_list,
                   pred_proba_long_list,
                   proba_weighted):
    
    
    if proba_weighted:
        year_dataset['Pred_proba_short'] = pred_proba_short_list
        year_dataset['Pred_proba_long'] = pred_proba_long_list
        
        df = year_dataset
        df2 = df[df['Price Y'] > short_limit]
        
        sum_proba_short = df2[df2['Predictions'] == 0]['Pred_proba_short'].sum()
        sum_proba_long = df[df['Predictions'] == 2]['Pred_proba_long'].sum()
        
        year_dataset['Weight'] = year_dataset.apply(lambda x: get_portfolio_weights_wProba(x['Predictions'], 
                                                                                   x['Price Y'], 
                                                                                   sum_proba_short,
                                                                                   sum_proba_long,
                                                                                   x['Pred_proba_short'],
                                                                                   x['Pred_proba_long'],
                                                                                   short_limit), axis=1)    
     
    else:
        df = year_dataset
        df2 = df[df['Price Y'] > short_limit]        
            
        nb_short = len(df2[df2['Predictions'] == 0])
        nb_long = len(df[df['Predictions'] == 2])
        
        year_dataset['Weight'] = year_dataset.apply(lambda x: get_portfolio_weights(x['Predictions'], 
                                                                                    x['Price Y'], 
                                                                                    nb_short,
                                                                                    nb_long,
                                                                                    short_limit), axis=1)
        
    return year_dataset 


def get_rf_rate(year,
                path_datafolder = PATH_DATAFOLDER,
                path_rf_rate = PATH_RF_RATE):
    
    rf_rates = pd.read_csv(path_datafolder + path_rf_rate)
    rf_rates['Date'] = pd.to_datetime(rf_rates['Date'])
    rf_rates['Year'] = pd.DatetimeIndex(rf_rates['Date']).year
    
    rf_year = rf_rates[rf_rates['Year'] == year]
  
    return float(rf_year[rf_year['Date'] == rf_year['Date'].min()]['10y_FR_Treasury_Bond_Rate'])
