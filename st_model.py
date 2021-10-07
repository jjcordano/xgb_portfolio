import numpy as np
import streamlit as st
import pandas as pd

from builder.helpers import *
from builder.portfolio_builder import PortfolioBuilder

def app():
    model = st.container()

    pb0 = PortfolioBuilder(probability_weighted=False).init_data()

    with model:
        st.header("Original model presented by Bloomberg (2020)")
        st.markdown("The proposed machine learning algorithm for this task is XGBoost as it is a high performing model and [it can handle missing values without preprocessing](https://xgboost.readthedocs.io/en/latest/faq.html).")
        #st.markdown("For a given year Y and a given company, the label is the class computed for Y+1. \n The classes are built using the annual returns of stocks in excess of Eurostoxx 600 returns. Excess returns above +18% are classified as 'long', \n those between +18% and -15% are classified as 'omit' and those below -12% are put in the 'short' class. ")
        st.markdown("In the original paper, 4 annual portfolios are built for 2014, 2015, 2016 and 2017. \n For each year, the model is trained on the 7 previous years.\n Depending on the class predicted by the model, a position is taken in each stock. \n In the original model, _each stock in the same class is gven the same weight_.")
        
        st.subheader("Choose model hyperparameters:")
        
        col_1, col_2, col_3 = st.columns(3)
        
        year = col_1.selectbox("Choose year of interest for annual portfolio:", list(range(2014,2018)), index=3, key='model')
        n_estimators = col_2.slider("Choose number of trees in XGBoost model:",min_value=100, max_value=250, value=200, step=10)
        max_depth = col_3.slider("Choose maximum depth of trees in XGBoost model",min_value=3, max_value=10, value=5, step=1)
        
        params = update_params(n_estimators=n_estimators, max_depth=max_depth)
        
        pb1 = PortfolioBuilder(probability_weighted=False).init_data().fit_portfolio(year=year, xgb_params=params)
        
        st.write('## Results')
        st.subheader("Portfolio Weights:")
        
        dict_weights = pb1.get_dict_weights()
        
        #print(dict_weights.keys())
        
        st.write(pd.DataFrame(np.array([list(dict_weights.values())]), 
                              columns=list(dict_weights.keys()),
                              index=['Weight']))
        
        st.subheader("Results from original XGBoost model:")
        
        col_1a, col_2a, col_3a = st.columns(3)
        
        col_1a.markdown("**Model accuracy (%):**")
        col_1a.write(round(pb1.get_model_accuracy() * 100,2))
        
        col_2a.markdown("**Portfolio return:**")
        col_2a.write(round(pb1.get_portfolio_return(),4))
        col_3a.markdown("**Portfolio Sharpe Ratio:**")
        col_3a.write(round(pb1.get_portfolio_sharpe_ratio(),4))
        
        col_2a.text("Benchmark return:")
        col_2a.write(round(pb1.get_benchmark_return(),4))
        col_3a.text("Benchmark Sharpe Ratio:")
        col_3a.write(round(pb1.get_benchmark_sharpe_ratio(),4))
        
        
        
        
        
        