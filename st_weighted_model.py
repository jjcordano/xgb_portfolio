import numpy as np
import streamlit as st
import plotly.express as px

from builder.helpers import *
from builder.portfolio_builder import PortfolioBuilder

def app():
    weighted_model = st.container()

    pb_original = PortfolioBuilder(probability_weighted=False).init_data()
    pb_weighted = PortfolioBuilder(probability_weighted=True).init_data()
    
    kpi_df = pd.DataFrame(columns=['Year','Portfolio','Annual Return','Sharpe Ratio'])
    
    for year in range(2014, 2018):
        pb_original.fit_portfolio(year)
        pb_weighted.fit_portfolio(year)
        
        kpi_df.loc[len(kpi_df)] = [year,'Benchmark',pb_original.get_benchmark_return(),pb_original.get_benchmark_sharpe_ratio()]
        kpi_df.loc[len(kpi_df)] = [year,'XGB_Original',pb_original.get_portfolio_return(),pb_original.get_portfolio_sharpe_ratio()]
        
        kpi_df.loc[len(kpi_df)] = [year,'XGB_Weighted',pb_weighted.get_portfolio_return(),pb_weighted.get_portfolio_sharpe_ratio()]
        
    print(kpi_df.groupby('Portfolio').mean())
        

    with weighted_model:
        st.header("The Probability-Weighted Model")
        #st.subheader("Intuition")
        st.markdown("In this section, we change the weight assignment methodology while keeping the same XGBoost models.")
        st.markdown("Using `xgboost.XGBoostClassifier.predict_proba(X_test)`, the position of each stock is weighted by the probability of belonging to the predicted class over the sum of probabilities of predictions in that class.")
        
        st.write("## Performance comparison")
        st.markdown('Select the desired years:')
        
        col_1, col_2, col_3, col_4 = st.columns(4)
        
        option_1 = col_1.checkbox('2014', value=True)
        option_2 = col_2.checkbox('2015', value=True)
        option_3 = col_3.checkbox('2016', value=True)
        option_4 = col_4.checkbox('2017', value=True)
        
        years = []
        options = [option_1, option_2, option_3, option_4]
        
        for i in list(range(4)):
            if options[i] == True:
                years.append(2014+i)
                
         
        selected_df = kpi_df.loc[kpi_df['Year'].isin(years)]
        
        st.subheader('Comparison of Annual Returns')
        st.markdown("As seen in the graphical representation of annual returns below, the weighted-probability method yields better returns than the equal-weight method in 3 out of 4 years.")
        st.markdown("Compared to the Eurostoxx 600 benchmark, both models surpass it in only 2 out of 4 years. However, expected returns are superior for both models over the 4 years with 5.48%, 8.43% and 8.49% for the benchmark, the equally weighted and the probability weighted portfolios respectively.")
        
        fig1 = px.bar(selected_df, x="Year", y="Annual Return",
             color='Portfolio', barmode='group',
             height=400)
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader('Comparison of Sharpe Ratios')
        st.markdown("in terms of Sharpe ratios, the expected values are 0.000047, 0.271189 and 0.269054 for the benchmark, the equally weighted and the probability weighted portfolios respectively.")
        
        fig2 = px.bar(selected_df, x="Year", y="Sharpe Ratio",
             color='Portfolio', barmode='group',
             height=400)
        
        st.plotly_chart(fig2, use_container_width=True)
