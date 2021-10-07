import streamlit as st

from builder.helpers import *
#from builder.helpers import companies_dict, get_year_end_prices
from builder.portfolio_builder import PortfolioBuilder


def app():    
    #header = st.container()
    dataset = st.container()

    #with header:
    #    st.title("Portfolio creation using XGBoost and ESG data")
        

    pb0 = PortfolioBuilder(probability_weighted=False).init_data()
    st.session_state.year_end_prices_df = get_year_end_prices(pb0.prices)
        
        
    with dataset:
        st.header("Introduction")
        st.markdown("In this project, we apply portfolio selection methodology presented in the 2020 paper _[Seeking Signals from ESG Data](https://www.bloomberg.com/professional/blog/seeking-signals-from-esg-data/)_ by Bloomberg Quant Research to a sample of 45 european large-cap stocks.")
        st.markdown("The presented model breaks down stock price prediction into a classification problem with 3 labels : positive returns ('long'), negative returns ('short') and no siginificant returns ('omit').")
        st.markdown("The independant variables are ESG metrics collected from Bloomberg for a given year Y, and the label is the level of return for the year Y+1.")
        st.markdown("Depending on the predictions, we build a long-short portfolio and hold it for one year, and compare observed returns and Sharpe ratio with the Eurostoxx 600 benchmark.")
        st.markdown("In later steps, we introduce a new way of giving weights to stocks within the protfolio and assess feature importance with SHAP values.")
        
        st.write('## Vizualising the data')
        st.write('Data ranges from 2007 to 2017. The dataset is ordered by year and company.')
        st.write(pb0.data.head(20))
        
        
        st.write('### The dependant variables: Bloomberg ESG metrics')
        st.write('21 annual ESG metrics make up the dependant variables.')
        st.write("""As stated in the original paper: "about 60% of ESG metrics have a Missing Rate (defined as the proportion of records for which the value is missing) of more than 50% in Bloomberg's ESG dataset on US companies over the past 14 years" (BQR, 2020). 
                 The reason for this is that companies do not have the obligation to publish information about Environmental, Social and Governance aspects of their business.""")
        st.write("In our ESG metric dataset, missing values presents approx. 40% of all values in a total of 17,940 data points.")
        
        st.write('### Labels: Stock Return Levels at Y+1')
        st.write('Labels can be found in the 2 last columns of the dataframe.')
        st.markdown("The classes are built using the annual returns of stocks in excess of Eurostoxx 600 returns.")
        st.markdown("- Excess returns above +18% are classified as '***long***' (***2***)")
        st.markdown("- Those between +18% and -15% are classified as '***omit***'(***1***)")
        st.markdown("- those below -15% are put in the '***short***' class (***0***).")
       
        
        st.write("## Compare companies' stock prices")
        viz1_stock_selection = st.multiselect('Choose the firms whose stock price you want to compare',
                    options = list(companies_dict.keys()),
                    default=list(companies_dict.keys())[:3])
        
        stock_list = [companies_dict[s] for s in viz1_stock_selection]
        
        st.line_chart(st.session_state.year_end_prices_df[stock_list])
