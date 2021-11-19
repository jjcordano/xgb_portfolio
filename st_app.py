import numpy as np
import streamlit as st

import st_intro, st_model, st_weighted_model, st_shap

from builder.helpers import *
from builder.portfolio_builder import PortfolioBuilder

PAGES = {
    "Introduction": st_intro,
    "Original Model (Bloomberg, 2020)": st_model,
    "Probability-Weighted Model": st_weighted_model,
    "Shap Values": st_shap
}

st.sidebar.title('Portfolio creation using XGBoost and ESG data')
st.sidebar.markdown('Based on 2020 paper \n _[Seeking Signals from ESG Data](https://www.bloomberg.com/professional/blog/seeking-signals-from-esg-data/)_ \n by Bloomberg Quantitative Research.')

#st.session_state.portfolio_dict = {'original':{}, 'proba_weighted':{}}

#for year in range(2014, 2018):
#    st.session_state.portfolio_dict['original'][year] = PortfolioBuilder().init_data().fit_portfolio(year)
#    st.session_state.portfolio_dict['proba_weighted'][year] = PortfolioBuilder(probability_weighted = True).init_data().fit_portfolio(year)


selection = st.sidebar.radio("Navigation", list(PAGES.keys()))
page = PAGES[selection]
page.app()

st.sidebar.text('Authors: \n Jean-Julien Cordano \n Nelson Castro \n Nirushanth Arishandra')
st.sidebar.text('Streamlit App by \n Jean-Julien Cordano')
st.sidebar.text('Université Paris 1 Panthéon-Sorbonne')

st.sidebar.markdown('Check out the code on [Github](https://github.com/jjcordano/xgb_portfolio).')
