import shap
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np

from builder.helpers import *
from builder.portfolio_builder import PortfolioBuilder

def app():
    shap_module = st.container()
    
    while shap_module:
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        st.header("Feature importance using SHAP values")
        st.markdown("We assess feature importance with SHAP values for the 2017 probability-weighted portfolio.")
        #col_1, col_2 = st.columns(2)
        
        #model = col_1.selectbox('Select model of interest',['Equal-Weighted','Probability-Weighted'],key=1)
        #year = col_2.selectbox('Select year of interest',[2014,2015,2016,2017],key=2)
        
        PROBA_WEIGHTED = True
        YEAR = 2017
        
        pb2 = PortfolioBuilder(probability_weighted=PROBA_WEIGHTED).init_data().fit_portfolio(YEAR)
        
        st.subheader('SHAP Barplot for 2017 Equally-Weighted Model')
        st.markdown("The plot below gives the cumulative impact of each ESG metric on the prediction.")
        st.markdown("We can see that *Average Board of Director Compensation* has the most impact over all 3 classes for the 2017 probability-weighted model.")
        
        X_Train = pb2.get_X_Train()
        xgb_model = pb2.get_model()
        
        #shap_values = shap.TreeExplainer(xgb_model).shap_values(X_Train)
        #fig1 = shap.summary_plot(shap_values, X_Train, plot_type="bar",auto_size_plot=False,show=False,matplotlib=True)
        
        #def st_shap(plot, height=None):
        #    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        #    components.html(shap_html, height=height)

        # visualize the training set predictions
        #st_shap(shap.summary_plot(shap_values, X_Train, plot_type="bar",auto_size_plot=False,show=False, matplotlib=True), 400)
        
        shap_values = shap.TreeExplainer(xgb_model).shap_values(X_Train)
        #explainer = shap.TreeExplainer(xgb_model)
        #shap_values = explainer(X_Train)
        #shap.plots.bar(st.session_state.shap_values)
        
        barplot = shap.summary_plot(shap_values,X_Train,plot_type="bar",show=False)
        st.pyplot(barplot)
        #plt.clf()
        
        #shap.summary_plot(shap_values, X_Train, plot_type="bar",auto_size_plot=False)
        #st.pyplot(fig1)
        #plt.clf()
        
        st.subheader('SHAP Swarplot for 2017 Equally-Weighted Model')
        st.markdown("The plot below tells us how each ESG metric or feature specifically impacts the class of our choice.")
        st.markdown("We can see that for the 'Long' class, high *Average Board of Director Compensation* values mean a higher chance of belonging to this class. A majority of blue dots on the right would mean the opposite : smaller values of the feature have a positive impact on the prediction being 'Long'.")
        
        class_ = st.selectbox('Select class of interest',['Long','Omit','Short'],index=0,key='key')
        
        class_ = 'Long'
        
        c = 2
        if class_ == 'Short':
            c = 0
        elif class_ == 'Omit':
            c = 1
        else:
            c = 2
        
        #shap.summary_plot(shap_values,X_Train,show=False)
        #st.pyplot(bbox_inches='tight')
        #plt.clf()
        
        beeswarmplot = shap.summary_plot(shap_values[c], X_Train, plot_type="dot",auto_size_plot=False)
        #shap.plots.beeswarm(shap_values)
        
        #fig2 = shap.summary_plot(shap_values[c], X_Train, plot_type="dot",auto_size_plot=False,show=False,matplotlib=True)
        st.pyplot(beeswarmplot)
        #plt.clf()
        
        break
        