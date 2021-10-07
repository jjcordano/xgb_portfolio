# Portfolio builder that uses Bloomberg ESG metrics and XGBoost classifier

Link to the Web Application: [HERE](https://share.streamlit.io/jjcordano/xgb_portfolio/main/st_app.py)

## Intuition
This model replicates a methodology presented in the 2020 paper _[Seeking Signals from ESG Data](https://www.bloomberg.com/professional/blog/seeking-signals-from-esg-data/)_ by Bloomberg Quant Research. 

The presented model breaks down stock price prediction into a classification problem with 3 labels : positive returns, negative returns and no siginificant returns. Depending of the predicitons, a long position, a short position or no position will be taken for the given stock.

To make this classification, the dependant variables are a set of 21 annual ESG metrics published by Bloomberg regarding the 45 Eurostoxx 600 quoted (European) companies considered, spanning from 2007 to 2017.

The proposed machine learning algorithm for this task is XGBoost as it is a high performing model and [it can handle missing values without preprocessing](https://xgboost.readthedocs.io/en/latest/faq.html).

In the first part of this project, we replicate the methodology presented by Bloomberg Quantitative Research that gives equal weights to all stocks and compare the observed annual returns and Sharpe ratios to the Eurostoxx 600 benchmark.\
In the second part, we implement a method that applies more weights to stocks that have a higher probability of being in the predicted class.

We find that this method consistently outperforms the equal-weight method.

## Data
For a given year Y and a given company, the label is the class computed for Y+1. The classes are built using the annual returns of stocks in excess of Eurostoxx 600 returns. Excess returns above +18% are classified as 'long', those between +18% and -15% are classified as 'omit' and those below -15% are put in the 'short' class.  Annual returns are computed from Yahoo Finance price data.

Available annual ESG metric data from 2007 to 2017 has been gathered using a Bloomberg subscription for the 45 considered European companies. 

As stated in the original paper: "about 60% of ESG metrics have a Missing Rate (defined as the proportion of records for which the value is missing) of more than 50% in Bloomberg's ESG dataset on US companies over the past 14 years" (BQR, 2020). The reason for this is that companies do not have the obligation to publish information about Environmental, Social and Governance aspects of their business.

In our ESG metric dataset, missing values presents approx. 40% of all values in a total of 17,940 data points.

## XGBoost Model & Results
Like in the original paper, we build 4 annual portfolios for 2014, 2015, 2016 and 2017. For each year, the model is trained on the 7 previous years.

Depending on the class predicted by the model, a position is taken in each stock. 
- __Original method__: each stock in the same class is gven the same weight.
- __Probability-Weighted method__: each stock's weight is multiplied by the stock's probability of being in the given class divided by the total probabilities of stocks in that predicted class.

## The `PortfolioBuilder` class

The required dependencies to use the PortfolioBuilder code are listed
in `requirements.txt`. You can install the dependencies with the
following command-line code:

```bash
pip install -U -r requirements.txt
```

The Kmeans class is accessible in the __code__ file.

### Instantiating a `PortfolioBuilder` object
To declare an instance of the `PortfolioBuilder` XGBoost model, the class takes the following arguments:
- _benchmark_: the desired benchmark index. Either 'eurostoxx_600' or 'russell_3000'. Defaults to 'eurostoxx_600'.
- _probability_weighted_: Boolean. If true, stock weights are weighted by the probability of each prediction to be in the class, else equally weighted portfolio (original method presented by Bloomberg Quantitative Research).
- _bucket_delimiters_: Threshold separators between _long_, _omit_ and _short_ classes.
- _short_limit_: Price under which stocks predicted into _short_ class are omitted.

### Methods:
- _PortfolioBuilder.init_data()_: loads ESG metric data and price data from CSV files and initializes dataset.
- _PortfolioBuilder.fit_portfolio()_: for agiven year, trains XGBoost model on 7 previous years, makes predictions for given year and evaluates portfolio's observed returns. Parameters are : _year_, _xgb_params_ and _print_accuracy_.
- _PortfolioBuilder.predict()_: takes an array of points to be labeled as input. Assigns each point to its closest centroid.
- _PortfolioBuilder.get_portfolio_return()_.
- _PortfolioBuilder.get_dict_weights()_.
- _PortfolioBuilder.get_portfolio_sharpe_ratio()_.


