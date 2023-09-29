
Portfolio Optimization with Sortino Index. v.0.1
================================================

Contents
========

* [Just for fun](#just-for-fun)
* [A walkthrough in the PortofolioOptimization class](#a-walkthrough-in-the-portofoliooptimization-class)
	* [Import the class](#import-the-class)
	* [Import the list of financial products](#import-the-list-of-financial-products)
	* [Run the Portfolio Analysis](#run-the-portfolio-analysis)
	* [Input the initial capital](#input-the-initial-capital)
	* [Final weights](#final-weights)
	* [Rebalance portofolio weights](#rebalance-portofolio-weights)
	* [Rebalanced weights](#rebalanced-weights)
* [PortfolioOptimization class features](#portfoliooptimization-class-features)
* [Have fun!](#have-fun)

# Just for fun


This file contains a brief overview of an active strategy of portfolio optimization based on Sortino index maximization and Student-t data generating process.

The classes in current repository allow for weights computation based on daily observations and any time interval specified in the format 'yy-mm-dd'. In particular: 
- Optimal weights
- Initial Capital
- Capital rebalancing
- Transaction costs as a percentage (%) of returns.



# A walkthrough in the PortofolioOptimization class

## Import the class


Import packages suit your project together with the PortofolioOptimization class.

```python
# Import packages 
from Maximization import PortfolioOptimization
```
## Import the list of financial products


Take the list of symbols from the list of financial products availables on Yahoo Finance.
For example, trending tickers:  https://finance.yahoo.com/lookup

```python
# Select a list of Green ETFs as specified on https://finance.yahoo.com
ETF_list = ['TAN', 'QCLN', 'CNRG', 'SMOG', 'ICLN', 'ACES', 'PBD', 'PBW', 'FAN', 'NZAC']
```
## Run the Portfolio Analysis


In order to run the class successfully, the following inputs must be specified:
- At least two stock symbols and date range in 'yy-mm-dd' format
- A boolean 'allow_for_shorting' is set to default to 'False'
- A value 'tc' between 0 and 1, as a fixed percentage of returns. It is set to default to 0.01.


```
# Include the ETF_list and the timespan of interest.
OptimizedPortfolio = PortfolioOptimization(ETF_list, ['2022-05-01', '2022-05-31'])
```
## Input the initial capital


When running the class with the 'rebalance' set to False, the workflow of the classwill return the following message:


*'To maximize the excessive returns with the current set of financial products, specify the initial capital: '*



**IMPORTANT**: Specify the initial capital with plain ciphers - neither commas nor other special characters.
## Final weights


Depending on whether the boolean 'allow_for_short' is set to 'True' or not, the workflow of the classwill return the following message:


***'With leverage, you should invest this month respectively: xxx euros.'***



***'With no leverage, you should invest this month respectively: xxx euros.'***


## Rebalance portofolio weights


To rebalance the portfolio, run again the Portfolio optimization at t=2.
Activate the rebalance mode by setting:
- The boolean 'rebalance' to True. The default is False.
- The 'rebalance_capital' equal to the capital of the previous optimization.
- The 'rebalance_weights' equal to the weights of the previous optimization


```
# Include the ETF_list and the timespan of interest and the elements specified above
PortfolioOptimization(ETF_list, ['2022-06-01', '2022-06-30'], rebalance=True, 
rebalance_capital=OptimizedPortfolio.capital,
rebalance_weights=OptimizedPortfolio.FinalWeights)
```
## Rebalanced weights


The current mode of the class will return the following messages:


***'With leverage, you should add/subtract to the previous portfolio weights respectively:: xxx euros.'***



***'With no leverage, you should add/subtract to the previous portfolio weights respectively:: xxx euros.'***


# PortfolioOptimization class features


The current version of the class allows to call the following object properties:

- PortfolioOptimization.capital: float, initial capital specified by the user
- PortfolioOptimization.FinalWeights: ndarray, array of portfolio weights.
- PortfolioOptimization.Results: class OptimizeResult from Scipy library, class containing all the relevant attributes of the maximization


The class runs with a SLSQP maximization algorithm and linear constraint imposed on the weights.The other requirements for this class to run are contained in 'requirements.txt'.
# Have fun!


In case you want to discuss this code further, feel free to reach me following the link on my GitHub profile.