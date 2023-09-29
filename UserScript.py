# Import packages you like the most on top of the class for PortofolioOptimization
from Maximization import PortfolioOptimization, PairTrading

# Take the list of symbols from the list of Stocks available on Yahoo Finance.
# For example, trending tickers:  https://finance.yahoo.com/lookup
ETF_list = ['TAN', 'QCLN', 'CNRG', "SMOG", "ICLN", "ACES", "PBD", "PBW", "FAN", "NZAC"]

# Run the Portfolio Analysis by setting:
# 1. At least two stock symbols and date range in yy-mm-dd format
# 2. A boolean 'allow_for_shorting' is set to default to 'False'.
# 3. A value 'tc' between 0 and 1, as a fixed share of returns. It is set to default to 0.01.
OptimizedPortfolio = PortfolioOptimization(ETF_list, ['2022-05-01', '2022-05-31'])

# Extract the optimizing weights
q_1 = OptimizedPortfolio.FinalWeights

# NOTE: After running the file, specify the initial capital with plain ciphers
# - neither commas nor other special characters.

# To rebalance the portfolio, run again the Portfolio optimization at t=2,
# Activate the rebalance mode by setting:
# 1. The boolean "rebalance" to True. The default is False.
# 2. The "rebalance_capital" equal to the capital of the previous optimization
# 3. The "rebalance_weights" equal to the weights of the previous optimization
OptimizedPortfolio = PortfolioOptimization(ETF_list, ['2022-06-01', '2022-06-30'],
                                           rebalance=True, rebalance_capital=OptimizedPortfolio.capital,
                                           rebalance_weights=OptimizedPortfolio.FinalWeights)



# Pair trading -strategy
PairTrading = PairTrading(ETF_list, ['2021-01-01', '2022-06-30'], m=1.75, significance=0.01)
