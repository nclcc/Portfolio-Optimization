# Import packages
import itertools
import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt

# Warnings
warnings.filterwarnings("ignore")

# Import global statement
from statistics import mean
from arch.unitroot.cointegration import phillips_ouliaris
from scipy.stats import t
from scipy.optimize import minimize, LinearConstraint


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
############################################# Sortino Index Maximization ##############################################

class PortfolioOptimization:
    def __init__(self, StockMnemonics: list, period: list, allow_for_short=False, tc=0.01,
                 rebalance=False, rebalance_capital=None, rebalance_weights=None):
        print("-" * 400, "\n New portfolio optimization has started for period", " until ".join(period),
              "\n", "Start downloading data.")
        # Initialize the time counting
        start = time.time()

        # Import data with the specified mnemonics
        InfoSet = ImportData(StockMnemonics, period)

        # Return classified DataSet
        ClassifiedInfoSet = StockRanking(InfoSet.returns)

        # Return the final weights
        self.Results = Maximization(ClassifiedInfoSet.share, InfoSet.returns, allow_for_short=allow_for_short, tc=tc)

        # Return the processing time
        stop = time.time()
        print("-" * 400, "\n With the current algorithm for the financial products:\n", ", ".join(StockMnemonics),
              "\n the portfolio optimization was implemented in seconds:", np.round(stop - start, 2), ".\n", "-" * 400)

        if rebalance:
            # Define weights
            self.FinalWeights = np.round(self.Results.result_max.x * rebalance_capital, 4)
            diff = np.round(self.FinalWeights - rebalance_weights, 4)
            val = [str(x) for x in diff.tolist()]

            # Store amount of capital
            self.capital = float(rebalance_capital)

            # Print results
            if allow_for_short:
                print(" With leverage, you should add/subtract to the previous portfolio weights respectively:\n",
                      ", ".join(val),
                      " euros.\n", "-" * 400)
            else:
                print(" With no leverage, you should add/subtract to the previous portfolio weights respectively:\n",
                      ", ".join(val),
                      " euros.\n", "-" * 400)

        else:

            InputString = input(" To maximize the excessive returns with the current set of financial products, "
                                "specify the initial capital:")

            # Store the Input string
            self.capital = float(InputString)

            # Compute shares
            if type(float(InputString)) == float:
                self.FinalWeights = np.round(self.Results.result_max.x * float(InputString), 4)

                if all(self.FinalWeights / InfoSet.dataframe.iloc[-1, :len(StockMnemonics)].to_numpy()
                       < 1):
                    warnings.warn("The initial capital does not allow to buy the full property of all the stocks."
                                  " Consider buying CFDs. ")
                elif any(self.FinalWeights / InfoSet.dataframe.iloc[-1, :len(StockMnemonics)].to_numpy()
                         < 1):
                    warnings.warn("The initial capital does not allow to buy"
                                  "the full property of stocks for all of them. Consider buying CFDs. ")

                # Change type per
                val = [str(x) for x in self.FinalWeights.tolist()]
            else:
                raise TypeError("Please specify the cipher without any comma or special characters.")

            # Print results
            if allow_for_short:
                print("-" * 400, "\n With leverage, you should invest this month respectively:\n", ", ".join(val),
                      " euros.")
            else:
                print("-" * 400, "\n With no leverage, you should invest this month respectively:\n", ", ".join(val),
                      " euros.")


class Maximization:
    def __init__(self, shares: dict, returns: pd.DataFrame, allow_for_short=True, tc=0.01):
        # Define initial conditions for maximization
        self.result_max = None
        weights = np.array([[0.1] * (returns.shape[1] - 1)]).transpose()
        self.maximization_problem(weights, returns, shares, allow_for_short=allow_for_short, tc=tc)

    def maximization_problem(self, weights: np.array, returns: pd.DataFrame, shares: dict, allow_for_short=True,
                             tc=0.01):

        # Define constraints
        # Matrix constraint
        A = np.ones([1, weights.shape[0]])

        # Upper and lower bound equal => equality constraint
        ub_lb = np.ones(1)

        # Call class linear constraint
        C = LinearConstraint(A, ub_lb, ub_lb)

        # Impose bound on weights depending on boolean: "allow_for_short".
        if allow_for_short:
            bounds = [(0.2, 0.9)] * weights.shape[0]
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, shares, allow_for_short),
                                       method='SLSQP', bounds=bounds,
                                       constraints=C)

        else:
            bounds = [(0.2, 0.9)] * weights.shape[0]
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, shares, tc),
                                       method='SLSQP', bounds=bounds,
                                       constraints=C)

    @staticmethod
    def maximand_function(weights: np.array, returns: pd.DataFrame, shares: dict, tc=0.01):

        # Define the excess returns with respect to the risk-free
        diff = returns.to_numpy()[:, :-1] - returns.to_numpy()[:, -1].reshape([returns.to_numpy()[:, -1].shape[0], 1])

        # Define the excess negative returns
        diff_min = returns[returns < 0].iloc[:, :-1].to_numpy() - returns.iloc[:, -1].to_numpy().reshape(
            [returns.to_numpy()[:, -1].shape[0], 1])

        # Extract the variance-covariance matrix
        var_cov = np.ma.cov(np.ma.masked_invalid(diff_min.transpose()))

        # Expand dimension of weights matrix
        weights = weights[:, np.newaxis]

        # Define the maximand function as a linear combination Sortino Ratio with t-Student Nuisances
        arg = - (1 - tc) * diff.mean(axis=0).reshape(
            [returns.iloc[:, :-1].shape[1], 1]).transpose() @ weights @ np.linalg.inv(
            weights.transpose() @ np.multiply(
                mean(shares['dof']) / (mean(shares['dof']) - 2),
                np.cov(np.ma.getdata(var_cov))) @ weights)
        return arg


class ImportData:
    def __init__(self, StockMnemonics: list, period: list):
        # Add the risk-free interest rate mnemonic symbol
        RiskFreeMnemonics = ['^TNX']  # Treasury Yield 10 Years
        StockMnemonics = StockMnemonics + RiskFreeMnemonics

        # Download the mnemonics specified by the user
        self.dataframe = yf.download(" ".join(StockMnemonics), start=period[0], end=period[1])
        self.returns = self.dataframe.iloc[:, :len(StockMnemonics) - 1].diff().dropna().assign(
            RiskFree=self.dataframe.iloc[1:, len(StockMnemonics) - 1])
        self.returns.columns = StockMnemonics

        # Redefine the columns
        self.prices = self.dataframe.iloc[:, :len(StockMnemonics)]
        self.prices.columns = StockMnemonics


class StockRanking:
    def __init__(self, returns: pd.DataFrame):
        # Create a set of labels for each return
        label_column = []
        for i in range(returns.shape[1] - 1):
            temp = self.classify_distribution(returns.iloc[:, i].to_numpy())
            label_column.append(temp)

        # Extract the alpha and beta
        self.share = self.alpha_beta(label_column)

    @staticmethod
    def alpha_beta(label_column: list):

        # Count the elements
        alpha = sum([x.count("Normal") for x in label_column])
        beta = sum([x.count("Student-t") for x in label_column])

        return {'alpha': float(alpha / (alpha + beta)), 'beta': float(beta / (alpha + beta)),
                "dof": [x[1] for x in label_column]}

    @staticmethod
    def classify_distribution(returns: np.array):

        # Extract parameters describing the centered t-student distribution
        # Compute observed variance
        varr = returns.var().round(3)

        # Observed degrees of freedom
        dff, _, _ = t.fit(returns, loc=returns.mean(), scale=returns.var())

        # Classify
        if dff > 30:
            return ["Normal", dff]
        else:
            return ["Student-t", dff]


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
############################################# Pairs Trading ###########################################################

class PairTrading:
    def __init__(self, StockMnemonics: list, period: list, thresholds=1, windows=[5, 30], m=0.4,
                 significance=0.05, path="Signals.png"):
        print("-" * 400, "\n New portfolio optimization has started for period", " until ".join(period),
              "\n", "Start downloading data.")

        # Initialize the time counting
        start = time.time()

        # Import data with the specified mnemonics
        InfoSet = ImportData(StockMnemonics, period)

        # Cointegrated_pairs
        self.cointegrated_pairs = self.cointegration(InfoSet.prices,
                                                     [list(x) for x in list(itertools.combinations(list(StockMnemonics),
                                                                                                   2))],
                                                     significance=significance)

        # Compute signals
        self.signals = self.compute_ratios(self.cointegrated_pairs, InfoSet,
                                           windows=windows, m=m)

        # Return signal
        self.save_results(InfoSet.prices)

        stop = time.time()
        print("-" * 400, "\n With the current algorithm for the financial products:\n", ", ".join(StockMnemonics),
              "\n the portfolio optimization was implemented in seconds:", np.round(stop - start, 2), ".\n", "-" * 400)

    def save_results(self, prices: pd.DataFrame, path="Signals.png"):
        Signals, axs = plt.subplots(len(self.cointegrated_pairs), 1, figsize=(10, 8))

        for j in range(len(self.cointegrated_pairs)):
            axs[j].plot(prices[self.cointegrated_pairs[j]])
            axs[j].plot(self.estimated[j].transpose().iloc[:, 0].replace(0, np.nan),
                        color='g', linestyle='None', marker='^', label="Buy signal")
            axs[j].plot(self.estimated[j].transpose().iloc[:, 1].replace(0, np.nan), color='r',
                        linestyle='None', marker='^', label="Sell signal")
            axs[j].set_title("Pairs strategy for the symbols %s" % self.cointegrated_pairs[j])
            axs[j].legend()

        plt.savefig(path)

    def compute_ratios(self, cointegrated_pairs: list, InfoSet: classmethod, windows=[5, 30], m=0.2):

        signal = []
        for j in range(len(cointegrated_pairs)):

            # Name the shares
            Shares_1 = InfoSet.prices[cointegrated_pairs[j]].iloc[:, 1]
            Shares_0 = InfoSet.prices[cointegrated_pairs[j]].iloc[:, 0]

            if all(Shares_0 > Shares_1):

                # Signals
                buyR, sellR = self.computation_utils(Shares_0, Shares_1, windows=windows, m=m)

                # Store results
                signal.append(pd.DataFrame([buyR, sellR]))

            elif all(Shares_0 < Shares_1):

                # Signals
                buyR, sellR = self.computation_utils(Shares_1, Shares_0, windows=windows, m=m)

                # Store results
                signal.append(pd.DataFrame([buyR, sellR]))

            else:
                raise Warning("This set of variables does not allow Pairs-trading. Financial variables overlap.")

        self.estimated = signal

    @staticmethod
    def computation_utils(Shares_higher: pd.DataFrame,
                          Shares_lower: pd.DataFrame, windows=[5, 30], m=0.4):

        # Volatility Threshold
        h = np.std(np.log(Shares_higher)) / np.std(np.log(Shares_lower))

        # Ratio value
        S2 = np.log(Shares_higher) - h * np.log(Shares_lower)

        # Compute moving averages rather than fixed ones
        ratios_mavg5 = S2.rolling(window=windows[0], center=False).mean()
        ratios_mavg60 = S2.rolling(window=windows[1], center=False).mean()
        std_60 = S2.rolling(window=windows[1], center=False).std(doff=1)
        zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60

        # Thresholds
        sma = S2.rolling(window=windows[1], min_periods=windows[1]).mean()[:windows[1]]
        rest = S2[windows[1]:]
        EXPWm = pd.concat([sma, rest]).ewm(span=windows[1], adjust=False).mean()
        # Final calculations for upper and lower threshold
        up = EXPWm + m * std_60
        down = EXPWm - m * std_60

        # Set the signal for buy
        buy = S2.copy()
        sell = S2.copy()
        buy[zscore_60_5 > -down] = 0
        sell[zscore_60_5 < up] = 0

        # Buy the ratios
        buyR = 0 * Shares_higher.copy()
        sellR = 0 * Shares_higher.copy()

        # When you buy the ratio, you buy stock Shares_1 and sell Shares_0
        buyR[buy != 0] = Shares_higher[buy != 0]
        sellR[buy != 0] = Shares_lower[buy != 0]

        # When you sell the ratio, you sell stock Shares_1 and buy Shares_0
        buyR[sell != 0] = Shares_lower[sell != 0]
        sellR[sell != 0] = Shares_higher[sell != 0]

        return buyR, sellR

    @staticmethod
    def cointegration(prices: pd.DataFrame, combine_coint: list, significance=0.05):

        # Perform cointegration matrix
        cointegration_matrix = []
        for j in combine_coint:
            cointegration_matrix.append(phillips_ouliaris(prices[j].iloc[:, 0], prices[j].iloc[:, 1]).pvalue)

        return list(map(lambda x: x[1],
                        [[x, h] for x, h in zip(cointegration_matrix,
                                                combine_coint) if x < significance]))
