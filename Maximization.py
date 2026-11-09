# Import packages
import itertools
import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt

# Warnings


# Import global statement
from statistics import mean

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
    """Maximize the empirical daily Sortino ratio under explicit weight bounds.

    The legacy ``shares`` argument is retained for compatibility; Student-t
    fits are descriptive and do not modify the empirical downside deviation.
    ``tc`` is a haircut on positive daily portfolio returns, not turnover fees.
    """

    def __init__(self, shares: dict, returns: pd.DataFrame, allow_for_short=True, tc=0.01):
        self.result_max = None
        weights = np.full(returns.shape[1] - 1, 1 / (returns.shape[1] - 1))
        self.maximization_problem(weights, returns, shares, allow_for_short, tc)

    def maximization_problem(self, weights, returns, shares, allow_for_short=True, tc=0.01):
        values = returns.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
            raise ValueError("returns must contain finite observations, risky assets and a final daily risk-free column")
        if not np.isfinite(tc) or not 0 <= tc < 1:
            raise ValueError("tc must be in [0, 1)")
        n_assets = values.shape[1] - 1
        initial = np.asarray(weights, dtype=float).reshape(-1)
        if initial.shape != (n_assets,) or not np.isfinite(initial).all():
            raise ValueError("Initial weights must match the number of risky assets")
        bounds = [(-1., 2.) if allow_for_short else (0., 1.)] * n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.}]
        if allow_for_short:
            # Net investment remains one; gross exposure is capped at two.
            constraints.append({'type': 'ineq', 'fun': lambda w: 2. - np.abs(w).sum()})
        result = minimize(self.maximand_function, initial, args=(returns, shares, tc),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-10})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
        if abs(result.x.sum() - 1.) > 1e-6 or (allow_for_short and np.abs(result.x).sum() > 2. + 1e-6):
            raise RuntimeError("Optimizer returned infeasible weights")
        portfolio = values[:, :-1] @ result.x
        excess = portfolio - tc * np.maximum(portfolio, 0.) - values[:, -1]
        if np.sqrt(np.mean(np.minimum(excess, 0.) ** 2)) <= 1e-12:
            raise ValueError("Optimal portfolio has zero empirical downside deviation; its Sortino ratio is undefined")
        self.result_max = result
        return result

    @staticmethod
    def maximand_function(weights, returns, shares=None, tc=0.01):
        values = returns.to_numpy(dtype=float)
        portfolio = values[:, :-1] @ np.asarray(weights).reshape(-1)
        net_excess = portfolio - tc * np.maximum(portfolio, 0.) - values[:, -1]
        downside = np.sqrt(np.mean(np.minimum(net_excess, 0.) ** 2))
        if downside <= 1e-12:
            raise ValueError("Portfolio has zero empirical downside deviation; its Sortino ratio is undefined")
        return float(-net_excess.mean() / downside)


class ImportData:
    """Adjusted simple daily returns with a daily rate derived from ^TNX.

    ^TNX is an annualized percentage yield proxy, not a tradable cash account.
    The prior observation is converted on a 252-trading-day convention.
    """

    def __init__(self, StockMnemonics: list, period: list):
        symbols = list(StockMnemonics)
        if not symbols or len(set(symbols)) != len(symbols) or '^TNX' in symbols:
            raise ValueError("Provide unique risky-asset tickers without ^TNX")
        raw = yf.download(symbols + ['^TNX'], start=period[0], end=period[1], auto_adjust=True)
        try:
            close = raw['Close'].reindex(columns=symbols + ['^TNX']).astype(float)
        except (KeyError, AttributeError) as error:
            raise ValueError("Price data must contain a Close column for each requested ticker") from error
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if len(close) < 3 or (close[symbols] <= 0).any().any():
            raise ValueError("At least three aligned positive price observations are required")
        if (close['^TNX'] <= -100).any():
            raise ValueError("Annual yield must exceed -100 percent")
        self.prices = close[symbols]
        self.dataframe = close
        returns = close[symbols].pct_change(fill_method=None)
        daily_risk_free = np.expm1(np.log1p(close['^TNX'].shift(1) / 100.) / 252.)
        self.returns = returns.assign(**{'^TNX': daily_risk_free}).dropna()


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
        if not self.cointegrated_pairs:
            return
        Signals, axes = plt.subplots(len(self.cointegrated_pairs), 1, figsize=(10, 8), squeeze=False)
        axs = axes[:, 0]

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
        return signal

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
        std_60 = S2.rolling(window=windows[1], center=False).std(ddof=1)
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

        from arch.unitroot.cointegration import phillips_ouliaris

        # Perform cointegration matrix
        cointegration_matrix = []
        for j in combine_coint:
            cointegration_matrix.append(phillips_ouliaris(prices[j].iloc[:, 0], prices[j].iloc[:, 1]).pvalue)

        return list(map(lambda x: x[1],
                        [[x, h] for x, h in zip(cointegration_matrix,
                                                combine_coint) if x < significance]))
