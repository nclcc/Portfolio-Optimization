# Empirical Sortino portfolio optimization

Python research utilities for constrained portfolio allocation, plus an experimental pairs-trading module.

## Install and test

Use Python 3.11 or 3.12 from the repository root:

```sh
python -m pip install -r Requirements.txt
python -m pytest -q
```

Tests use synthetic returns and mocked downloads, so they do not require network access or current market data. GitHub Actions runs the same numerical checks on Python 3.11 and 3.12. `arch` is needed for cointegration in the optional pairs module; it is imported only when that functionality is called.

## Allocation example

```python
from Maximization import PortfolioOptimization

assets = ['TAN', 'QCLN', 'CNRG', 'SMOG', 'ICLN', 'ACES', 'PBD', 'PBW', 'FAN', 'NZAC']
portfolio = PortfolioOptimization(assets, ['2022-05-01', '2022-06-01'], allow_for_short=False, tc=.01)
weights = portfolio.Results.result_max.x  # dimensionless, sum to one
amounts = portfolio.FinalWeights        # currency allocations after the capital prompt
```

Long-only bounds are [0,1], which permits the ten-asset example. Short-enabled bounds are [-1,2], with net weights summing to one and gross exposure `sum(abs(weights)) <= 2`. These are explicit research constraints, not an automatic choice of leverage for a real account. Optimizer failures and zero-downside portfolios raise errors instead of returning apparently valid allocations. SLSQP finds a local solution; global optimality is not established.

To work without downloads or a capital prompt:

```python
import pandas as pd
from Maximization import Maximization

returns = pd.DataFrame({'A': [.02, -.01, .01, -.02],
                        'B': [.01, -.02, .02, -.01],
                        'risk_free': [.0001] * 4})
result = Maximization({}, returns, allow_for_short=False, tc=0).result_max
```

The last column is always a **daily** risk-free return; preceding columns are simple daily risky-asset returns. For weights w, the objective is the negative sample mean of net excess returns divided by `sqrt(mean(min(net_excess, 0)**2))`. It uses realized portfolio downside, not the covariance of a covariance matrix or an ad hoc Student-t inflation. Student-t fits in `StockRanking` remain descriptive diagnostics and do not change this objective.

`tc` is a fraction in [0,1) applied as a haircut to positive daily portfolio returns before subtracting the daily risk-free return. It is **not** a brokerage fee on turnover. Modeling turnover fees requires previous holdings and a separate cost model. Both shorting branches use the supplied `tc` unchanged.

## Price alignment

`ImportData` explicitly selects adjusted Close prices in requested ticker order and computes percentage changes, rather than price differences. The ^TNX annual percentage yield is converted using `(1 + yield/100)**(1/252) - 1`, with the previous available observation used for the current interval. ^TNX is a ten-year yield proxy, not a realized daily cash-account return. Use an appropriate externally prepared risk-free series for substantive research.

## Rebalancing

```python
next_portfolio = PortfolioOptimization(
    assets, ['2022-06-01', '2022-07-01'], rebalance=True,
    rebalance_capital=portfolio.capital, rebalance_weights=portfolio.FinalWeights
)
```

`FinalWeights` retains its legacy name but contains currency amounts; `Results.result_max.x` contains proportions. Rebalancing assumes the supplied capital and holdings are already marked to the desired rebalance date.

## Scope of the evidence

Regression checks cover feasibility for ten assets, actual negative positions, gross exposure, the objective against hand calculations, cost handling, return ordering, daily-rate conversion, and error handling. They do not establish out-of-sample investment performance. No transaction-cost-aware backtest or execution system is provided. The pairs-trading code remains an exploratory in-sample heuristic; its signals are not evidence of a profitable or causally valid strategy. Historical results must be rerun after the return and objective corrections.
