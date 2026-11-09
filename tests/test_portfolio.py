import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from Maximization import Maximization, ImportData


def observations(n=10):
    rng = np.random.default_rng(42)
    # A common loss factor guarantees observed downside for all fully invested portfolios.
    common = rng.normal(.0004, .012, 400)
    risky = common[:, None] + rng.normal(0., .002, (400, n))
    return pd.DataFrame(np.c_[risky, np.full(400, .0001)])


def test_documented_ten_asset_case_is_feasible():
    result = Maximization({}, observations(), allow_for_short=False, tc=.01).result_max
    assert result.success
    assert_allclose(result.x.sum(), 1., atol=1e-7)
    assert np.all(result.x >= 0)


def test_shorting_actually_permits_negative_weights():
    # Equal downside exposures and opposite means make shorting the second asset useful.
    common = np.tile([-.02, .01, -.01, .02], 100)
    data = pd.DataFrame({'good': common + .002, 'bad': common - .002, 'rf': 0.})
    long_only = Maximization({}, data, allow_for_short=False, tc=0).result_max
    short = Maximization({}, data, allow_for_short=True, tc=0).result_max
    assert short.x[1] < -.1
    assert np.abs(short.x).sum() <= 2 + 1e-7
    assert short.fun < long_only.fun


def test_objective_matches_hand_computed_sortino():
    data = pd.DataFrame([[.02, -.01, .001], [-.01, -.03, .001], [.01, .02, .001]])
    portfolio = np.array([.0125, -.015, .0125])
    excess = portfolio - .01 * np.maximum(portfolio, 0) - .001
    expected = -excess.mean() / np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    assert_allclose(Maximization.maximand_function([.75, .25], data, {}, .01), expected)


def test_tc_is_used_in_shorting_branch():
    data = observations(3)
    result = Maximization({}, data, allow_for_short=True, tc=.02).result_max
    assert_allclose(result.fun, Maximization.maximand_function(result.x, data, {}, .02))


@pytest.mark.parametrize('tc', [-1., 1., np.nan])
def test_invalid_cost_is_rejected(tc):
    with pytest.raises(ValueError, match='tc'):
        Maximization({}, observations(2), tc=tc)


def test_zero_downside_is_reported_as_undefined():
    with pytest.raises(ValueError, match='undefined'):
        Maximization({}, pd.DataFrame([[.01, .02, 0.]] * 5), allow_for_short=False)


def test_download_uses_close_prices_in_requested_order_and_daily_risk_free(monkeypatch):
    dates = pd.date_range('2024-01-01', periods=4)
    closes = pd.DataFrame({'B': [200, 202, 198, 204], '^TNX': [5., 5.1, 5.2, 5.3], 'A': [100, 102, 99, 103]}, index=dates)
    downloaded = pd.concat({'Close': closes, 'Open': closes * .9}, axis=1)
    monkeypatch.setattr('Maximization.yf.download', lambda *a, **kw: downloaded)
    imported = ImportData(['A', 'B'], ['2024-01-01', '2024-02-01'])
    assert list(imported.returns.columns) == ['A', 'B', '^TNX']
    assert_allclose(imported.returns.iloc[0, :2], [.02, .01])
    assert_allclose(imported.returns.iloc[0, 2], 1.05 ** (1/252) - 1)
    assert imported.prices.columns.tolist() == ['A', 'B']


def test_optimizer_failure_is_not_exposed_as_valid_weights(monkeypatch):
    from scipy.optimize import OptimizeResult
    monkeypatch.setattr('Maximization.minimize', lambda *a, **kw: OptimizeResult(success=False, message='failed', fun=0.))
    with pytest.raises(RuntimeError, match='failed'):
        Maximization({}, observations(2))
