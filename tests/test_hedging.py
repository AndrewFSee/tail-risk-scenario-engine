import numpy as np
import pytest

from tailrisk.config import HedgeInstrument
from tailrisk.hedging import InverseETF, PutOption, black_scholes_put, build_hedges, evaluate_hedges


def test_black_scholes_put_matches_reference_and_parity():
    # S=100, K=95, 1 year, 20% vol, 5% rate: d1=0.6065, d2=0.4065 -> 3.71
    p = float(black_scholes_put(100.0, 95.0, 1.0, 0.20, 0.05))
    assert p == pytest.approx(3.71, abs=0.01)
    # put-call parity with the matching call from the same d1/d2
    from scipy import stats
    d1 = (np.log(100 / 95) + (0.05 + 0.02) * 1.0) / 0.2
    call = 100 * stats.norm.cdf(d1) - 95 * np.exp(-0.05) * stats.norm.cdf(d1 - 0.2)
    assert call - p == pytest.approx(100 - 95 * np.exp(-0.05), abs=1e-9)
    assert float(black_scholes_put(0.9, 0.95, 0.0, 0.2)) == pytest.approx(0.05)   # expired -> intrinsic


def test_put_premium_follows_implied_vol():
    calm = PutOption(asset_index=0, moneyness=0.95, implied_vol=0.15).premium()
    stressed = PutOption(asset_index=0, moneyness=0.95, implied_vol=0.40).premium()
    assert 0.001 < calm < 0.01 and stressed > 3 * calm
    assert PutOption(asset_index=0, premium_pct=0.012).premium() == 0.012


def test_put_payoff_at_expiry_and_before():
    terminal = np.log(np.array([[0.80], [1.10]]))          # -20% and +10% paths
    put = PutOption(asset_index=0, moneyness=0.95, implied_vol=0.2, tenor_days=30)
    expired = put.payoff(terminal, horizon_days=21)          # 30 calendar ~ 21 trading days
    assert expired[0] == pytest.approx(0.15 - put.premium())
    assert expired[1] == pytest.approx(-put.premium())
    longer = PutOption(asset_index=0, moneyness=0.95, implied_vol=0.2, tenor_days=91).payoff(terminal, 21)
    assert longer[1] > -PutOption(asset_index=0, moneyness=0.95, implied_vol=0.2, tenor_days=91).premium()


def test_inverse_etf_uses_simple_returns():
    r = np.log(np.array([[[1.10], [0.90]]]))                # +10% then -10% (simple)
    got = InverseETF(asset_index=0, allocation=1.0).payoff_from_steps(r)
    assert got[0] == pytest.approx((1 - 0.10) * (1 + 0.10) - 1)   # -1%: daily-rebalanced decay


def test_build_hedges_skips_unpriceable_put():
    specs = [HedgeInstrument(kind="put", underlying="SPY", moneyness=0.95, tenor_days=30),
             HedgeInstrument(kind="inverse_etf", ticker="SPY", allocation=0.05),
             HedgeInstrument(kind="put", underlying="XYZ")]   # not in the portfolio
    with pytest.warns(UserWarning):
        hedges = build_hedges(specs, ["SPY", "TLT"], implied_vol=None)
    assert [type(h).__name__ for h in hedges] == ["InverseETF"]
    hedges = build_hedges(specs, ["SPY", "TLT"], implied_vol=0.18)
    assert isinstance(hedges[0], PutOption) and hedges[0].implied_vol == 0.18


def test_evaluate_hedges_put_cuts_the_deep_tail():
    rng = np.random.default_rng(0)
    steps = rng.standard_t(4, size=(20_000, 21, 1)) * 0.012
    rows = {r.name: r for r in evaluate_hedges(steps, np.array([1.0]),
                                               [PutOption(asset_index=0, implied_vol=0.2)])}
    put = PutOption(asset_index=0, implied_vol=0.2)
    assert rows["put_0.95m"].cvar_99 < rows["unhedged"].cvar_99
    # a put on the whole position floors the loss at the strike plus the premium
    assert rows["put_0.95m"].var_99 <= 0.05 + put.premium() + 1e-9
    assert rows["put_0.95m"].cvar_99 <= 0.05 + put.premium() + 1e-9
