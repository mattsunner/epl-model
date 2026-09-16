import numpy as np

from plforecast.evaluate.calibration import calibration_by_outcome, calibration_curve


def test_calibration_curve_perfect_calibration():
    rng = np.random.default_rng(0)
    n = 2000
    # Half the matches: model says 80% home win, and home actually wins 80% of the time.
    # Half: model says 20% home win, and home wins 20% of the time. Perfectly calibrated
    # on the home-win column by construction.
    probs = np.zeros((n, 3))
    outcomes = np.zeros(n, dtype=int)
    for i in range(n):
        p = 0.8 if i < n // 2 else 0.2
        probs[i] = [p, (1 - p) / 2, (1 - p) / 2]
        outcomes[i] = 0 if rng.random() < p else rng.choice([1, 2])

    curve = calibration_curve(probs, outcomes, n_buckets=10)

    near_80 = curve.filter((curve["mean_predicted"] > 0.75) & (curve["mean_predicted"] < 0.85))
    assert near_80.height == 1
    row = near_80.row(0, named=True)
    assert abs(row["empirical_frequency"] - 0.8) < 0.05
    assert row["ci_low"] <= row["empirical_frequency"] <= row["ci_high"]


def test_calibration_curve_rows_cover_only_buckets_with_data():
    probs = np.array([[0.5, 0.3, 0.2]])
    outcomes = np.array([0])

    curve = calibration_curve(probs, outcomes, n_buckets=10)

    assert curve.height <= 3  # at most one bucket per one of the three probability values
    assert (curve["n"] > 0).all()


def test_calibration_curve_confidence_interval_widens_with_fewer_samples():
    rng = np.random.default_rng(1)

    def _make(n: int):
        probs = np.tile([0.5, 0.25, 0.25], (n, 1))
        outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
        return calibration_curve(probs, outcomes, n_buckets=10)

    small = _make(20)
    large = _make(2000)

    small_width = (small["ci_high"] - small["ci_low"]).to_list()[0]
    large_width = (large["ci_high"] - large["ci_low"]).to_list()[0]
    assert small_width > large_width


def test_calibration_by_outcome_has_pooled_and_one_curve_per_class():
    rng = np.random.default_rng(3)
    n = 600
    probs = np.tile([0.5, 0.3, 0.2], (n, 1))
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])

    curves = calibration_by_outcome(probs, outcomes, n_buckets=10)

    assert set(curves["outcome"].to_list()) == {"pooled", "home", "draw", "away"}
    draw = curves.filter(curves["outcome"] == "draw")
    assert draw.height == 1  # every draw probability is 0.3 -> one bucket
    assert draw["n"].item() == n
    assert abs(draw["empirical_frequency"].item() - 0.3) < 0.06
