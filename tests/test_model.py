import numpy as np

from mccast import ca, markov
from mccast.suitability import Suitability, distance_to
from mccast.validate import urban_change_metrics


def _city(n=60):
    """Urban core in a 60x60 agro grid with a water strip."""
    g = np.full((n, n), 2, dtype=np.uint8)
    g[:, :3] = 4
    g[25:35, 25:35] = 3
    return g


def test_transition_matrix_rows_sum_to_one():
    a, b = _city(), _city()
    b[20:25, 25:35] = 3
    P = markov.transition_matrix(a, b, 5)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[1, 2] > 0 and P[2, 2] == 1  # agro -> urban happened; urban stayed


def test_rescale_is_stochastic():
    P = markov.transition_matrix(_city(), _city(), 5)
    Q = markov.rescale(P, 30, 8)
    assert np.allclose(Q.sum(axis=1), 1) and (Q >= 0).all()


def test_ca_allocates_exact_demand_near_existing_urban():
    g = _city()
    allowed = (g > 0) & (g != 4)
    sim = ca.simulate(g, allowed, lambda u: np.ones(g.shape, np.float32), 100, steps=5)
    assert (sim == 3).sum() - (g == 3).sum() == 100
    assert (sim[g == 4] == 4).all()  # water untouched
    assert distance_to(g == 3, 1.0)[(sim == 3) & (g != 3)].max() < 6  # grows from the edge


def test_suitability_prefers_cells_near_urban():
    a = _city()
    b = a.copy()
    b[22:25, 25:35] = 3  # growth hugs the core
    allowed = a != 4
    f = {"dist_urban": distance_to(a == 3, 30.0)}
    s = Suitability.fit(a == 3, b == 3, allowed, f, bins=6)
    score = s.predict(f)
    assert score[23, 30] > score[5, 50]
    assert Suitability.from_dict(s.to_dict()).base_rate == s.base_rate


def test_perfect_simulation_scores_one():
    a = _city()
    b = a.copy()
    b[20:25, 25:35] = 3
    m = urban_change_metrics(a, b, b)
    assert m["figure_of_merit"] == 1.0 and m["kappa_urban"] == 1.0


def test_ca_warns_and_carries_over_when_fringe_is_exhausted():
    import pytest

    g = _city()
    allowed = g == 3  # nowhere to grow
    with pytest.warns(UserWarning):
        ca.simulate(g, allowed, lambda u: np.ones(g.shape, np.float32), 10, steps=2)
