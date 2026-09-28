import numpy as np
import pytest
from evbtp import config
from evbtp.ev_model import build_from_random_inputs, generate_ev_data, load_matlab_ev_data
from evbtp.loadflow import RadialLoadFlow
from evbtp.markov import markov_weights
from evbtp.optimizers import run_gapso, run_hoa, run_mpa
from evbtp.problem import Problem
from evbtp.system import load_system
from evbtp.utils import mround


def scalar_loadflow(P, Q, sysd, vbase=12660.0):
    """Literal transcription of the MATLAB `lf` function in base_case_INDIA.m (independent of BIBC/Path)."""
    FB, TB, R, X, nb, nl = sysd.FB, sysd.TB, sysd.R, sysd.X, sysd.nb, sysd.nl
    V = np.ones(nb, complex); Z = R + 1j * X
    for _ in range(100):
        Vp = V.copy()
        Il = np.conj((P + 1j * Q) / (V * vbase))
        Ib = np.zeros(nl, complex)
        for k in range(nl - 1, -1, -1):
            Ib[k] = Il[TB[k] - 1]
            for m in range(k + 1, nl):
                if FB[m] == TB[k]:
                    Ib[k] += Ib[m]
        V[0] = 1.0
        for k in range(nl):
            V[TB[k] - 1] = V[FB[k] - 1] - Ib[k] * Z[k] / vbase
        if np.max(np.abs(np.abs(V) - np.abs(Vp))) < 1e-6:
            break
    return V, float(np.sum(np.abs(Ib) ** 2 * R))


@pytest.mark.parametrize("system", [33, 69])
def test_vectorised_loadflow_matches_scalar_reference(system):
    s = load_system(system)
    lf = RadialLoadFlow(s.FB, s.TB, s.R, s.X, s.nb)
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.6, 1.2, (6, 1))
    P = s.P_load[None, :] * scale; Q = s.Q_load[None, :] * scale
    P[:, 10] -= 8e5                                    # some reverse flow
    V, loss = lf.solve(P, Q)
    for t in range(6):
        Vr, lr = scalar_loadflow(P[t], Q[t], s)
        assert np.allclose(V[t], Vr, atol=1e-9)
        assert loss[t] == pytest.approx(lr, rel=1e-9)


def test_system_totals():
    assert load_system(33).P_load.sum() == pytest.approx(3.715e6)
    assert load_system(69).P_load.sum() == pytest.approx(3.8021e6, rel=1e-4)


def test_ev_deterministic_pipeline_reproduces_matlab_fleet():
    ev = load_matlab_ev_data()
    o = build_from_random_inputs(ev.Md, ev.EV_type, ev.t_arriv, ev.t_depart, ev.SOC_init)
    for k in ("P_EV_hourly", "V2G_avail_hourly", "SOC_desired", "E_demand", "t_dur"):
        assert np.array_equal(o[k], getattr(ev, k)), k
    assert ev.P_EV_hourly.max(axis=0) == pytest.approx([489.9, 517.8])


def test_generated_ev_fleet_is_plausible():
    ev = generate_ev_data(seed=7)
    assert ev.N_EV == 160 and ev.P_EV_hourly.shape == (24, 2)
    assert 350 < ev.P_EV_hourly.max(axis=0).max() < 700          # India-hub scale (README: ~450-550 kW)


# F1..F4 saved by the MATLAB project at each optimizer's gbest (GAPSO/HOA/MPA_INDIA_results.mat)
MATLAB_REF = {
    "gapso": ([33.0, 17.0, 0.3002377243204411, 1.0, 2.0, 19.0, 0.49103906144096343, 0.5179476684646652],
              (3.588128361615807, 33.68326391442743, 583114.8198088376, 6352.507719996784)),
    "hoa": ([33.0, 18.0, 0.3016287458916353, 0.9985319721341521, 19.0, 2.0, 0.49177878836750283, 0.5244107496852837],
            (3.592468138015326, 33.66343265934105, 583121.984922702, 6395.846308949338)),
    "mpa": ([16.0, 18.0, 0.7771884567156966, 0.5230614723133731, 19.0, 2.0, 0.4899000772420569, 0.5178061967064803],
            (3.6215112891600745, 33.41324682120661, 583338.106721899, 6344.803342306517)),
}


@pytest.mark.parametrize("name", list(MATLAB_REF))
def test_objective_matches_matlab_saved_results(name):
    x, ref = MATLAB_REF[name]
    out = Problem(quiet=True).detailed(np.array(x))
    assert out[1:] == pytest.approx(ref, rel=1e-9)


def test_repair_makes_buses_unique_and_in_range():
    pr = Problem(quiet=True)
    x = np.array([5, 5, 0.3, 0.4, 5, 5, 0.3, 0.3], float)
    r = pr.repair_solution(x)
    buses = r[[0, 1, 4, 5]]
    assert len(set(buses)) == 4 and buses.min() >= 2 and buses.max() <= 33
    assert pr.objective(x) == pr.objective(r)                    # repaired layout scores identically


def test_f4_depends_on_cs_capacity_and_hosting_cap_binds():
    pr = Problem(quiet=True)
    small = pr.detailed(np.array([10, 20, 0.3, 0.3, 5, 25, 0.06, 0.06]))[4]
    big = pr.detailed(np.array([10, 20, 0.3, 0.3, 5, 25, 0.7, 0.7]))[4]
    assert big > small * 2                                        # F4 no longer decision-invariant
    ok = pr.objective(np.array([10, 20, 0.6, 0.6, 5, 25, 0.55, 0.55]))
    over = pr.objective(np.array([10, 20, 1.0, 1.0, 5, 25, 0.55, 0.55]))
    assert over > ok + 10                                         # steep penalty past 35% of 3.715 MW


def test_markov_weights_sum_to_one():
    w = markov_weights(np.random.default_rng(1).random((200, 3)))
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_mround_half_away_from_zero():
    assert list(mround([0.5, 1.5, 2.5, 12.5])) == [1, 2, 3, 13]


@pytest.mark.parametrize("fn", [run_gapso, run_hoa, run_mpa])
def test_optimizers_respect_budget_bounds_and_seed(fn):
    pr = Problem(quiet=True)
    kw = dict(max_iter=None, nofe_budget=1500, seed=5, verbose=False)
    if fn is run_gapso:
        kw["track_details"] = False
    a, b = fn(pr, **kw), fn(pr, **kw)
    assert 1500 <= a.NOFE <= 1500 + 60                            # small overshoot inside one inner loop is allowed
    assert a.gbest_fit == b.gbest_fit and np.array_equal(a.gbest, b.gbest)   # deterministic per seed
    lb, ub = config.bounds(33)
    assert (a.gbest >= lb - 1e-9).all() and (a.gbest <= ub + 1e-9).all()
    assert a.gbest_fit == pytest.approx(pr.objective(a.gbest))    # reported layout == evaluated layout
    assert (np.diff(a.trace[:, 1]) <= 1e-12).all()                # best-so-far is monotone


def test_standalone_mode_runs_fixed_iterations():
    pr = Problem(quiet=True)
    r = run_mpa(pr, max_iter=5, seed=1, verbose=False)
    assert len(r.convergence) == 5 and r.NOFE == 50 + 5 * 100
