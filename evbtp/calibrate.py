"""Calibration: normalisation bases + objective weights (port of calibrate_and_weights_INDIA.m).

Samples N feasible random solutions, takes the 5th/95th percentile of F1..F4 as
range-normalisation bases, computes every weighting method in evbtp/weights.py from the same
samples (w_all), copies the one named by config.WEIGHT_METHOD (default 'ahp_critic') into
w_fixed, and saves everything to results/calibration_<system>bus.npz.
"""
from dataclasses import asdict
import numpy as np
from . import config
from .config import DEFAULT_WEIGHTS, N_VAR, VAR_IS_INT
from .problem import Problem, Calibration, calibration_path, EPS
from .utils import mround
from .weights import compute_all


def calibrate(problem: Problem, n_samples=1000, seed=42, verbose=True, save=True,
              weight_method=None) -> Calibration:
    weight_method = weight_method or config.WEIGHT_METHOD
    if verbose:
        print("=" * 45); print(f"  CALIBRATION - INDIA {problem.system_id}-bus ({n_samples} samples)"); print("=" * 45)
    rng = np.random.default_rng(seed)          # fixed seed: calibration is reproducible run to run
    lb, ub = problem.lb, problem.ub
    vals = np.zeros((n_samples, 4))
    for i in range(n_samples):
        x = lb + rng.random(N_VAR) * (ub - lb)
        x[VAR_IS_INT] = mround(x[VAR_IS_INT])
        _, *vals[i] = problem.detailed(x, DEFAULT_WEIGHTS)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples} samples done")
    F1, F2, F3, F4 = vals.T
    lo = np.percentile(vals, 5, axis=0); hi = np.percentile(vals, 95, axis=0)   # linear (R-7) as in my_prctile

    if verbose:
        print(f"\n{'Term':<6} {'P5 (lo)':<12} {'P95 (hi)':<12} {'Mean':<12} {'Std':<12}")
        for j, n in enumerate(("F1", "F2", "F3", "F4")):
            f = "{:<12.0f}" if n == "F3" else "{:<12.4f}"
            print(f"{n:<6} " + " ".join(f.format(v) for v in (lo[j], hi[j], vals[:, j].mean(), vals[:, j].std(ddof=1))))

    w_all, ahp_info = compute_all(vals, lo, hi, verbose=verbose)
    if weight_method not in w_all:
        raise ValueError(f"weight_method '{weight_method}' is not available (unknown, or BASE_PAPER_W not set)")
    w_fixed = w_all[weight_method]

    r12 = 0.0
    xm, ym = F1 - F1.mean(), F2 - F2.mean()
    den = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if den >= EPS:
        r12 = float((xm * ym).sum() / den)

    if verbose:
        print(f"\n{'Method':<16} {'tau':<8} {'beta':<8} {'gamma':<8} {'alpha':<8}")
        for k, w in w_all.items():
            print(f"{k:<16} " + " ".join(f"{v:<8.4f}" for v in w))
        if "base_paper" not in w_all:
            print(f"{'base_paper':<16} (not set -- fill BASE_PAPER_W in evbtp/config.py)")
        print(f"\nACTIVE weight_method = '{weight_method}' -> w_fixed = [" + " ".join(f"{v:.4f}" for v in w_fixed) + "]")
        print(f"(Sum check: {w_fixed.sum():.6f} -- should equal 1.0)")
        print(f"\nCorrelation(F1,F2) = {r12:.3f}" + ("  <-- high; F1 and F2 are largely redundant here." if abs(r12) > 0.7 else ""))

    cal = Calibration(F1_lo=lo[0], F1_hi=hi[0], F2_lo=lo[1], F2_hi=hi[1], F3_lo=lo[2], F3_hi=hi[2],
                      F4_lo=lo[3], F4_hi=hi[3], w_fixed=w_fixed, n_samples=n_samples,
                      F1_vals=F1, F2_vals=F2, F3_vals=F3, F4_vals=F4,
                      weight_method=weight_method, w_all=w_all, ahp_info=asdict(ahp_info))
    if save:
        p = calibration_path(problem.results_dir, problem.system_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        cal.save(p)
        if verbose:
            print(f"\nSaved to {p}\nRe-run this after any change to the objective, bounds, EV/RES data, weights or system,\n"
                  "then re-run every optimizer (weights and the scale of F both change).")
    return cal
