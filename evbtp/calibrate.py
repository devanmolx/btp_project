"""Calibration: normalisation bases + Markov weights (port of calibrate_and_weights_INDIA.m).

Samples N feasible random solutions, takes the 5th/95th percentile of F1..F4 as
range-normalisation bases, derives Markov weights from the range-normalised F1..F3, fixes
F4's weight at 0.05, and saves everything to results/calibration_<system>bus.npz.
"""
import numpy as np
from .config import ALPHA_FIXED, DEFAULT_WEIGHTS, N_VAR, VAR_IS_INT
from .markov import markov_weights
from .problem import Problem, Calibration, calibration_path, EPS
from .utils import mround


def calibrate(problem: Problem, n_samples=1000, seed=42, verbose=True, save=True) -> Calibration:
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

    norm = np.column_stack([(vals[:, j] - lo[j]) / max(hi[j] - lo[j], EPS) for j in range(3)])
    w3 = markov_weights(norm)
    w_fixed = np.append(w3 * (1 - ALPHA_FIXED), ALPHA_FIXED)

    r12 = 0.0
    xm, ym = F1 - F1.mean(), F2 - F2.mean()
    den = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if den >= EPS:
        r12 = float((xm * ym).sum() / den)

    if verbose:
        print(f"\n{'Term':<6} {'P5 (lo)':<12} {'P95 (hi)':<12} {'Mean':<12} {'Std':<12}")
        for j, n in enumerate(("F1", "F2", "F3", "F4")):
            f = "{:<12.0f}" if n == "F3" else "{:<12.4f}"
            print(f"{n:<6} " + " ".join(f.format(v) for v in (lo[j], hi[j], vals[:, j].mean(), vals[:, j].std(ddof=1))))
        print(f"\nMarkov weights (F1,F2,F3 ranked, range-normalized) + fixed F4={ALPHA_FIXED:.2f}:")
        print("[tau, beta, gamma, alpha] = [" + " ".join(f"{v:.4f}" for v in w_fixed) + "]")
        print(f"(Sum check: {w_fixed.sum():.6f} -- should equal 1.0)")
        print(f"\nCorrelation(F1,F2) = {r12:.3f}" + ("  <-- high; F1 and F2 are largely redundant here." if abs(r12) > 0.7 else ""))

    cal = Calibration(F1_lo=lo[0], F1_hi=hi[0], F2_lo=lo[1], F2_hi=hi[1], F3_lo=lo[2], F3_hi=hi[2],
                      F4_lo=lo[3], F4_hi=hi[3], w_fixed=w_fixed, n_samples=n_samples,
                      F1_vals=F1, F2_vals=F2, F3_vals=F3, F4_vals=F4)
    if save:
        p = calibration_path(problem.results_dir, problem.system_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        cal.save(p)
        if verbose:
            print(f"\nSaved to {p}\nRe-run this after any change to the objective, bounds, EV/RES data or system,\n"
                  "then re-run every optimizer (weights and the scale of F both change).")
    return cal
