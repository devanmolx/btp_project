"""Fair comparison GA-PSO vs HOA vs MPA (port of fair_compare_GAPSO_HOA_MPA_INDIA.m).

Every algorithm gets the SAME total number of objective evaluations (NOFE_BUDGET) and
N_RUNS independent seeds; each run stops the moment the budget is spent.  Same seed 1000+r is
given to all three algorithms in run r, as in the MATLAB harness.
Addition over the MATLAB version: a Wilcoxon rank-sum (Mann-Whitney U) test between algorithm
pairs -- the MATLAB README listed this as "not done".
"""
import time
import numpy as np
from scipy import stats
from .optimizers import run_gapso, run_hoa, run_mpa
from .results_io import dump_json

ALGOS = ("GAPSO", "HOA", "MPA")


def _run(name, problem, budget, seed):
    if name == "GAPSO":
        return run_gapso(problem, max_iter=None, nofe_budget=budget, seed=seed, verbose=False, track_details=False)
    if name == "HOA":
        return run_hoa(problem, max_iter=None, nofe_budget=budget, seed=seed, verbose=False)
    return run_mpa(problem, max_iter=None, nofe_budget=budget, schedule_iters=max(1, budget // 100), seed=seed, verbose=False)


def fair_compare(problem, nofe_budget=20000, n_runs=10, seed_base=1000, verbose=True):
    if verbose:
        print("=" * 45 + f"\n  FAIR COMPARISON (equal NOFE budget = {nofe_budget}, {n_runs} runs)\n" + "=" * 45 + "\n")
    best = {a: np.zeros(n_runs) for a in ALGOS}
    nofe = {a: np.zeros(n_runs) for a in ALGOS}
    traces = {a: [] for a in ALGOS}
    for r in range(1, n_runs + 1):
        if verbose:
            print(f"--- Run {r}/{n_runs} ---")
        for a in ALGOS:
            res = _run(a, problem, nofe_budget, seed_base + r)
            best[a][r - 1] = res.gbest_fit; nofe[a][r - 1] = res.NOFE; traces[a].append(res.trace)
            if verbose:
                print(f"  {a:<6}: F={res.gbest_fit:.6f}  NOFE={res.NOFE}")

    if verbose:
        print("\n" + "=" * 69 + f"\n  SUMMARY OVER {n_runs} RUNS (equal NOFE budget = {nofe_budget} each)\n" + "=" * 69)
        print(f"{'Algo':<8} {'Best':<12} {'Mean':<12} {'Worst':<12} {'Std':<12} {'Avg NOFE':<10}")
        for a in ALGOS:
            v = best[a]
            print(f"{a:<8} {v.min():<12.6f} {v.mean():<12.6f} {v.max():<12.6f} {v.std(ddof=1) if n_runs > 1 else 0:<12.6f} {nofe[a].mean():<10.0f}")
        print("=" * 69 + "\nLower is better (F is minimised). Compare MEAN and STD, not just the best run.")
        if n_runs >= 5:
            print("\nWilcoxon rank-sum test (two-sided) on best fitness per run:")
            for i, a in enumerate(ALGOS):
                for b in ALGOS[i + 1:]:
                    p = stats.mannwhitneyu(best[a], best[b], alternative="two-sided").pvalue
                    print(f"  {a:<6} vs {b:<6}: p = {p:.4g}" + ("  (p<0.05)" if p < 0.05 else ""))
        else:
            print("(use >= 5 runs for a rank-sum test; 30 is the usual choice for a write-up)")

    out = dict(best_fit=best, nofe_used=nofe, conv_all={a: [t.tolist() for t in traces[a]] for a in ALGOS},
               NOFE_BUDGET=nofe_budget, N_RUNS=n_runs, system=problem.system_id)
    return out


def mean_convergence(conv_all, budget, n_grid=200):
    """Interpolate each run's [NOFE, best] trace onto a common grid ('previous' rule), average over runs."""
    grid = np.linspace(0, budget, n_grid)
    curves = {}
    for a, runs in conv_all.items():
        M = np.full((len(runs), n_grid), np.nan)
        for r, tr in enumerate(runs):
            tr = np.asarray(tr)
            idx = np.searchsorted(tr[:, 0], grid, side="right") - 1
            ok = idx >= 0                                       # MATLAB 'previous' gives NaN before first sample
            M[r, ok] = tr[idx[ok], 1]
        curves[a] = M.mean(axis=0)
    return grid, curves
