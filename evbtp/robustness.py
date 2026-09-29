"""Weight-robustness study (port of run_mpa_once_INDIA.m + weight_robustness_INDIA.m).

For each weight set in config.ROBUST_SETS, run MPA (the NOFE-budget mode used by fair-compare,
unchanged) for every seed in config.ROBUST_SEEDS, and report the optimised layout and F1-F4/Vmin
as mean +/- std.  Every run is kept.  A weight set that is not defined (base_paper while
config.BASE_PAPER_W is None) is reported as "not run".
"""
from collections import Counter
import numpy as np

from . import config
from .base_case import eval_scenario
from .optimizers import run_mpa
from .results_io import dump_json


def run_mpa_once(problem, w, seed, nofe_budget=20000):
    """One MPA run with weights w; returns the repaired best x, F1..F4 and Vmin (dict)."""
    r = run_mpa(problem, w, max_iter=None, nofe_budget=nofe_budget,
                schedule_iters=max(1, nofe_budget // 100), seed=seed, verbose=False)
    s = eval_scenario(problem, True, True, x=r.gbest)          # same 24-h network snapshot as the objective
    return dict(x=r.gbest, F=r.gbest_fit, NOFE=r.NOFE, F1=r.F1, F2=r.F2, F3=r.F3, F4=r.F4,
                Vmin=s["Vmin"], Vmin_bus=s["Vmin_bus"], seed=int(seed), w=np.asarray(w, float))


def _pm(v, dec):
    v = np.asarray(v, float)
    return f"{v.mean():.{dec}f}+/-{v.std(ddof=1) if len(v) > 1 else 0.0:.{dec}f}"


def _units(runs, bi, ci):
    """Per run: the two units sorted by bus number -> (bus pairs, kW pairs)."""
    B, K = [], []
    for r in runs:
        b = r["x"][list(bi)].astype(int); k = r["x"][list(ci)] * 1000
        o = np.argsort(b, kind="stable")
        B.append(tuple(b[o])); K.append(k[o])
    return B, np.array(K)


def summarise(runs, n_seeds):
    B, K = _units(runs, (0, 1), (2, 3)); (rb, rf), = Counter(B).most_common(1)
    Bc, Kc = _units(runs, (4, 5), (6, 7)); (cb, cf), = Counter(Bc).most_common(1)
    col = lambda f: [r[f] for r in runs]
    return dict(
        res=f"{rb[0]},{rb[1]}({rf}/{n_seeds})|{_pm(K[:, 0], 0)}/{_pm(K[:, 1], 0)}",
        cs=f"{cb[0]},{cb[1]}({cf}/{n_seeds})|{_pm(Kc[:, 0], 0)}/{_pm(Kc[:, 1], 0)}",
        **{f: (float(np.mean(col(f))), float(np.std(col(f), ddof=1) if len(runs) > 1 else 0.0))
           for f in ("F1", "F2", "F3", "F4", "Vmin")})


def weight_robustness(problem, sets=None, seeds=None, nofe_budget=None, verbose=True):
    cal = problem.calibration
    if cal is None or not cal.w_all:
        raise SystemExit("Calibration has no w_all -- re-run `python -m evbtp calibrate`.")
    sets = sets or config.ROBUST_SETS
    seeds = seeds or config.ROBUST_SEEDS
    nofe_budget = nofe_budget or config.ROBUST_NOFE
    if verbose:
        print("=" * 69 + f"\n  WEIGHT ROBUSTNESS -- IEEE {problem.max_bus}-bus | MPA, {nofe_budget} NOFE, "
              f"{len(seeds)} seeds per set\n" + "=" * 69)

    runs, not_run = {}, []
    for name in sets:
        if name not in cal.w_all:
            not_run.append(name)
            if verbose:
                print(f"{name:<12} NOT RUN (weights not defined)")
            continue
        w = np.asarray(cal.w_all[name], float)
        runs[name] = []
        for sd in seeds:
            o = run_mpa_once(problem, w, sd, nofe_budget)
            runs[name].append(o)
            if verbose:
                x = o["x"]
                print(f"{name:<12} seed {sd}: F={o['F']:.6f} F1={o['F1']:.4f} F2={o['F2']:.4f} F3={o['F3']:.0f} "
                      f"F4={o['F4']:.2f} Vmin={o['Vmin']:.4f} x=[{x[0]:.0f} {x[1]:.0f} {x[2]:.3f} {x[3]:.3f} "
                      f"{x[4]:.0f} {x[5]:.0f} {x[6]:.3f} {x[7]:.3f}]")

    summary = {n: summarise(r, len(seeds)) for n, r in runs.items()}
    m1 = np.array([s["F1"][0] for s in summary.values()]); mv = np.array([s["Vmin"][0] for s in summary.values()])
    spread_F1 = float((m1.max() - m1.min()) / m1.mean() * 100)
    spread_Vmin = float((mv.max() - mv.min()) / mv.mean() * 100)

    if verbose:
        print("\n" + table(summary, problem.max_bus))
        print(f"\nSpread across weight sets (max-min of the per-set means, % of their average):")
        print(f"  F1  : {spread_F1:.2f} %  ({m1.min():.4f} .. {m1.max():.4f} MWh/d)")
        print(f"  Vmin: {spread_Vmin:.2f} %  ({mv.min():.4f} .. {mv.max():.4f} pu)")
        if not_run:
            print("Not run (weights undefined): " + ", ".join(not_run))

    return dict(system=problem.system_id, seeds=list(seeds), nofe=nofe_budget, runs=runs, summary=summary,
                not_run=not_run, spread_F1_pct=spread_F1, spread_Vmin_pct=spread_Vmin,
                w_all={k: np.asarray(v).tolist() for k, v in cal.w_all.items()},
                weight_method=cal.weight_method)


def table(summary, max_bus):
    h = (f"{'Weight set':<11}| {'RES buses (freq) | kW':<27}| {'CS buses (freq) | kW':<25}| {'F1 MWh/d':<16}| "
         f"{'F2 pu':<16}| {'Energy cost Rs/d':<17}| {'F4 Rs':<13}| {'Vmin pu':<15}")
    L = [f"IEEE {max_bus}-bus", h, "-" * len(h)]
    for n, s in summary.items():
        f = lambda k, d: f"{s[k][0]:.{d}f}+/-{s[k][1]:.{d}f}"
        L.append(f"{n:<11}| {s['res']:<27}| {s['cs']:<25}| {f('F1', 4):<16}| {f('F2', 3):<16}| "
                 f"{f('F3', 0):<17}| {f('F4', 1):<13}| {f('Vmin', 4):<15}")
    L += ["-" * len(h),
          "Buses: most frequent pair over seeds (units sorted by bus number); kW per unit as mean+/-std.",
          "Energy cost = F3 (grid energy - V2G revenue + daily capex)."]
    return "\n".join(L)


def weights_table(cal, max_bus):
    """compare_weighting_methods_INDIA.m: every method's weights, the active one, and the AHP check."""
    labels = dict(base_paper="Base paper (2017)", equal="Equal", markov="Markov", critic="CRITIC",
                  critic_f4fixed="CRITIC (F4 fixed)", combined="Markov+CRITIC", ahp="AHP", ahp_critic="AHP x CRITIC")
    L = ["=" * 69, f"  OBJECTIVE WEIGHTS BY METHOD  (IEEE {max_bus}-bus calibration)", "=" * 69,
         f"{'Method':<20} {'F1 loss':<9} {'F2 volt':<9} {'F3 cost':<9} {'F4 batt':<9}", "-" * 69]
    for m, lab in labels.items():
        if cal.w_all and m in cal.w_all:
            mark = "<-- active (w_fixed)" if m == cal.weight_method else ""
            L.append(f"{lab:<20} " + " ".join(f"{v:<9.4f}" for v in cal.w_all[m]) + mark)
        else:
            L.append(f"{lab:<20} (not set -- fill BASE_PAPER_W in evbtp/config.py)")
    L.append("-" * 69)
    i = cal.ahp_info
    L.append(f"AHP: lambda_max = {i['lambda_max']:.4f}, CI = {i['CI']:.4f}, CR = {i['CR']:.4f} "
             f"({'consistent' if i['CR'] < 0.10 else 'INCONSISTENT'})")
    L.append("AHP geometric-mean weights: [" + " ".join(f"{v:.4f}" for v in i["w_gm"]) + "]")
    L.append("AHP matrix is a PLACEHOLDER until confirmed by the guide.")
    if cal.weight_method != config.WEIGHT_METHOD:
        L.append(f"WARNING: config says '{config.WEIGHT_METHOD}' but this calibration was built with "
                 f"'{cal.weight_method}' -- re-run `python -m evbtp calibrate`.")
    L.append("=" * 69)
    return "\n".join(L)
