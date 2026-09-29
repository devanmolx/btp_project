"""Command-line interface:  python -m evbtp <command> [options]

Run order (same as the MATLAB MANIFEST):
    ev-model  ->  calibrate  ->  optimize gapso|hoa|mpa  ->  compare  ->  fair-compare  ->  base-case
or just:  python -m evbtp all
"""
import argparse
from pathlib import Path

from . import config


def _problem(a, quiet=False):
    from .problem import Problem
    from .ev_model import generate_ev_data, load_matlab_ev_data
    ev = generate_ev_data(seed=a.ev_seed) if a.ev_source == "generate" else load_matlab_ev_data()
    return Problem(system=a.system, ev=ev, results_dir=a.results_dir, quiet=quiet)


def _fig(a, fig, name):
    import matplotlib.pyplot as plt
    if a.show:
        plt.show()
    plt.close(fig)


def cmd_ev_model(a):
    from . import ev_model, plotting
    ev = ev_model.generate_ev_data(seed=a.ev_seed) if (a.regenerate or a.ev_source == "generate") else ev_model.load_matlab_ev_data()
    print(ev_model.summary(ev))
    if not a.no_plots:
        out = Path(a.results_dir) / "figures" / "EV_Model_INDIA_Plots.png"
        _fig(a, plotting.plot_ev_model(ev, out), "ev"); print(f"Saved {out}")


def cmd_profiles(a):
    from . import plotting
    from .profiles import build_profiles
    p = build_profiles()
    print(f"Load {p.load_profile.min():.2f}-{p.load_profile.max():.2f} MW | price Rs {p.price_TOU.min():.3f}-{p.price_TOU.max():.3f}/kWh")
    out = Path(a.results_dir) / "figures" / "profiles_24hr_INDIA.png"
    _fig(a, plotting.plot_profiles(p, out), "prof"); print(f"Saved {out}")


def cmd_calibrate(a):
    from .calibrate import calibrate
    calibrate(_problem(a, quiet=True), n_samples=a.samples, seed=a.seed if a.seed is not None else 42)


def _need_calibration(a):
    from .problem import calibration_path
    if not calibration_path(a.results_dir, a.system).exists():
        raise SystemExit(f"No calibration for the {a.system}-bus system in {a.results_dir}/. "
                         "Run `python -m evbtp calibrate` first.")


def cmd_optimize(a):
    from .optimizers import ALGORITHMS
    from .results_io import save_result
    from . import plotting
    _need_calibration(a)
    pr = _problem(a)
    print(f"Using calibrated weights: [" + " ".join(f"{v:.4f}" for v in pr.w_fixed) + "]\n")
    kw = dict(seed=a.seed, max_iter=a.max_iter, pop_size=a.pop_size)
    if a.algo == "mpa":
        kw["schedule_iters"] = None
    res = ALGORITHMS[a.algo](pr, **kw)
    save_result(res, a.algo, a.results_dir)
    if not a.no_plots:
        out = Path(a.results_dir) / "figures" / f"{res.algo}_{pr.system_id}bus_diagnostics.png"
        _fig(a, plotting.plot_algo_diagnostics(res, pr.max_bus, out), "diag"); print(f"Saved {out}")


def _load_three(a):
    from .results_io import load_result
    out = [load_result(k, a.results_dir, a.system) for k in ("gapso", "hoa", "mpa")]
    missing = [n for n, r in zip(("gapso", "hoa", "mpa"), out) if r is None]
    if missing:
        raise SystemExit(f"Missing results for: {', '.join(missing)}. Run `python -m evbtp optimize <algo>` first.")
    return out


def cmd_compare(a):
    from .reports import comparison_table
    from . import plotting
    g, h, m = _load_three(a)
    print(comparison_table(g, h, m))
    if not a.no_plots:
        out = Path(a.results_dir) / "figures" / f"comparison_{a.system}bus.png"
        _fig(a, plotting.plot_comparison(g, h, m, config.get_max_bus(a.system), out), "cmp"); print(f"Saved {out}")


def cmd_fair(a):
    from .fair_compare import fair_compare, mean_convergence
    from .results_io import dump_json
    from . import plotting
    _need_calibration(a)
    pr = _problem(a)
    res = fair_compare(pr, nofe_budget=a.budget, n_runs=a.runs)
    p = Path(a.results_dir) / f"fair_compare_{a.system}bus_results.json"
    dump_json(res, p); print(f"\nSaved to {p}")
    if not a.no_plots:
        grid, curves = mean_convergence(res["conv_all"], a.budget)
        out = Path(a.results_dir) / "figures" / f"fair_compare_{a.system}bus.png"
        _fig(a, plotting.plot_fair_convergence(grid, curves, a.budget, a.runs, out), "fair"); print(f"Saved {out}")


def cmd_base(a):
    from .base_case import run_base_case
    from . import plotting
    _need_calibration(a)
    pr = _problem(a)
    A, B, C, algos = run_base_case(pr, a.results_dir)
    if not a.no_plots:
        cols = ["#0000ff", "#1a9950", "#0080cc"]
        sc = [("Base case (no RES, no EV)", A["Vprofile"], dict(color="k", marker="o", ms=4, lw=2)),
              ("EV only (no RES)", B["Vprofile"], dict(color="r", ls="--", marker="s", ms=4, lw=1.6))]
        sc += [(n, c["Vprofile"], dict(color=cols[min(i, 2)], lw=2)) for i, (n, c) in enumerate(zip(algos, C))]
        out = Path(a.results_dir) / "figures" / f"voltage_profile_{a.system}bus.png"
        fig = plotting.plot_voltage_profiles(sc, out); fig.axes[0].set_title(f"Voltage Profile Improvement -- IEEE {pr.max_bus}-bus (India)")
        fig.savefig(out, dpi=130, bbox_inches="tight"); _fig(a, fig, "v"); print(f"Saved {out}")


def cmd_loss(a):
    from .loss_priority import run_loss_priority
    run_loss_priority(_problem(a), a.results_dir)


def cmd_weights(a):
    from .robustness import weights_table
    _need_calibration(a)
    pr = _problem(a, quiet=True)
    if not pr.calibration.w_all:
        raise SystemExit("This calibration predates w_all -- re-run `python -m evbtp calibrate`.")
    print(weights_table(pr.calibration, pr.max_bus))


def cmd_robustness(a):
    import json
    from .robustness import weight_robustness
    from .results_io import dump_json
    from . import plotting
    _need_calibration(a)
    pr = _problem(a, quiet=True)
    res = weight_robustness(pr, seeds=tuple(range(1, a.seeds + 1)), nofe_budget=a.budget)
    p = Path(a.results_dir) / "weight_robustness_INDIA.json"          # both feeders side by side
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres[f"bus{pr.system_id}"] = res
    dump_json(allres, p); print(f"\nSaved to {p} (key bus{pr.system_id})")
    if not a.no_plots:
        out = Path(a.results_dir) / "figures" / f"weight_robustness_F1_{pr.system_id}bus.png"
        _fig(a, plotting.plot_robustness_F1(res["summary"], pr.max_bus, res["nofe"], len(res["seeds"]), out), "wr")
        print(f"Saved {out}")


def cmd_all(a):
    cmd_ev_model(a); cmd_calibrate(a)
    for k in ("gapso", "hoa", "mpa"):
        a.algo = k; cmd_optimize(a)
    cmd_compare(a); cmd_fair(a); cmd_base(a)


def build_parser():
    p = argparse.ArgumentParser(prog="evbtp", description="EV charging-station + RES siting (India) -- Python port")
    g = p.add_argument_group("global options")
    g.add_argument("--system", type=int, choices=(33, 69), default=config.SYSTEM_CHOICE, help="IEEE test system (the MATLAB SYSTEM_CHOICE switch)")
    g.add_argument("--results-dir", default=str(config.DEFAULT_RESULTS_DIR))
    g.add_argument("--ev-source", choices=("matlab", "generate"), default="matlab",
                   help="matlab = the exact fleet from the MATLAB project (default); generate = fresh fleet from NumPy RNG")
    g.add_argument("--ev-seed", type=int, default=42)
    g.add_argument("--no-plots", action="store_true"); g.add_argument("--show", action="store_true", help="open plot windows")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ev-model", help="EV fleet summary + plots"); s.add_argument("--regenerate", action="store_true"); s.set_defaults(f=cmd_ev_model)
    s = sub.add_parser("profiles", help="plot the 24-h profiles"); s.set_defaults(f=cmd_profiles)
    s = sub.add_parser("calibrate", help="normalisation bases + objective weights (run first)")
    s.add_argument("--samples", type=int, default=1000); s.add_argument("--seed", type=int); s.set_defaults(f=cmd_calibrate)
    s = sub.add_parser("optimize", help="run one optimizer once")
    s.add_argument("algo", choices=("gapso", "hoa", "mpa")); s.add_argument("--seed", type=int)
    s.add_argument("--max-iter", type=int, default=200); s.add_argument("--pop-size", type=int, default=50); s.set_defaults(f=cmd_optimize)
    s = sub.add_parser("compare", help="single-run 3-way table + plots"); s.set_defaults(f=cmd_compare)
    s = sub.add_parser("fair-compare", help="equal-NOFE, multi-seed comparison (the number to report)")
    s.add_argument("--budget", type=int, default=20000); s.add_argument("--runs", type=int, default=30); s.set_defaults(f=cmd_fair)
    s = sub.add_parser("base-case", help="improvement vs base case + voltage profile"); s.set_defaults(f=cmd_base)
    s = sub.add_parser("loss-priority", help="balanced vs loss-priority scenario"); s.set_defaults(f=cmd_loss)
    s = sub.add_parser("weights", help="weights of every weighting method + AHP consistency"); s.set_defaults(f=cmd_weights)
    s = sub.add_parser("robustness", help="weight-robustness study (MPA per weight set x seeds)")
    s.add_argument("--budget", type=int, default=config.ROBUST_NOFE); s.add_argument("--seeds", type=int, default=len(config.ROBUST_SEEDS))
    s.set_defaults(f=cmd_robustness)
    s = sub.add_parser("all", help="whole pipeline"); s.add_argument("--samples", type=int, default=1000)
    s.add_argument("--seed", type=int); s.add_argument("--max-iter", type=int, default=200); s.add_argument("--pop-size", type=int, default=50)
    s.add_argument("--budget", type=int, default=20000); s.add_argument("--runs", type=int, default=30); s.add_argument("--regenerate", action="store_true"); s.set_defaults(f=cmd_all)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if not a.show:
        from . import plotting
        plotting.use_headless()
    a.f(a)


if __name__ == "__main__":
    main()
