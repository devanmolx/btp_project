"""matplotlib versions of the MATLAB figures.  All functions save a PNG and return the Figure."""
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COL = {"GA-PSO": (0, 0, 1), "HOA": (0.1, 0.6, 0.3), "MPA": (0.0, 0.5, 0.8)}


def use_headless():
    matplotlib.use("Agg")


def _save(fig, out):
    if out:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def plot_algo_diagnostics(res, max_bus, out=None):
    """3-panel diagnostic: best vs mean, semilog convergence, siting matrix."""
    col = COL.get(res.algo, (0, 0, 1)); g = res.gbest
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    it = np.arange(1, len(res.convergence) + 1)
    ax[0].plot(it, res.convergence, color=col, lw=2, label="Global Best")
    ax[0].plot(it, res.mean_history, "r--", lw=1, label="Population Mean")
    ax[0].set(xlabel="Iteration Index", ylabel="Objective Function (F)", title="Global Best & Population Mean"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogy(it, res.convergence - res.convergence.min() + 1e-6, color=col, lw=2)
    ax[1].set(xlabel="Iteration Index", ylabel=r"$F - F_{min} + \epsilon$", title="Semi-Log Convergence Trajectory"); ax[1].grid(alpha=.3, which="both")
    cap = np.zeros(max_bus); colors = ["#cccccc"] * max_bus
    for bi, ci, c in ((0, 2, "#33cc33"), (1, 3, "#33cc33"), (4, 6, "#3366e6"), (5, 7, "#3366e6")):
        cap[int(g[bi]) - 1] = g[ci] * 1000; colors[int(g[bi]) - 1] = c
    ax[2].bar(np.arange(1, max_bus + 1), cap, color=colors)
    ax[2].axhline(1000, color="k", ls="--", lw=1); ax[2].text(1, 1010, "RES max = 1000 kW", fontsize=8)
    ax[2].axhline(700, color="r", ls=":", lw=1); ax[2].text(1, 710, "CS max = 700 kW", fontsize=8)
    ax[2].set(xlabel="Bus Number", ylabel="Installed Capacity (kW)", ylim=(0, 1100),
              title="Optimized Siting Matrix (India)\nGreen = RES | Blue = EV Station"); ax[2].grid(alpha=.3)
    fig.suptitle(f"{res.algo} India ({max_bus}-bus) | Weights [" + " ".join(f"{v:.3f}" for v in res.w_fixed) + "]")
    fig.tight_layout()
    return _save(fig, out)


def plot_comparison(g, h, m, max_bus, out=None):
    """4-panel comparison of the three single-run results."""
    rs = [("GA-PSO", g), ("HOA", h), ("MPA", m)]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for n, r in rs:
        ax[0, 0].plot(np.arange(1, len(r.convergence) + 1), r.convergence, color=COL[n], lw=2, label=n)
        ax[0, 1].plot(np.linspace(0, r.NOFE, len(r.convergence)), r.convergence, color=COL[n], lw=2, label=n)
    ax[0, 0].set(xlabel="Iteration", ylabel="Best Fitness (F)", title="Convergence vs Iteration"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 1].set(xlabel="Cumulative Function Evaluations (NOFE)", ylabel="Best Fitness (F)",
                 title="Convergence vs NOFE\n(fairer: iterations cost different compute)"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    F = np.array([[r.F1 for _, r in rs], [r.F2 for _, r in rs], [r.F3 for _, r in rs], [r.F4 for _, r in rs]])
    Fn = F / F.max(axis=1, keepdims=True)
    w = 0.2
    for k, (lab, c) in enumerate(zip(["F1 Loss", "F2 Voltage", "F3 Cost", "F4 Battery"],
                                     ["#d95319", "#edb120", "#7e2f8e", "#77ac30"])):
        ax[1, 0].bar(np.arange(3) + (k - 1.5) * w, Fn[k], w, color=c, label=lab)
    ax[1, 0].set_xticks(range(3), [n for n, _ in rs]); ax[1, 0].set(ylim=(0.9, 1.02), ylabel="Value (normalized to row max)",
                 title="Objective Breakdown (lower = better)"); ax[1, 0].legend(loc="upper center", ncol=4, fontsize=8); ax[1, 0].grid(alpha=.3)
    cap = np.zeros((max_bus, 3))
    for a, (_, r) in enumerate(rs):
        x = r.gbest
        for bi, ci in ((0, 2), (1, 3), (4, 6), (5, 7)):
            cap[int(x[bi]) - 1, a] += x[ci] * 1000
    active = np.flatnonzero(cap.sum(axis=1) > 0)
    for a, (n, _) in enumerate(rs):
        ax[1, 1].bar(np.arange(len(active)) + (a - 1) * 0.27, cap[active, a], 0.27, color=COL[n], label=n)
    ax[1, 1].set_xticks(range(len(active)), [str(b + 1) for b in active])
    ax[1, 1].set(xlabel="Bus Number (only buses used)", ylabel="Installed Capacity (kW)", title="Siting & Sizing Comparison"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    fig.suptitle(f"GA-PSO vs HOA vs MPA -- India ({max_bus}-bus system)")
    fig.tight_layout()
    return _save(fig, out)


def plot_fair_convergence(grid, curves, budget, n_runs, out=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for a, c in zip(("GAPSO", "HOA", "MPA"), ("b", "g", "r")):
        ax.plot(grid, curves[a], c, lw=2, label=a)
    ax.set(xlabel="Cumulative Function Evaluations (NOFE)", ylabel="Mean Best Fitness (F) across runs",
           title=f"Fair comparison: equal NOFE budget ({budget}), {n_runs} runs each")
    ax.legend(); ax.grid(alpha=.3)
    return _save(fig, out)


def plot_ev_model(ev, out=None):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].hist(ev.Md, 40, color=(.3, .7, .4)); ax[0, 0].set(xlabel="Daily Distance (km)", ylabel="Number of EVs", title="EV Daily Distance (Log-Normal)")
    ax[0, 1].hist(ev.E_demand, 30, color=(.8, .3, .3)); ax[0, 1].set(xlabel="Charging Demand (kWh)", ylabel="Number of EVs", title="EV Charging Demand Distribution")
    ax[1, 0].hist(ev.t_arriv, 24, color=(.2, .5, .9)); ax[1, 0].set(xlabel="Arrival Time (hour)", ylabel="Number of EVs", title="EV Arrival Time Distribution", xlim=(0, 24))
    ax[1, 1].bar(np.arange(1, 25), ev.P_EV_total, color=(.6, .2, .8)); ax[1, 1].set(xlabel="Hour of Day", ylabel="EV Charging Demand (kW)", title="Hourly Total EV Charging Demand", xlim=(0, 25))
    for a in ax.ravel(): a.grid(alpha=.3)
    fig.suptitle("EV Model Results - India (Tata/Mahindra/MG Fleet)"); fig.tight_layout()
    return _save(fig, out)


def plot_profiles(prof, out=None):
    h = np.arange(1, 25)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    ax[0, 0].plot(h, prof.load_profile, "r-o", lw=2.5); ax[0, 0].set(xlabel="Hour of Day", ylabel="Demand (MW)", title="System Load Profile\nNorthern Region (Delhi) - Jun 10 2025")
    ax[0, 1].fill_between(h, prof.solar_profile, color=(1, .85, .1), alpha=.85); ax[0, 1].set(xlabel="Hour of Day", ylabel="Solar Output (normalized)", title="Solar PV Profile\nDelhi - Jun 10 2019", ylim=(0, 1.2))
    ax[1, 0].plot(h, prof.wind_profile, "b-s", lw=2.5); ax[1, 0].set(xlabel="Hour of Day", ylabel="Wind Output (normalized)", title="Wind Profile\nDelhi - Jun 10 2019", ylim=(0, 1.2))
    ax[1, 1].plot(h, prof.price_TOU, "k-^", lw=2.5, label="Charging (ToD)"); ax[1, 1].plot(h, prof.price_V2G, "g--v", lw=1.5, label="V2G (+10%)")
    ax[1, 1].set(xlabel="Hour of Day", ylabel="Price (Rs/kWh)", title="Electricity Price (BRPL EV ToD Tariff)"); ax[1, 1].legend()
    for a in ax.ravel(): a.grid(alpha=.3); a.set_xlim(1, 24)
    fig.suptitle("24-Hour System Profiles - India (Delhi), Jun 10 2025 - Real Data"); fig.tight_layout()
    return _save(fig, out)


def plot_voltage_profiles(scen, out=None):
    """scen: list of (label, Vprofile, style dict)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for lab, vp, kw in scen:
        ax.plot(np.arange(1, len(vp) + 1), vp, label=lab, **kw)
    ax.axhline(0.95, color="m", ls=":", lw=1.4, label="0.95 pu guideline")
    ax.set(xlabel="Bus Number", ylabel="Voltage Magnitude (pu), 24-h average"); ax.legend(loc="lower left"); ax.grid(alpha=.3)
    return _save(fig, out)
