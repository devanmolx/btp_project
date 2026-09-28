"""Base-case & improvement analysis (port of base_case_INDIA.m).

A. Base case  : network alone (no RES, no EV)
B. EV only    : EV charging at the optimiser's CS buses, no RES
C. Optimised  : best solution from each algorithm
NB (as in the MATLAB script): A/B "F3" is grid-energy cost only (no capex, no V2G revenue),
while C's F3 comes from the full objective, so the F3 column is not like-for-like.
"""
import numpy as np
from .problem import Problem
from .results_io import load_result, dump_json, KEYS
from .utils import mround


def eval_scenario(pr: Problem, use_RES, use_EV, cs_buses=None, x=None, use_statcom=True):
    P = pr._P_base.copy(); Q = pr._Q_base
    e, p = pr.ev, pr.prof
    if use_EV:
        if x is not None:
            b1, b2 = int(mround(x[4])), int(mround(x[5]))
            v1 = np.where(pr._peak_mask, np.minimum(e.V2G_avail_hourly[:, 0], x[6] * 1000 * 0.35), 0.0)
            v2 = np.where(pr._peak_mask, np.minimum(e.V2G_avail_hourly[:, 1], x[7] * 1000 * 0.35), 0.0)
            P[:, b1 - 1] += (e.P_EV_hourly[:, 0] - v1) * 1000
            P[:, b2 - 1] += (e.P_EV_hourly[:, 1] - v2) * 1000
        else:
            P[:, cs_buses[0] - 1] += e.P_EV_hourly[:, 0] * 1000
            P[:, cs_buses[1] - 1] += e.P_EV_hourly[:, 1] * 1000
    if use_RES and x is not None:
        r1, r2 = int(mround(x[0])), int(mround(x[1]))
        P[:, r1 - 1] -= x[2] * 1e6 * pr._gen_pu
        P[:, r2 - 1] -= x[3] * 1e6 * pr._gen_pu
    if pr.statcom and use_statcom and use_RES and x is not None:
        Q = Q - pr.statcom_q(*pr.decode_statcom(x))
    V, loss = pr.lf.solve(P, Q)
    vm = np.abs(V)
    F1 = float(np.sum(loss / 1e6)); F2 = float(np.sum(np.abs(1 - vm[:, 1:])))
    F3 = float(np.sum(np.maximum(0, P.sum(axis=1) + loss) / 1000 * p.price_TOU))
    Vmin = float(vm.min()); Vmin_bus = int(np.unravel_index(np.argmin(vm), vm.shape)[1]) + 1
    return dict(F1=F1, F2=F2, F3=F3, Vprofile=vm.mean(axis=0), Vmin=Vmin, Vmin_bus=Vmin_bus)


def run_base_case(pr: Problem, results_dir, verbose=True):
    A = eval_scenario(pr, False, False)
    ref = load_result("gapso", results_dir, pr.system_id)
    ref = load_result("gapso", results_dir, pr.system_id, pr.statcom)
    cs = [int(round(ref.gbest[4])), int(round(ref.gbest[5]))] if ref else [2, int(round(pr.max_bus / 2))]
    B = eval_scenario(pr, False, True, cs_buses=cs)

    algos, sols = [], []
    for k in ("gapso", "hoa", "mpa"):
        r = load_result(k, results_dir, pr.system_id, pr.statcom)
        if r is not None:
            algos.append(r.algo); sols.append(r.gbest)
    if not algos:
        raise FileNotFoundError("No optimiser result files found. Run the optimisers first.")
    C = []
    for x in sols:
        _, f1, f2, f3, f4 = pr.detailed(x)
        s = eval_scenario(pr, True, True, x=x)
        s.update(F1=f1, F2=f2, F3=f3, F4=f4)
        if pr.statcom:                                   # same RES/CS layout with the DSTATCOMs removed = their benefit
            s["Vprofile_noSC"] = eval_scenario(pr, True, True, x=x, use_statcom=False)["Vprofile"]
            s["metrics"], s["metrics_noSC"] = pr.metrics(x), pr.metrics(x, without_statcom=True)
            s["F_noSC"] = list(pr._evaluate(np.where(np.arange(len(x)) >= 10, 0.0, x), pr.w_fixed, 0.05)[1:])
        C.append(s)

    if verbose:
        print("=" * 53 + f"\n  BASE CASE ANALYSIS -- IEEE {pr.max_bus}-bus (India)\n" + "=" * 53 + "\n")
        print(f"{'Scenario':<14} {'F1 (MWh/day)':<14} {'F2 (pu)':<14} {'F3 (Rs/day)':<14} {'Vmin (pu)':<12}"); print("-" * 70)
        print(f"{'Base case':<14} {A['F1']:<14.4f} {A['F2']:<14.4f} {A['F3']:<14.0f} {A['Vmin']:<12.4f}")
        print(f"{'EV only':<14} {B['F1']:<14.4f} {B['F2']:<14.4f} {B['F3']:<14.0f} {B['Vmin']:<12.4f}")
        for n, c in zip(algos, C):
            print(f"{n:<14} {c['F1']:<14.4f} {c['F2']:<14.4f} {c['F3']:<14.0f} {c['Vmin']:<12.4f}")
        print("(F3: Base/EV-only = grid energy cost only; optimised rows include capex and V2G revenue.)")
        print("\n" + "=" * 53 + "\n  IMPROVEMENT vs BASE CASE  (the headline numbers)\n" + "=" * 53)
        print(f"{'Algorithm':<14} {'Loss reduction':<16} {'Voltage dev.':<16} {'Vmin improvement':<16}"); print("-" * 66)
        for n, c in zip(algos, C):
            dl = f"{(A['F1'] - c['F1']) / A['F1'] * 100:.2f}%"; dv = f"{(A['F2'] - c['F2']) / A['F2'] * 100:.2f}%"
            print(f"{n:<14} {dl:<16} {dv:<16} {A['Vmin']:.4f} -> {c['Vmin']:.4f}")
        print("\n--- Impact of EV charging alone (no RES) vs base case ---")
        print(f"Loss increase   : {(B['F1'] - A['F1']) / A['F1'] * 100:.2f}%")
        print(f"Vmin drops from : {A['Vmin']:.4f} to {B['Vmin']:.4f} pu")
        print("This quantifies the problem that RES allocation is solving.")
        if pr.statcom:
            print("\n" + "=" * 66 + "\n  DSTATCOM BENEFIT  (same RES/CS layout, DSTATCOMs removed -> installed)\n" + "=" * 66)
            for n, c, x in zip(algos, C, sols):
                print(f"\n{n}: DSTATCOM 1 bus {int(x[8])} {x[10] * 1000:.0f} kVAr | DSTATCOM 2 bus {int(x[9])} {x[11] * 1000:.0f} kVAr")
                print(f"  {'Metric':<26} {'without':<12} {'with':<12} {'change':<10}")
                rows = (("Mean P loss (kW)", "P_loss_kW", ".2f"), ("Mean Q loss (kVAr)", "Q_loss_kVAr", ".2f"),
                        ("VDI (mean sum|1-V|)", "VDI", ".4f"), ("Min VSI", "VSI_min", ".4f"), ("Vmin (pu)", "Vmin", ".4f"))
                for lab, k, f in rows:
                    a0, a1 = c["metrics_noSC"][k], c["metrics"][k]
                    print(f"  {lab:<26} {a0:<12{f}} {a1:<12{f}} {(a1 - a0) / abs(a0) * 100:+.1f}%")
                f0, f1 = c["F_noSC"], [c["F1"], c["F2"], c["F3"], c["F4"]]
                print(f"  {'F3 daily cost (Rs)':<26} {f0[2]:<12.0f} {f1[2]:<12.0f} {(f1[2] - f0[2]) / abs(f0[2]) * 100:+.1f}%   (incl. DSTATCOM capex)")

    dump_json(dict(A=A, B=B, C=C, algos=algos, MAX_BUS=pr.max_bus), f"{results_dir}/base_case_{pr.tag}_results.json")
    return A, B, C, algos
