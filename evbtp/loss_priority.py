"""Balanced vs loss-priority scenario (port of run_loss_priority_scenario_INDIA.m).

The MATLAB script set RES to 2.0 MW per unit -- but the objective clamps RES capacity to its
1.0 MW bound, so that scenario was silently evaluated at 1.0 MW.  Here the loss-priority case
uses the actual upper bound (1.0 MW) so the printed sizes are the sizes really evaluated.
Results are identical to what the MATLAB script computed.
"""
import numpy as np
from .config import RES_CAP_MAX
from .problem import Problem
from .results_io import load_result, dump_json


def run_loss_priority(pr: Problem, results_dir, verbose=True):
    ref = load_result("gapso", results_dir, pr.system_id, pr.statcom)
    if ref is None:
        raise FileNotFoundError("GA-PSO results not found. Run `python -m evbtp optimize gapso` first.")
    w, x_base = ref.w_fixed, ref.gbest
    _, F1b, F2b, F3b, F4b = pr.detailed(x_base, w, 0.05)
    x_loss = x_base.copy(); x_loss[2] = x_loss[3] = RES_CAP_MAX
    _, F1l, F2l, F3l, F4l = pr.detailed(x_loss, w, 0.05)
    if verbose:
        print("=" * 45 + "\n  SCENARIO COMPARISON: Balanced vs Loss-Priority\n" + "=" * 45)
        print(f"\nScenario A (Balanced): RES {x_base[2]:.2f}/{x_base[3]:.2f} MW | F1={F1b:.4f} F2={F2b:.4f} F3=Rs.{F3b:.0f} F4=Rs.{F4b:.2f}")
        print(f"Scenario B (Loss-priority, RES at {RES_CAP_MAX:.1f}/{RES_CAP_MAX:.1f} MW bound): F1={F1l:.4f} F2={F2l:.4f} F3=Rs.{F3l:.0f} F4=Rs.{F4l:.2f}")
        print("\n--- Improvement (Scenario A -> B) ---")
        print(f"F1 (loss)    : {(F1b - F1l) / F1b * 100:.2f}% reduction")
        print(f"F2 (voltage) : {(F2b - F2l) / F2b * 100:.2f}% reduction")
        print(f"F3 (cost)    : {(F3b - F3l) / F3b * 100:.2f}% change")
        print(f"F4 (battery) : {(F4b - F4l) / F4b * 100:.2f}% change")
        print(f"\nTrade-off: Scenario B installs {(x_loss[2] + x_loss[3]) - (x_base[2] + x_base[3]):.2f} MW more RES")
        print("NOTE: the RES hosting-factor cap (35% of load) is a soft penalty in F; B may violate it.")
    out = dict(x_base=x_base, x_loss=x_loss, F_bal=[F1b, F2b, F3b, F4b], F_loss=[F1l, F2l, F3l, F4l])
    dump_json(out, f"{results_dir}/loss_priority_{pr.tag}.json")
    return out
