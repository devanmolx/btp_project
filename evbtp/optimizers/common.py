"""Shared plumbing for the three optimizers."""
import time
from dataclasses import dataclass, field, asdict
import numpy as np


class Evaluator:
    """Counts objective evaluations (NOFE) -- every call is one evaluation."""
    def __init__(self, problem, w):
        self.problem, self.w, self.count = problem, np.asarray(w, float), 0

    def __call__(self, x):
        self.count += 1
        return self.problem.objective(x, self.w)


@dataclass
class OptResult:
    algo: str
    gbest: np.ndarray            # repaired layout (what was actually evaluated)
    gbest_fit: float
    NOFE: int
    time: float
    F1: float = np.nan; F2: float = np.nan; F3: float = np.nan; F4: float = np.nan
    w_fixed: np.ndarray = None
    convergence: np.ndarray = None      # best-so-far per iteration
    mean_history: np.ndarray = None     # population mean per iteration
    trace: np.ndarray = None            # (n, 2) [NOFE, best-so-far] per iteration (for NOFE-aligned plots)
    F_history: np.ndarray = None        # (iters, 4) F1..F4 of gbest per iteration (GA-PSO only)
    system: int = 33

    def to_dict(self):
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("gbest", "w_fixed", "convergence", "mean_history", "trace", "F_history"):
            if d.get(k) is not None:
                d[k] = np.asarray(d[k], float)
        return cls(**d)


def finalize(algo, problem, w, gbest, gbest_fit, ev, t0, conv, mean_hist, trace, F_hist=None, verbose=True):
    elapsed = time.perf_counter() - t0
    gbest = problem.repair_solution(gbest)              # report the layout actually evaluated
    _, F1, F2, F3, F4 = problem.detailed(gbest, w)
    res = OptResult(algo=algo, gbest=gbest, gbest_fit=float(gbest_fit), NOFE=ev.count, time=elapsed,
                    F1=F1, F2=F2, F3=F3, F4=F4, w_fixed=np.asarray(w, float),
                    convergence=np.asarray(conv, float), mean_history=np.asarray(mean_hist, float),
                    trace=np.asarray(trace, float).reshape(-1, 2),
                    F_history=None if F_hist is None else np.asarray(F_hist, float), system=problem.system_id)
    if verbose:
        print_result(res)
    return res


def print_result(r: OptResult):
    g = r.gbest
    print("\n" + "=" * 45 + f"\n  COMPLETE - {r.algo} India ({r.system}-bus)\n" + "=" * 45)
    print("Weights [tau,beta,gamma,alpha]: [" + " ".join(f"{v:.4f}" for v in r.w_fixed) + "]")
    print(f"Best Fitness: {r.gbest_fit:.6f} | NOFE: {r.NOFE} | Time: {r.time:.2f} s")
    print(f"RES 1: Bus {int(g[0])}, {g[2] * 1000:.1f} kW | RES 2: Bus {int(g[1])}, {g[3] * 1000:.1f} kW")
    print(f"CS 1 : Bus {int(g[4])}, {g[6] * 1000:.1f} kW | CS 2 : Bus {int(g[5])}, {g[7] * 1000:.1f} kW")
    print(f"\nF1={r.F1:.4f} MWh/day  F2={r.F2:.4f} pu (24h sum)  F3=Rs.{r.F3:.0f}  F4=Rs.{r.F4:.2f}")
