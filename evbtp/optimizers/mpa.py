"""Marine Predators Algorithm (Faramarzi et al. 2020) -- port of MPA_INDIA_optimizer.m and
run_mpa_fair.  Same dual-mode design as run_gapso.

`schedule_iters` sets the horizon the three velocity-ratio phases and CF are scheduled over:
standalone -> max_iter (200); fair-comparison -> floor(budget/100), as in the MATLAB harness.
"""
import time
import numpy as np
from ..utils import fix_solution, levy_step, make_rng, random_population
from .common import Evaluator, finalize


def run_mpa(problem, w=None, *, pop_size=50, max_iter=200, nofe_budget=None, schedule_iters=None,
            seed=None, verbose=True):
    rng = make_rng(seed)
    w = problem.w_fixed if w is None else np.asarray(w, float)
    lb, ub, mb = problem.lb, problem.ub, problem.max_bus
    n = len(lb)
    budget = np.inf if nofe_budget is None else nofe_budget
    iter_cap = np.inf if max_iter is None else max_iter
    if schedule_iters is None:
        schedule_iters = max_iter if max_iter is not None else max(1, int(budget // 100))
    P_const, FADs = 0.5, 0.2
    ev = Evaluator(problem, w)
    fx = lambda x: fix_solution(x, lb, ub, mb)

    prey = random_population(rng, pop_size, lb, ub, mb)
    fit = np.array([ev(p) for p in prey])
    gi = int(np.argmin(fit)); gbest_fit = fit[gi]; gbest = prey[gi].copy()
    conv, mean_hist, trace = [], [], []
    if verbose:
        print(f"Initial best fitness: {gbest_fit:.6f}\n\n{'Iter':<8} {'Best F':<14} {'NOFE':<10}\n" + "-" * 40)
    t0 = time.perf_counter(); it = 0

    def update_best():
        nonlocal gbest_fit, gbest
        ci = int(np.argmin(fit))
        if fit[ci] < gbest_fit:
            gbest_fit = fit[ci]; gbest = prey[ci].copy()

    while it < iter_cap and ev.count < budget:
        it += 1
        CF = max(0.0, 1 - it / schedule_iters) ** (2 * it / schedule_iters)
        new = prey.copy()
        for i in range(pop_size):
            if it < schedule_iters / 3:                              # phase 1: Brownian
                RB = rng.standard_normal(n)
                step = RB * (gbest - RB * prey[i])
                new[i] = prey[i] + P_const * rng.random(n) * step
            elif it < 2 * schedule_iters / 3:                        # phase 2
                if i < pop_size / 2:                                 # MATLAB: i <= pop_size/2 (1-based)
                    RL = levy_step(rng, n)
                    step = RL * (gbest - RL * prey[i])
                    new[i] = prey[i] + P_const * rng.random(n) * step
                else:
                    RB = rng.standard_normal(n)
                    step = RB * (RB * gbest - prey[i])
                    new[i] = gbest + P_const * CF * step
            else:                                                    # phase 3: Levy exploitation
                RL = levy_step(rng, n)
                step = RL * (RL * gbest - prey[i])
                new[i] = gbest + P_const * CF * step
            new[i] = fx(new[i])

        for i in range(pop_size):
            nf = ev(new[i])
            if nf < fit[i]:
                fit[i] = nf; prey[i] = new[i]
            if ev.count >= budget:
                break
        update_best()
        if ev.count >= budget:
            trace.append((ev.count, gbest_fit)); break

        # ---- FADs effect ----
        if rng.random() < FADs:
            U = rng.random((pop_size, n)) < FADs
            rand_pos = lb + rng.random((pop_size, n)) * (ub - lb)
            fads = prey + CF * (rand_pos * U)
        else:
            r = rng.random()
            i1, i2 = rng.permutation(pop_size), rng.permutation(pop_size)
            fads = prey + (FADs * (1 - r) + r) * (prey[i1] - prey[i2])
        for i in range(pop_size):
            fads[i] = fx(fads[i])
            nf = ev(fads[i])
            if nf < fit[i]:
                fit[i] = nf; prey[i] = fads[i]
            if ev.count >= budget:
                break
        update_best()

        conv.append(gbest_fit); mean_hist.append(fit.mean()); trace.append((ev.count, gbest_fit))
        if verbose and (it % 10 == 0 or it == 1):
            print(f"{it:<8d} {gbest_fit:<14.6f} {ev.count:<10d}")

    return finalize("MPA", problem, w, gbest, gbest_fit, ev, t0, conv, mean_hist, trace, None, verbose)
