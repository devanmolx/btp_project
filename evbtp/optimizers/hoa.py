"""Hippopotamus Optimization Algorithm (v3; Amiri et al. 2024) -- port of HOA_INDIA_optimizer.m
and run_hoa_fair.  Same dual-mode design as run_gapso (fixed max_iter, or NOFE budget).

Phase 1 (first half of herd): male-line and female/immature-line candidates, keep the best of
    {current, male, female}.  NOTE (as in the MATLAB code): the paper's Eq. 29 stochastic
    parameter h is implemented as a random draw over five qualitatively similar forms, not a
    byte-exact transcription.
Phase 2 (second half): predator-fitness comparison + Levy-flight defence.
Phase 3 (whole herd): escape inside a window shrinking around the CURRENT position (v3a fix).
"""
import time
import numpy as np
from ..utils import fix_solution, levy_step, make_rng, random_population
from .common import Evaluator, finalize


def run_hoa(problem, w=None, *, pop_size=50, max_iter=200, nofe_budget=None, seed=None, verbose=True):
    rng = make_rng(seed)
    w = problem.w_fixed if w is None else np.asarray(w, float)
    lb, ub, mb = problem.lb, problem.ub, problem.max_bus
    n = len(lb)
    budget = np.inf if nofe_budget is None else nofe_budget
    iter_cap = np.inf if max_iter is None else max_iter
    T_horizon = 200 if max_iter is None else max_iter          # MATLAB: T = exp(-iter/max(iter,200))
    ev = Evaluator(problem, w)
    fx = lambda x: fix_solution(x, lb, ub, mb)

    pop = random_population(rng, pop_size, lb, ub, mb)
    fit = np.array([ev(p) for p in pop])
    gi = int(np.argmin(fit)); gbest_fit = fit[gi]; gbest = pop[gi].copy()
    half = pop_size // 2
    conv, mean_hist, trace = [], [], []
    if verbose:
        print(f"Initial best fitness: {gbest_fit:.6f}\n\n{'Iter':<8} {'Best F':<14} {'NOFE':<10}\n" + "-" * 40)
    t0 = time.perf_counter(); it = 0

    def update_best():
        nonlocal gbest_fit, gbest
        ci = int(np.argmin(fit))
        if fit[ci] < gbest_fit:
            gbest_fit = fit[ci]; gbest = pop[ci].copy()

    while it < iter_cap and ev.count < budget:
        it += 1
        T = np.exp(-it / max(it, T_horizon))
        herd_mean = pop.mean(axis=0)
        window = (ub - lb) / it

        # ---- Phase 1 ----
        for i in range(half):
            hb = rng.integers(1, 6)
            if hb == 1:   hvec = 2 * rng.integers(1, 3) * rng.random(n) - 1
            elif hb == 2: hvec = 2 * rng.random(n) - 1
            elif hb == 3: hvec = rng.random(n)
            elif hb == 4: hvec = rng.integers(1, 3) * rng.random(n) + (1 - rng.integers(1, 3))
            else:         hvec = rng.standard_normal(n) * 0.5
            Z1 = rng.integers(1, 3)
            xM = fx(pop[i] + hvec * (gbest - Z1 * pop[i]))
            fM = ev(xM)
            if ev.count >= budget:
                if fM < fit[i]:
                    fit[i] = fM; pop[i] = xM
                break

            Z2 = rng.integers(1, 3)
            if T > 0.6:
                xFB = pop[i] + rng.random(n) * (gbest - Z2 * herd_mean)
            elif rng.random() > 0.5:
                xFB = pop[i] + rng.random(n) * (herd_mean - gbest)
            else:
                xFB = lb + rng.random(n) * (ub - lb)
            xFB = fx(xFB)
            fFB = ev(xFB)
            which = int(np.argmin([fit[i], fM, fFB]))
            if which == 1:   fit[i] = fM;  pop[i] = xM
            elif which == 2: fit[i] = fFB; pop[i] = xFB
            if ev.count >= budget:
                break
        update_best()
        if ev.count >= budget:
            trace.append((ev.count, gbest_fit)); break

        # ---- Phase 2 ----
        for i in range(half, pop_size):
            predator = lb + rng.random(n) * (ub - lb)
            F_pred = ev(fx(predator))
            if ev.count >= budget:
                break
            dist = np.abs(predator - pop[i]) + 1e-6
            RL = levy_step(rng, n)
            scale = rng.random() / (2 - rng.random() * np.cos(2 * np.pi * rng.random()))
            if F_pred < fit[i]:
                xHR = RL * predator + scale * (1 / dist)
            else:
                xHR = RL * predator + scale * (1 / (2 * dist + rng.random()))
            xHR = fx(xHR)
            fHR = ev(xHR)
            if fHR < fit[i]:
                fit[i] = fHR; pop[i] = xHR
            if ev.count >= budget:
                break
        update_best()
        if ev.count >= budget:
            trace.append((ev.count, gbest_fit)); break

        # ---- Phase 3 ----
        for i in range(pop_size):
            l_loc = np.maximum(lb, pop[i] - window / 2)
            u_loc = np.minimum(ub, pop[i] + window / 2)
            sb = rng.integers(1, 4)
            if sb == 1:   s1 = 2 * rng.random(n) - 1
            elif sb == 2: s1 = rng.random(n)
            else:         s1 = rng.standard_normal(n) * 0.3
            target = l_loc + s1 * (u_loc - l_loc)
            new_pos = fx(pop[i] + rng.random(n) * (target - pop[i]))
            nf = ev(new_pos)
            if nf < fit[i]:
                fit[i] = nf; pop[i] = new_pos
            if ev.count >= budget:
                break
        update_best()

        conv.append(gbest_fit); mean_hist.append(fit.mean()); trace.append((ev.count, gbest_fit))
        if verbose and (it % 10 == 0 or it == 1):
            print(f"{it:<8d} {gbest_fit:<14.6f} {ev.count:<10d}")

    return finalize("HOA", problem, w, gbest, gbest_fit, ev, t0, conv, mean_hist, trace, None, verbose)
