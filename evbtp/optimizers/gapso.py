"""GA-PSO optimizer (v2 logic; port of GAPSO_INDIA_optimizer.m + run_gapso_fair).

The MATLAB project had two copies of this algorithm: the standalone script (fixed max_iter) and a
budget-driven copy inside the fair-comparison harness.  This single function does both:
    run_gapso(problem, max_iter=200)                      -> standalone behaviour
    run_gapso(problem, max_iter=None, nofe_budget=20000)  -> fair-comparison behaviour
With no budget the budget checks never fire, so the two paths are the same code.

Bug fixes carried over from v2: parents/replacement targets tracked by ORIGINAL index (no
sorted-copy index mismatch); fitness[i] synced unconditionally after every PSO evaluation;
no redundant full-population re-evaluation.
"""
import time
import numpy as np
from ..config import CAP_IDX, N_VAR, VAR_IS_INT
from ..utils import fix_solution, make_rng, mround, random_population
from .common import Evaluator, finalize


def run_gapso(problem, w=None, *, pop_size=50, max_iter=200, nofe_budget=None, seed=None,
              verbose=True, track_details=True):
    rng = make_rng(seed)
    w = problem.w_fixed if w is None else np.asarray(w, float)
    lb, ub, mb = problem.lb, problem.ub, problem.max_bus
    budget = np.inf if nofe_budget is None else nofe_budget
    iter_cap = np.inf if max_iter is None else max_iter
    ev = Evaluator(problem, w)

    W, C1, C2 = 1.0, 2.0, 2.0
    Pc_init, Pm_init = 0.8, 0.1
    GA_repeat, PSO_repeat = 5, 2
    streak_limit, reinject_frac = 3, 0.25
    shrink_prob, shrink_min, shrink_max = 0.3, 0.5, 0.95
    v_max = 0.3 * (ub - lb)

    pop = random_population(rng, pop_size, lb, ub, mb)
    fit = np.array([ev(p) for p in pop])
    pbest, pbest_fit = pop.copy(), fit.copy()
    gi = int(np.argmin(fit)); gbest_fit = fit[gi]; gbest = pop[gi].copy()
    vel = (rng.random((pop_size, N_VAR)) - 0.5) * (ub - lb) * 0.1

    Pc, Pm, eps_prev, streak = Pc_init, Pm_init, rng.random(), 0
    conv, mean_hist, F_hist, trace = [], [], [], []
    t0 = time.perf_counter()
    it = 0

    def replace(idx, cand, nf):
        pop[idx] = cand; fit[idx] = nf
        if nf < pbest_fit[idx]:
            pbest[idx] = cand; pbest_fit[idx] = nf

    while it < iter_cap and ev.count < budget:
        it += 1
        # ---------------- GA stage ----------------
        for _ in range(GA_repeat):
            sort_idx = np.argsort(fit, kind="stable")           # best ... worst, ORIGINAL indices
            top_n = max(1, int(mround(pop_size / 2)))
            top_pool = sort_idx[:top_n]
            worst_ptr = pop_size - 1

            n_cross = int(mround(Pc * pop_size))
            for k in range(1, n_cross + 1, 2):                   # MATLAB k = 1:2:n_cross (1-based k)
                if k + 1 <= pop_size and worst_ptr >= 0:
                    p1 = top_pool[rng.integers(top_n)]; p2 = top_pool[rng.integers(top_n)]
                    cp = int(rng.integers(1, N_VAR))
                    c1 = np.concatenate([pop[p1, :cp], pop[p2, cp:]])
                    c2 = np.concatenate([pop[p2, :cp], pop[p1, cp:]])
                    c1 = np.where(VAR_IS_INT, mround(np.clip(c1, lb, ub)), np.clip(c1, lb, ub))
                    c2 = np.where(VAR_IS_INT, mround(np.clip(c2, lb, ub)), np.clip(c2, lb, ub))

                    t1 = sort_idx[worst_ptr]; worst_ptr -= 1
                    nf1 = ev(c1)
                    if nf1 < fit[t1]:
                        replace(t1, c1, nf1)
                    if worst_ptr >= 0:
                        t2 = sort_idx[worst_ptr]; worst_ptr -= 1
                        nf2 = ev(c2)
                        if nf2 < fit[t2]:
                            replace(t2, c2, nf2)

            n_mut = max(1, int(mround(Pm * pop_size)))
            elite_idx = sort_idx[0]
            for _m in range(n_mut):
                m_idx = int(rng.integers(pop_size))
                if m_idx == elite_idx:
                    continue                                     # protect the current best
                m_var = int(rng.integers(N_VAR))
                cand = pop[m_idx].copy()
                if m_var in CAP_IDX and rng.random() < shrink_prob:
                    sf = shrink_min + rng.random() * (shrink_max - shrink_min)
                    cand[m_var] = max(lb[m_var], cand[m_var] * sf)
                else:
                    cand[m_var] = lb[m_var] + rng.random() * (ub[m_var] - lb[m_var])
                    if VAR_IS_INT[m_var]:
                        cand[m_var] = max(2, min(mb, mround(cand[m_var])))
                nf = ev(cand)
                if nf < fit[m_idx]:
                    replace(m_idx, cand, nf)

            ci = int(np.argmin(fit))
            if fit[ci] < gbest_fit:
                gbest_fit = fit[ci]; gbest = pop[ci].copy()

            # stagnation check on the MEDIAN
            med = np.median(fit)
            if np.sum(np.abs(fit - med) < 1e-4) >= 0.4 * pop_size:
                streak += 1
                eps_i = rng.random()
                Pc = np.clip(Pc_init * (eps_i - eps_prev + 0.5), 0.30, 0.95)
                Pm = np.clip(Pm_init * (eps_i - eps_prev + 0.3), 0.05, 0.40)
                eps_prev = eps_i
                if streak >= streak_limit:
                    worst_to_best = np.argsort(-fit, kind="stable")
                    for ridx in worst_to_best[:int(mround(reinject_frac * pop_size))]:
                        pop[ridx] = random_population(rng, 1, lb, ub, mb)[0]
                        nf = ev(pop[ridx]); fit[ridx] = nf
                        if nf < pbest_fit[ridx]:
                            pbest[ridx] = pop[ridx]; pbest_fit[ridx] = nf
                        vel[ridx] = (rng.random(N_VAR) - 0.5) * (ub - lb) * 0.1
                    streak = 0
                    Pm = min(0.40, Pm * 1.5)
            else:
                streak = 0
                Pc = min(0.9, Pc * 1.05); Pm = min(0.3, Pm * 1.05)
            if ev.count >= budget:
                break

        if ev.count >= budget:
            trace.append((ev.count, gbest_fit)); break

        # ---------------- PSO stage ----------------
        for _ in range(PSO_repeat):
            for i in range(pop_size):
                r1, r2 = rng.random(N_VAR), rng.random(N_VAR)
                vel[i] = np.clip(W * vel[i] + C1 * r1 * (pbest[i] - pop[i]) + C2 * r2 * (gbest - pop[i]), -v_max, v_max)
                pop[i] = fix_solution(pop[i] + vel[i], lb, ub, mb)
                nf = ev(pop[i])
                fit[i] = nf                                       # v2a: ALWAYS sync
                if nf < pbest_fit[i]:
                    pbest_fit[i] = nf; pbest[i] = pop[i]
                if ev.count >= budget:
                    break
            ci = int(np.argmin(pbest_fit))
            if pbest_fit[ci] < gbest_fit:
                gbest_fit = pbest_fit[ci]; gbest = pbest[ci].copy()
            if ev.count >= budget:
                break

        conv.append(gbest_fit); mean_hist.append(fit.mean()); trace.append((ev.count, gbest_fit))
        if track_details:
            F_hist.append(problem.detailed(gbest, w)[1:])         # diagnostic; not counted in NOFE
            if verbose and (it % 10 == 0 or it == 1):
                f = F_hist[-1]
                print(f"Iter {it:3d} | Best F: {gbest_fit:.6f} | Mean F: {fit.mean():.6f} | "
                      f"F1={f[0]:.3f} F2={f[1]:.3f} F3={f[2]:.0f} F4={f[3]:.2f} | NOFE: {ev.count}")

    return finalize("GA-PSO", problem, w, gbest, gbest_fit, ev, t0, conv, mean_hist, trace,
                    F_hist if track_details else None, verbose)
