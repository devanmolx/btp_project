"""Backward/forward-sweep radial load flow, vectorised over the 24 hours.

Mathematically identical to run_loadflow_india() in the MATLAB code:
  * branch currents  I_branch = BIBC @ I_load          (BIBC built once per topology)
  * voltage update   V[TB[k]] = V[FB[k]] - I_branch[k]*Z[k]/Vbase   swept k = 1..nl
The sequential sweep telescopes to  V = 1 - Path @ (I_branch * Z / Vbase),  where Path[b, k]
is 1 when branch k lies on the root->b path.  That turns the inner Python loop into one
matrix product, and lets all 24 hourly snapshots be solved at once.  Each hour still stops
at its own convergence (|dV| < 1e-6, max 100 sweeps), exactly like the scalar MATLAB loop.
"""
import numpy as np

VBASE_V = 12660.0


class RadialLoadFlow:
    def __init__(self, FB, TB, R, X, nb, vbase=VBASE_V, tol=1e-6, max_iter=100):
        FB = np.asarray(FB, int); TB = np.asarray(TB, int)
        nl = len(FB)
        self.nb, self.nl, self.R, self.vbase, self.tol, self.max_iter = nb, nl, np.asarray(R, float), vbase, tol, max_iter

        # BIBC: bus-injection -> branch-current (same construction as the .m file)
        B = np.zeros((nl, nb))
        for k in range(nl - 1, -1, -1):
            B[k, TB[k] - 1] = 1.0
            for m in range(k + 1, nl):
                if FB[m] == TB[k]:
                    B[k] += B[m]
        self.BIBC_T = np.ascontiguousarray(B.T)                 # (nb, nl)

        # Path matrix; requires parent-before-child branch ordering (true for both IEEE files,
        # and required by the original sequential sweep as well).
        seen = {1}
        Path = np.zeros((nb, nl))
        for k in range(nl):
            if FB[k] not in seen:
                raise ValueError("branches must be ordered parent-before-child")
            Path[TB[k] - 1] = Path[FB[k] - 1]
            Path[TB[k] - 1, k] = 1.0
            seen.add(int(TB[k]))
        self.Path_T = np.ascontiguousarray(Path.T)              # (nl, nb)
        self.Zc = (np.asarray(R, float) + 1j * np.asarray(X, float)) / vbase

    def solve(self, P, Q):
        """P, Q: arrays (T, nb) in W / VAr.  Returns V (T, nb) complex [pu] and P_loss (T,) [W]."""
        P = np.atleast_2d(P); Q = np.atleast_2d(Q)
        T = P.shape[0]
        S = P + 1j * Q
        V = np.ones((T, self.nb), dtype=complex)
        I_branch = np.zeros((T, self.nl), dtype=complex)
        active = np.arange(T)
        for _ in range(self.max_iter):
            Va = V[active]
            I_load = np.conj(S[active] / (Va * self.vbase))
            Ib = I_load @ self.BIBC_T
            Vn = 1.0 - (Ib * self.Zc) @ self.Path_T
            done = np.max(np.abs(np.abs(Vn) - np.abs(Va)), axis=1) < self.tol
            I_branch[active] = Ib
            V[active] = Vn
            active = active[~done]
            if active.size == 0:
                break
        P_loss = np.sum(np.abs(I_branch) ** 2 * self.R, axis=1)
        return V, P_loss
