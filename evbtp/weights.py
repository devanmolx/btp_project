"""Objective-weighting methods (port of ahp_weights.m, critic_weights.m and the w_all block of
calibrate_and_weights_INDIA.m).  Every vector is ordered [F1 loss, F2 voltage dev., F3 cost, F4 battery].

    ahp_critic      (DEFAULT) w_AHP * w_CRITIC, normalised
    ahp             AHP principal-eigenvector weights of config.AHP_MATRIX
    critic          CRITIC (Diakoulaki et al., 1995) on the F1..F4 calibration samples
    critic_f4fixed  CRITIC on F1..F3, F4 fixed at ALPHA_FIXED
    markov          Markov-chain weights on range-normalised F1..F3, F4 fixed at ALPHA_FIXED
    combined        normalised mean of markov and critic
    base_paper      fixed weights of the 2017 base paper (config.BASE_PAPER_W = 0.4, 0.3, 0.2, 0.1)
    equal           [0.25 0.25 0.25 0.25]
"""
from dataclasses import dataclass
import numpy as np

from . import config
from .markov import markov_weights

RI_TABLE = (0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)   # Saaty random index, n = 1..10
METHODS = ("ahp_critic", "ahp", "critic", "critic_f4fixed", "markov", "combined", "base_paper", "equal")


class AHPInconsistentError(ValueError):
    pass


@dataclass
class AHPInfo:
    lambda_max: float
    CI: float
    CR: float
    RI: float
    w_gm: np.ndarray


def ahp_weights(A, verbose=True, tol=1e-3):
    """Returns (w, AHPInfo).  Raises ValueError on an invalid matrix and AHPInconsistentError if CR >= 0.10."""
    A = np.asarray(A, float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"A must be square (got {A.shape})")
    if n != 4:
        raise ValueError(f"expected a 4x4 matrix for F1..F4 (got {n}x{n})")
    if not np.all(np.isfinite(A)) or np.any(A <= 0):
        raise ValueError("all entries of A must be positive and finite")
    if np.any(np.abs(np.diag(A) - 1) > tol):
        raise ValueError("diagonal of A must be 1")
    bad = np.argwhere(np.abs(A * A.T - 1) > tol)
    if bad.size:
        i, j = bad[0]
        raise ValueError(f"A is not reciprocal: A[{i},{j}]={A[i, j]:.4g} but 1/A[{j},{i}]={1 / A[j, i]:.4g}")

    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lambda_max = float(vals.real[k])
    v = np.abs(vecs[:, k].real)
    w = v / v.sum()

    gm = np.prod(A, axis=1) ** (1 / n)
    w_gm = gm / gm.sum()

    RI = RI_TABLE[n - 1]
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI
    info = AHPInfo(lambda_max, CI, CR, RI, w_gm)
    if verbose:
        print("AHP weights (eigenvector)    : [" + " ".join(f"{x:.4f}" for x in w) + "]")
        print("AHP weights (geometric mean) : [" + " ".join(f"{x:.4f}" for x in w_gm) + "]")
        print(f"lambda_max = {lambda_max:.4f} | CI = {CI:.4f} | CR = {CR:.4f} (RI = {RI:.2f})")
    if CR >= 0.10:
        raise AHPInconsistentError(f"CR = {CR:.4f} >= 0.10 -- the pairwise judgements are inconsistent "
                                   "and must be revised.")
    return w, info


def critic_weights(F):
    """CRITIC: min-max normalise (1 = best, all objectives minimised), C_j = sigma_j * sum_k (1 - r_jk)."""
    F = np.asarray(F, float)
    N, m = F.shape
    X = np.zeros((N, m))
    for j in range(m):
        col = F[:, j]
        rng_j = col.max() - col.min()
        X[:, j] = 0.0 if rng_j < 1e-12 else (col.max() - col) / rng_j
    sigma = X.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    C = sigma * np.sum(1 - R, axis=0)
    return np.ones(m) / m if C.sum() < np.finfo(float).eps else C / C.sum()


def compute_all(F_vals, lo, hi, ahp_matrix=None, alpha_fixed=None, base_paper_w=None, verbose=True):
    """All weighting methods from the calibration samples.  F_vals: (N, 4).  Returns (w_all dict, AHPInfo)."""
    ahp_matrix = config.AHP_MATRIX if ahp_matrix is None else ahp_matrix
    alpha = config.ALPHA_FIXED if alpha_fixed is None else alpha_fixed
    base_paper_w = config.BASE_PAPER_W if base_paper_w is None else base_paper_w
    F_vals = np.asarray(F_vals, float)
    eps = np.finfo(float).eps

    norm = np.column_stack([(F_vals[:, j] - lo[j]) / max(hi[j] - lo[j], eps) for j in range(3)])
    w_markov = np.append(markov_weights(norm) * (1 - alpha), alpha)
    w_critic = critic_weights(F_vals)
    w_critic_f4fixed = np.append(critic_weights(F_vals[:, :3]) * (1 - alpha), alpha)
    w_combined = (w_markov + w_critic) / 2
    w_combined /= w_combined.sum()
    if verbose:
        print("\nAHP (pairwise matrix from config.AHP_MATRIX):")
    w_ahp, info = ahp_weights(ahp_matrix, verbose=verbose)
    w_ahp_critic = w_ahp * w_critic
    w_ahp_critic /= w_ahp_critic.sum()

    w_all = dict(ahp_critic=w_ahp_critic, ahp=w_ahp, critic=w_critic, critic_f4fixed=w_critic_f4fixed,
                 markov=w_markov, combined=w_combined, equal=np.full(4, 0.25))
    if base_paper_w is not None:
        b = np.asarray(base_paper_w, float)
        w_all["base_paper"] = b / b.sum()
    return w_all, info
