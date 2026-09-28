"""Markov-chain objective-weight determination (port of markov_weights.m)."""
import numpy as np


def markov_weights(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, float)
    N, m = F.shape
    G = np.zeros((N, m))
    for j in range(m):
        col = F[:, j]
        rng_j = col.max() - col.min()
        G[:, j] = 0.5 if rng_j < 1e-12 else 1 - (col - col.min()) / rng_j

    M = np.zeros((m, m))
    for j in range(m):
        for l in range(m):
            if j != l:
                M[j, l] = np.sum(G[:, l] > G[:, j]) / N
    for j in range(m):
        s = M[j].sum()
        M[j] = 1.0 / m if s < 1e-12 else M[j] / s

    pi = np.ones(m) / m
    for _ in range(500):
        new = pi @ M
        if np.max(np.abs(new - pi)) < 1e-10:
            pi = new
            break
        pi = new
    return pi / pi.sum()
