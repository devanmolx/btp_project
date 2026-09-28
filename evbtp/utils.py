"""Small helpers that replicate MATLAB semantics where they differ from NumPy."""
import math
import numpy as np


def mround(x):
    """MATLAB round(): halves round AWAY from zero (NumPy's np.round rounds half to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def levy_step(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    """Levy flight step via Mantegna's algorithm (identical to levy_step*.m)."""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(dim) * sigma_u
    v = rng.standard_normal(dim)
    return 0.05 * u / (np.abs(v) ** (1 / beta))


def make_rng(seed=None) -> np.random.Generator:
    return np.random.default_rng(seed)


def fix_solution(x, lb, ub, max_bus):
    """Clamp to bounds, round integer variables, clamp bus variables to [2, MAX_BUS].

    This is the block of MATLAB lines repeated after every position update:
        x = max(lb,min(ub,x)); x(var_type==1)=round(...); x(1:2)/x(5:6) clamp
    """
    from .config import bus_slices, var_is_int
    x = np.minimum(ub, np.maximum(lb, x))
    x = np.where(var_is_int(len(lb)), mround(x), x)
    for sl in bus_slices(len(lb)):
        x[sl] = np.maximum(2, np.minimum(max_bus, x[sl]))
    return x


def random_population(rng, n, lb, ub, max_bus):
    from .config import bus_slices, var_is_int
    is_int = var_is_int(len(lb))
    pop = lb + rng.random((n, len(lb))) * (ub - lb)
    pop[:, is_int] = mround(pop[:, is_int])
    for sl in bus_slices(len(lb)):
        pop[:, sl] = np.maximum(2, np.minimum(max_bus, pop[:, sl]))
    return pop
