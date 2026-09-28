"""Project-wide constants.  Replaces SYSTEM_CHOICE_INDIA.m / get_max_bus_INDIA.m and the
bounds that were copy-pasted at the top of every MATLAB optimizer.

Decision vector (8 variables, same layout as MATLAB but 0-based positions):
    x[0], x[1]  RES bus 1, 2          (integer, 2..MAX_BUS -- bus numbers stay 1-based)
    x[2], x[3]  RES capacity 1, 2     (MW, 0.01..1.0)
    x[4], x[5]  charging-station bus 1, 2
    x[6], x[7]  charging-station capacity 1, 2  (MW, 0.05..0.7)
"""
from pathlib import Path
import numpy as np

# ---- THE switch: 33 or 69.  Override per-run with `--system` on the CLI. ----
SYSTEM_CHOICE = 33

N_VAR = 8
VAR_IS_INT = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
BUS_IDX = (0, 1, 4, 5)          # positions of the four bus variables
CAP_IDX = (2, 3, 6, 7)          # positions of the four capacity variables

RES_CAP_MIN, RES_CAP_MAX = 0.01, 1.0     # MW per unit (net-metering ceiling)
CS_CAP_MIN,  CS_CAP_MAX  = 0.05, 0.7     # MW per unit

DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
ALPHA_FIXED = 0.05                        # fixed F4 weight used by calibration

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RESULTS_DIR = Path("results")


def get_max_bus(system: int = SYSTEM_CHOICE) -> int:
    if system not in (33, 69):
        raise ValueError(f"system must be 33 or 69 (got {system!r})")
    return int(system)


def bounds(system: int = SYSTEM_CHOICE):
    """(lb, ub) arrays for the 8 decision variables."""
    mb = get_max_bus(system)
    lb = np.array([2, 2, RES_CAP_MIN, RES_CAP_MIN, 2, 2, CS_CAP_MIN, CS_CAP_MIN], float)
    ub = np.array([mb, mb, RES_CAP_MAX, RES_CAP_MAX, mb, mb, CS_CAP_MAX, CS_CAP_MAX], float)
    return lb, ub
