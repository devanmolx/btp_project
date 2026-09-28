"""The optimisation problem: objective function v9 (port of objective_function_INDIA.m,
objective_function_detailed_INDIA.m and repair_solution_INDIA.m).

The MATLAB code kept two hand-synchronised copies of the objective (plain + detailed) and
cached data in `persistent` variables (hence "run clear functions after recalibrating").
Here there is ONE implementation, `Problem._evaluate`, and the cache is simply the Problem
object -- build a new one after recalibrating or switching system.
"""
from dataclasses import dataclass
import warnings
import numpy as np

from . import config
from .config import DEFAULT_WEIGHTS, RES_CAP_MAX, RES_CAP_MIN, CS_CAP_MAX, CS_CAP_MIN
from .ev_model import EVData, load_matlab_ev_data
from .loadflow import RadialLoadFlow
from .profiles import Profiles, build_profiles
from .system import SystemData, load_system
from .utils import mround

# ---- model constants (all as in the MATLAB objective) ----
RES_HF_LIMIT = 0.35            # RES hosting factor: combined RES <= 35% of system load
RES_HF_PENALTY_SLOPE = 50.0
V2G_RATING_FRAC = 0.35         # discharge rating as a fraction of station capacity
RES_CAPEX_PER_KW = 45000.0     # Rs/kW
CS_CAPEX_PER_KW = 30000.0      # Rs/kW
DISCOUNT_RATE = 0.09
N_RES_YEARS, N_CS_YEARS = 20, 10
DOD = 0.8
FEAS_PENALTY_SLOPE = 1.2
REG_WEIGHT_CS = 0.05
EPS = np.finfo(float).eps

_PLACEHOLDER_BASES = dict(F1_lo=0.0, F1_hi=6.917742, F2_lo=0.0, F2_hi=45.8954,
                          F3_lo=0.0, F3_hi=649289.0, F4_lo=0.0, F4_hi=25000.0)


def _crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


@dataclass
class Calibration:
    F1_lo: float; F1_hi: float
    F2_lo: float; F2_hi: float
    F3_lo: float; F3_hi: float
    F4_lo: float; F4_hi: float
    w_fixed: np.ndarray
    n_samples: int = 0
    F1_vals: np.ndarray = None
    F2_vals: np.ndarray = None
    F3_vals: np.ndarray = None
    F4_vals: np.ndarray = None

    def save(self, path):
        np.savez(path, **{k: (np.asarray(v) if v is not None else np.array([])) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        kw = {k: d[k] for k in d.files}
        for k in ("F1_lo F1_hi F2_lo F2_hi F3_lo F3_hi F4_lo F4_hi".split()):
            kw[k] = float(kw[k])
        kw["n_samples"] = int(kw["n_samples"])
        return cls(**kw)

    def bases(self):
        return {k: getattr(self, k) for k in _PLACEHOLDER_BASES}


def calibration_path(results_dir, system):
    from pathlib import Path
    return Path(results_dir) / f"calibration_{system}bus.npz"


class Problem:
    def __init__(self, system=config.SYSTEM_CHOICE, ev: EVData = None, profiles: Profiles = None,
                 calibration: Calibration = None, results_dir=config.DEFAULT_RESULTS_DIR, quiet=False):
        self.system_id = config.get_max_bus(system)
        self.sys: SystemData = load_system(system)
        self.max_bus = self.sys.max_bus
        self.prof = profiles or build_profiles()
        self.ev = ev or load_matlab_ev_data()
        self.lf = RadialLoadFlow(self.sys.FB, self.sys.TB, self.sys.R, self.sys.X, self.sys.nb)
        self.lb, self.ub = config.bounds(system)
        self.results_dir = results_dir

        if calibration is None:
            p = calibration_path(results_dir, self.system_id)
            if p.exists():
                calibration = Calibration.load(p)
        self.calibration = calibration
        if calibration is None:
            if not quiet:
                warnings.warn(f"No calibration file for the {self.system_id}-bus system -- using placeholder "
                              "normalisation bases. Run `python -m evbtp calibrate` first.")
            self._bases = dict(_PLACEHOLDER_BASES)
            self.w_fixed = np.array(DEFAULT_WEIGHTS)
        else:
            self._bases = calibration.bases()
            self.w_fixed = np.asarray(calibration.w_fixed, float)

        # ---- precomputed constants ----
        p, e = self.prof, self.ev
        self._ratio = p.load_profile / p.load_profile.max()
        self._gen_pu = p.solar_profile * 0.5 + p.wind_profile * 0.5
        self._peak_mask = p.price_TOU > p.price_TOU.mean()
        self._batt_coeff = (e.Cb * e.avg_BCAP + e.Cr) / (e.Lc * e.avg_BCAP * DOD)
        self._P_base = self.sys.P_load[None, :] * self._ratio[:, None]      # (24, nb) W
        self._Q_base = self.sys.Q_load[None, :] * self._ratio[:, None]
        self._res_hf_cap_MW = RES_HF_LIMIT * self.sys.P_load.sum() / 1e6
        self._EV_peak = e.P_EV_hourly.max(axis=0)
        self._daily_capex_res_per_MW = _crf(DISCOUNT_RATE, N_RES_YEARS) * RES_CAPEX_PER_KW * 1000 / 365
        self._daily_capex_cs_per_MW = _crf(DISCOUNT_RATE, N_CS_YEARS) * CS_CAPEX_PER_KW * 1000 / 365
        self.nofe = 0                                   # running evaluation counter (optional use)

    # ------------------------------------------------------------------ decoding / repair
    def decode(self, x):
        mb = self.max_bus
        x = np.asarray(x, float)
        b = [int(max(2, min(mb, mround(x[i])))) for i in (0, 1, 4, 5)]
        caps = (max(RES_CAP_MIN, min(RES_CAP_MAX, x[2])), max(RES_CAP_MIN, min(RES_CAP_MAX, x[3])),
                max(CS_CAP_MIN, min(CS_CAP_MAX, x[6])), max(CS_CAP_MIN, min(CS_CAP_MAX, x[7])))
        return (*self._repair_bus_collisions(b), *caps)   # rb1, rb2, cb1, cb2, rc1, rc2, cc1, cc2

    def _repair_bus_collisions(self, buses):
        """Reassign any colliding bus to the nearest free bus in [2, MAX_BUS] (tries +d before -d)."""
        mb = self.max_bus
        used = []
        for b in buses:
            if b in used:
                found = False
                for d in range(1, mb):
                    for s in (1, -1):
                        cand = b + s * d
                        if 2 <= cand <= mb and cand not in used:
                            b = cand; found = True; break
                    if found:
                        break
            used.append(b)
        return used

    def repair_solution(self, x):
        """Same clamp + collision repair the objective applies; returns the layout actually evaluated."""
        rb1, rb2, cb1, cb2, rc1, rc2, cc1, cc2 = self.decode(x)
        return np.array([rb1, rb2, rc1, rc2, cb1, cb2, cc1, cc2], float)

    # ------------------------------------------------------------------ objective
    def _evaluate(self, x, w, reg_weight_RES):
        tau, beta, gamma, alpha = w
        rb1, rb2, cb1, cb2, rc1, rc2, cc1, cc2 = self.decode(x)
        p, e = self.prof, self.ev

        hf_pen = max(0.0, (rc1 + rc2) - self._res_hf_cap_MW) * RES_HF_PENALTY_SLOPE

        # --- vectorised 24-hour network snapshots ---
        v1 = np.where(self._peak_mask, np.minimum(e.V2G_avail_hourly[:, 0], cc1 * 1000 * V2G_RATING_FRAC), 0.0)
        v2 = np.where(self._peak_mask, np.minimum(e.V2G_avail_hourly[:, 1], cc2 * 1000 * V2G_RATING_FRAC), 0.0)
        net1 = e.P_EV_hourly[:, 0] - v1
        net2 = e.P_EV_hourly[:, 1] - v2

        P = self._P_base.copy(); Q = self._Q_base
        P[:, cb1 - 1] += net1 * 1000
        P[:, cb2 - 1] += net2 * 1000
        P[:, rb1 - 1] -= rc1 * 1e6 * self._gen_pu
        P[:, rb2 - 1] -= rc2 * 1e6 * self._gen_pu

        V, P_loss = self.lf.solve(P, Q)

        F1 = float(np.sum(P_loss / 1e6))
        F2 = float(np.sum(np.abs(1 - np.abs(V[:, 1:]))))
        P_sub_kW = np.maximum(0.0, P.sum(axis=1) + P_loss) / 1000
        P_V2G = v1 + v2
        F3 = float(np.sum(P_sub_kW * p.price_TOU - P_V2G * p.price_V2G))
        F4 = float(self._batt_coeff * P_V2G.sum())

        # annualised capital cost -> daily equivalent
        F3 += (rc1 + rc2) * self._daily_capex_res_per_MW + (cc1 + cc2) * self._daily_capex_cs_per_MW

        # feasibility: charging station must cover the EV peak
        short = max(0.0, self._EV_peak[0] - cc1 * 1000) + max(0.0, self._EV_peak[1] - cc2 * 1000)
        feas = short / 1000 * FEAS_PENALTY_SLOPE

        reg = (reg_weight_RES * ((rc1 + rc2) / (2 * RES_CAP_MAX))
               + REG_WEIGHT_CS * ((cc1 + cc2) / (2 * CS_CAP_MAX)))

        b = self._bases
        F1p = (F1 - b["F1_lo"]) / max(b["F1_hi"] - b["F1_lo"], EPS)
        F2p = (F2 - b["F2_lo"]) / max(b["F2_hi"] - b["F2_lo"], EPS)
        F3p = (F3 - b["F3_lo"]) / max(b["F3_hi"] - b["F3_lo"], EPS)
        F4p = (F4 - b["F4_lo"]) / max(b["F4_hi"] - b["F4_lo"], EPS)
        F = tau * F1p + beta * F2p + gamma * F3p + alpha * F4p + feas + reg + hf_pen
        return F, F1, F2, F3, F4

    def objective(self, x, w=None, reg_weight_RES=0.05) -> float:
        return self._evaluate(x, self.w_fixed if w is None else w, reg_weight_RES)[0]

    def detailed(self, x, w=None, reg_weight_RES=0.05):
        """Returns (F_total, F1, F2, F3, F4) like objective_function_detailed_INDIA."""
        return self._evaluate(x, self.w_fixed if w is None else w, reg_weight_RES)

    # convenience used by optimizers
    __call__ = objective
