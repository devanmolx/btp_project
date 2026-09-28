"""EV fleet model (port of EV_model_INDIA.m).

Two entry points
----------------
load_matlab_ev_data()   -> the exact fleet the MATLAB project generated (shipped as .npz).
                           This is the default so results match the MATLAB project.
generate_ev_data(seed)  -> a fresh fleet from NumPy's RNG.  Same model, different random
                           draws (MATLAB's rng(42) stream is not reproducible in NumPy).

Both go through `build_from_random_inputs`, which is the deterministic part of the model.
The tests feed it the MATLAB fleet's random draws and confirm it reproduces MATLAB's
P_EV_hourly / V2G_avail_hourly exactly.
"""
from dataclasses import dataclass, field
import numpy as np
from .config import DATA_DIR


@dataclass(frozen=True)
class EVSpec:
    brand: str
    BCAP: float          # kWh
    range: float         # km
    max_charge: float    # kW (AC)

    @property
    def consumption(self):   # kWh/km
        return self.BCAP / self.range


EV_SPECS = (
    EVSpec("Tata Tiago EV", 24.0, 280, 3.3),
    EVSpec("Tata Nexon EV", 40.5, 465, 7.2),
    EVSpec("Mahindra XUV400", 39.4, 456, 7.2),
    EVSpec("MG ZS EV", 50.3, 461, 7.4),
)

# Travel statistics (India)
MU_MD, SIG_MD = 30.0, 12.0
MU_ARRIV, SIG_ARRIV = 19.0, 1.5
MU_DEPART, SIG_DEPART = 9.0, 1.5
# Battery economics (Rs)
CB, CR, LC = 9000.0, 6500.0, 2000.0
N_EV_DEFAULT = 160     # 80 per station -> ~450-550 kW peak per station


@dataclass
class EVData:
    specs: tuple
    N_EV: int
    Md: np.ndarray
    EV_type: np.ndarray        # 1..4 (MATLAB numbering kept for parity)
    E_demand: np.ndarray
    t_arriv: np.ndarray
    t_depart: np.ndarray
    t_dur: np.ndarray
    SOC_init: np.ndarray
    SOC_desired: np.ndarray
    P_EV_hourly: np.ndarray        # (24, 2) kW
    V2G_avail_hourly: np.ndarray   # (24, 2) kW
    T_avg_hourly: np.ndarray
    Cb: float = CB
    Cr: float = CR
    Lc: float = LC
    extra: dict = field(default_factory=dict)

    @property
    def P_EV_total(self):
        return self.P_EV_hourly.sum(axis=1)

    @property
    def avg_BCAP(self):
        return float(np.mean([s.BCAP for s in self.specs]))


def build_from_random_inputs(Md, EV_type, t_arriv, t_depart, SOC_init, specs=EV_SPECS, n_stations=2):
    """Deterministic part of the EV model (MATLAB sections 3-4)."""
    Md = np.asarray(Md, float); EV_type = np.asarray(EV_type, int)
    t_arriv = np.asarray(t_arriv, float); t_depart = np.asarray(t_depart, float)
    SOC_init = np.asarray(SOC_init, float)
    N = len(Md)
    bcap = np.array([specs[k - 1].BCAP for k in EV_type])
    em = np.array([specs[k - 1].consumption for k in EV_type])
    chg = np.array([specs[k - 1].max_charge for k in EV_type])

    Md_max = bcap / em
    E_demand = np.where(Md >= Md_max, bcap, Md * em)

    t_dur = np.mod(t_depart - t_arriv, 24)
    t_dur[t_dur == 0] = 24
    SOC_desired = np.minimum(np.minimum(SOC_init + E_demand / bcap, SOC_init + chg * t_dur / bcap), 1.0)

    per = N // n_stations
    P = np.zeros((24, n_stations)); V2G = np.zeros((24, n_stations))
    Tsum = np.zeros((24, n_stations)); Tcnt = np.zeros((24, n_stations))
    for t in range(1, 25):                       # hour index as in MATLAB (1..24)
        for s in range(n_stations):
            for k in range(s * per, (s + 1) * per):
                if t_depart[k] > t_arriv[k]:
                    parked = t_arriv[k] <= t <= t_depart[k]
                else:                            # overnight
                    parked = (t >= t_arriv[k]) or (t <= t_depart[k])
                if not parked:
                    continue
                if SOC_desired[k] > SOC_init[k]:
                    P[t - 1, s] += chg[k]; Tsum[t - 1, s] += t_dur[k]; Tcnt[t - 1, s] += 1
                if SOC_desired[k] >= 0.50:
                    V2G[t - 1, s] += chg[k] * 0.85
    return dict(Md_max=Md_max, E_demand=E_demand, t_dur=t_dur, SOC_desired=SOC_desired,
                P_EV_hourly=P, V2G_avail_hourly=V2G, T_avg_hourly=Tsum / np.maximum(Tcnt, 1))


def _pack(specs, N, Md, EV_type, t_arriv, t_depart, SOC_init, Cb=CB, Cr=CR, Lc=LC):
    o = build_from_random_inputs(Md, EV_type, t_arriv, t_depart, SOC_init, specs)
    return EVData(specs=tuple(specs), N_EV=N, Md=np.asarray(Md), EV_type=np.asarray(EV_type, int),
                  E_demand=o["E_demand"], t_arriv=np.asarray(t_arriv), t_depart=np.asarray(t_depart),
                  t_dur=o["t_dur"], SOC_init=np.asarray(SOC_init), SOC_desired=o["SOC_desired"],
                  P_EV_hourly=o["P_EV_hourly"], V2G_avail_hourly=o["V2G_avail_hourly"],
                  T_avg_hourly=o["T_avg_hourly"], Cb=Cb, Cr=Cr, Lc=Lc,
                  extra={"Md_max": o["Md_max"]})


def generate_ev_data(seed=42, N_EV=N_EV_DEFAULT) -> EVData:
    """Fresh synthetic fleet (log-normal daily distance, normal arrival/departure)."""
    rng = np.random.default_rng(seed)
    mu_ln = np.log(MU_MD ** 2 / np.sqrt(SIG_MD ** 2 + MU_MD ** 2))
    sig_ln = np.sqrt(np.log(1 + (SIG_MD / MU_MD) ** 2))
    Md = np.maximum(np.exp(mu_ln + sig_ln * rng.standard_normal(N_EV)), 1.0)
    EV_type = rng.integers(1, 5, N_EV)
    t_arriv = np.mod(MU_ARRIV + SIG_ARRIV * rng.standard_normal(N_EV), 24)
    t_depart = np.mod(MU_DEPART + SIG_DEPART * rng.standard_normal(N_EV), 24)
    SOC_init = 0.2 + 0.6 * rng.random(N_EV)
    return _pack(EV_SPECS, N_EV, Md, EV_type, t_arriv, t_depart, SOC_init)


def load_matlab_ev_data() -> EVData:
    d = np.load(DATA_DIR / "ev_model_matlab.npz", allow_pickle=False)
    specs = tuple(EVSpec(str(b), float(c), float(r), float(m)) for b, c, r, m in
                  zip(d["brand"], d["spec_BCAP"], d["spec_range"], d["spec_max_charge"]))
    ev = EVData(specs=specs, N_EV=int(d["N_EV"]), Md=d["Md"], EV_type=d["EV_type"].astype(int),
                E_demand=d["E_demand"], t_arriv=d["t_arriv"], t_depart=d["t_depart"], t_dur=d["t_dur"],
                SOC_init=d["SOC_init"], SOC_desired=d["SOC_desired"], P_EV_hourly=d["P_EV_hourly"],
                V2G_avail_hourly=d["V2G_avail_hourly"], T_avg_hourly=d["T_avg_hourly"],
                Cb=float(d["Cb"]), Cr=float(d["Cr"]), Lc=float(d["Lc"]),
                extra={"Md_max": d["Md_max"]})
    return ev


def summary(ev: EVData) -> str:
    lines = [
        f"EV POPULATION SUMMARY ({ev.N_EV} EVs)",
        f"  Avg daily distance : {ev.Md.mean():.2f} km",
        f"  Avg charging demand: {ev.E_demand.mean():.2f} kWh",
        f"  Avg SOC initial    : {ev.SOC_init.mean() * 100:.2f} %",
        f"  Avg SOC desired    : {ev.SOC_desired.mean() * 100:.2f} %",
        f"  Avg parking time   : {ev.t_dur.mean():.2f} hrs",
        f"  Peak EV demand     : {ev.P_EV_total.max():.2f} kW (combined)",
        f"  Peak per station   : {ev.P_EV_hourly[:, 0].max():.2f} / {ev.P_EV_hourly[:, 1].max():.2f} kW",
    ]
    for i, s in enumerate(ev.specs, 1):
        c = int(np.sum(ev.EV_type == i))
        lines.append(f"  {s.brand}: {c} EVs ({c / ev.N_EV * 100:.1f}%)")
    return "\n".join(lines)
