"""Quick numerical check of the port against numbers saved in the MATLAB .mat files."""
import sys, numpy as np, scipy.io as sio
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from evbtp.problem import Problem
from evbtp.ev_model import load_matlab_ev_data, build_from_random_inputs, generate_ev_data
from evbtp.profiles import build_profiles

M = sys.argv[1]
# 1) profiles
d = sio.loadmat(M + "/profiles_24hr_INDIA.mat", squeeze_me=True); p = build_profiles()
for k in ["load_profile", "solar_profile", "wind_profile", "price_TOU", "price_V2G"]:
    print("profile", k, "max|diff| =", np.abs(getattr(p, k) - d[k]).max())
# 2) EV deterministic pipeline
ev = load_matlab_ev_data()
o = build_from_random_inputs(ev.Md, ev.EV_type, ev.t_arriv, ev.t_depart, ev.SOC_init)
for k in ["P_EV_hourly", "V2G_avail_hourly", "SOC_desired", "E_demand", "t_dur"]:
    print("ev", k, "max|diff| =", np.abs(o[k] - getattr(ev, k)).max())
# 3) objective at saved optimizer solutions
prob = Problem(quiet=True)
for f, key in [("GAPSO_INDIA_results.mat", "results"), ("HOA_INDIA_results.mat", "hoa_results"), ("MPA_INDIA_results.mat", "mpa_results")]:
    r = sio.loadmat(M + "/" + f, squeeze_me=True, struct_as_record=False)[key]
    x = np.atleast_1d(r.gbest).astype(float)
    F, F1, F2, F3, F4 = prob.detailed(x, np.atleast_1d(r.w_fixed))
    print(f"{key:12s} x={np.round(x,3)}\n   saved F1..F4 = {r.F1:.6f} {r.F2:.6f} {r.F3:.4f} {r.F4:.4f}\n   python       = {F1:.6f} {F2:.6f} {F3:.4f} {F4:.4f}")
