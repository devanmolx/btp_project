"""One-off converter: MATLAB .mat data files -> the files shipped in evbtp/data.

Usage:  python tools/convert_mat_data.py /path/to/matlab/project

Why this exists: MATLAB's rng(42) stream cannot be reproduced in NumPy, so the
EV fleet that the original project generated (EV_model_data_INDIA.mat) is
shipped verbatim as an .npz.  That keeps every number identical to the MATLAB
project.  Use `python -m evbtp ev-model --regenerate` for a fresh Python-RNG fleet.
"""
import json, sys
from pathlib import Path
import numpy as np
import scipy.io as sio

src = Path(sys.argv[1]); out = Path(__file__).resolve().parents[1] / "evbtp" / "data"

for n in (33, 69):
    d = sio.loadmat(src / f"IEEE{n}_system_data.mat", squeeze_me=True)
    js = {k: np.atleast_1d(d[k]).tolist() for k in ["FB", "TB", "R", "X", "P_load", "Q_load"]}
    js["FB"] = [int(v) for v in js["FB"]]; js["TB"] = [int(v) for v in js["TB"]]
    js["nb"] = int(d["nb"]); js["nl"] = int(d["nl"])
    js["note"] = "P_load in W, Q_load in VAr, R/X in ohm; base 12.66 kV"
    (out / f"ieee{n}.json").write_text(json.dumps(js, indent=1))

d = sio.loadmat(src / "EV_model_data_INDIA.mat", squeeze_me=True, struct_as_record=False)
specs = d["EV_specs"]
np.savez(out / "ev_model_matlab.npz",
         brand=np.array([s.brand for s in specs]),
         spec_BCAP=np.array([s.BCAP for s in specs], float),
         spec_range=np.array([s.range for s in specs], float),
         spec_max_charge=np.array([s.max_charge for s in specs], float),
         spec_consumption=np.array([s.consumption for s in specs], float),
         **{k: np.asarray(d[k], float) for k in
            ["Md", "EV_type", "E_demand", "Md_max", "BCAP_vec", "t_arriv", "t_depart", "t_dur",
             "SOC_init", "SOC_desired", "P_EV_hourly", "P_EV_total", "T_avg_hourly",
             "V2G_avail_hourly"]},
         N_EV=int(d["N_EV"]), Cb=float(d["Cb"]), Cr=float(d["Cr"]), Lc=float(d["Lc"]),
         mu_md=float(d["mu_md"]), sig_md=float(d["sig_md"]))
print("done ->", out)
