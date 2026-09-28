# EV-BTP India — Python port

Optimal siting and sizing of **2 renewable (RES) units + 2 EV charging stations (with V2G)** on the
IEEE 33-bus or 69-bus feeder, for Delhi/BRPL data. Three metaheuristics — **GA-PSO, HOA (Hippopotamus),
MPA (Marine Predators)** — are compared at equal evaluation budgets.
This is a Python port of the MATLAB project `EVBTPINDIAFinal_v2`.

## Install & run

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib (pytest for tests)

python -m evbtp all                    # whole pipeline, 33-bus
python -m evbtp --system 69 all        # whole pipeline, 69-bus
```

or step by step (same order as the MATLAB MANIFEST):

```bash
python -m evbtp ev-model               # EV fleet summary + plot
python -m evbtp calibrate              # RUN FIRST: normalisation bases + objective weights
python -m evbtp optimize gapso         # also: hoa, mpa   (--seed N, --max-iter 200, --pop-size 50)
python -m evbtp compare                # single-run 3-way table + plots
python -m evbtp fair-compare --runs 30 # equal-NOFE, multi-seed comparison  <- the number to report
python -m evbtp base-case              # improvement vs no-RES/no-EV base case + voltage profile
python -m evbtp loss-priority          # balanced vs loss-priority scenario
python -m evbtp weights                # weights of every weighting method + AHP consistency ratio
python -m evbtp robustness             # MPA per weight set x 5 seeds (--budget 20000, --seeds 5)
```

## Objective weights

`calibrate` computes every weighting method from the same 1,000 samples (`evbtp/weights.py`) and copies
the one named by `WEIGHT_METHOD` in `evbtp/config.py` into `w_fixed`, which the optimizers use:

| method | definition |
|---|---|
| `ahp_critic` **(default)** | `w_AHP * w_CRITIC`, normalised |
| `ahp` | principal eigenvector of `AHP_MATRIX` (Saaty 1-9); errors if CR >= 0.10 |
| `critic` | CRITIC (Diakoulaki et al., 1995) on F1..F4 |
| `critic_f4fixed` | CRITIC on F1..F3, F4 fixed at 0.05 |
| `markov` | Markov-chain weights on F1..F3, F4 fixed at 0.05 (the original method) |
| `combined` | normalised mean of `markov` and `critic` |
| `base_paper` | `BASE_PAPER_W` = [0.4, 0.3, 0.2, 0.1], the 2017 base paper |
| `equal` | 0.25 each |

`AHP_MATRIX` is a **placeholder** until the pairwise judgements are confirmed by the guide.
The MATLAB versions are `ahp_weights.m`, `critic_weights.m`, `SCENARIO_SETTINGS_INDIA.m`,
`compare_weighting_methods_INDIA.m`, `run_mpa_once_INDIA.m` and `weight_robustness_INDIA.m`.

Global flags go **before** the command: `--system 33|69`, `--results-dir DIR`, `--no-plots`, `--show`,
`--ev-source matlab|generate`, `--ev-seed N`.

Outputs land in `results/` (JSON + calibration `.npz`) and `results/figures/` (PNG). Files carry the system
in their name (`GAPSO_33bus_results.json`, `calibration_69bus.npz`, …).

## 33 ↔ 69 bus

MATLAB: edit `SYSTEM_CHOICE_INDIA.m`. Python: pass `--system 69` (default lives in `evbtp/config.py`,
`SYSTEM_CHOICE`). As before, a different system is a different problem — run `calibrate` and the optimizers
again for it. Because results/calibration are keyed by system, you can no longer accidentally mix them, and
the MATLAB "run `clear functions`" caveat is gone (the cache is just the `Problem` object).

## DSTATCOM extension (optional, `--statcom`)

Adds **two DSTATCOMs** (bus + kVAr rating) to the decision vector, following Abdelaziz et al., *Sci Rep* 14:28974
(2024). Every command now runs the DSTATCOM problem first and then the original one (`--statcom` = DSTATCOM only,
`--no-statcom` = original only). The original problem is still exactly the MATLAB one (regression tests unchanged).

```bash
python -m evbtp --system 69 all                      # DSTATCOM run, then original run (each has its own calibration)
python -m evbtp --system 69 --statcom optimize hoa   # DSTATCOM only; also gapso, mpa, `fair-compare`, `base-case`
```

* Decision vector grows 8 → 12: `x[8:10]` DSTATCOM buses, `x[10:12]` ratings (0–1.0 MVAr each). A DSTATCOM may share a bus with a RES/CS unit.
* Model: reactive injection at the bus, peaking at the rated value at the load peak and following the load shape
  (`Problem.statcom_q`). It enters the load flow through Q, so F1 (loss) and F2 (voltage) improve directly.
* Cost: Rs 4,200/kVAr (~US$50/kVAr as in the paper), annualised with the same CRF machinery as RES/CS over 10 years
  (the paper uses 1 year), added to F3. The four-term objective is unchanged.
* Diagnostics (`Problem.metrics`, printed by `base-case`): mean P/Q loss, VDI, minimum VSI (Chakravorty form) and Vmin,
  for the optimised layout **with vs without** its DSTATCOMs — the direct measure of what they buy.
* Files get a `_statcom` suffix (`HOA_69bus_statcom_results.json`, `calibration_69bus_statcom.npz`, ...), so they never
  mix with the base results.

## MATLAB → Python map

| MATLAB | Python |
|---|---|
| `SYSTEM_CHOICE_INDIA`, `get_max_bus_INDIA`, bounds copy-pasted in each script | `evbtp/config.py` |
| `get_system_data_INDIA`, `IEEE33/69_system_data.mat` | `evbtp/system.py`, `evbtp/data/ieee33.json`, `ieee69.json` |
| `IEEE33_bus_data.m`, `IEEE69_bus_data.m` | data converted once (`tools/convert_mat_data.py`) |
| `profiles_24hr_INDIA.m` | `evbtp/profiles.py` |
| `EV_model_INDIA.m`, `EV_model_data_INDIA.mat` | `evbtp/ev_model.py`, `evbtp/data/ev_model_matlab.npz` |
| `run_loadflow_india` (BIBC backward/forward sweep) | `evbtp/loadflow.py` — vectorised over the 24 h |
| `objective_function_INDIA` + `_detailed_INDIA` + `repair_solution_INDIA` | `evbtp/problem.py` (`Problem.objective / .detailed / .repair_solution`) — **one** implementation instead of two synced copies |
| `markov_weights`, `calibrate_and_weights_INDIA` | `evbtp/markov.py`, `evbtp/calibrate.py` |
| `ahp_weights`, `critic_weights`, `SCENARIO_SETTINGS_INDIA`, `compare_weighting_methods_INDIA` | `evbtp/weights.py`, `evbtp/config.py`, `evbtp/robustness.py` (`weights` command) |
| `run_mpa_once_INDIA`, `weight_robustness_INDIA` | `evbtp/robustness.py` (`robustness` command) |
| `GAPSO_/HOA_/MPA_INDIA_optimizer` **and** the three copies inside `fair_compare_…` | `evbtp/optimizers/{gapso,hoa,mpa}.py` — each algorithm exists once and runs in fixed-iteration mode (`max_iter=200`) or NOFE-budget mode (`nofe_budget=…`) |
| `fair_compare_GAPSO_HOA_MPA_INDIA` | `evbtp/fair_compare.py` |
| `compare_ALL_INDIA`, `compare_GAPSO_MPA_INDIA` | `evbtp/reports.py` (`compare_GAPSO_MPA` is a subset of `compare_ALL`, so it is not ported separately) |
| `plot_algo_diagnostics_INDIA`, `plot_comparison_INDIA`, plots in the EV/profile scripts | `evbtp/plotting.py` |
| `base_case_INDIA`, `run_loss_priority_scenario_INDIA` | `evbtp/base_case.py`, `evbtp/loss_priority.py` |
| `retired/`, `*.asv`, `*.mat` results | not ported (superseded / autosave / stale) |

## How faithful is it? (verified)

* **Objective function:** at the GA-PSO, HOA and MPA solutions saved in your `.mat` files, Python reproduces
  F1–F4 to ≤1e-9 relative error and the total fitness to ~1e-15 (using your saved calibration). This is a
  regression test (`tests/test_port.py`), with the MATLAB numbers hard-coded.
* **Load flow:** checked against an independent, literal transcription of the MATLAB sweep on both systems.
* **Profiles & EV model:** profiles are bit-identical; the deterministic EV pipeline reproduces MATLAB's
  `P_EV_hourly` / `V2G_avail_hourly` exactly when fed the same random draws.
* `python tools/verify_against_matlab.py /path/to/matlab/project` re-runs these checks against your `.mat` files.

Run the tests with `python -m pytest -q tests`.

## Things that are *not* bit-identical (and why)

1. **Random numbers.** MATLAB's `rng(seed)` stream can't be reproduced in NumPy, so a given seed gives a
   different (statistically equivalent) run. Individual run results — and a freshly calibrated
   `INDIA_calibration` — will differ slightly from MATLAB's; means over many seeds should agree.
   For the EV fleet specifically, the **default is the exact fleet MATLAB generated** (shipped as `.npz`), so the
   problem data is identical. `--ev-source generate` (or `ev-model --regenerate`) builds a new one from NumPy's RNG.
2. **MATLAB `round`** rounds halves away from zero (NumPy rounds to even); `utils.mround` replicates MATLAB
   (matters e.g. for `round(0.25*50)=13` in GA-PSO's reinjection).
3. **Speed:** ~0.4 ms per objective evaluation (24 load flows solved as one batch); a 200-iteration GA-PSO run
   (~65k evaluations) takes ~25 s.

## Small differences from the MATLAB scripts (all deliberate)

* Results/calibration are per-system files (see above); results are JSON rather than `.mat`.
* `fair-compare` additionally prints a Wilcoxon rank-sum test between algorithms (your README listed it as not done).
* `loss-priority`: the MATLAB script set RES to 2.0 MW/unit, but the objective clamps RES to its 1.0 MW bound, so
  it silently evaluated 1.0 MW. The port uses 1.0 MW explicitly; the numbers are the same, the labels are honest.
* `base-case`: unchanged logic, plus a footnote — its "F3" for *Base case / EV only* is grid-energy cost only,
  while the optimised rows use the full objective (capex, V2G revenue), so that column isn't like-for-like.
  This was already true in the MATLAB version.

## Layout

```
evbtp/
  config.py  system.py  profiles.py  ev_model.py  loadflow.py  problem.py
  markov.py  calibrate.py  fair_compare.py  reports.py  plotting.py
  base_case.py  loss_priority.py  results_io.py  cli.py
  optimizers/{gapso,hoa,mpa,common}.py
  data/{ieee33.json, ieee69.json, ev_model_matlab.npz}
tests/test_port.py        tools/convert_mat_data.py   tools/verify_against_matlab.py
```

Library use:

```python
from evbtp.problem import Problem
from evbtp.calibrate import calibrate
from evbtp.optimizers import run_mpa

pr = Problem(system=33, results_dir="results")
calibrate(pr); pr = Problem(system=33, results_dir="results")   # reload to pick up the calibration
res = run_mpa(pr, seed=1)
print(res.gbest, res.gbest_fit)
```

## Known modelling caveats carried over unchanged from the MATLAB project

Documented in your `README_FIXES_AND_MPA.md` and left as-is: the HOA Eq. 29 `h` parameter is an approximation;
the capex figures and 35% hosting-factor are assumptions, not sourced values; F1 and F2 are highly correlated
(the calibration step prints this, ≈0.97 here); EV charging demand is a fixed profile per station, independent
of siting.
