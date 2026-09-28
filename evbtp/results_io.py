"""Save/load optimizer results as JSON (replaces the *_INDIA_results.mat files).
Files are keyed by system, e.g. results/GAPSO_33bus_results.json, so a 33-bus and a 69-bus run
can never overwrite or be mistaken for each other."""
import json
from pathlib import Path
import numpy as np
from . import config
from .optimizers.common import OptResult

KEYS = {"gapso": "GAPSO", "hoa": "HOA", "mpa": "MPA"}


def _default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    raise TypeError(type(o))


def dump_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=_default, indent=1))


def result_path(results_dir, key, system, statcom=False):
    return Path(results_dir) / f"{KEYS[key]}_{config.case_tag(system, statcom)}_results.json"


def save_result(res: OptResult, key, results_dir):
    p = result_path(results_dir, key, res.system, res.statcom)
    dump_json(res.to_dict(), p)
    print(f"\nSaved to {p}")
    return p


def load_result(key, results_dir, system, statcom=False):
    p = result_path(results_dir, key, system, statcom)
    return OptResult.from_dict(json.loads(p.read_text())) if p.exists() else None
