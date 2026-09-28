"""IEEE 33/69-bus network data (replaces get_system_data_INDIA.m)."""
import json
from dataclasses import dataclass
import numpy as np
from .config import DATA_DIR, get_max_bus


@dataclass(frozen=True)
class SystemData:
    FB: np.ndarray      # from-bus (1-based bus numbers)
    TB: np.ndarray      # to-bus   (1-based)
    R: np.ndarray       # ohm
    X: np.ndarray       # ohm
    P_load: np.ndarray  # W  (index = bus-1)
    Q_load: np.ndarray  # VAr
    nb: int
    nl: int
    max_bus: int


def load_system(system: int) -> SystemData:
    mb = get_max_bus(system)
    d = json.loads((DATA_DIR / f"ieee{mb}.json").read_text())
    return SystemData(
        FB=np.array(d["FB"], int), TB=np.array(d["TB"], int),
        R=np.array(d["R"], float), X=np.array(d["X"], float),
        P_load=np.array(d["P_load"], float), Q_load=np.array(d["Q_load"], float),
        nb=int(d["nb"]), nl=int(d["nl"]), max_bus=mb)
