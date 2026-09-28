from .gapso import run_gapso
from .hoa import run_hoa
from .mpa import run_mpa
from .common import OptResult

ALGORITHMS = {"gapso": run_gapso, "hoa": run_hoa, "mpa": run_mpa}
DISPLAY = {"gapso": "GA-PSO", "hoa": "HOA", "mpa": "MPA"}
