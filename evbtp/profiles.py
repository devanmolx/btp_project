"""24-hour profiles -- Delhi, 10 June (port of profiles_24hr_INDIA.m).

Sources (unchanged from the MATLAB project):
  load   : NITI Aayog ICED, Northern Region, Daily_Demand_Profile, 10 Jun 2025
  solar  : Renewables.ninja, Delhi, 10 Jun 2019 (MERRA-2)
  wind   : Renewables.ninja, same site/date (Vestas V90 2000 @ 80 m)
  price  : BRPL EV tariff FY2021-22 + ToD (DERC) + surcharges + PPAC
"""
from dataclasses import dataclass
import numpy as np

NORTHERN_DEMAND_RAW = np.array([
    84856.49, 83541.50, 81830.62, 80669.45, 79440.29, 77913.59,
    74899.10, 71450.09, 71360.78, 74196.04, 78876.06, 80805.87,
    82617.41, 84316.17, 86301.69, 86177.85, 82414.23, 82055.60,
    80635.75, 77255.95, 82678.55, 84544.05, 86087.62, 85662.14])
SOLAR_RAW = np.array([
    0.000, 0.000, 0.000, 0.000, 0.000, 0.019, 0.119, 0.267,
    0.404, 0.513, 0.582, 0.630, 0.621, 0.575, 0.480, 0.360,
    0.216, 0.078, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000])
WIND_RAW = np.array([
    0.047, 0.024, 0.022, 0.020, 0.010, 0.004, 0.001, 0.001,
    0.013, 0.034, 0.076, 0.144, 0.228, 0.323, 0.419, 0.498,
    0.544, 0.519, 0.471, 0.494, 0.497, 0.478, 0.460, 0.437])

D_MAX_SYSTEM_MW = 3.75
BASE_RATE = 4.50        # Rs/kWh, EV charging LT category
SURCHARGE_PCT = 0.15    # 8% + 7%
PPAC_PCT = 0.17


@dataclass(frozen=True)
class Profiles:
    load_profile: np.ndarray   # MW (24,)
    solar_profile: np.ndarray  # pu of nameplate
    wind_profile: np.ndarray   # pu of nameplate
    price_TOU: np.ndarray      # Rs/kWh
    price_V2G: np.ndarray      # Rs/kWh


def build_profiles() -> Profiles:
    load_profile = NORTHERN_DEMAND_RAW / NORTHERN_DEMAND_RAW.max() * D_MAX_SYSTEM_MW
    rate = BASE_RATE * (1 + SURCHARGE_PCT)
    price = np.zeros(24)
    for i in range(24):
        t = i + 1                                   # MATLAB hour 1..24
        is_peak = (14 <= t < 17) or (t >= 22 or t < 1)
        is_off = 4 <= t < 10
        r = rate * 1.20 if is_peak else rate * 0.80 if is_off else rate
        price[i] = r * (1 + PPAC_PCT)               # PPAC applied last
    return Profiles(load_profile, SOLAR_RAW.copy(), WIND_RAW.copy(), price, price * 1.10)
