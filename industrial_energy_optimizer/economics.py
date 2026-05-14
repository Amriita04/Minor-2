"""Convert electrical quantities to INR using configurable tariffs."""

from __future__ import annotations

import numpy as np


def plant_energy_cost_inr(
    pe_mw: np.ndarray,
    hours: float,
    tariff_inr_per_kwh: float,
) -> np.ndarray:
    """Hourly plant energy in MW → kWh → ₹."""
    kwh = pe_mw * MW_TO_KWH_PER_HOUR * hours
    return kwh * tariff_inr_per_kwh


def machine_idle_wastage_inr(
    power_kw: np.ndarray,
    is_peak: np.ndarray,
    is_idle: np.ndarray,
    hours_each: float,
    peak_tariff: float,
    offpeak_tariff: float,
    waste_fraction_when_idle: float = 0.35,
) -> np.ndarray:
    """
    Demo cost model: a fraction of observed mechanical proxy power is treated as
    wasted kWh when `is_idle`, priced at peak vs off-peak tariff.
    """
    tariff = np.where(is_peak, peak_tariff, offpeak_tariff)
    wasted_kwh = np.clip(power_kw * waste_fraction_when_idle * hours_each, 0, None)
    return wasted_kwh * tariff * is_idle.astype(float)
