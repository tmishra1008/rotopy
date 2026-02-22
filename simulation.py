"""
Simulation utilities for ground-based observation effects.
"""

import numpy as np


def simulate_ground_observation(lc, night_hours=8, bad_weather_periods=None):
    """
    Simulate ground-based observations from space-based light curve.

    Parameters:
    -----------
    lc : light curve object
        Single light curve with .time.value and .flux.value
    night_hours : float
        Hours of observation per 24-hour period. Default: 8
    bad_weather_periods : list of tuples, optional
        List of (start, end) tuples defining bad weather periods in days
        relative to start of quarter. Default uses paper values:
        [(18.5, 22.5), (34.5, 37.5), (48.5, 52.5), (62.5, 64.5), (76.5, 81.5)]

    Returns:
    --------
    time_ground, flux_ground : arrays
        Simulated ground-based time and flux arrays
    """
    if bad_weather_periods is None:
        bad_weather_periods = [
            (18.5, 22.5),
            (34.5, 37.5),
            (48.5, 52.5),
            (62.5, 64.5),
            (76.5, 81.5)
        ]

    time = lc.time.value.copy()
    flux = lc.flux.value.copy()

    # Remove NaNs from original
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    # Get start time of this quarter
    time_start = time.min()

    # Convert to relative time (days since start of quarter)
    time_rel = time - time_start

    # Create mask for observable data
    observable_mask = np.ones(len(time), dtype=bool)

    # 1. Apply night-time only
    hours_per_day = 24.0
    night_duration = night_hours / hours_per_day

    for day in range(int(time_rel.max()) + 1):
        # Night window: centered in each day
        night_start = day + (1 - night_duration) / 2
        night_end = night_start + night_duration

        # Mask out daytime (keep only night)
        daytime_mask = ((time_rel >= day) & (time_rel < night_start)) | \
                       ((time_rel >= night_end) & (time_rel < day + 1))
        observable_mask &= ~daytime_mask

    # 2. Apply bad weather periods
    for start, end in bad_weather_periods:
        weather_mask = (time_rel >= start) & (time_rel <= end)
        observable_mask &= ~weather_mask

    # Keep only observable points
    time_ground = time[observable_mask]
    flux_ground = flux[observable_mask]

    return time_ground, flux_ground


def simulate_ground_light_curves(light_curves, night_hours=8, bad_weather_periods=None):
    """
    Simulate ground-based observations for multiple light curves.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects
    night_hours : float
        Hours of observation per night
    bad_weather_periods : list of tuples, optional
        Bad weather periods

    Returns:
    --------
    list of GroundLC objects
    """
    from .data_utils import GroundLC

    ground_light_curves = []

    for lc in light_curves:
        time_ground, flux_ground = simulate_ground_observation(
            lc, night_hours=night_hours, bad_weather_periods=bad_weather_periods
        )
        ground_light_curves.append(GroundLC(time_ground, flux_ground))

    return ground_light_curves
