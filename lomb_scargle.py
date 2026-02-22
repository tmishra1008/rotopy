"""
Lomb-Scargle periodogram methods for period detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from gatspy.periodic import LombScargle, LombScargleFast, TrendedLombScargle


def process_light_curves_ls(light_curves, algorithm="general", period_range=(0.1, 100),
                            n_periods=10000, plot=True):
    """
    Process light curves using Lomb-Scargle periodogram to estimate rotation periods.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects with .time.value and .flux.value attributes
    algorithm : str
        Algorithm to use: "general", "fast", or "trended"
    period_range : tuple
        (min_period, max_period) in days
    n_periods : int
        Number of period points in the grid
    plot : bool
        Whether to display the periodogram for the first light curve

    Returns:
    --------
    dict with keys:
        'periods': list of best periods for each light curve
        'mean_period': mean period across all light curves
        'std_period': standard deviation of periods
    """
    best_periods = []

    for i, lc in enumerate(light_curves):
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time = np.asarray(time[mask])
        flux = np.asarray(flux[mask])
        flux = (flux / np.nanmedian(flux)) - 1

        # Instantiate model
        if algorithm == "general":
            model = LombScargle()
        elif algorithm == "fast":
            model = LombScargleFast()
        elif algorithm == "trended":
            model = TrendedLombScargle()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'general', 'fast', or 'trended'")

        model.fit(time, flux)

        # Define period grid
        periods = np.linspace(period_range[0], period_range[1], n_periods)
        power = model.score(periods)

        # Find best period
        best_period = periods[np.argmax(power)]
        best_periods.append(best_period)

        # Plot periodogram for first light curve
        if i == 0 and plot:
            plt.figure(figsize=(10, 4))
            plt.plot(periods, power)
            plt.axvline(best_period, color='r', linestyle='--',
                       label=f'Best period: {best_period:.3f} days')
            plt.xlabel("Period [days]")
            plt.ylabel("Lomb-Scargle Power")
            plt.title(f"Lomb-Scargle Periodogram ({algorithm})")
            plt.xscale("log")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    mean_period = np.mean(best_periods)
    std_period = np.std(best_periods)

    if plot:
        print(f"Period: {mean_period:.4f} +/- {std_period:.4f} days")

    return {
        'periods': best_periods,
        'mean_period': mean_period,
        'std_period': std_period
    }
