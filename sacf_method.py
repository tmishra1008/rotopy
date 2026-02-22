"""
SACF (Selective Autocorrelation Function) method for period detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sacf import SACF
from scipy.signal import find_peaks


def process_light_curve_sacf(light_curves, max_lag=50, lag_resolution=0.1,
                              min_period=1.0, height_threshold=0.1, plot=True):
    """
    Process light curves using SACF to estimate the rotation period.

    Parameters:
    -----------
    light_curves : list or single lightkurve object
        Light curve(s) to process. Can be a single light curve or list of light curves.
    max_lag : float
        Maximum lag to search for period (in days). Default: 50
    lag_resolution : float
        Resolution of the lag grid (in days). Default: 0.1
    min_period : float
        Minimum period to consider (in days). Default: 1.0
    height_threshold : float
        Minimum correlation height for peak detection. Default: 0.1
    plot : bool
        Whether to display the ACF plot. Default: True

    Returns:
    --------
    dict with keys:
        'period': estimated period in days
        'correlation': correlation strength at the period
    """
    # Handle single light curve or list
    if not isinstance(light_curves, list):
        light_curves = [light_curves]

    # Combine all light curves into single arrays
    times = np.concatenate([lc.time.value for lc in light_curves])
    fluxes = np.concatenate([lc.flux.value for lc in light_curves])

    # Check if flux_err exists
    has_errors = hasattr(light_curves[0], 'flux_err') and light_curves[0].flux_err is not None
    if has_errors:
        errors = np.concatenate([lc.flux_err.value for lc in light_curves])

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]
    if has_errors:
        errors = errors[sort_idx]

    # Normalize flux
    flux_normalized = (fluxes - np.mean(fluxes)) / np.std(fluxes)

    # Convert to contiguous float64 arrays (required by SACF)
    times = np.ascontiguousarray(times, dtype=np.float64)
    flux_normalized = np.ascontiguousarray(flux_normalized, dtype=np.float64)

    # Initialize SACF
    if has_errors:
        errors_normalized = np.ascontiguousarray(errors / np.std(fluxes), dtype=np.float64)
        sacf = SACF(timeseries=times, values=flux_normalized, errors=errors_normalized)
    else:
        sacf = SACF(timeseries=times, values=flux_normalized)

    # Compute autocorrelation
    lags, correlations = sacf.autocorrelation(
        min_lag=0.5,
        max_lag=max_lag,
        lag_resolution=lag_resolution
    )

    # Find peaks
    peak_indices, _ = find_peaks(correlations, height=height_threshold,
                                  distance=int(1.0/lag_resolution))

    # Filter peaks above minimum period and sort by correlation
    valid_peaks = [(lags[i], correlations[i]) for i in peak_indices if lags[i] >= min_period]
    valid_peaks_sorted = sorted(valid_peaks, key=lambda x: x[1], reverse=True)

    # Get best period
    if valid_peaks_sorted:
        estimated_period = valid_peaks_sorted[0][0]
        peak_correlation = valid_peaks_sorted[0][1]
    else:
        estimated_period = None
        peak_correlation = None

    # Plot if requested
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(lags, correlations, 'b-', linewidth=0.8)
        if estimated_period is not None:
            plt.axvline(x=estimated_period, color='r', linestyle='--',
                       label=f'Period = {estimated_period:.2f} days')
            plt.legend()
        plt.xlabel('Lag (days)')
        plt.ylabel('Autocorrelation')
        plt.title('SACF - Selective Autocorrelation Function')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print results
        if estimated_period is not None:
            print(f"Estimated Period: {estimated_period:.3f} days")
            print(f"Peak correlation: {peak_correlation:.3f}")
        else:
            print("No significant peaks found.")

    return {
        'period': estimated_period,
        'correlation': peak_correlation
    }
