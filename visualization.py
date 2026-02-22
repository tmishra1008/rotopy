"""
Visualization utilities for phase folding and period analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_phase_folded(light_curves, period, nbins=50, ax=None, title=None):
    """
    Create a phase-folded plot of light curves at a given period.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects with .time.value and .flux.value
    period : float
        Period to fold at (in days)
    nbins : int
        Number of phase bins
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure
    title : str, optional
        Custom title. If None, uses default

    Returns:
    --------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Get time range for coloring
    all_times = []
    for lc in light_curves:
        all_times.append(np.median(lc.time.value))
    all_times = np.array(all_times)
    time_min, time_max = all_times.min(), all_times.max()

    cmap = cm.get_cmap('viridis')

    # Plot each quarter
    for i, lc in enumerate(light_curves):
        time = lc.time.value
        flux = lc.flux.value

        mask = np.isfinite(time) & np.isfinite(flux)
        time = time[mask]
        flux = flux[mask]

        if len(flux) < 10:
            continue

        # Normalize to ppt
        flux = (flux / np.nanmedian(flux) - 1) * 1e3

        # Phase fold
        phase = (time % period) / period

        # Sort by phase
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        flux = flux[sort_idx]

        # Bin the data
        phase_bins = np.linspace(0, 1, nbins)
        binned_flux = []
        binned_phases = []

        for j in range(len(phase_bins)-1):
            mask_bin = (phase >= phase_bins[j]) & (phase < phase_bins[j+1])
            if np.sum(mask_bin) > 0:
                binned_flux.append(np.median(flux[mask_bin]))
                binned_phases.append((phase_bins[j] + phase_bins[j+1])/2)

        if len(binned_phases) == 0:
            continue

        # Color by time (blue = early, yellow = late)
        median_time = np.median(time)
        color_val = (median_time - time_min) / (time_max - time_min) if time_max > time_min else 0.5
        color = cmap(color_val)

        ax.plot(binned_phases, binned_flux, '-', color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Relative Flux (ppt)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Phase-Folded Light Curve (P = {period:.4f} d)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 100)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    return ax


def plot_phase_folded_comparison(light_curves, periods, titles, nbins=50):
    """
    Compare phase-folded plots for multiple periods side by side.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects
    periods : list
        List of periods to compare
    titles : list
        List of titles for each subplot
    nbins : int
        Number of phase bins
    """
    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(6*n_periods, 6))

    if n_periods == 1:
        axes = [axes]

    for ax, period, title in zip(axes, periods, titles):
        plot_phase_folded(light_curves, period, nbins=nbins, ax=ax,
                         title=f'{title}\nP = {period:.4f} d')
        ax.set_ylim(-60, 60)

    plt.tight_layout()
    plt.show()


def plot_period_distribution(periods, literature_period=None, method_name=""):
    """
    Plot the distribution of detected periods across quarters.

    Parameters:
    -----------
    periods : list
        List of periods from different quarters
    literature_period : float, optional
        Known literature value for comparison
    method_name : str
        Name of the method used
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    quarters = np.arange(1, len(periods) + 1)
    mean_period = np.mean(periods)
    std_period = np.std(periods)

    # Quarter-by-quarter periods
    ax1.plot(quarters, periods, 'o-', markersize=8, linewidth=2, label=f'{method_name} periods')
    if literature_period:
        ax1.axhline(literature_period, color='red', linestyle='--', linewidth=2,
                   label=f'Literature ({literature_period:.4f} d)')
    ax1.axhline(mean_period, color='blue', linestyle=':', linewidth=2,
               label=f'Mean: {mean_period:.4f} d')
    ax1.fill_between(quarters,
                      mean_period - std_period,
                      mean_period + std_period,
                      alpha=0.2, color='blue')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Period [days]')
    ax1.set_title(f'{method_name} Period Detection by Quarter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(quarters)

    # Histogram
    ax2.hist(periods, bins=min(15, len(periods)), alpha=0.7, edgecolor='black')
    if literature_period:
        ax2.axvline(literature_period, color='red', linestyle='--', linewidth=2,
                   label='Literature')
    ax2.axvline(mean_period, color='blue', linestyle=':', linewidth=2,
               label=f'Mean: {mean_period:.4f} d')
    ax2.set_xlabel('Period [days]')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of {method_name} Periods')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"Period statistics ({method_name}):")
    print(f"  Mean: {mean_period:.4f} days")
    print(f"  Std:  {std_period:.4f} days")
    print(f"  Min:  {np.min(periods):.4f} days")
    print(f"  Max:  {np.max(periods):.4f} days")
    if literature_period:
        print(f"  Difference from literature: {abs(mean_period - literature_period):.4f} days")
