# rotolib

A Python library for stellar rotation period detection from light curve data.

## Overview

rotolib provides multiple methods for detecting rotation periods in stellar light curves, along with utilities for visualization and ground-based observation simulation. It is designed to work with Kepler/K2 light curve data but can be adapted for other time-series photometry.

## Installation

```bash
# Clone or copy the rotolib folder to your project
# Then install dependencies:
pip install numpy matplotlib scipy astropy lightkurve gatspy

# For Gaussian Process method:
pip install jax jaxlib tinygp

# For SACF method:
pip install sacf
# Or from source: pip install git+https://github.com/tmishra1008/sacf.git
```

## Methods

### 1. Lomb-Scargle Periodogram
Classical frequency-domain method for period detection. Supports three variants:
- **general**: Standard Lomb-Scargle
- **fast**: Optimized implementation for large datasets
- **trended**: Accounts for long-term trends in the data

### 2. Gaussian Process (GP)
Uses a quasi-periodic kernel to model stellar variability. Combines:
- ExpSineSquared kernel for periodic signal
- Matern32 kernel for aperiodic variations

### 3. SACF (Selective Autocorrelation Function)
Time-domain autocorrelation method optimized for unevenly-sampled data. Particularly effective for ground-based observations with gaps.

## Quick Start

```python
from rotolib import (
    load_kepler_light_curves,
    process_light_curves_ls,
    process_light_curve_sacf,
    find_period_gp,
    plot_phase_folded
)

# Load Kepler light curves from FITS files
light_curves = load_kepler_light_curves("/path/to/fits/files/")

# Method 1: Lomb-Scargle
ls_result = process_light_curves_ls(light_curves, algorithm="general")
print(f"LS Period: {ls_result['mean_period']:.4f} +/- {ls_result['std_period']:.4f} days")

# Method 2: SACF
sacf_result = process_light_curve_sacf(light_curves)
print(f"SACF Period: {sacf_result['period']:.4f} days")

# Method 3: Gaussian Process (single quarter)
gp_result = find_period_gp(light_curves[0], initial_period=3.5)
print(f"GP Period: {gp_result['period']:.4f} days")

# Visualize with phase folding
plot_phase_folded(light_curves, period=3.47)
```

## API Reference

### Data Loading

```python
from rotolib import load_kepler_light_curves, GroundLC

# Load from FITS files
light_curves = load_kepler_light_curves(
    path_to_files="/path/to/fits/",
    use_pdcsap=True  # Use PDCSAP_FLUX (True) or SAP_FLUX (False)
)

# Create custom light curve object
from rotolib import GroundLC
lc = GroundLC(time_array, flux_array)
```

### Lomb-Scargle

```python
from rotolib import process_light_curves_ls

result = process_light_curves_ls(
    light_curves,
    algorithm="general",      # "general", "fast", or "trended"
    period_range=(0.1, 100),  # Min/max period in days
    n_periods=10000,          # Grid resolution
    plot=True                 # Show periodogram
)

# Returns:
# result['periods']      - List of best periods per quarter
# result['mean_period']  - Mean period
# result['std_period']   - Standard deviation
```

### SACF

```python
from rotolib import process_light_curve_sacf

result = process_light_curve_sacf(
    light_curves,
    max_lag=50,            # Maximum lag in days
    lag_resolution=0.1,    # Lag grid resolution
    min_period=1.0,        # Minimum period to consider
    height_threshold=0.1,  # Peak detection threshold
    plot=True              # Show ACF plot
)

# Returns:
# result['period']       - Best period estimate
# result['correlation']  - Correlation strength at peak
```

### Gaussian Process

```python
from rotolib import find_period_gp, process_light_curves_gp

# Single light curve
result = find_period_gp(
    lc,
    initial_period=3.5,  # Initial guess
    verbose=True,
    plot=False
)

# Multiple light curves
result = process_light_curves_gp(light_curves, initial_period=3.5)

# Returns:
# result['period']   - Best period
# result['success']  - Optimization convergence
# result['params']   - Optimized kernel parameters
```

### Visualization

```python
from rotolib import plot_phase_folded, plot_phase_folded_comparison, plot_period_distribution

# Single phase-folded plot
plot_phase_folded(light_curves, period=3.47, nbins=50)

# Compare multiple periods
plot_phase_folded_comparison(
    light_curves,
    periods=[3.47, 3.53],
    titles=['Literature', 'Lomb-Scargle'],
    nbins=50
)

# Period distribution across quarters
plot_period_distribution(
    periods=[3.44, 3.45, 3.47, ...],
    literature_period=3.4693,
    method_name="Lomb-Scargle"
)
```

### Ground-Based Simulation

```python
from rotolib import simulate_ground_observation, simulate_ground_light_curves

# Single light curve
time_ground, flux_ground = simulate_ground_observation(
    lc,
    night_hours=8,  # Observable hours per night
    bad_weather_periods=[  # Weather gaps (start, end) in days
        (18.5, 22.5),
        (34.5, 37.5),
        (48.5, 52.5),
    ]
)

# Multiple light curves
ground_lcs = simulate_ground_light_curves(light_curves, night_hours=8)

# Use with period detection
sacf_result = process_light_curve_sacf(ground_lcs)
```

## Example Workflow

```python
import numpy as np
from rotolib import (
    load_kepler_light_curves,
    process_light_curves_ls,
    process_light_curve_sacf,
    plot_phase_folded_comparison,
    simulate_ground_light_curves
)

# Load data
path = "/path/to/kepler/fits/"
light_curves = load_kepler_light_curves(path)

# Compare methods on space-based data
ls_result = process_light_curves_ls(light_curves, algorithm="trended", plot=False)
sacf_result = process_light_curve_sacf(light_curves, plot=False)

print(f"Lomb-Scargle: {ls_result['mean_period']:.4f} days")
print(f"SACF: {sacf_result['period']:.4f} days")

# Compare phase-folded plots
plot_phase_folded_comparison(
    light_curves,
    periods=[ls_result['mean_period'], sacf_result['period']],
    titles=['Lomb-Scargle', 'SACF']
)

# Test on simulated ground-based data
ground_lcs = simulate_ground_light_curves(light_curves, night_hours=8)
ground_result = process_light_curve_sacf(ground_lcs)
print(f"Ground-based SACF: {ground_result['period']:.4f} days")
```

## Dependencies

**Required:**
- numpy
- matplotlib
- scipy
- astropy
- lightkurve
- gatspy

**Optional (for specific methods):**
- jax, jaxlib, tinygp (Gaussian Process)
- sacf (SACF method)

## License

MIT

## References

- SACF method: Briegal et al. - "A selective estimator of the autocorrelation function"
- Lomb-Scargle: VanderPlas (2018) - "Understanding the Lomb-Scargle Periodogram"
- Gaussian Process: Angus et al. - "Inferring stellar rotation periods with Gaussian processes"
