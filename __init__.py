"""
rotolib - A library for stellar rotation period detection

Methods:
- Lomb-Scargle periodogram (general, fast, trended)
- Gaussian Process with quasi-periodic kernel
- SACF (Selective Autocorrelation Function)

Utilities:
- Phase folding visualization
- Ground-based observation simulation
"""

from .lomb_scargle import process_light_curves_ls
from .gaussian_process import find_period_gp
from .sacf_method import process_light_curve_sacf
from .visualization import plot_phase_folded, plot_phase_folded_comparison
from .simulation import simulate_ground_observation
from .data_utils import GroundLC, load_kepler_light_curves

__all__ = [
    'process_light_curves_ls',
    'find_period_gp',
    'process_light_curve_sacf',
    'plot_phase_folded',
    'plot_phase_folded_comparison',
    'simulate_ground_observation',
    'GroundLC',
    'load_kepler_light_curves',
]

__version__ = '0.1.0'
