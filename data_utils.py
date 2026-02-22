"""
Data loading and utility classes.
"""

import numpy as np
from pathlib import Path
from lightkurve import KeplerLightCurveFile


class GroundLC:
    """
    Simple light curve container for ground-based simulated data.

    Attributes:
    -----------
    time : object with .value attribute
        Time array
    flux : object with .value attribute
        Flux array
    """

    def __init__(self, time, flux):
        """
        Initialize GroundLC.

        Parameters:
        -----------
        time : array
            Time values
        flux : array
            Flux values
        """
        self.time = type('obj', (object,), {'value': time})()
        self.flux = type('obj', (object,), {'value': flux})()


def load_kepler_light_curves(path_to_files, use_pdcsap=True):
    """
    Load Kepler light curves from a folder of FITS files.

    Parameters:
    -----------
    path_to_files : str or Path
        Path to folder containing FITS files
    use_pdcsap : bool
        Whether to use PDCSAP_FLUX (True) or SAP_FLUX (False)

    Returns:
    --------
    list of light curve objects
    """
    folder = Path(path_to_files)
    fits_files = list(folder.glob("*.fits"))

    if len(fits_files) == 0:
        raise ValueError(f"No FITS files found in {path_to_files}")

    light_curves = []
    for file in sorted(fits_files):
        lcfile = KeplerLightCurveFile(file)
        if use_pdcsap:
            lc = lcfile.PDCSAP_FLUX.remove_nans()
        else:
            lc = lcfile.SAP_FLUX.remove_nans()
        light_curves.append(lc)

    return light_curves


def combine_light_curves(light_curves):
    """
    Combine multiple light curves into single time/flux arrays.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects

    Returns:
    --------
    time, flux : arrays
        Combined and sorted time and flux arrays
    """
    times = np.concatenate([lc.time.value for lc in light_curves])
    fluxes = np.concatenate([lc.flux.value for lc in light_curves])

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]

    return times, fluxes


def normalize_flux(flux, method='median'):
    """
    Normalize flux values.

    Parameters:
    -----------
    flux : array
        Flux values
    method : str
        'median' - divide by median
        'zscore' - subtract mean, divide by std
        'ppt' - convert to parts per thousand relative to median

    Returns:
    --------
    array : normalized flux
    """
    if method == 'median':
        return flux / np.nanmedian(flux)
    elif method == 'zscore':
        return (flux - np.nanmean(flux)) / np.nanstd(flux)
    elif method == 'ppt':
        return (flux / np.nanmedian(flux) - 1) * 1e3
    else:
        raise ValueError(f"Unknown method: {method}")
