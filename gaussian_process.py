"""
Gaussian Process method for period detection using quasi-periodic kernel.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
from tinygp import kernels, GaussianProcess
from scipy.optimize import minimize


@jit
def _neg_log_likelihood(params_array, time, flux, flux_err):
    """
    Negative log likelihood for optimization.

    params_array: [log_period, log_length, log_gamma]
    """
    period = jnp.exp(params_array[0])
    length = jnp.exp(params_array[1])
    gamma = jnp.exp(params_array[2])

    kernel_periodic = kernels.ExpSineSquared(scale=period, gamma=gamma)
    kernel_smooth = kernels.Matern32(scale=length)
    kernel = kernel_periodic + kernel_smooth

    gp = GaussianProcess(kernel, time, diag=flux_err)
    return -gp.log_probability(flux)


_grad_nll = jax.grad(_neg_log_likelihood, argnums=0)


def find_period_gp(lc, initial_period=3.5, verbose=True, plot=False):
    """
    Find rotation period using Gaussian Process with quasi-periodic kernel.

    Parameters:
    -----------
    lc : lightkurve object
        Light curve with .time.value, .flux.value, and .flux_err.value
    initial_period : float
        Initial guess for period in days
    verbose : bool
        Whether to print progress
    plot : bool
        Whether to plot the optimization trajectory

    Returns:
    --------
    dict with keys:
        'period': estimated period in days
        'params': optimized parameters [log_period, log_length, log_gamma]
        'success': whether optimization converged
        'trajectory': dict with 'periods' and 'losses' lists
    """
    # Extract and clean data
    time = lc.time.value
    flux = lc.flux.value

    # Normalize to ppt
    flux = (flux / np.nanmedian(flux) - 1) * 1e3
    flux_err = lc.flux_err.value

    # Convert to JAX
    time_jax = jnp.array(time)
    flux_jax = jnp.array(flux)
    flux_err_jax = jnp.array(flux_err)

    # Initial parameters [log_period, log_length, log_gamma]
    initial_params = np.array([
        np.log(initial_period),
        np.log(50.0),
        np.log(0.5),
    ])

    # Storage for trajectory
    trajectory_periods = []
    trajectory_losses = []

    def loss_function(params):
        loss = float(_neg_log_likelihood(params, time_jax, flux_jax, flux_err_jax))
        period = np.exp(params[0])
        trajectory_periods.append(period)
        trajectory_losses.append(loss)
        return loss

    if verbose:
        print("Optimizing GP parameters...")

    result = minimize(
        fun=loss_function,
        x0=initial_params,
        jac=lambda p: np.array(_grad_nll(p, time_jax, flux_jax, flux_err_jax)),
        method='L-BFGS-B'
    )

    period = np.exp(result.x[0])

    if verbose:
        print(f"Success: {result.success}")
        print(f"Period: {period:.4f} days")

    if plot and trajectory_periods:
        plt.figure(figsize=(10, 4))
        log_probs = -np.array(trajectory_losses)
        probs = np.exp(log_probs - np.max(log_probs))
        plt.plot(trajectory_periods, probs, 'o-', color='orange', linewidth=1)
        plt.axvline(period, color='red', linestyle='--', label=f'Period: {period:.3f}d')
        plt.xlabel('Period (days)')
        plt.ylabel('Relative Probability')
        plt.title('GP Period Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        'period': period,
        'params': result.x,
        'success': result.success,
        'trajectory': {
            'periods': trajectory_periods,
            'losses': trajectory_losses
        }
    }


def process_light_curves_gp(light_curves, initial_period=3.5, verbose=True):
    """
    Process multiple light curves using Gaussian Process.

    Parameters:
    -----------
    light_curves : list
        List of light curve objects
    initial_period : float
        Initial guess for period
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict with keys:
        'periods': list of periods for each light curve
        'mean_period': mean period
        'std_period': standard deviation
        'results': list of full result dicts from find_period_gp
    """
    gp_periods = []
    gp_results = []

    for i, lc in enumerate(light_curves):
        if verbose:
            print(f"\nQuarter {i+1}")

        result = find_period_gp(lc, initial_period=initial_period, verbose=verbose)
        gp_periods.append(result['period'])
        gp_results.append(result)

    mean_period = np.mean(gp_periods)
    std_period = np.std(gp_periods)

    if verbose:
        print(f"\nMean period: {mean_period:.4f} +/- {std_period:.4f} days")

    return {
        'periods': gp_periods,
        'mean_period': mean_period,
        'std_period': std_period,
        'results': gp_results
    }
