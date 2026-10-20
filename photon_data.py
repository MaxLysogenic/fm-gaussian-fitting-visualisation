from pathlib import Path

import numpy as np

GRID_SIZE = 6
PIXEL_SIZE_NM = 107.0
PSF_X0, PSF_Y0 = 2.3, 3.8
PSF_SIGMA = 0.8
PHOTON_FILE = Path(__file__).resolve().parent / "simulated_photons.npz"
READ_NOISE_SIGMA = 7.0
READ_NOISE_SEED = 43


def load_camera_data():
    """Load the Phase 2 photon sample and digitize it reproducibly."""
    if not PHOTON_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PHOTON_FILE.name}; run smt_2.py before Phases 3-5."
        )

    with np.load(PHOTON_FILE) as data:
        all_photons_x = data["photons_x"]
        all_photons_y = data["photons_y"]

    bins = np.arange(GRID_SIZE + 1)
    counts, _, _ = np.histogram2d(
        all_photons_x,
        all_photons_y,
        bins=[bins, bins],
    )
    photon_counts = counts.T.astype(int)
    print(f"Total photons generated: {len(all_photons_x)}")
    print(f"Total photons binned in 6x6 grid: {photon_counts.sum()}")
    read_noise = np.random.default_rng(READ_NOISE_SEED).normal(
        loc=0.0,
        scale=READ_NOISE_SIGMA,
        size=(GRID_SIZE, GRID_SIZE),
    )
    digitized_grid = np.clip(photon_counts + read_noise, 0, None)
    return all_photons_x, all_photons_y, photon_counts, digitized_grid


def gaussian_2d(coordinates, amplitude, x0, y0, sigma, offset):
    """Evaluate an isotropic 2D Gaussian with a constant camera offset."""
    x, y = coordinates
    return offset + amplitude * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
    )


def fit_gaussian(digitized_grid):
    """Fit a continuous Gaussian model to the digitized camera pixels."""
    from scipy.optimize import curve_fit

    pixel_centers = np.arange(GRID_SIZE) + 0.5
    x_grid, y_grid = np.meshgrid(pixel_centers, pixel_centers)
    initial_offset = float(np.percentile(digitized_grid, 20))
    initial_amplitude = max(float(np.max(digitized_grid)) - initial_offset, 1.0)
    parameters, _ = curve_fit(
        gaussian_2d,
        (x_grid.ravel(), y_grid.ravel()),
        digitized_grid.ravel(),
        p0=(initial_amplitude, PSF_X0, PSF_Y0, PSF_SIGMA, initial_offset),
        bounds=(
            (0.0, 0.0, 0.0, 0.1, 0.0),
            (np.inf, GRID_SIZE, GRID_SIZE, 3.0, np.inf),
        ),
        maxfev=20000,
    )
    fitted_grid = gaussian_2d((x_grid, y_grid), *parameters)
    return parameters, x_grid, y_grid, fitted_grid
