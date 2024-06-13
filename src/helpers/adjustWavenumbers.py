import numpy as np
from scipy.interpolate import interp1d

def adjustWavenumbers(RefSpec, wnRefSpec, RawSpec, wnRawSpec):
    """
    Adjust the wavenumbers of a reference spectrum to be compatible with the measured data set.

    Input:
    RefSpec      - Reference spectrum (1D array)
    wnRefSpec    - Wavenumbers corresponding to the reference spectrum (1D array)
    RawSpec      - Raw spectra (2D array, each row corresponds to one spectrum)
    wnRawSpec    - Wavenumbers corresponding to the raw spectra (1D array)

    Output:
    RefSpecFitted    - Reference spectrum adjusted to the wavenumbers in wn (1D array)
    RawSpecFitted    - Raw spectra adjusted to the wavenumbers in wn (2D array)
    wn               - Wavenumbers in the range where wnRefSpec and wnRawSpec overlap, with the same spacing as wnRawSpec (1D array)
    
    # Example data
    RefSpec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    wnRefSpec = np.array([400, 500, 600, 700, 800])
    RawSpec = np.random.rand(5, 5)
    wnRawSpec = np.array([350, 450, 550, 650, 750])

    RefSpecFitted, RawSpecFitted, wn = adjustWavenumbers(RefSpec, wnRefSpec, RawSpec, wnRawSpec)

    print("RefSpecFitted:", RefSpecFitted)
    print("RawSpecFitted:", RawSpecFitted)
    print("Wavenumbers:", wn)

    """
    minWavenumber = max(min(wnRefSpec), min(wnRawSpec))
    maxWavenumber = min(max(wnRefSpec), max(wnRawSpec))

    i1 = np.argmin(np.abs(wnRefSpec - minWavenumber))
    i2 = np.argmin(np.abs(wnRefSpec - maxWavenumber))
    RefSpec = RefSpec[0,i1:i2 + 1].reshape(1,-1)
    wnRefSpec = wnRefSpec[i1:i2 + 1].reshape(1,-1)

    j1 = np.argmin(np.abs(wnRawSpec - minWavenumber))
    j2 = np.argmin(np.abs(wnRawSpec - maxWavenumber))
    RawSpec = RawSpec[:, j1:j2 + 1]
    wnRawSpec = wnRawSpec[j1:j2 + 1].reshape(1,-1)

    # Interpolate the reference spectrum to match the raw spectrum wavenumbers
    RefSpecFitted = interp1d(wnRefSpec.flatten(), RefSpec.flatten(), kind='linear', fill_value="extrapolate")(wnRawSpec.flatten())

    RawSpecFitted = RawSpec
    wn = wnRawSpec

    # Handle any NaN values at the boundaries
    if np.isnan(RefSpecFitted).any():
        if np.isnan(RefSpecFitted[0]):
            RefSpecFitted[0] = RefSpecFitted[1]
        if np.isnan(RefSpecFitted[-1]):
            RefSpecFitted[-1] = RefSpecFitted[-2]

    return RefSpecFitted.reshape(1,-1), RawSpecFitted, wn.flatten()
