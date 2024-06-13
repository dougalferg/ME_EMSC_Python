import numpy as np

def Mie_hulst_complex_scaled(RefIndexComplex, wn, gammaCoef, alphaCoef):
    """
    Calculates approximate extinction of electromagnetic radiation by a sphere.
    
    Input: 
    RefIndexComplex  - Imaginary and real fluctuating part of refractive index (row vector)
    wn               - Wavenumbers corresponding to RefIndexComplex (column vector)
    gammaCoef        - Physical parameter gamma (float)
    alphaCoef        - Physical parameter alpha0 (float)
    
    Output: 
    Q    - Mie extinction curve (row vector)
    
    # Example usage
    RefIndexComplex = np.array([1.5 + 0.1j, 1.6 + 0.1j, 1.7 + 0.1j])  # example data
    wn = np.array([400, 500, 600])  # example wavenumbers
    gammaCoef = 0.5
    alphaCoef = 0.1

    Q = Mie_hulst_complex_scaled(RefIndexComplex, wn, gammaCoef, alphaCoef)
    print(Q)

    """
    
    nv = np.real(RefIndexComplex)
    nvp = np.imag(RefIndexComplex)

    rhov = (alphaCoef * (1.0 + gammaCoef * nv)) *wn* 100  # 100 is correcting for the unit cm^-1

    divider = (1.0 / gammaCoef) + nv
    tanbeta = nvp / divider
    beta0 = np.arctan2(nvp, divider)

    cosB = np.cos(beta0)

    Q = 2.0 + (4 / rhov) * cosB * (
        -np.exp(-rhov * tanbeta) * (np.sin(rhov - beta0)
        + (1.0 / rhov) * cosB * np.cos(rhov - 2 * beta0))
        + (1.0 / rhov) * cosB * np.cos(2 * beta0)
    )
    
    return Q

