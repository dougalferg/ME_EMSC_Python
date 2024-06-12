import numpy as np

import Mie_hulst_complex_scaled
import kkre_hilbert

def make_MieExtinction_curves(absorbanceSpectrum, wn, alpha0, gamma):
    """
    Calculate Mie extinction curves from a given imaginary part of the complex refractive index.
    
    Input: 
    absorbanceSpectrum - Absorbance spectrum, input for the imaginary part of the refractive index (row vector)
    wn                 - Wavenumbers corresponding to absorbanceSpectrum (column vector)
    alpha0             - Physical parameter alpha0 (range of values, row vector)
    gamma              - Physical parameter gamma (range of values, row vector)
    
    Output: 
    ExtinctionCurves - Matrix containing the Mie extinction curves (matrix, each row corresponds to one curve)
    
    Example usage
    absorbanceSpectrum = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # example data
    wn = np.array([400, 500, 600, 700, 800])  # example wavenumbers
    alpha0 = np.array([0.1, 0.2, 0.3])  # example alpha0
    gamma = np.array([0.4, 0.5, 0.6])  # example gamma

    ExtinctionCurves = make_MieExtinction_curves(absorbanceSpectrum, wn, alpha0, gamma)
    print("ExtinctionCurves:", ExtinctionCurves)
    
    """
    
    # Imaginary part of the refractive index
    RefIndexABS = absorbanceSpectrum
    
    # Number of points for extending the absorbance spectrum
    xts = 200
    
    # Extend the absorbance spectrum in both directions
    dx = wn[1] - wn[0]
    wn = np.concatenate([(dx * np.linspace(1, xts, xts) + (wn[0] - dx * (xts + 1))), wn, 
                         (dx * np.linspace(1, xts, xts) + wn[-1])])
    RefIndexABS = np.concatenate([np.full(xts, RefIndexABS[0]), RefIndexABS, 
                                  np.full(xts, RefIndexABS[-1])])
    
    # Calculate real fluctuating part
    RefIndexN = kkre_hilbert.kkre_hilbert(RefIndexABS / (wn * 100))
    
    RefIndexN = RefIndexN[xts:-xts]
    RefIndexABS = RefIndexABS[xts:-xts]
    wn = wn[xts:-xts]
    
    if abs(min(RefIndexN)) > 1:
        RefIndexIN = RefIndexN / abs(min(RefIndexN))
        RefIndexABS = RefIndexABS / (wn * 100) / abs(min(RefIndexN))
    else:
        RefIndexIN = RefIndexN
        RefIndexABS = RefIndexABS / (wn * 100)
    
    refIndexComplex = RefIndexIN + 1j * RefIndexABS
    
    ExtinctionCurves = np.zeros((len(gamma) * len(alpha0), len(wn)))
    
    ifunc = 0
    for i in range(len(gamma)):
        for j in range(len(alpha0)):
            ExtinctionCurves[ifunc, :] = Mie_hulst_complex_scaled.Mie_hulst_complex_scaled(refIndexComplex, wn, gamma[i], alpha0[j])
            ifunc = ifunc+1
    
    return ExtinctionCurves


