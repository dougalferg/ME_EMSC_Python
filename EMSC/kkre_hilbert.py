import numpy as np
from scipy.signal import hilbert

def kkre_hilbert(absorbance):
    """
    Calculates the real part of the refractive index using the Hilbert transform of the absorbance.
    
    Input:
    absorbance - Absorbance data (array)
    
    Output:
    rindex - Real part of the refractive index (array)
    """
    
    rindex = -np.imag(hilbert(absorbance))
    return rindex

