import numpy as np

def make_basicEMSCmodel(referenceSpectra, wn, EMSCoption=1):
    """
    Establishes the basic EMSC model without any extensions in addition to the linear and quadratic term.
    
    Input: 
    referenceSpectra - Reference spectrum (row vector, or matrix containing one spectrum per row)
    wn               - Wavenumbers corresponding to referenceSpectra (column vector)
    EMSCoption       - 1 (Basic EMSC, default) or 2 (MSC)
    
    Output: 
    EMSCmodel    - Matrix containing the elements of the EMSC model as column vectors: 
                   Column number 1: constant baseline 
                   Column number 2: linear
                   Column number 3: quadratic    
                   Column number 4: reference spectrum (mean of all spectra in referenceSpectra)
                   
    Example usage
    referenceSpectra = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                 [0.2, 0.3, 0.4, 0.5, 0.6]])  # example data
    wn = np.array([400, 500, 600, 700, 800])  # example wavenumbers
    EMSCoption = 1  # example option

    EMSCmodel = make_basicEMSCmodel(referenceSpectra, wn, EMSCoption)
    print("EMSCmodel:", EMSCmodel)
                   
    """
    
    # Calculate the basic model functions
    _, Ny = referenceSpectra.shape
    
    Start = wn[0]
    End = wn[-1]
    
    C = 0.5 * (Start + End)
    M0 = 2.0 / (Start - End)
    M = 4.0 / ((Start - End) * (Start - End))
    
    WaveNumT = wn.reshape(-1, 1)
    Baseline = np.ones((1, Ny))
    Mean = np.mean(referenceSpectra, axis=0)
    
    if EMSCoption == 1:
        Linear = M0 * (Start - WaveNumT) - 1
        Quadratic = M * (WaveNumT - C) ** 2
        MModel = np.vstack([Baseline, Linear.T, Quadratic.T, Mean])
    elif EMSCoption == 2:
        MModel = np.vstack([Baseline, Mean])
    
    EMSCmodel = MModel.T
    
    return EMSCmodel
