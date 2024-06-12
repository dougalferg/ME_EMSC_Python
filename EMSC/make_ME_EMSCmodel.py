import numpy as np
from sklearn.decomposition import PCA
import make_MieExtinction_curves
import make_basicEMSCmodel

def make_ME_EMSCmodel(referenceSpectrum, pureAbsorbanceSpectrum, wn, alpha0, gamma, options, PCnumber=None):
    """
    Establishes the ME-EMSC model.
    
    Input: 
    referenceSpectrum        - Reference spectrum for ME-EMSC model (row vector) 
    pureAbsorbanceSpectrum   - Input spectrum for calculation of n' (row vector)  
    wn                       - Wavenumbers corresponding to referenceSpectrum and pureAbsorbanceSpectrum (column vector) 
    alpha0                   - Physical parameter alpha0 (range of values, row vector)
    gamma                    - Physical parameter gamma (range of values, row vector)
    options                  - options for the correction, specified in the ME_EMSC function (dict)
    PCnumber                 - Number of principal components in the Mie meta-model (int) 
    
    Output: 
    MieEMSCmodel     - Matrix containing the elements of the ME-EMSC as column vectors 
                       Column number 1: baseline 
                       Column number 2: reference spectrum 
                       Column number 3 - end: loadings from PCA on Mie extinction curves 
    PCnumber         - Number of principal components in the Mie meta-model, if not specified as input (int)
    
    # Example usage
    referenceSpectrum = np.array([1, 2, 3])  # example data
    pureAbsorbanceSpectrum = np.array([0.1, 0.2, 0.3])  # example data
    wn = np.array([400, 500, 600])  # example wavenumbers
    alpha0 = np.array([0.1, 0.2, 0.3])  # example alpha0
    gamma = np.array([0.4, 0.5, 0.6])  # example gamma
    options = {'PCnumber': None, 'ExplainedVariance': 95}  # example options
    PCnumber = None

    MieEMSCmodel, PCnumber = make_ME_EMSCmodel(referenceSpectrum, pureAbsorbanceSpectrum, wn, alpha0, gamma, options, PCnumber)
    print("MieEMSCmodel:", MieEMSCmodel)
    print("PCnumber:", PCnumber)

    """
    
    # Create meta model (100 extinction curves)
    MieExtinctionCurves = make_MieExtinction_curves.make_MieExtinction_curves(pureAbsorbanceSpectrum, wn, alpha0, gamma)

    # Remove the mean-centered reference spectrum from the model spectra
    m = np.dot(referenceSpectrum, referenceSpectrum.T)
    norm = np.sqrt(m)
    rnorm = referenceSpectrum / norm
    s = np.dot(MieExtinctionCurves, rnorm.T)
    MieExtinctionCurves = MieExtinctionCurves - np.dot(s.reshape(-1,1), rnorm.reshape(1,-1))

    # Decompose the set of Mie functions
    pca_model = PCA().fit(MieExtinctionCurves)
    lds = pca_model.components_.T
    latent = pca_model.explained_variance_

    # Construct the Mie model with the new reference spectrum
    EMSCfunctions = make_basicEMSCmodel.make_basicEMSCmodel(referenceSpectrum, wn, 2)

    if PCnumber is None:
        t = np.zeros(len(latent) - 1)
        for i in range(len(latent) - 1):
            s = np.sum(latent[:i + 1])
            t[i] = (s / np.sum(latent)) * 100

        if 'PCnumber' in options:
            PCnumber = options['PCnumber']
        elif 'ExplainedVariance' in options:
            id2 = np.where(t > options['ExplainedVariance'])[0]
            PCnumber = min(id2)
        else:
            raise ValueError("Set options['PCnumber'] or options['ExplainedVariance']")
    
    nRow, nCol = EMSCfunctions.shape
    PCzeros = np.zeros((nRow, PCnumber))
    MieEMSCmodel = np.hstack([EMSCfunctions, PCzeros])
    for i in range(PCnumber):
        MieEMSCmodel[:, nCol + i] = lds[:, i]

    return MieEMSCmodel, PCnumber