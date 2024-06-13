import numpy as np

def basicEMSCsolver(RawSpectra, EMSCModel):
    """
    Solves the basic EMSC correction.
    
    Input: 
    RawSpectra   - Raw spectra (matrix containing one spectrum per row)
    EMSCmodel    - Matrix containing the elements of the EMSC model as column vectors: 
                   Column number 1: constant baseline 
                   Column number 2: linear
                   Column number 3: quadratic    
                   Column number 4: reference spectrum 
    
    Output: 
    Corrected    - Corrected spectra (matrix containing one spectrum per row)
    Residuals    - Residuals after correction (matrix containing one spectrum per row)
    Parameters   - EMSC parameters in the following order: constant baseline, linear, quadratic, reference spectrum (parameter b)
    
    # Example usage
    RawSpectra = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # example data
    EMSCModel = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])  # example model
    
    Corrected, Residuals, Parameters = basicEMSCsolver(RawSpectra, EMSCModel)
    print("Corrected:", Corrected)
    print("Residuals:", Residuals)
    print("Parameters:", Parameters)

    """
    
    Model = EMSCModel
    # Solve for Parameters using least squares
    Parameters = np.linalg.lstsq(Model, RawSpectra.T, rcond=None)[0].T
    
    #take first 3 columns of the parameters
    # Calculate corrected spectra
    Corrected = RawSpectra - np.dot(Parameters[:, 0:3], EMSCModel[:, 0:3].T)
    Corrected = Corrected / Parameters[:, 3].reshape(-1, 1)  # correct for multiplicative effects
    
    # Calculate residuals
    Residuals = RawSpectra - np.dot(Parameters, EMSCModel.T)
    
    return Corrected, Residuals, Parameters

