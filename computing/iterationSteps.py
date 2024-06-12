import numpy as np
import basicEMSCsolver
import make_ME_EMSCmodel
import ME_EMSCsolver

def iterationSteps(spectrumRowForCorrection, spectraNumber, correctedSpectraForIteration,
                   EMSCScaleModel, maxIterationNumber, weights, wn, alpha0, gamma,
                   options, PCnumber):
    """
    Perform iteration steps for ME-EMSC correction.

    Input:
    spectrumRowForCorrection - Spectrum row for correction (1D array)
    spectraNumber            - Spectrum number (int)
    correctedSpectraForIteration - Corrected spectra for iteration (2D array)
    EMSCScaleModel           - EMSC scale model (2D array)
    maxIterationNumber       - Maximum number of iterations (int)
    weights                  - Weights (1D array)
    wn                       - Wavenumbers (1D array)
    alpha0                   - Alpha0 parameter (1D array)
    gamma                    - Gamma parameter (1D array)
    options                  - Options (dict with keys: maxIterationNumber, scaleRef, PositiveRefSpectrum, fixIterationNumber, ExplainedVariance, PCnumber)
    PCnumber                 - Number of principal components (int)

    Output:
    correctedSpectraForIteration - Corrected spectra after iteration (2D array)
    residualsFromIteration        - Residuals after iteration (2D array)
    parameters                   - EMSC parameters (2D array)
    numberOfIterations           - Number of iterations performed (int)
    
    
    # Example options
    options = {
        'maxIterationNumber': 10,
        'scaleRef': True,
        'PositiveRefSpectrum': True,
        'fixIterationNumber': False,
        'ExplainedVariance': 95.0,
        'PCnumber': None
    }

    # Example data
    spectrumRowForCorrection = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    spectraNumber = 1
    correctedSpectraForIteration = np.random.rand(5, 5)
    EMSCScaleModel = np.random.rand(5, 5)
    maxIterationNumber = 10
    weights = np.array([1, 1, 1, 1, 1])
    wn = np.array([400, 500, 600, 700, 800])
    alpha0 = np.array([0.1, 0.2])
    gamma = np.array([0.1, 0.2])
    PCnumber = 3

    correctedSpectraForIteration, residualsFromIteration, parameters, numberOfIterations = iterationSteps(
        spectrumRowForCorrection, spectraNumber, correctedSpectraForIteration,
        EMSCScaleModel, maxIterationNumber, weights, wn, alpha0, gamma,
        options, PCnumber)

    print("Corrected Spectra:", correctedSpectraForIteration)
    print("Residuals:", residualsFromIteration)
    print("Parameters:", parameters)
    print("Number of Iterations:", numberOfIterations)

    
    """
    
    RMSE = np.zeros(options['maxIterationNumber'])
    
    for iterationNumber in range(2, maxIterationNumber + 1):
        # Scale the reference spectrum in each iteration
        if options['scaleRef']:
            correctedSpectraForIteration, _, _ = basicEMSCsolver(correctedSpectraForIteration, EMSCScaleModel)
        
        # Apply weights to the corrected spectra
        correctedSpectraForIteration *= weights
        
        # Reference spectrum for EMSC with weights applied
        referenceSpectrum = correctedSpectraForIteration.copy()
        
        # Set negative parts to zero
        correctedSpectraForIteration[correctedSpectraForIteration < 0] = 0
        
        # Handle positive reference spectrum option
        if options['PositiveRefSpectrum']:
            referenceSpectrum = correctedSpectraForIteration.copy()
        
        # Create ME-EMSC model
        MieEMSCmodel, PCnumber = make_ME_EMSCmodel(referenceSpectrum, correctedSpectraForIteration, wn, alpha0, gamma, options, PCnumber)
        
        # Calculate corrected spectrum and parameters from EMSC
        correctedSpectraForIteration, residualsFromIteration, parameters = ME_EMSCsolver(spectrumRowForCorrection, MieEMSCmodel)
        
        # Calculate root mean square error
        if not options['fixIterationNumber']:
            RMSE[iterationNumber - 1] = np.round(np.sqrt(np.mean(residualsFromIteration ** 2)), 4)
        
        # Stop criterion
        if iterationNumber == options['maxIterationNumber']:
            print(f"Spectrum no. {spectraNumber}: Number of iterations (maxIterationNumber) should be bigger.")
            numberOfIterations = options['maxIterationNumber']
        elif options['fixIterationNumber'] and iterationNumber < options['fixIterationNumber']:
            continue
        elif iterationNumber == options['fixIterationNumber']:
            numberOfIterations = iterationNumber
            break
        elif iterationNumber > 2 and not options['fixIterationNumber']:
            if (RMSE[iterationNumber - 1] == RMSE[iterationNumber - 2] == RMSE[iterationNumber - 3]) or (RMSE[iterationNumber - 1] > RMSE[iterationNumber - 2]):
                numberOfIterations = iterationNumber
                break
    return correctedSpectraForIteration, residualsFromIteration, parameters, numberOfIterations


