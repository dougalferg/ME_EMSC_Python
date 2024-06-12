import sys
import os

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\PyIR\src'
sys.path.append(full_path)

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\ME_EMSC_Python\\'
sys.path.append(full_path)


import ME_EMSC
import numpy as np
import pickle
import adjustWavenumbers

options = {}

# Load data
MatrigelSpectrum = np.load(r'C:\Users\Dougal\Documents\GitHub\ME_EMSC_Python\data\MatrigelSpectrum.npy')

with open(r'C:\Users\Dougal\Documents\GitHub\ME_EMSC_Python\data\measuredSpectra.pkl', 'rb') as f:
    measuredSpectraData = pickle.load(f)

# Set options
options['mode'] = 'Correction'
options['PCnumber'] = 12
options['ExplainedVariance'] = 99.99
options['Weights_InflectionPoints'] = [[3700, 2550], [1900, 1000]]
options['Weights_Kappa'] = [[1, 1], [1, 1]]
options['plotResults'] = True
options['maxIterationNumber'] = 15
options['minRadius'] = 3
options['maxRadius'] = 9.1
options['minRefractiveIndex'] = 1.2
options['maxRefractiveIndex'] = 1.5

# Quality test parameters
RMSE_limit = 1000

# Data converting
referenceSpectrum = MatrigelSpectrum[:, 1]
normalizedReferenceSpectrum = referenceSpectrum / max(referenceSpectrum)
wn_ref = MatrigelSpectrum[:, 0]
measuredSpectra = measuredSpectraData['measured_spec']
wn_raw = np.array(list(map(float,  measuredSpectraData['measured_wavs'])))

# Adjust wavenumbers
normalizedReferenceSpectrum, measuredSpectra, wn = adjustWavenumbers.adjustWavenumbers(normalizedReferenceSpectrum, wn_ref,
                                                                      measuredSpectra, wn_raw)

# Selected spectra for correction
selectedSpectraForCorrection = measuredSpectra

normalizedReferenceSpectrum = normalizedReferenceSpectrum-np.min(normalizedReferenceSpectrum)
selectedSpectraForCorrection = selectedSpectraForCorrection-np.min(selectedSpectraForCorrection)

# Run Mie correction
correctedSpectra, residuals, EMSCparameters, numberOfIterations, options = ME_EMSC.ME_EMSC(normalizedReferenceSpectrum,
                                                                                   selectedSpectraForCorrection,
                                                                                   wn, options)

# Calculate RMSE for all spectra for quality test
RMSE = np.sqrt((1 / selectedSpectraForCorrection.shape[1]) * np.sum(residuals ** 2, axis=1))

# Remove spectra with RMSE > RMSE_limit
ProcessedQT = np.full_like(correctedSpectra, np.nan)
ResidualsQT = np.full_like(residuals, np.nan)
EMSCParametersQT = np.full_like(EMSCparameters, np.nan)
numberOfIterationsQT = np.full_like(numberOfIterations, np.nan)
ProcessedQT[RMSE < RMSE_limit, :] = correctedSpectra[RMSE < RMSE_limit, :]
ResidualsQT[RMSE < RMSE_limit, :] = residuals[RMSE < RMSE_limit, :]
EMSCParametersQT[RMSE < RMSE_limit, :] = EMSCparameters[RMSE < RMSE_limit, :]
numberOfIterationsQT[RMSE < RMSE_limit] = numberOfIterations[RMSE < RMSE_limit]
discardedSpectraNumber = np.where(RMSE > RMSE_limit)[0]

# Plot results
if options['plotResults']:
    referenceSpectrumPlot(normalizedReferenceSpectrum, wn, options)
    correctedSpectrumPlot(ProcessedQT, normalizedReferenceSpectrum, wn)
    RMSEvaluesPlot(RMSE, RMSE_limit, range(1, selectedSpectraForCorrection.shape[0] + 1))
    residualsPlot(ResidualsQT, wn)
