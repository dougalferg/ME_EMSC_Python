"""
This script will go over how to use the python version of Johanne Solheim's
ME-EMSC scattering correction algorithm. ORIGINAL WORKS/CREDIT SHOULD BE 
DIRECTED TO - https://github.com/BioSpecNorway/ME-EMSC

The justification for porting this into a new repository (the tools already 
exist in python in BioSpecNorway's biospectools kit - 
https://github.com/BioSpecNorway/biospectools) was because there was no clear
example script or documentation to help me understand how to use their package.

I had to port things across and rebuild it in python so I know how to use it.

The below will load in the example measured spectra, and correct to a matrigel
reference.

"""

import sys
import os
import matplotlib.pyplot as plt

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\PyIR\src'
sys.path.append(full_path)

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\ME_EMSC_Python\src'
sys.path.append(full_path)


import ME_EMSC
import numpy as np
import pickle
import adjustWavenumbers

#initiate an empty options directory to then store your parameters in
options = {}

# Load in the data (matrigel reference and some scattered spectra to correct)
MatrigelSpectrum = np.load(r'C:\Users\Dougal\Documents\GitHub\ME_EMSC_Python\src\data\MatrigelSpectrum.npy')

with open(r'C:\Users\Dougal\Documents\GitHub\ME_EMSC_Python\src\data\measuredSpectra.pkl', 'rb') as f:
    measuredSpectraData = pickle.load(f)

# Set options - not sure what options are optimal.
options['mode'] = 'Correction'
options['PCnumber'] = 15
options['ExplainedVariance'] = 99
options['Weights_InflectionPoints'] = [[3800, 2700], [2000, 900]]
options['Weights_Kappa'] = [[0.5, 0.5], [1, 0.5]]
options['maxIterationNumber'] = 100
options['minRadius'] = 3
options['maxRadius'] = 9
options['minRefractiveIndex'] = 1.4
options['maxRefractiveIndex'] = 1.7

# Set the quality test parameters for removal of certain spectra 
# (calculated as RMSE)
RMSE_limit = 500

# Data converting - THE REFERENCE SPECTRUM MUST BE A 1XN FORMAT i.e (1, 1556)
# Data must also be normalised.
referenceSpectrum = MatrigelSpectrum[:, 1].reshape(1,-1)
normalizedReferenceSpectrum = referenceSpectrum / np.max(referenceSpectrum)
#wn_ref is the wavenumbers of the reference spectrum - we need this to 
#interpolate the data/wavenumbers of the reference to be the same size as our
#measured spectra
wn_ref = MatrigelSpectrum[:, 0].reshape(-1,1)
#measured spectra and wavenumbers for our raw (to be corrected) data
measuredSpectra = measuredSpectraData['measured_spec']
wn_raw = np.array(list(map(float,  measuredSpectraData['measured_wavs']))).reshape(-1,1)

# Adjust wavenumbers - We reshape/interpolate the reference data to be the 
#same size as our raw data
normalizedReferenceSpectrum, measuredSpectra, wn = adjustWavenumbers.adjustWavenumbers(normalizedReferenceSpectrum, wn_ref,
                                                                      measuredSpectra, wn_raw)

# Selected spectra for correction
selectedSpectraForCorrection = measuredSpectra

#I add a min2zero step to make sure there are no negative absorbances
normalizedReferenceSpectrum = normalizedReferenceSpectrum-np.min(normalizedReferenceSpectrum)
selectedSpectraForCorrection = selectedSpectraForCorrection-np.min(selectedSpectraForCorrection)

#Run the Mie correction
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

# Plot the results to compare 
plt.figure()
plt.plot(wn.reshape(-1,1), selectedSpectraForCorrection.T)
plt.title("Spectra to be corrected")
plt.ylabel("Absorbance (arb.)")
plt.xlabel("Wavenumber (cm-1)")

plt.figure()
plt.plot(wn.reshape(-1,1), correctedSpectra.T)
plt.title("Corrected spectra")
plt.ylabel("Absorbance (arb.)")
plt.xlabel("Wavenumber (cm-1)")
