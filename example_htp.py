# Copyright (c) 2021 Charlie Callahan

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

# import gaas as gs
import gaas_ocl as gs
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import published_validation.htpValidation as htpVal

startWavenum = 7000
endWavenum = 18000
wavenumStep = 0.01 #wavenums per simulation step
molarMass = 1.0
tempK = 300
feat_wavenums = np.linspace(startWavenum,endWavenum,3000) #generate 3000 evenly spaced transitions between 7000 and 18000
feat_data = []
for lc in feat_wavenums[:-1]:
    feat_data.append(gs.HTPFeatureData(lc,0.5,0.1,0.1,0.1,0.1,0.1,1.0))
t0 = time.time()
wvn, spec = gs.simHTP(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
gaasTime = time.time()-t0
print("gaas time: ",gaasTime)

t0=time.time()
wvnh, spech = htpVal.hapiSimHTP(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
hapiTime = time.time()-t0
print("hapi time: ",hapiTime)
print("GAAS is ",hapiTime/gaasTime, " times faster")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

idx0 = 0
idx1 = len(spec)-1
# --- TOP SUBPLOT: GAAS & HAPI ---
ax1.plot(wvn[idx0:idx1], spec[idx0:idx1], label="GAAS")
ax1.plot(wvnh[idx0:idx1], spech[idx0:idx1], label="HAPI")
ax1.legend()
ax1.set_title("Synthetic HTP Spectrum Comparison")
ax1.set_ylabel("Absorbance")

# --- BOTTOM SUBPLOT: Residual ---
ax2.plot(wvn[idx0:idx1], spec[idx0:idx1] - spech[idx0:idx1], label="Residual")
ax2.legend()
ax2.set_xlabel("Wavenumber")
ax2.set_ylabel("Residual")

plt.tight_layout()
plt.show()
