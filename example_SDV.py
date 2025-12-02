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
    feat_data.append(gs.SDVFeatureData(lc,0.5,0.1,0.1,0.1,0.0,1.0))
t0 = time.time()
wvn, spec = gs.simSDV(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
gaasTime = time.time()-t0
print("gaas time: ",gaasTime)

t0=time.time()
wvnh, spech = htpVal.hapiSimSDV(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
hapiTime = time.time()-t0
print("hapi time: ",hapiTime)
print("GAAS is ",hapiTime/gaasTime, " times faster")

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

ax1, ax2 = axs[0]  # top-left (real), top-right (imag)
ax3, ax4 = axs[1]  # bottom-left (real residual), bottom-right (imag residual)

idx0 = 0
idx1 = 20000

# -------------------------------------
# TOP LEFT: GAAS & HAPI (REAL PART)
# -------------------------------------
ax1.plot(wvn[idx0:idx1], spec[idx0:idx1].real, label="GAAS (real)")
ax1.plot(wvnh[idx0:idx1], spech[idx0:idx1].real, label="HAPI (real)")
ax1.legend()
ax1.set_title("HTP Comparison — REAL part")
ax1.set_ylabel("Absorbance")

# -------------------------------------
# TOP RIGHT: GAAS & HAPI (IMAG PART)
# -------------------------------------
ax2.plot(wvn[idx0:idx1], spec[idx0:idx1].imag, label="GAAS (imag)")
ax2.plot(wvnh[idx0:idx1], spech[idx0:idx1].imag, label="HAPI (imag)")
ax2.legend()
ax2.set_title("HTP Comparison — IMAG part")

# -------------------------------------
# BOTTOM LEFT: Residual (REAL)
# -------------------------------------
ax3.plot(wvn[idx0:idx1], 
         spec[idx0:idx1].real - spech[idx0:idx1].real,
         label="Residual (real)")
ax3.legend()
ax3.set_ylabel("Residual")

# -------------------------------------
# BOTTOM RIGHT: Residual (IMAG)
# -------------------------------------
ax4.plot(wvn[idx0:idx1], 
         spec[idx0:idx1].imag - spech[idx0:idx1].imag,
         label="Residual (imag)")
ax4.legend()
ax4.set_xlabel("Wavenumber")

plt.tight_layout()
plt.show()
