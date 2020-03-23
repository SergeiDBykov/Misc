# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

import stingray
import os


os.chdir('/Users/s.bykov/work/xray_pulsars/rxte/results/out90089-11-04-03/products/se_0.1sec_lc')

lc1=fits.open('ch1314.lc')
lc2=fits.open('ch15.lc')
#lc2=fits.open('ch1819.lc')

from stingray import Lightcurve
lc=Lightcurve(lc1[1].data['time'], lc1[1].data['rate'],
              err=lc1[1].data['error'],input_counts=0)
lc1=lc.rebin(0.15)

from stingray.events import EventList
evt1=EventList.from_lc(lc1)
#evt1.simulate_energies([[5.5],[1]])

lc=Lightcurve(lc2[1].data['time'], lc2[1].data['rate'],
              err=lc2[1].data['error'],input_counts=0)
lc2=lc.rebin(0.15)

evt2=EventList.from_lc(lc2)
#evt2.simulate_energies([6.5,1])
#evt=evt1.join(evt2)


#%% cross
from stingray import Crossspectrum,AveragedCrossspectrum

avg_cs=AveragedCrossspectrum(lc1,lc2,5)

freq_lags, freq_lags_err = avg_cs.time_lag()


fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.hlines(0, avg_cs.freq[0], avg_cs.freq[-1], color='black', linestyle='dashed', lw=2)
ax.errorbar(avg_cs.freq, freq_lags, yerr=freq_lags_err,fmt="o", lw=1, color='blue')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Time lag (s)")
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(which='major', width=1.5, length=7)
ax.tick_params(which='minor', width=1.5, length=4)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)
plt.show()


#%% lag en spe
stop
from stingray import  LagEnergySpectrum as les

lag_en=les(evt,freq_interval=[0,100],energy_spec=(5,6,1,'lin'))

