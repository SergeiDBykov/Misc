# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import  stats
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import pearsonr
from Misc.TimeSeries.cross_corr import *
import seaborn as sns
sns.set(style='ticks', palette='deep',context='notebook')


#%% init_data

rxte_67_rate=290.21
rxte_67_std=14.780

nustar_67_rate=62.443
nustar_67_std=6.2950


#%% generate some lc


mean,std=rxte_67_rate,rxte_67_std

T=500
binsize=1
P=30

ampl=std*5

N=T/binsize

t=np.linspace(0, T, int(N))


rate1=np.sin(2*np.pi*t/P)*ampl+mean
err=np.random.normal(t-t,std)
rate1+=err

plt.errorbar(t,rate1,std,color='g',alpha=0.7)




rate2=np.sin(2*np.pi*t/P)*ampl+mean+70
err=np.random.normal(t-t,std+5)
rate2+=err

#plt.errorbar(t,rate2,std+5,color='c',alpha=0.7)


#%% fe line with shift will be added to the second curve
fe_fraction=1


rate_fe=(np.sin(2*np.pi*(t-15)/P)*ampl+mean+70)*fe_fraction
#err=np.random.normal(t-t,std+5)
rate2+=rate_fe


plt.errorbar(t,rate2,std+5,color='c',alpha=0.7)


#%%cross corr

fig,axs=plt.subplots(2,1)
fig.subplots_adjust(hspace=0.5)
my_crosscorr(t,rate1,rate2,axs[0],axs[1],subtract_mean=1,divide_by_mean=1,
             y1label='rate1',y2label='rate2',
             only_pos_delays=0,my_only_pos_delays=0,
             my_ccf=0)
plt.show()





