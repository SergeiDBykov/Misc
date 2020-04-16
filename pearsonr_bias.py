#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:00:34 2020

@author: s.bykov
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import  stats
from scipy.stats import pearsonr

#%% generate data

x=np.linspace(0,1,16)

yerr=0.1+x/5

plt.plot(x,10+x,'r-',label='ground truth')

y=10+x+np.random.normal(x-x,yerr)
plt.errorbar(x,y,yerr)

#%%generate new data

#test eqw trials
N=1000
y_trials=np.zeros(shape=(N,len(x)))
for i in range(N):
    #y_trial=y+np.random.normal(loc=0,scale=yerr)
    y_trial=10+x+np.random.normal(loc=0,scale=yerr)
    y_trials[i]=y_trial
    if i<10:
        1
        #plt.plot(x,y_trial,'gray',alpha=0.7)
plt.errorbar(x,y,yerr,color='r',zorder=10,capsize=3,label='data')
plt.errorbar(x,y_trials.mean(axis=0),y_trials.std(axis=0),color='c',zorder=15,capsize=3,label='bootstrap mean and error')

plt.legend()

#%% trial pearson coefficients
pearsonr_arr=np.zeros(N)

for i in range(N):
    pearsonr_arr[i]=pearsonr(y_trials[i], x)[0]


plt.figure()
plt.hist(pearsonr_arr,bins=25,label='pearson r for trials')
plt.axvline(pearsonr(x,y)[0],color='r',label='real pearson r')
plt.legend()

print(f'mean of trial pearson: {pearsonr_arr.mean()}')
print(f'median of trial pearson: {np.median(pearsonr_arr)}')

print(f'real pearson corr coef (x,y): {pearsonr(x,y)[0]}')