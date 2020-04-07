#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:57:40 2020

@author: s.bykov
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import  stats
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage.interpolation import shift


def cross_correlation(x,y1,y2,
                      circular=1,
                      divide_by_mean=1, subtract_mean=1):
    dx=np.median(np.diff(x))

    if divide_by_mean:
        y1=y1/np.mean(y1)
        y2=y2/np.mean(y2)

    if subtract_mean:
        y1=y1-np.mean(y1)
        y2=y2-np.mean(y2)

    lags_index=np.arange(len(y1))
    if circular:
        ccf_pos=np.array([stats.pearsonr(y1,np.roll(y2,lag))[0] for lag in lags_index])

        ccf_neg=np.array([stats.pearsonr(y1,np.roll(y2,lag))[0] for lag in -lags_index])

        ccf=np.concatenate((ccf_neg,ccf_pos))

        lags=np.concatenate((-lags_index[lags_index],lags_index[lags_index]))*dx

        tmp = lags.argsort()
        return lags[tmp],ccf[tmp]
    if not circular:
        # def my_roll(y,lag):
        #     tmp=np.roll(y,lag)
        #     if lag<=0:
        #         tmp[lag:]=0
        #     else:
        #         tmp[0:lag]=0
        #     return tmp

        # ccf_pos=np.array([stats.pearsonr(y1,my_roll(y2,lag))[0] for lag in lags_index])

        # ccf_neg=np.array([stats.pearsonr(y1,my_roll(y2,lag))[0] for lag in -lags_index])

        # ccf=np.concatenate((ccf_neg,ccf_pos))

        # lags=np.concatenate((-lags_index[lags_index],lags_index[lags_index]))*dx

        # tmp = lags.argsort()

        def xcorr(x, y, normed=True, maxlags=10):
            # Cross correlation of two signals of equal length
            # Returns the coefficients when normed=True
            # Returns inner products when normed=False
            # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
            # Optional detrending e.g. mlab.detrend_mean
            #https://github.com/colizoli/xcorr_python

            Nx = len(x)
            if Nx != len(y):
                raise ValueError('x and y must be equal length')


            c = np.correlate(x, y, mode='full')

            if normed:
                n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
                c = np.true_divide(c,n)

            if maxlags is None:
                maxlags = Nx - 1

            if maxlags >= Nx or maxlags < 1:
                raise ValueError('maglags must be None or strictly '
                                 'positive < %d' % Nx)

            lags = np.arange(-maxlags, maxlags + 1)
            c = c[Nx - 1 - maxlags:Nx + maxlags]
            return lags, c

        lags,ccf=xcorr(y1,y2,maxlags=len(y1)-1)
        plt.cla()
        lags=lags*dx
        return lags,ccf



class CrossCorrelation():
    '''
    Computes ccf of two timeseries.
    lags are those of the second array relative to the first. I.E.
    peaking on the negative delays means that the seconds LAGS the first
    peaking on the positive values means that the second precedes the first
    '''
    def __init__(self,x,y1,y2,circular=1):
        self.x=x
        self.y1=y1
        self.y2=y2
        self.circular=circular

    def calc_ccf(self):
        ccf=cross_correlation(self.x, self.y1, self.y2,circular=self.circular)
        self.lag,self.ccf=ccf
        return ccf
    def find_max(self):
        indeces=np.where(self.ccf==self.ccf.max())[0]
        return self.lag[indeces],self.ccf[indeces],indeces



if __name__=='main':
#%%test
    x=np.linspace(0,4,1000)*np.pi
    #y1=np.sin(x)+np.random.normal(x-x,0.5)
    #y2=np.cos(x)+np.random.normal(x-x,0.5)
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    y1=gaussian(x, 2, 0.4)
    y2=gaussian(x, 3, 0.4)
    dt=np.median(np.diff(x))
    fig,[ax1,ax2]=plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(x,y1,x,y2)


    CCF_obj=CrossCorrelation(x, y1, y2,circular=0)
    CCF_obj.calc_ccf()

    ax2.plot(CCF_obj.lag,CCF_obj.ccf,'g-.',ms=1)

    CCF_obj=CrossCorrelation(x, y1, y2,circular=1)
    CCF_obj.calc_ccf()
    ax2.plot(CCF_obj.lag,CCF_obj.ccf,'b-.',ms=1)
    plt.show()
    max_stuff=CCF_obj.find_max()

    fig,ax=plt.subplots()

    ax.plot(CCF_obj.y1,np.roll(CCF_obj.y2,max_stuff[-1][-1]))

    plt.show()

