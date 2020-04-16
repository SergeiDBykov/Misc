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
    '''
    Computes ccf of two timeseries.
    lags are those of the second array relative to the first. I.E.
    peaking on the negative delays means that the seconds LAGS the first
    peaking on the positive values means that the second precedes the first
    '''
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

    import numpy, scipy.optimize

    def fit_sin(tt, yy):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = numpy.array(tt)
        yy = numpy.array(yy)
        ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(numpy.fft.fft(yy))
        guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = numpy.std(yy) * 2.**0.5
        guess_offset = numpy.mean(yy)
        guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*numpy.pi)
        fitfunc = lambda t: A * numpy.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

    #data=np.genfromtxt('/Users/s.bykov/work/xray_pulsars/rxte/results/out90089-11-03-01G/products/fasebin/cutoffpl/ph_res_cutoffpl.dat')
    data=np.genfromtxt('/Users/s.bykov/work/xray_pulsars/rxte/results/out90427-01-03-02/products/fasebin/cutoffpl/ph_res_cutoffpl.dat')

    N_sp=(data[0,1]-1)/2
    spe_num=data[:,0]

    data=np.vstack((data,data))
    nph=data[0,1]
    data[:,0]=np.arange(1,nph) #create new spe_num
    spe_num=data[:,0]
    phase=((spe_num-1)/(N_sp))

    eqw=data[:,4]
    eqw_low=eqw-data[:,5]
    eqw_hi=data[:,6]-eqw

    eqw=eqw*1e3
    eqw_low=eqw_low*1e3
    eqw_hi=eqw_hi*1e3
    eqw_err=np.vstack((eqw_low, eqw_hi)).max(axis=0)

    flux712=data[:,7]
    flux712_low=flux712-data[:,8]
    flux712_hi=data[:,9]-flux712

    flux712=flux712/1e-8
    flux712_hi=flux712_hi/1e-8
    flux712_low=flux712_low/1e-8

    flux712_err=np.vstack((flux712_low, flux712_hi)).max(axis=0)

    figure()
    fit=fit_sin(phase,eqw)
    plt.plot(phase,eqw)
    plt.plot(np.linspace(0,2,100),fit['fitfunc'](np.linspace(0,2,100)))


#test eqw trials
    N=500
    eqw_trials=np.zeros(shape=(N,len(phase)))
    for i in range(N):
        eqw_trial=eqw+np.random.normal(loc=0,scale=eqw_err)
        eqw_trials[i]=eqw_trial
        #plt.plot(phase,eqw_trial,'gray',alpha=0.7)
    plt.errorbar(phase,eqw,eqw_err,color='r',zorder=10,capsize=3)
    plt.errorbar(phase,eqw_trials.mean(axis=0),eqw_trials.std(axis=0),color='c',zorder=15,capsize=3)

    pearsonr_arr=np.zeros(N)

    for i in range(N):
        pearsonr_arr[i]=pearsonr(eqw_trials[i], flux712)[0]

    N_trials=500

    ccf_trials=np.zeros(shape=(N_trials,2*len(phase)))

    test_eqw=np.zeros(N_trials)
    test_eqw_ind=12

    test_lag_of_max=np.zeros(N_trials)

    for i in range(N_trials):
        print(i)
        eqw_trial=np.random.normal(loc=eqw,scale=eqw_err)
        test_eqw[i]=eqw_trial[test_eqw_ind]
        #flux_trial=np.random.normal(loc=flux712,scale=flux712_err)
        flux_trial=flux712
        CCF=CrossCorrelation(phase,eqw_trial,flux_trial,circular=1)
        lag,ccf=CCF.calc_ccf()
        test_lag_of_max[i]=lag[np.argmax(ccf)]
        ccf_trials[i]=ccf

    fig,ax=plt.subplots()
    ax.errorbar(lag,ccf_trials.mean(axis=0),ccf_trials.std(axis=0)*1.645)

    CCF=CrossCorrelation(phase,eqw,flux712,circular=True)
    lag_orig,ccf_orig=CCF.calc_ccf()
    ax.plot(lag_orig,ccf_orig,'r:')


    plt.figure()
    plt.plot(lag,ccf-ccf_trials.mean(axis=0),'k:')
    plt.plot(lag,ccf-np.median(ccf_trials,axis=0),'c-.')

    fig,ax=plt.subplots()
    plt.hist(test_eqw,bins=50)
    plt.axvline(eqw[test_eqw_ind])

    plt.figure()
    from scipy import stats
    import seaborn as sns

    sns.distplot(ccf_trials[:,11],fit=stats.norm,kde=0)
    plt.axvline(ccf[11])

    plt.figure()
    from scipy import stats
    import seaborn as sns

    sns.distplot(test_lag_of_max,fit=stats.norm,kde=0)




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

