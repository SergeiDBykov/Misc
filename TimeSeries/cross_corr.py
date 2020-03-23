#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:42:32 2020

@author: s.bykov
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import  stats
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import pearsonr


def my_wrap_ccf(x,y1,y2,only_pos_delays=1):
    lags=np.arange(len(x))

    def my_wrap_pos(y1,y2,lags):
        return np.array([pearsonr(np.roll(y1,-lag),y2)[0] for lag in lags])
    def my_wrap_neg(y1,y2,lags):
        return np.array([pearsonr(np.roll(y1,lag),y2)[0] for lag in lags])
    def my_wrap(y1,y2,lags):
        return np.concatenate((my_wrap_neg(y1,y2,lags),my_wrap_pos(y1,y2,lags)))

#    def my_wrap_zero_padd(y1,y2,lags):
#        def my_roll(y,lag):
#            tmp=np.roll(y,lag)
#            if lag<=0:
#                tmp[lag:]=0
#            else:
#                tmp[0:lag]=0
#            return tmp
#
#        return np.array([pearsonr(my_roll(y1,-lag),y2)[0] for lag in lags])

    if only_pos_delays:
        ccf=my_wrap_pos(y1,y2,lags)
        delay=x[lags]
    else:
        ccf=my_wrap(y1,y2,lags)
        delay=np.concatenate((-x[lags],x[lags]))
    args=delay.argsort()
    return delay[args],ccf[args]
    #return x[lags],np.array([pearsonr(np.roll(y1,-lag),y2)[0] for lag in lags])
    #return x[lags],np.array([pearsonr(np.concatenate((np.roll(x,-lag)[0:len(x)-lag],np.zeros(lag))),y2)[0] for lag in lags])


def my_crosscorr(x,y1,y2,ax1,ax2,
                 divide_by_mean=0,subtract_mean=1,
                 maxbinslag=None,
                 only_pos_delays=0,my_only_pos_delays=0,
                 y1label='y1',y2label='y2',
                 my_ccf=1):
    binsize=np.median(np.diff(x))

    if ax1!=None:
        ax1_tw=ax1.twinx()
        ax1.plot(x,y1,label=y1label,color='r')
        ax1_tw.plot(x,y2,label=y2label,color='k')
        ax1.legend()
        ax1_tw.legend()
        ax1.set_xlabel('time')
        ax1.set_ylabel('signal')
        ax1.grid()
    if ax1==None:
        pass

    if divide_by_mean:
        y1=y1/np.mean(y1)
        y2=y2/np.mean(y2)

    if subtract_mean:
        y1=y1-np.mean(y1)
        y2=y2-np.mean(y2)



    lag,corr,_,_=ax2.xcorr(y1,y2,maxlags=maxbinslag,lw=2,usevlines=1)

    if only_pos_delays:
        corr=corr[lag>=0]
        lag=lag[lag>=0]

    ax2.cla()
    timelag=lag*binsize
    ax2.plot(timelag,corr)
    if my_ccf:
        my_delta,my_corr=my_wrap_ccf(x,y1,y2,only_pos_delays=my_only_pos_delays)
        ax2.plot(my_delta,my_corr,'r:')#,label='wrap padding')
        my_tdelay=my_delta[np.argmax(my_corr)]
        print(np.where(my_corr==max(my_corr)))
        #my_tdelay *=binsize
        ax2.axvline(my_tdelay,color='r',alpha=0.5,label=f'lag {"{0:.2f}".format(my_tdelay)} sec')
        ax2.legend()
    else:
        my_tdelay=0

    ax2.plot()
    ax2.grid()

    ax2.set_xlabel('lag, sec ')
    ax2.set_ylabel('Cross correlation')
    ax2.set_title(f' {y1label} leads <---0---> {y2label} leads')
    if only_pos_delays:
        ax2.set_title(f'0---> {y2label} leads')

    delay=lag[np.argmax(corr)]
    #print(f'lag, number of indeces: {delay}')
    tdelay=delay*binsize
    #print(f'cross corr lag: {tdelay}')

    ax2.axvline(tdelay,color='b',alpha=0.5,label=f'lag {"{0:.2f}".format(tdelay)} sec')
    ax2.legend()

    return np.array([tdelay,my_tdelay]),lag*binsize,corr

#
#
##def crosscorr(datax, datay, lag=0, wrap=False):
##    """ Lag-N cross correlation.
##    Shifted data filled with NaNs
##
##    Parameters
##    ----------
##    lag : int, default 0
##    datax, datay : pandas.Series objects of equal length
##    Returns
##    ----------
##    crosscorr : float
##    """
##    if wrap:
##        shiftedy = datay.shift(lag)
##        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
##        return datax.corr(shiftedy)
##    else:
##        return datax.corr(datay.shift(lag))
#
#
#
#stop
#
#
#%% test

if __name__=='main':
    ObsID='90089-11-04-02G'#90427-01-03-12 90427-01-03-11  90089-11-03-03  90089-11-02-06 90089-11-02-05
    filepath=f'/Users/s.bykov/work/xray_pulsars/rxte/results/out{ObsID}/products/fasebin/cutoffpl/ph_res_cutoffpl.dat'

    data=np.genfromtxt(filepath)

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
    eqw_err=np.vstack((eqw_low, eqw_hi))


    flux712=data[:,7]
    flux712_low=flux712-data[:,8]
    flux712_hi=data[:,9]-flux712

    flux712=flux712/1e-8
    flux712_hi=flux712_hi/1e-8
    flux712_low=flux712_low/1e-8

    flux712_err=np.vstack((flux712_low, flux712_hi))



    #dphase,r=my_crosscorr(phase*4.37415680102,eqw,flux712,only_pos_delays=0)
    #plot(dphase,r,'g.:')
    #plt.show()

    fig,axs=plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    my_crosscorr(phase*4.37415680102,eqw,flux712,axs[0],axs[1],subtract_mean=1,divide_by_mean=1,
                 y1label='eqw',y2label='F(7-12)',
                 only_pos_delays=0,my_only_pos_delays=0)
    plt.show()

    ##fig,ax0=plt.subplots()
    ##fig,ax=plt.subplots()
    ##fig.subplots_adjust(hspace=0.5)
    ##my_crosscorr(phase*4.3745,eqw,flux712,ax0,ax,subtract_mean=0)
    ##plt.show()