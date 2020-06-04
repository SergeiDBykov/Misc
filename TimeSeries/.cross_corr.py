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
from scipy.ndimage.interpolation import shift

def my_ccf(x,y1,y2):
    lags=np.arange(len(x)-1)
    def my_wrap_zero_padd(y1,y2,lags):
        def my_roll(y,lag):
            tmp=np.roll(y,lag)
            if lag<=0:
                tmp[lag:]=0
            else:
                tmp[0:lag]=0
            return tmp
        #my_roll=np.roll
        return np.array([pearsonr(my_roll(y1,-lag),y2)[0] for lag in lags])
    return lags,my_wrap_zero_padd(y1, y2, lags)

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real



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
        fig,[ax1,ax2]=plt.subplots(2,1)
        fig.subplots_adjust(hspace=0.5)


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

    plt.show()

    return np.array([tdelay,my_tdelay]),lag*binsize,corr



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



#%%test
if __name__=='main':
    x=np.linspace(0,2,1000)*np.pi
    y1=np.sin(x)
    y2=np.cos(x)
    dt=np.median(np.diff(x))
    fig,[ax1,ax2]=plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(x,y1,x,y2)
    np_lag,np_corr,_,_=ax2.xcorr(y1,y2,maxlags=len(y1)-1,lw=2,usevlines=False,ls='-')
    ax2.cla()


    plt.show()

    lag,ccf=cross_correlation(x, y1, y2,circular=0)

    ax2.plot(lag,ccf,'g-.',ms=1)

    lag,ccf=cross_correlation(x, y1, y2,circular=1)

    ax2.plot(lag,ccf,'m-.',ms=1)
    ax2.plot(np_lag*dt,np_corr,'b-.')


#%% test 2

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