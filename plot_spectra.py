#!/Users/s.bykov/anaconda/bin/python

#####!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:49:40 2020

@author: s.bykov
"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import random


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def read_data(filename):
    data=np.genfromtxt(filename)
    data=data.T
    x=data[:,0]
    y=data[:,1]
    yerr=data[:,2]
    mo=data[:,3]
    delchi=data[:,4]
    return x,y,yerr,mo,delchi


def make_figure(log=1):
    fig = plt.figure(figsize=(6.6, 6.6))
    rows=3
    cols=3
    ax_spe = plt.subplot2grid((rows,cols), (0, 0), rowspan=2, colspan=3)
    ax_del = plt.subplot2grid((rows,cols), (2, 0), rowspan=1, colspan=3)
    plt.subplots_adjust(hspace=0)
    ax_del.axhline(0,color='k')
    ax_del.set_xlabel('E, keV')
    ax_del.set_ylabel('$\chi$')
    ax_spe.set_ylabel('$EF_E, keV^2 (phot cm^{-2} s^{-1} keV^{-1})$')
    if log:
        for ax in [ax_spe,ax_del]:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax_del.set_yscale('linear')

    return fig,ax_spe,ax_del

class Spectra():
    def __init__(self,filename):
        temp=read_data(filename)
        self.en=temp[0]
        self.data=temp[1]
        self.data_err=temp[2]
        self.mo=temp[3]
        self.delchi=temp[4]

    def plot_spe(self,ax,
                 marker='s',mfc='k',mec='k',
                 ecolor='gray',mew=1,ls='None',
                 label='spectra',alpha=0.6,mocolor='r'):
        ax.plot(self.en,self.data,marker=marker,mfc=mfc,mec=mec,mew=mew,ls=ls,label=label,alpha=alpha)
        ax.errorbar(self.en,self.data,self.data_err,ecolor=ecolor,fmt='none',alpha=0.5)

        ax.plot(self.en,self.mo,marker='',ls=':',color=mocolor,alpha=alpha)

        ax.legend(loc='best')



    def plot_del(self,ax,
                 marker='s',mfc='k',mec='k',
                 ecolor='gray',mew=1,ls='None',
                 label='spectra',alpha=0.6):
        ax.plot(self.en,self.delchi,marker=marker,mfc=mfc,mec=mec,mew=mew,ls=ls,label=label,alpha=alpha)
        ax.errorbar(self.en,self.delchi,self.en/self.en,ecolor=ecolor,fmt='none',alpha=0.5)
        ax.legend(loc='best')


    def plot_all(self,ax_spe,ax_del,
                 marker='s',mfc='k',mec='k',
                 ecolor='gray',mew=1,ls='None',
                 label='spectra',alpha=0.6,mocolor='b'):
        self.plot_spe(ax_spe,marker=marker,mfc=mfc,mec=mec,mew=mew,ls=ls,label=label,alpha=alpha,mocolor=mocolor)
        self.plot_del(ax_del,marker=marker,mfc=mfc,mec=mec,mew=mew,ls=ls,label=label,alpha=alpha)

if __name__ == "__main__":
    specs =sys.argv[1:]
    fig,ax_spe,ax_del=make_figure()
    for spefile in specs:
        sp=Spectra(spefile)
        r, g, b = np.random.uniform(0, 1, 3)
        sp.plot_all(ax_spe,ax_del,mfc=(r, g, b, 1),label=spefile)
    plt.show()


'''
filename='/Users/s.bykov/work/xray_pulsars/nustar/results/out90202031004/products/mean_spectra/spectra2.dat'
sp=Spectra(filename)
#f,a,b=make_figure()
r, g, bl = np.random.uniform(0, 1, 3)
sp.plot_all(a,b,mfc=(r, g, bl, 1))

'''