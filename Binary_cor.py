#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:29:13 2020

@author: s.bykov
"""


import numpy as np
try:
    from PyAstronomy import pyasl
    PyAstronomy_ok=1
except:
    PyAstronomy_ok=0
    print('PyAstronomy not installed, the process will be slower')

day2sec=86400
try:
    del e
except:
    pass

orb_params_v0332_sec={'P':33.850,'e':0.3713,'asini':77.81,
'w':277.43,'T_p':57157.88}


def Binary_orbit(time,orb_params,pporb=0.0,limit=1.0*10**-6,maxiter=20):

    porb=orb_params['P']
    ecc=orb_params['e']
    asini=orb_params['asini']
    omega_d=orb_params['w']
    try:
        t0=orb_params['T_p']
        t90=-1
    except:
        t90=orb_params['T90']
        t0=-1


    #%\function{BinaryCor}
    #%\synopsis{Removes the influence of the doublestar motion for circular or eliptical orbits.}
    #%\usage{Array_Type t = BinaryCor(Array_Type OR Double_Type time), time in MJD}
    #%\qualifiers{
    #%\qualifier{asini}{: Projected semi-major axis [lt-secs], Mandatory}
    #%\qualifier{porb}{: Orbital period at the epoch [days], Mandatory}
    #%\qualifier{eccentricity}{: Eccentricity, (0<=e<1)}
    #%\qualifier{omega}{: Longitude of periastron [degrees], mandatory}
    #%\qualifier{t0}{: epoch for mean longitude of 0 degrees (periastron, MJD)}
    #%\qualifier{t90}{: epoch for mean longitude of 90 degrees (MJD)}
    #%\qualifier{pporb}{: rate of change of the orbital period (s/s) (default 0)}
    #%\qualifier{limit}{: absolute precision of the correction of the  computation (days, default: 1D-6)}
    #%\qualifier{maxiter}{: stop Newton-Raphson iteration after maxiter steps if limit is not reached (default: 20)}

    #%\description
    #%    (transcribed from IDL-programm BinaryCor.pro)
    #%
    #%    For each time, the z-position of the emitting object is computed
    #%    and the time is adjusted accordingly. This is iterated until
    #%    convergence is reached (usually only one iteration is necessary,
    #%    even in high elliptic cases).
    #%
    #%    Follows equations from Hilditch's book and has also been
    #%    checked against fasebin/axBary. All codes give identical results,
    #%    (to better than 1d-7s) as checked by a Monte Carlo search using
    #%    1d7 different orbits.
    #%
    #%    qualifiers t90 and t0 have to be in days and in the same time
    #3%    system as time (e.g. JD or MJD)
    #%
    #%    Circular orbits:
    #%         * if time of lower conjunction Tlow is known, set
    ##%           t0=Tlow and omega=0
    #%         * if time of ascending node is known, Tasc, set
    #%           t90=Tasc and omega=0
    #%         * if time of mid eclipse is known, Tecl, set
    #%           t0=Tecl-0.25*porb and omega=0
    #%!%-

    time=np.array(time)  #make sure that time is an array
    if t90==-1 and t0==-1:
        print("error: need t90 or t0 value")
        return

    if t90!=-1 and t0!=-1:
        print("error: Only one of the t90 and t0 arguments is allowed")
        return

    if ecc<0 :
        print("error: eccentricity must be positive!")
        return
    if ecc>=1:
        print("error: Orbit correction has only been implemented for circular and elliptic orbits")
        return
    if ecc==0:
        omega_d=0 #circular orbit


    if t0==-1:
        t0 = t90+(omega_d-90.)/360. * porb

    if maxiter <=0:
        maxiter=20

    asini_d = asini/86400. #86400 segundos en un día
    t= time

    #Corrections for eccentricity 0<=ecc<1
    omega = omega_d * np.pi/180.0
    sinw = np.sin(omega)
    cosw = np.cos(omega)
    sq = ((1.-ecc)*(1.+ecc))**0.5
    cor =np.array([2.*limit]*len(t))


    #start with number of iterations = zero
    numiter=0

    if PyAstronomy_ok==0:
        contada=0
        while((abs(np.amax(cor)) > limit) and (numiter < maxiter)):
            tper = (t-t0)/porb
            m = 2*np.pi*(tper*(1.-0.5*pporb*tper))
            m=np.array(m)
            eanom=np.array([1.0]*len(t))
            #eanom = KeplerEquation(m,ecc)  #use this command for a faster solution
            eanom = KeplerEquation1(m,ecc)  #use this command for a better solution
            sin_e = np.sin(eanom)
            cos_e = np.cos(eanom)
            z = asini_d*(sinw*(cos_e-ecc)+sq*cosw*sin_e)
            f = (t-time)+z
            df = (sq*cosw*cos_e - sinw*sin_e)*(2*np.pi*asini_d/(porb*(1.0-ecc*cos_e)))
            cor =f/(1.0+df)
            t = t-cor
            numiter=numiter+1
            contada=contada+1
            print(100*contada/20,"%")
            if numiter >= maxiter:
                print("Exceeded maxiter iterations and did not reach convergence");
                break
    return(t)

def KeplerEquation(m,ecc):#http://astro.uni-tuebingen.de/software/idl/aitlib/astro/binarycor.html
    m=np.array(m)
    if ecc<0 :
        print("error: eccentricity must be positive!")
        return
    if ecc>=1:
        print("error: Orbit correction has only been implemented for circular and elliptic orbits")
    print('etapa 1')
    for j in range(0,len(m)):
        mod_m=m[j]/2/np.pi
        m[j]=m[j]-2*np.pi*round(mod_m)
        if j==3:print(len(m))
        if j==round(len(m)*0.05):print(5,"%")
        if j==round(len(m)*0.25):print(25,"%")
        if j==round(len(m)*0.5):print(50,"%")
        if j==round(len(m)*0.8):print(80,"%")
        if j==len(m)-1:print(100,"%")
        while m[j]>np.pi:
            print(m[j])
            m[j]=m[j]-2*np.pi
            print(m[j])

        while m[j]<-np.pi:
            print(m[j])
            m[j]=m[j]+2*np.pi
            print(m[j])
    if ecc==0:
        E=m
    aux=4.0*ecc+0.5
    alpha=(1.0-ecc)/aux

    Beta=m/(2.0*aux)
    aux=np.sqrt(Beta**2+alpha**3)

    z=Beta+aux
    test=np.array([1.0]*len(z))
    for j in range(0,len(m)):
        if z[j]<=0.0:
            z[j]=Beta[j]-aux[j]

        test[j]=abs(z[j])**(1/3)
    z=test
    for j in range(0,len(m)):
        if z[j]<0.0:
            z[j]=-z[j]
    s0=z-alpha/z

    s1=s0-(0.078*s0**5)/(1.0+ecc)
    e0=m+ecc*(3.0*s1-4.0*s1**3)

    se0=np.sin(e0)
    ce0=np.cos(e0)

    f  = e0-ecc*se0-m
    f1 = 1.0-ecc*ce0
    f2 = ecc*se0
    f3 = ecc*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.50*f2*u1)
    u3 = -f/(f1+0.50*f2*u2+.16666666666667*f3*u2*u2)
    u4 = -f/(f1+0.50*f2*u3+.16666666666667*f3*u3*u3+.041666666666667*f4*u3**3)

    eccanom=e0+u4

    for j in range(0,len(m)):
        while eccanom[j]>=2.0*np.pi:
            eccanom[j]=eccanom[j]-2.0*np.pi
        while eccanom[j]<2.0*np.pi:
            eccanom[j]=eccanom[j]+2.0*np.pi


    return(eccanom)

def KeplerEquation1(m,ecc):
    m=np.array(m)
    if ecc<0 :
        print("error: eccentricity must be positive!")
        return
    if ecc>=1:
        print("error: Orbit correction has only been implemented for circular and elliptic orbits")
    for j in range(0,len(m)):
        while m[j]>np.pi:
            m[j]=m[j]-2*np.pi
        while m[j]<-np.pi:
            m[j]=m[j]+2*np.pi
    if ecc==0:
        E=m
    aux=4.0*ecc+0.5
    alpha=(1.0-ecc)/aux

    Beta=m/(2.0*aux)
    aux=np.sqrt(Beta**2+alpha**3)

    z=Beta+aux
    test=np.array([1.0]*len(z))
    for j in range(0,len(m)):
        if z[j]<=0.0:
            z[j]=Beta[j]-aux[j]

        test[j]=abs(z[j])**(1/3)
    z=test
    for j in range(0,len(m)):
        if z[j]<0.0:
            z[j]=-z[j]
    s0=z-alpha/z

    s1=s0-(0.078*s0**5)/(1.0+ecc)
    e0=m+ecc*(3.0*s1-4.0*s1**3)

    se0=np.sin(e0)
    ce0=np.cos(e0)

    f  = e0-ecc*se0-m
    f1 = 1.0-ecc*ce0
    f2 = ecc*se0
    f3 = ecc*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.50*f2*u1)
    u3 = -f/(f1+0.50*f2*u2+.16666666666667*f3*u2*u2)
    u4 = -f/(f1+0.50*f2*u3+.16666666666667*f3*u3*u3+.041666666666667*f4*u3**3)

    eccanom=e0+u4

    for j in range(0,len(m)):
        while eccanom[j]>=2.0*np.pi:
            eccanom[j]=eccanom[j]-2.0*np.pi
        while eccanom[j]<2.0*np.pi:
            eccanom[j]=eccanom[j]+2.0*np.pi
    ##better solution
    CONT=True
    thresh=10**-5
    for j in range(0,len(m)):
        if m[j]<0:
            m[j]=m[j]+2.0*np.pi
    diff=eccanom-np.sin(eccanom)-m
    for j in range(0,len(m)):
        if abs(diff[j])>10**-10:
            I=diff[j]
            while CONT==True:
                fe=eccanom[j]-ecc*np.sin(eccanom[j])-m[j]
                fs=1.0-ecc*np.cos(eccanom[j])
                oldval=eccanom[j]
                eccanom[j]=oldval-fe/fs
                if abs(oldval-eccanom[j])<thresh :CONT=False
            while eccanom[j]>= np.pi:eccanom[j]=eccanom[j]-2.0*np.pi
            while eccanom[j]< np.pi:eccanom[j]=eccanom[j]+2.0*np.pi

    return(eccanom)




stop

#%% compare my and binary_corr
orb_params_v0332_days={'P':33.850,'e':0.3713,'asini':77.81,
'w':277.43,'T_p':57157.88}



orb_params_v0332_seconds={'P':33.850*day2sec,'e':0.3713,'asini':77.81,
'w':np.deg2rad(277.43),'T_p':57157.88*day2sec}


time0=57223.41519888889+33.850/4*3 #MJD


time0_corr=Binary_orbit(np.array([time0]),orb_params_v0332_days)[0]

from Miscellaneous.doppler_correction import kepler_solution
dt=kepler_solution(time0*day2sec, orb_params_v0332_seconds)[-1][0]
time0_mycorr=time0-dt/day2sec

print('BinaryCor, MJD ', time0_corr)

print('MyCor, MJD', time0_mycorr)

print('Diff MyCor - Binary Cor, sec', (time0_mycorr-time0_corr)*day2sec)