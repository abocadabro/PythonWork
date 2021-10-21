#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:15:20 2021

@author: andrewjacobs
"""

import xlrd
import math
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from os.path import expanduser as ospath
from scipy.optimize import minimize
from scipy import optimize
from scipy import stats
from numpy import log,sqrt,inf,exp
from scipy.misc import derivative
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import QuantLib as ql
import time   
import csv

        
    
def SABR_imp_var(alpha,rho,nu,F,K,t):
    """SABR formula. alpha is vol level, rho is
    correlation and nu is vol of vol. F is the forward rate, K the strike and t
    the maturity"""
    A = 1 + (0.25 * alpha * nu * rho + (2 - 3 * (rho**2)) * (nu**2) / 24.) * t
    if K <= 0:
        sabr_vol = -1
    elif F == K: # ATM formula      
        sabr_vol = alpha * A
    elif F != K: # not-ATM formula
        y = math.log(K/F)
        z = - (nu / alpha) * y
        chiz = math.log( ( math.sqrt(1 - 2 * rho * z + z**2) + z - rho ) / (1 - rho) )
        sabr_vol = alpha * ( z / chiz ) * A
    return sabr_vol * sabr_vol * t

def SSVI_pf(theta,eta,gamma):
    """helper function for SSVI1"""
    return eta /( (theta**gamma) * (1 + theta)**(1 - gamma))

def SSVI1_imp_var(y,theta,eta,rho, gamma):
    """SSVI power law"""
    return (theta / 2) * (1 + rho * SSVI_pf(theta,eta,gamma) * y 
                          + math.sqrt((SSVI_pf(theta,eta,gamma) * y + rho)**2 + (1 - rho**2)))

def SSVI1_Objective(SSVI1_params,vols,strikes,F,t):
    obj=0
    for i in range(len(strikes)):
        y=math.log(strikes[i] / F)
        diff=math.sqrt(SSVI1_imp_var(y,SSVI1_params[0],SSVI1_params[1],SSVI1_params[2],
                           SSVI1_params[3])/t) - vols[i] 
        obj+=diff * diff
    return  math.sqrt(obj)
    
def SSVI1_constraint(SSVI1_params):
    return 2 - SSVI1_params[1] * (1 + abs(SSVI1_params[2]))

    
def calibrate_SSVI1(vols,strikes,F,t,initial_guess):
    bnds=((0.00001,None),(None,None),(-0.99999,0.99999),(0.00001,0.5))
    cons=({'type':'ineq','fun':SSVI1_constraint})
    result=minimize(SSVI1_Objective,initial_guess,(vols,strikes,F,t),
                    method='SLSQP',bounds=bnds,constraints=cons)
    SSVI1_params=result.x
    error=0
    ssvi1_vols=[]
    for i in range(len(strikes)):
        ssvi1_vols.append(math.sqrt(SSVI1_imp_var(math.log(strikes[i]/F),SSVI1_params[0],
                                         SSVI1_params[1],SSVI1_params[2],
                                         SSVI1_params[3]) / t))
        vol_diff=ssvi1_vols[i]-vols[i]
        error+=vol_diff * vol_diff
    error=math.sqrt(error)
    return (SSVI1_params,ssvi1_vols,error)

def SSVI1_w(y,theta,eta,rho, gamma):
    return SSVI1_imp_var(y,theta,eta,rho, gamma)

def SSVI1_wp(y,theta,eta,rho, gamma):
    pht=SSVI_pf(theta,eta,gamma)
    wp = (theta / 2) * (rho * pht + ((pht * y + rho)**2 + (1 - rho**2))**(-0.5) *
                          (pht * y + rho) * pht)
    return wp
    
def SSVI1_wpp(y,theta,eta,rho, gamma):
    pht=SSVI_pf(theta,eta,gamma)
    wpp = (theta / 2) * (rho * pht + ((pht * y + rho)**2 + (1 - rho**2))**(-0.5) * pht**2 - 
                         ((pht * y + rho)**2 + (1 - rho**2))**(-1.5) * (pht * y + rho)**2 *
                         pht**2)
    return wpp                     
    
def pdf_smile(y,w,wp,wpp):
    d2=-(w**(-0.5))*(y+0.5*w)
    g=(1 - (y * wp)/(2 * w))**2 - 0.25 * ((1/w) + 0.25) * (wp * wp) + 0.5 * wpp
    return (w**(-0.5)) * g * stats.norm.pdf(d2)

def pdf_SSVI1(y,params):
    w=SSVI1_w(y,params[0],params[1],params[2], params[3])
    wp=SSVI1_wp(y,params[0],params[1],params[2], params[3])
    wpp=SSVI1_wpp(y,params[0],params[1],params[2], params[3])
    pdf=pdf_smile(y,w,wp,wpp)
    return pdf

def BS_Call_Price(F,K,var):
    """Black-Scholoes forward value (un-discoounted) call option price.
    Forward F, strike K, implied variance var."""
    d1=(log(F/K) + 0.5 * var)/(math.sqrt(var))
    d2=d1 - math.sqrt(var)
    call_price = F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
    return call_price

def BS_Put_Price(F,K,var):
    """Black-Scholoes forward value (un-discoounted) put option price.
    Forward F, strike K, implied variance var."""
    d1=(log(F/K) + 0.5 * var)/(math.sqrt(var))
    d2=d1 - math.sqrt(var)
    put_price = -F * stats.norm.cdf(-d1) + K * stats.norm.cdf(-d2)
    return put_price

def pdf_BS(y,w):
    """Black-Scholes pdf at value y=logK/F. Working in log space means this 
    is a normal """
    d2=-(w**(-0.5)) * (y + 0.5*w)
    return (w**(-0.5)) * stats.norm.pdf(d2)

def deltafp_BS(cp,y,w):
    d1 = (-y + 0.5*w) / math.sqrt(w)
    return cp * stats.norm.cdf(cp * d1)

def main():
    
    alpha =0.11
    nu=0.8
    rho=-0.2
    
    f=1
    s=1
    t=1
    
    dnk=optimize.brentq(lambda k : log(k/f) - 0.5 * SABR_imp_var(alpha,rho,nu,f,k,t),
                        0.001,10)
    
    dnvol=math.sqrt(SABR_imp_var(alpha,rho,nu,f,dnk,t) / t)
    
    ck=optimize.brentq(lambda k : (log(k/f) - 0.5 * SABR_imp_var(alpha,rho,nu,f,k,t)
                                   + stats.norm.ppf(0.25) * math.sqrt(SABR_imp_var(alpha,rho,nu,f,k,t))),
                        0.001,10)
    
    ck_high=optimize.brentq(lambda k : (log(k/f) - 0.5 * SABR_imp_var(alpha,rho,nu,f,k,t)
                                   + stats.norm.ppf(0.01) * math.sqrt(SABR_imp_var(alpha,rho,nu,f,k,t))),
                        0.001,10)
    
    ckvol=math.sqrt(SABR_imp_var(alpha,rho,nu,f,ck,t) / t)
    
    pk=optimize.brentq(lambda k : (log(k/f) - 0.5 * SABR_imp_var(alpha,rho,nu,f,k,t)
                                   - stats.norm.ppf(0.25) * math.sqrt(SABR_imp_var(alpha,rho,nu,f,k,t))),
                        0.001,10)
    
    pk_low=optimize.brentq(lambda k : (log(k/f) - 0.5 * SABR_imp_var(alpha,rho,nu,f,k,t)
                                   - stats.norm.ppf(0.01) * math.sqrt(SABR_imp_var(alpha,rho,nu,f,k,t))),
                        0.001,10)
    
    pkvol=math.sqrt(SABR_imp_var(alpha,rho,nu,f,pk,t) / t)
    
    
    rr=ckvol-pkvol
    fly=0.5*(ckvol+pkvol)-dnvol
    
    print("DN ", dnvol, " 25RR ", rr, " 25Fly ", fly)
    
    
    dn_test=(deltafp_BS(1,log(dnk/f),SABR_imp_var(alpha,rho,nu,f,dnk,t)) + 
             deltafp_BS(-1,log(dnk/f),SABR_imp_var(alpha,rho,nu,f,dnk,t)))
    
    c_test = deltafp_BS(1,log(ck/f),SABR_imp_var(alpha,rho,nu,f,ck,t)) 
    
    p_test = deltafp_BS(-1,log(pk/f),SABR_imp_var(alpha,rho,nu,f,pk,t)) 
    
    # print(p_test)
    # print(dn_test)
    # print(c_test)
    
    # print(pk)
    # print(pkvol)
    # print(ck)
    # print(ckvol)
    # print(dnk)
    # print(dnvol)
    
    vols=(pkvol,dnvol,ckvol)
    strikes=(pk,dnk,ck)
    params=[0.1,1.,0,0.25]
    calibration=calibrate_SSVI1(vols,strikes,f,t,params)
    
    params=calibration[0]
    vols=calibration[1]
    error=calibration[2]
    
    # print(params)
    # print(vols)
    # print(error)
    
    noise=0.15


    strikes=np.linspace(pk_low,ck_high,100)
    aprx1=[]
    aprx2=[]
    aprx3=[]
    true=[]
    SSVIvols=[]
    SABRvols=[]

    
    for k in strikes:
        lower =max(-k,-5*noise)
        upper =-lower
        
        check = integrate.quad(lambda z : stats.norm.pdf(z,0,noise),lower,upper)[0]
        
        base_val = BS_Call_Price(f,k,dnvol*dnvol*t)
        int_val = integrate.quad(lambda z : (BS_Call_Price(f,k+z,dnvol*dnvol*t) 
                                     * stats.norm.pdf(z,0,noise)),lower,upper)[0]
    
    
        effk = optimize.brentq(lambda x : int_val - BS_Call_Price(f,x,dnvol*dnvol*t),
                           0.001,10)
        
        
        base_smile_val = BS_Call_Price(f,k,
                         SSVI1_imp_var(log(k/f),params[0],params[1],params[2], params[3]))
        
        smile_val = integrate.quad(lambda z : (BS_Call_Price(f,k+z,
                                           SSVI1_imp_var(log((k+z)/f),params[0],params[1],params[2], params[3]))
                                     * stats.norm.pdf(z,0,noise)),lower,upper)[0]
        
        impact1 = base_smile_val-base_val
        scale = (int_val/base_val)
        impact2 = scale*impact1
               
        impact3 = (BS_Call_Price(f,effk,
                            SSVI1_imp_var(log((effk)/f),params[0],params[1],params[2], params[3]))
               -BS_Call_Price(f,effk,dnvol*dnvol*t))
    
        impact4 = smile_val- int_val
        
        aprx1.append(impact1*1000)
        aprx2.append(impact2*1000)
        aprx3.append(impact3*1000)
        true.append(impact4*1000)
        
        vol=math.sqrt(SSVI1_imp_var(log(k/f),params[0],params[1],params[2], params[3])/t)
        SSVIvols.append(vol)
        vol=math.sqrt(SABR_imp_var(alpha,rho,nu,f,k,t))
        SABRvols.append(vol)

        

    plt.figure(0)
    plt.plot(strikes,aprx1,'b')
    #plt.plot(strikes,aprx2,'y')
    #plt.plot(strikes,aprx3,'g')
    plt.plot(strikes,true,'r')
    
    plt.legend(["approximation","integrated"])
    plt.savefig('Impacts3.pdf')
    plt.figure(1)
    plt.plot(strikes,SSVIvols,'r')
    #plt.savefig('SmileB.pdf')
    
  
if __name__ == "__main__":
    main()
    
    