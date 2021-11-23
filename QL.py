#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:08:13 2021

@author: andrewjacobs
"""

import datetime
import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple 
import math
from math import exp
from pandas import DataFrame
from os.path import expanduser as ospath



def initfracodes():
    global fra_codes
    fra_codes = ('A','B','C','D','E','F','G','H','I','J','K','1')    

def initswapdefaults():
    global swap_defaults
    swap_defaults={}
    swap_defaults['USD']=(ql.JointCalendar(ql.UnitedStates(
        ql.UnitedStates.FederalReserve), ql.UnitedKingdom()),
        (ql.Period(ql.Semiannual), ql.Thirty360(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),
        (ql.Period(ql.Quarterly), ql.Actual360(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),2)
    swap_defaults['EUR']=(ql.TARGET(),
        (ql.Period(ql.Annual), ql.Thirty360(ql.Thirty360.BondBasis), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),
        (ql.Period(ql.Semiannual), ql.Actual360(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),2)
    swap_defaults['AUD']=(ql.Australia(),
        (ql.Period(ql.Semiannual), ql.Actual365Fixed(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),
        (ql.Period(ql.Semiannual), ql.Actual365Fixed(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),2)
    swap_defaults['GBP']=(ql.UnitedKingdom(),
        (ql.Period(ql.Semiannual), ql.Actual365Fixed(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),
        (ql.Period(ql.Semiannual), ql.Actual365Fixed(), 
         ql.ModifiedFollowing, ql.DateGeneration.Backward, False),0)
 
"""These mean reversions chosen to be in line with QuIC"""
def initccymrs():
    global ccy_meanreversions
    ccy_meanreversions={}
    ccy_meanreversions['USD']=0.01
    ccy_meanreversions['EUR']=0.01
    ccy_meanreversions['AUD']=0.01
    ccy_meanreversions['GBP']=0.01
 
"""These correlations chosen to be in line with QuIC"""
def initcorrs():
    global correlations
    correlations={}
    correlations[('AUD','USD')]=0.5891
    correlations[('AUD','AUDUSD')]=0.3634
    correlations[('USD','AUDUSD')]=0.0661
    correlations[('EUR','USD')]=0.6043
    correlations[('EUR','EURUSD')]=0.1467
    correlations[('USD','EURUSD')]=-0.0948
    correlations[('GBP','USD')]=0.4872
    correlations[('GBP','GBPUSD')]=0.2117
    correlations[('USD','GBPUSD')]=-0.054
    

"""Note that we build curves corresponding to the standard swaption contracts"""
def buildcurve(ccy,date):
    """Read BBG style curve data, assemble helpers and build a pair of OIS and 
    Libor curves. Input date is today date (a ql date)"""
    initfracodes()
    ois_helpers=[]
    libor_helpers=[]
    fut_date=date
    spread = ql.QuoteHandle(ql.SimpleQuote(0.0))
    period = ql.Period()
    if ccy == "USD":
        ois = ql.Sofr()
        libor = ql.USDLibor(ql.Period('3M'))
        calendar=ql.UnitedStates(ql.UnitedStates.FederalReserve)
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="USD SOFR")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '1D':
                (ois_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
            else:
                (ois_helpers.append(ql.OISRateHelper(2,ql.Period(tenor),ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
        ois_curve = ql.PiecewiseFlatForward(0, calendar, ois_helpers, ql.Actual365Fixed())
        ois_curve.enableExtrapolation()
        ois_curve = ql.YieldTermStructureHandle(ois_curve)
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="USD LIBOR")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '3M':
                (libor_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), libor)))
            elif tenor[0:2] == 'ED':
                fut_date=ql.IMM.nextDate(fut_date)
                (libor_helpers.append(ql.FuturesRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(100 - rate)), fut_date, libor)))
            else:
                (libor_helpers.append(ql.SwapRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ql.Period(tenor), calendar, 
                    ql.Quarterly, ql.ModifiedFollowing, ql.Thirty360(), libor,
                    spread,period,ois_curve)))
        libor_curve = ql.PiecewiseFlatForward(0, calendar, libor_helpers, ql.Actual365Fixed())
        libor_curve.enableExtrapolation()
        libor_curve= ql.YieldTermStructureHandle(libor_curve)
        libor = ql.USDLibor(ql.Period('3M'),libor_curve)
        return (ois_curve,libor_curve,libor)
    elif ccy == "EUR":
        ois = ql.Eonia()
        libor = ql.Euribor6M()
        calendar=ql.TARGET()
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="EUR ESTR")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '1D':
                (ois_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
            else:
                (ois_helpers.append(ql.OISRateHelper(2,ql.Period(tenor),ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
        ois_curve = ql.PiecewiseFlatForward(0, calendar, ois_helpers, ql.Actual365Fixed())
        ois_curve.enableExtrapolation()
        ois_curve = ql.YieldTermStructureHandle(ois_curve)
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="EURIBOR")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '6M':
                (libor_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), libor)))
            elif tenor[0:5] == 'EUFR0':
                fra_month=fra_codes.index(tenor[5])+1
                (libor_helpers.append(ql.FraRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), fra_month, libor)))
            else:
                (libor_helpers.append(ql.SwapRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ql.Period(tenor), calendar, 
                    ql.Annual, ql.ModifiedFollowing, ql.Thirty360(ql.Thirty360.BondBasis), libor,
                    spread,period,ois_curve)))
        libor_curve = ql.PiecewiseFlatForward(0, calendar, libor_helpers, ql.Actual365Fixed())
        libor_curve.enableExtrapolation()
        libor_curve= ql.YieldTermStructureHandle(libor_curve)
        libor = ql.Euribor6M(libor_curve)
        return (ois_curve,libor_curve,libor)
    elif ccy == "AUD":
        ois = ql.Aonia()
        libor = ql.AUDLibor(ql.Period('6M'))
        calendar=ql.Australia()
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="AUD OIS")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '1D':
                (ois_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
            else:
                (ois_helpers.append(ql.OISRateHelper(1,ql.Period(tenor),ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
        ois_curve = ql.PiecewiseFlatForward(0, calendar, ois_helpers, ql.Actual365Fixed())
        ois_curve.enableExtrapolation()
        ois_curve = ql.YieldTermStructureHandle(ois_curve)
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="AUD 6M")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '6M':
                (libor_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), libor)))
            else:
                (libor_helpers.append(ql.SwapRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ql.Period(tenor), calendar, 
                    ql.Semiannual, ql.ModifiedFollowing, ql.Actual365Fixed(), libor,
                    spread,period,ois_curve)))
        libor_curve = ql.PiecewiseFlatForward(0, calendar, libor_helpers, ql.Actual365Fixed())
        libor_curve.enableExtrapolation()
        libor_curve= ql.YieldTermStructureHandle(libor_curve)
        libor = ql.AUDLibor(ql.Period('6M'),libor_curve)
        return (ois_curve,libor_curve,libor)
    elif ccy == "GBP":
        ois = ql.Sonia()
        libor = ql.GBPLibor(ql.Period('6M'))
        calendar=ql.UnitedKingdom()
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="GBP SONIA")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '1D':
                (ois_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
            else:
                (ois_helpers.append(ql.OISRateHelper(0,ql.Period(tenor),ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ois)))
        ois_curve = ql.PiecewiseFlatForward(0, calendar, ois_helpers, ql.Actual365Fixed())
        ois_curve.enableExtrapolation()
        ois_curve = ql.YieldTermStructureHandle(ois_curve)
        df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="GBP 6M")
        for i in df.index:
            tenor = df.at[i,'Term']
            tenor = tenor.replace(" ","")
            tenor = tenor[:-1]
            rate = df.at[i,'Market Rate']
            if tenor == '6M':
                (libor_helpers.append(ql.DepositRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), libor)))
            elif tenor[0:5] == 'BPFR0':
                fra_month=fra_codes.index(tenor[5])+1
                (libor_helpers.append(ql.FraRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), fra_month, libor)))
            else:
                (libor_helpers.append(ql.SwapRateHelper(ql.QuoteHandle(
                    ql.SimpleQuote(rate / 100)), ql.Period(tenor), calendar, 
                    ql.Semiannual, ql.ModifiedFollowing, ql.Actual365Fixed(), libor,
                    spread,period,ois_curve)))
        libor_curve = ql.PiecewiseFlatForward(0, calendar, libor_helpers, ql.Actual365Fixed())
        libor_curve.enableExtrapolation()
        libor_curve= ql.YieldTermStructureHandle(libor_curve)  
        libor=ql.GBPLibor(ql.Period('6M'),libor_curve)
        return (ois_curve,libor_curve,libor)
    else:
        print("No construction implemented for currency ", ccy)

"""Function to set up vanilla libor swap --- used for testing curves."""
def vanillaiborswap(ccy,date,maturity,notional,rate,payreceive,index):
    calendar = swap_defaults[ccy][0]
    fixed_freq = swap_defaults[ccy][1][0]
    fixed_count = swap_defaults[ccy][1][1]
    fixed_busdayadj = swap_defaults[ccy][1][2]
    fixed_rollconv = swap_defaults[ccy][1][3]
    fixed_eom = swap_defaults[ccy][1][4]
    float_freq = swap_defaults[ccy][2][0]
    float_count = swap_defaults[ccy][2][1]
    float_busdayadj = swap_defaults[ccy][2][2]
    float_rollconv = swap_defaults[ccy][2][3]
    float_eom = swap_defaults[ccy][2][4]
    tpdays = swap_defaults[ccy][3]
    effective_date = calendar.advance(date,tpdays,ql.Days)
    fixed_schedule = ql.Schedule(effective_date, maturity, fixed_freq, calendar, 
                                 fixed_busdayadj, fixed_busdayadj, 
                                 fixed_rollconv, fixed_eom)
    float_schedule = ql.Schedule(effective_date, maturity, float_freq, calendar, 
                                 float_busdayadj, float_busdayadj, 
                                 float_rollconv, float_eom)
    
    fixed_leg_daycount = fixed_count
    float_leg_daycount = float_count

    fixed_rate = rate

    float_spread = 0

    swap = ql.VanillaSwap(payreceive, notional, fixed_schedule, fixed_rate, fixed_leg_daycount, float_schedule, index, float_spread, float_leg_daycount)
    return swap

"""Value swap"""
def priceswap(swap,disc_curve):
    swap_engine = ql.DiscountingSwapEngine(disc_curve)
    swap.setPricingEngine(swap_engine)
    
    npv =swap.NPV()
    return npv

"""More detail on individual swap legs"""
def swaplegsdetail(swap,disc_curve):
    notional = swap.nominal()
    # Fixed leg cash flows
    fixed_leg = swap.fixedLeg()
    cash_flows = list(map(ql.as_coupon, fixed_leg))
    dates = [datetime.datetime(x.date().year(), x.date().month(), x.date().dayOfMonth()) for x in cash_flows]
    amounts = [x.amount() for x in cash_flows]
    discounts = [disc_curve.discount(x.date()) for x in cash_flows]
    df = pd.DataFrame(zip(amounts, discounts), index=dates, columns=['Amount', 'Discount'])
    print(df)
    print('Fixed Leg NPV: {}'.format(sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1]))
    print('Fixed Leg Premium: {}'.format((sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1])/notional * 100))

    # Float leg cash flows
    float_leg = swap.floatingLeg()
    cash_flows = list(map(ql.as_coupon, float_leg))
    dates = [datetime.datetime(x.date().year(), x.date().month(), x.date().dayOfMonth()) for x in cash_flows]
    amounts = [x.amount() for x in cash_flows]
    discounts = [disc_curve.discount(x.date()) for x in cash_flows]
    df = pd.DataFrame(zip(amounts, discounts), index=dates, columns=['Amount', 'Discount'])
    print(df)
    print('Float Leg NPV: {}'.format(sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1]))
    print('Float Leg Premium: {}'.format((sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1])/notional * 100))


"""Set up swaptions to calibrate to --- not very precise about swaption expiry here... 
should probably try to do this more precisely."""
def create_swaption_helpers(ccy, today, index, ois, engine):
    swaptions = []
    exercise_dates = []
    mkt_vols = []
    fixed_leg_tenor = swap_defaults[ccy][1][0]
    fixed_leg_daycounter = swap_defaults[ccy][1][1]
    float_leg_daycounter = swap_defaults[ccy][2][1]
    df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="SwaptionData")
    curve=index.forwardingTermStructure()
    volfactors=[]
    for i in df.index:
        exptenor = df.at[i,'ExpTenor']
        expiry=exptenor.partition('x')[0]
        tenor=exptenor.partition('x')[2]
        expdate = today + ql.Period(expiry) 
        swapmat = expdate + ql.Period(tenor)
       
        vol = df.at[i,ccy]/10000
        mkt_vols.append(vol)
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(vol))
        helper = ql.SwaptionHelper(ql.Period(expiry),
                                   ql.Period(tenor),
                                   vol_handle,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   float_leg_daycounter,
                                   ois,
                                   ql.BlackCalibrationHelper.RelativePriceError,
                                   ql.nullDouble(),
                                   1.0,
                                   ql.Normal
                                   )
        
        helper.setPricingEngine(engine)
        swaptions.append(helper)
        swaption = helper.swaption()
        expiry = swaption.exercise().date(0)
        exercise_dates.append(expiry)
    return (swaptions,exercise_dates,mkt_vols)
    
def calibration_report(swaptions, vols):
    columns = ["Model Price", "Market Price", "Implied Vol", "Market Vol",
                       "Rel Error Price", "Rel Error Vols"]
    report_data = []
    cum_err = 0.0
    cum_err2 = 0.0
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = vols[i]
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error*rel_error
        cum_err2 += rel_error2*rel_error2
        report_data.append((model_price, black_price, implied_vol, market_vol, rel_error, rel_error2)) 
        
    print("Cumulative Error Price: %7.5f" % math.sqrt(cum_err))

    print("Cumulative Error Vols : %7.5f" % math.sqrt(cum_err2)) 
    
    return DataFrame(report_data, columns=columns,
                             index=['']*len(report_data))    

"""Two different models for HW - flat params and term stucture. For flat params 
calibrating both mr and vol is default."""        
def HWFlatCalibrate(model,swaptions,vols):
    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 1000, 1e-8, 1e-8, 1e-8)
    model.calibrate(swaptions, optimization_method, end_criteria)
    a, sigma = model.params()
    report = calibration_report(swaptions, vols)
    return (a,sigma,report)

"""Two different models for HW - flat params and term stucture. For flat params 
calibrating both mr and vol is default. This function for calibrating just vol."""    
def HWFlatCalibrateConstrained(model,swaptions,vols,flags):
    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 1000, 1e-8, 1e-8, 1e-8)
    model.calibrate(swaptions, optimization_method, end_criteria,ql.NoConstraint(),[],flags)
    a, sigma = model.params()
    report = calibration_report(swaptions, vols)
    return (a,sigma,report)

"""This is for calibration of full term structure HW."""
def GSRCalibrate(model,swaptions,vols):
    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 1000, 1e-8, 1e-8, 1e-8)
    model.calibrateVolatilitiesIterative(swaptions, optimization_method, end_criteria)
    sigmas = model.volatility()
    report = calibration_report(swaptions, vols)
    return (sigmas,report)


"""Helper functions for the calibration of fx spot vols. These are the functions
used for summing up ir variances and ir-ir, fx-ir covariances."""

def phi(vols,mr,times,a,b,n):
    x=0  
    for i in range(a,b,1):
        tin = times[n] - times[i]
        ti1n = times[n] - times[i+1]
        ti1i = times[i+1] - times[i]
        if mr == 0:           
            x += (1/2) * vols[i] * ( tin*2 - ti1n**2 )
        else:
            x += (vols[i]/mr) * ( ti1i 
                             - (1/mr) * ( exp(-mr * ti1n) - exp(-mr * tin)) )
            
    return x

def psi(vols,mr,times,a,b,n):
    x=0  
    for i in range(a,b,1):
        tin = times[n] - times[i]
        ti1n = times[n] - times[i+1]
        ti1i = times[i+1] - times[i]
        if mr == 0:           
            x += (1/3) * vols[i]**2 * ( tin**3 - ti1n**3 )
        else:
            x += (vols[i]**2/mr**2) * ( ti1i 
                             - (2/mr) * ( exp(-mr * ti1n) - exp(-mr * tin)) 
                             + (1/(2*mr)) * ( exp(-2 * mr * ti1n) - exp(-2 *mr * tin)) )            
    return x    

def xsi(vols1,mr1,vols2,mr2,times,a,b,n):
    x=0
    mr12=mr1+mr2
    for i in range(a,b,1):
        tin = times[n] - times[i]
        ti1n = times[n] - times[i+1]
        ti1i = times[i+1] - times[i]
        if mr1 == 0 and mr2 == 0:
            x += (1/3) * vols1[i] * vols2[i] * ( tin**3 - ti1n**3 )
        elif mr1 == 0 and mr2 != 0:
            x += (vols1[i] * vols2[i] / mr2) * ( (1/2) * (tin**2 - ti1n**2)
                                    + (1/mr2**2) * ( (1 + mr2 * tin) * exp(-mr2 * tin) 
                                                    - (1 + mr2 * ti1n) * exp(-mr2 * ti1n)))
        elif mr1 != 0 and mr2 == 0:
            x += (vols1[i] * vols2[i] / mr1) * ( (1/2) * (tin**2 - ti1n**2)
                                    + (1/mr1**2) * ( (1 + mr1 * tin) * exp(-mr1 * tin) 
                                                    - (1 + mr1 * ti1n) * exp(-mr1 * ti1n))) 
        else:
            x += (vols1[i] * vols2[i] / (mr1 * mr2)) * ( ti1i 
                                            - (1/mr1) * ( exp(-mr1 * ti1n) - exp(-mr1 * tin))
                                            - (1/mr2) * ( exp(-mr2 * ti1n) - exp(-mr2 * tin))
                                            + (1/mr12) * ( exp(-mr12 * ti1n) - exp(-mr12 * tin)))
    return x

"""This is the function to calibrate fx spot vol given fx term vols and calibrated
ir vols. It also extends the fx term vol curve based on holding the fx spot vol
fixed at its last calibrated level."""  
def calibratefxspotvol(und_vols,acc_vols,und_mr,acc_mr,fx_vols,
                       und_fx_corr,acc_fx_corr,und_acc_corr,times,segments):
    
    spot_vols=[]
    ts = times.copy()
    ts.insert(0,0)
    numvols = len(fx_vols)
    new_fx_vols=fx_vols.copy()
    spotvar=0
    
    for i in range(segments):
        fxvar = fx_vols[i] * fx_vols[i] * ts[i+1]
        undvar = psi(und_vols,und_mr,ts,0,i+1,i+1)
        accvar = psi(acc_vols,acc_mr,ts,0,i+1,i+1)
        undacccovar = und_acc_corr * xsi(und_vols,und_mr,acc_vols,acc_mr,ts,0,i+1,i+1)
        
        cadd=0
        for j in range(i):
            cadd += 2 * spot_vols[j] * (acc_fx_corr * phi(acc_vols,acc_mr,ts,j,j+1,i+1)
                                     - und_fx_corr * phi(und_vols,und_mr,ts,j,j+1,i+1))
        cadd += spotvar
            
        A = ts[i+1] - ts[i]
        B = (2 * acc_fx_corr * phi(acc_vols,acc_mr,ts,i,i+1,i+1)
             - 2 * und_fx_corr * phi(und_vols,und_mr,ts,i,i+1,i+1))
        
        C = (undvar + accvar - 2 * undacccovar - fxvar + cadd)
        
        
        disc = (B**2) - 4 * A * C
        
        vol = (- B + math.sqrt(disc)) / (2 * A)
    
        spot_vols.append(vol)
        
        
        spotvar += spot_vols[i] * spot_vols[i] * (ts[i+1] - ts[i])
        
        accfxcovar=0
        undfxcovar=0
        for j in range(i+1):
            accfxcovar += acc_fx_corr * spot_vols[j] * phi(acc_vols,acc_mr,ts,j,j+1,i+1)
            undfxcovar += und_fx_corr * spot_vols[j] * phi(und_vols,und_mr,ts,j,j+1,i+1)
        
        
        check = fxvar - spotvar - undvar - accvar - 2 * accfxcovar + 2 * undfxcovar + 2 * undacccovar
        
        if abs(check) < 0.000001:
            print("pass")
  
    for i in range(segments, numvols,1):
        spot_vols.append(vol)
        spotvar += spot_vols[i] * spot_vols[i] * (ts[i+1] - ts[i])
        undvar = psi(und_vols,und_mr,ts,0,i+1,i+1)
        accvar = psi(acc_vols,acc_mr,ts,0,i+1,i+1)
        undacccovar = und_acc_corr * xsi(und_vols,und_mr,acc_vols,acc_mr,ts,0,i+1,i+1)
        accfxcovar=0
        undfxcovar=0
        for j in range(i+1):
            accfxcovar += acc_fx_corr * spot_vols[j] * phi(acc_vols,acc_mr,ts,j,j+1,i+1)
            undfxcovar += und_fx_corr * spot_vols[j] * phi(und_vols,und_mr,ts,j,j+1,i+1)
        
        fxvar = spotvar + undvar + accvar + 2 * accfxcovar - 2 * undfxcovar - 2 * undacccovar
        fxvol = math.sqrt(fxvar/ts[i+1])
        new_fx_vols[i]=fxvol
    
    # spotvar=0.0
    # test_vols=[]
    # for i in range(numvols):
    #     spotvar += spot_vols[i] * spot_vols[i] * (ts[i+1] - ts[i])
    #     undvar = psi(und_vols,und_mr,ts,0,i+1,i+1)
    #     accvar = psi(acc_vols,acc_mr,ts,0,i+1,i+1)
    #     undacccovar = und_acc_corr * xsi(und_vols,und_mr,acc_vols,acc_mr,ts,0,i+1,i+1)
    #     accfxcovar=0
    #     undfxcovar=0
    #     for j in range(i+1):
    #         accfxcovar += acc_fx_corr * spot_vols[j] * phi(acc_vols,acc_mr,ts,j,j+1,i+1)
    #         undfxcovar += und_fx_corr * spot_vols[j] * phi(und_vols,und_mr,ts,j,j+1,i+1)
        
    #     fxvar = spotvar + undvar + accvar + 2 * accfxcovar - 2 * undfxcovar - 2 * undacccovar
    #     fxvol = math.sqrt(fxvar/ts[i+1])
    #     test_vols.append(fxvol)
        
    return (spot_vols,new_fx_vols)

"""Reads in the fx term vol curve. Note that at present there is an assumption that
the ir vol maturities and fx vol maturities coincide. Would be nice to decouple this."""
def readfxvolcurve(und,acc):
    ccypair = und+acc
    volcurve = {}
    df=pd.read_excel(ospath('~/Documents/Python/Curves.xlsx'),sheet_name="FX Vols")
    for i in df.index:
        volcurve[df.at[i,'Tenor']]=df.at[i,ccypair]/100
        
    return volcurve


def main():
    
    today_date = ql.Date(8, 10, 2021)
    ql.Settings.instance().evaluationDate = today_date
    
    """Initialise some ccy convention static data"""
    initswapdefaults()
    """Initialise some correlation static data"""   
    initcorrs()
    initccymrs()
    """Flag to value to test swap to check curve setup"""
    test_curves = True
    
    """Can either use mr from flat HW calib or set by hand"""    
    use_flat_calib_mr = False
    
    """Set currency pair to investigate"""
    und_ccy="EUR"
    acc_ccy="USD"
    
    """Build yield curves - data is loaded from pre-populated spreadsheet containing 
    Bloomberg default instruments"""
    und_curves = buildcurve(und_ccy,today_date)
    acc_curves = buildcurve(acc_ccy,today_date)
    
    if test_curves ==  True:
        #Test swap is 5y 
        acc_maturity=swap_defaults[acc_ccy][0].advance(today_date,ql.Period('5Y'))
        und_maturity=swap_defaults[und_ccy][0].advance(today_date,ql.Period('5Y'))
        notional = 10000000
        payreceive = ql.VanillaSwap.Receiver
        
        acc_rate = 1.01865/100        
        acc_swap = vanillaiborswap(acc_ccy,today_date,acc_maturity,notional,
                                   acc_rate,payreceive,acc_curves[2])
        acc_test_npv = priceswap(acc_swap,acc_curves[0])
        
        print("Acc swap npv: ", acc_test_npv)
        
        und_rate = -0.20675/100
        und_swap = vanillaiborswap(und_ccy,today_date,und_maturity,notional,
                                   und_rate,payreceive,und_curves[2])
        und_test_npv = priceswap(und_swap,und_curves[0])
        
        print("Und swap npv: ", und_test_npv)
        
        # # Fixed leg cash flows
        # leg = und_swap.fixedLeg()
        # cash_flows = list(map(ql.as_coupon, leg))
        # dates = [datetime.datetime(x.date().year(), x.date().month(), x.date().dayOfMonth()) for x in cash_flows]
        # amounts = [x.amount() for x in cash_flows]
        # discounts = [und_curves[0].discount(x.date()) for x in cash_flows]
        # df = pd.DataFrame(zip(amounts, discounts), index=dates, columns=['Amount', 'Discount'])
        # print(df)
        # print('Fixed Leg NPV: {}'.format(sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1]))
        # print('Fixed Leg Premium: {}'.format((sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1])/notional * 100))
        
        # # Float leg cash flows
        # leg = und_swap.floatingLeg()
        # cash_flows = list(map(ql.as_coupon, leg))
        # dates = [datetime.datetime(x.date().year(), x.date().month(), x.date().dayOfMonth()) for x in cash_flows]
        # amounts = [x.amount() for x in cash_flows]
        # discounts = [und_curves[0].discount(x.date()) for x in cash_flows]
        # df = pd.DataFrame(zip(amounts, discounts), index=dates, columns=['Amount', 'Discount'])
        # print(df)
        # print('Float Leg NPV: {}'.format(sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1]))
        # print('Float Leg Premium: {}'.format((sum(df['Amount']*df['Discount'])+notional*df['Discount'][-1])/notional * 100))
        
    """Flat HW calibration - both a flat mr and flat vol are calibrated"""
    und_hw_flat = ql.HullWhite(und_curves[1])
    #und_hw_flat_fixedmr = ql.HullWhite(und_curves[1],0.00001,0.001)
    und_hw_flat_engine = ql.JamshidianSwaptionEngine(und_hw_flat)
    #und_hw_flat_engine_fixedmr = ql.JamshidianSwaptionEngine(und_hw_flat_fixedmr)
    
    acc_hw_flat = ql.HullWhite(acc_curves[1])
    #acc_hw_flat_fixedmr = ql.HullWhite(acc_curves[1],0.00001,0.001)
    acc_hw_flat_engine = ql.JamshidianSwaptionEngine(acc_hw_flat)
    #acc_hw_flat_engine_fixedmr = ql.JamshidianSwaptionEngine(acc_hw_flat_fixedmr)
    
    und_swaption_data = create_swaption_helpers(und_ccy, today_date, und_curves[2], und_curves[0], und_hw_flat_engine)
    acc_swaption_data = create_swaption_helpers(acc_ccy, today_date, acc_curves[2], acc_curves[0], acc_hw_flat_engine)
    
    #single_und_swaption = [und_swaption_data[0][5].SetPricingEngine(und_hw_flat_engine_fixedmr)]
    #single_acc_swaption = [acc_swaption_data[0][5].SetPricingEngine(acc_hw_flat_engine_fixedmr)]
    #single_und_vol = [und_swaption_data[2][5]]
    #single_acc_vol = [acc_swaption_data[2][5]]
    
    #und_flat_calib_fixedmr = HWFlatCalibrateConstrained(und_hw_flat_fixedmr,single_und_swaption,single_und_vol,[True,False])
    #acc_flat_calib_fixedmr = HWFlatCalibrateConstrained(acc_hw_flat_fixedmr,single_acc_swaption,single_acc_vol,[True,False])
    
    #fixed_und_mr = und_flat_calib_fixedmr[0]
    #fixed_und_vol = und_flat_calib_fixedmr[1]
    
    und_flat_calib = HWFlatCalibrate(und_hw_flat,und_swaption_data[0],und_swaption_data[2])
    acc_flat_calib = HWFlatCalibrate(acc_hw_flat,acc_swaption_data[0],acc_swaption_data[2])
    
   
    und_flat_mr = und_flat_calib[0]
    und_flat_vol = und_flat_calib[1]
    und_flat_report = und_flat_calib[2]
    
    acc_flat_mr = acc_flat_calib[0]
    acc_flat_vol = acc_flat_calib[1]
    acc_flat_report = acc_flat_calib[2]
    
    print(und_ccy, "a = %6.5f, sigma = %6.5f" % (und_flat_mr, und_flat_vol))
    print(acc_ccy, "a = %6.5f, sigma = %6.5f" % (acc_flat_mr, acc_flat_vol))
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(und_flat_report)
        
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(acc_flat_report)
    
    exercise_dates = und_swaption_data[1]
    
    und_sigmas = [ql.QuoteHandle(ql.SimpleQuote(und_flat_vol)) for x in range(1, len(exercise_dates)+2)]
    acc_sigmas = [ql.QuoteHandle(ql.SimpleQuote(acc_flat_vol)) for x in range(1, len(exercise_dates)+2)]
    
    """Term HW calibration - flat mr is an input, term structure of piecewise constant vols are calibrated"""
    if und_flat_mr<0.0001:
        und_flat_mr=0.0
    if acc_flat_mr<0.0001:
        acc_flat_mr=0.0
    
    if use_flat_calib_mr == False:        
        und_flat_mr=ccy_meanreversions[und_ccy]
        acc_flat_mr=ccy_meanreversions[acc_ccy]
                
    und_reversion = [ql.QuoteHandle(ql.SimpleQuote(und_flat_mr))]
    acc_reversion = [ql.QuoteHandle(ql.SimpleQuote(acc_flat_mr))]
    und_gsr_model = ql.Gsr(und_curves[1], exercise_dates, und_sigmas, und_reversion)
    acc_gsr_model = ql.Gsr(acc_curves[1], exercise_dates, acc_sigmas, acc_reversion)
    
    #und_gsr_engine = ql.Gaussian1dJamshidianSwaptionEngine(und_gsr_model)
    #acc_gsr_engine = ql.Gaussian1dJamshidianSwaptionEngine(acc_gsr_model)
    und_gsr_engine = ql.Gaussian1dSwaptionEngine(und_gsr_model, 64, 7.0, True, False, und_curves[0])
    acc_gsr_engine = ql.Gaussian1dSwaptionEngine(acc_gsr_model, 64, 7.0, True, False, acc_curves[0])
    
    und_swaption_data = create_swaption_helpers(und_ccy, today_date, und_curves[2], und_curves[0], und_gsr_engine)
    acc_swaption_data = create_swaption_helpers(acc_ccy, today_date, acc_curves[2], acc_curves[0], acc_gsr_engine)
    
    und_gsr_calib = GSRCalibrate(und_gsr_model,und_swaption_data[0],und_swaption_data[2])
    acc_gsr_calib = GSRCalibrate(acc_gsr_model,acc_swaption_data[0],acc_swaption_data[2])
    
    """Some test code --- leaving here since provides examples of some useful
    ql functionality."""
    # num_swaptions = len(und_swaption_data[0])
    # undfacs = []
    # for i in range(num_swaptions):
    #     swaption = und_swaption_data[0][i].swaption()
    #     swap = swaption.underlyingSwap()        
    #     exp = swaption.exercise().date(0)
    #     end = swap.maturityDate()
    #     swaprate = swap.fairRate()
    #     dfexp = und_curves[1].discount(exp)
    #     dfend = und_curves[1].discount(end)
    #     time = ql.Actual360().yearFraction(exp,end)
    #     factor = swaprate * (dfexp / (dfexp - dfend)) * time
    #     undfacs.append(factor)
        
    # num_swaptions = len(acc_swaption_data[0])
    # accfacs = []
    # for i in range(num_swaptions):
    #     swaption = acc_swaption_data[0][i].swaption()
    #     swap = swaption.underlyingSwap()        
    #     exp = swaption.exercise().date(0)
    #     end = swap.maturityDate()
    #     swaprate = swap.fairRate()
    #     dfexp = acc_curves[1].discount(exp)
    #     dfend = acc_curves[1].discount(end)
    #     time = ql.Actual360().yearFraction(exp,end)
    #     factor = swaprate * (dfexp / (dfexp - dfend)) * time
    #     accfacs.append(factor)
        
    und_vols=und_gsr_calib[0]
    acc_vols=acc_gsr_calib[0]
    
    und_report = und_gsr_calib[1]
    acc_report = acc_gsr_calib[1]
    
    print(und_vols)
    #print(undfacs)
    print(acc_vols)
    #print(accfacs)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(und_report)
        
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(acc_report)
    
    """Read in ATM curve from pre-populated spreadsheet. Bloomberg vols"""
    fxvols_dict = readfxvolcurve(und_ccy,acc_ccy)
    times = []
    for date in exercise_dates:
        times.append(ql.Actual365Fixed().yearFraction(today_date,date))
            
    fxvols = list(fxvols_dict.values())
    
    """Calibrate spot vols to chosen tenor"""
    final_tenor = '5Y'
    tenors = list(fxvols_dict.keys())
    intervals = tenors.index(final_tenor) + 1
    
    testvols=[]
    testvols.extend(0 for i in range(len(und_vols)))
    
    spotvols, newfxvols = calibratefxspotvol(und_vols,acc_vols,
                                      und_flat_mr,acc_flat_mr,fxvols, 
                                      correlations[(und_ccy,und_ccy+acc_ccy)], 
                                      correlations[(acc_ccy,und_ccy+acc_ccy)],
                                      correlations[(und_ccy,acc_ccy)],times,intervals)
    

    
    
    yr1_index=tenors.index('1Y')
    longvolcurve=[]
    for i in range(10):
        longvolcurve.append(newfxvols[yr1_index+i])
    yr10_index=tenors.index('10Y')
    yr15_index=tenors.index('15Y')
    yr20_index=tenors.index('20Y')
    yr25_index=tenors.index('25Y')
    yr30_index=tenors.index('30Y')
    
    var10=newfxvols[yr10_index]**2 * 10
    var15=newfxvols[yr15_index]**2 * 15
    var20=newfxvols[yr20_index]**2 * 20
    var25=newfxvols[yr25_index]**2 * 25
    var30=newfxvols[yr30_index]**2 * 30
    
    time = 10
    for i in range(5):
        time=time+i+1
        var=var10+(var15-var10)/5 * (i+1)
        vol = math.sqrt(var/time)
        longvolcurve.append(vol)
    for i in range(5):
        time=time+i+1
        var=var10+(var15-var10)/5 * (i+1)
        vol = math.sqrt(var/(time+i+1))
        longvolcurve.append(vol)
    for i in range(5):
        var=var10+(var15-var10)/5 * (i+1)
        vol = math.sqrt(var/(time+i+1))
        longvolcurve.append(vol)
    for i in range(5):
        var=var10+(var15-var10)/5 * (i+1)
        vol = math.sqrt(var/(time+i+1))
        longvolcurve.append(vol)
    
    var1520=newfxvols[yr20_index]**2 * 20 - newfxvols[yr15_index]**2 * 15
    var2025=newfxvols[yr25_index]**2 * 25 - newfxvols[yr20_index]**2 * 20
    var2530=newfxvols[yr30_index]**2 * 30 - newfxvols[yr25_index]**2 * 25
    
    

if __name__ == "__main__":
    main()
