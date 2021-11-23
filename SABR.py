import xlrd
import math
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from os.path import expanduser as ospath
from scipy.optimize import minimize
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

def SVI_imp_var(y,a,b,rho,m,gamma):
    """Gatheral's 'raw' SVI"""
    return a + b * ( rho * (y - m) + math.sqrt((y - m)**2 + gamma**2))

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
    
        
def SABR_Objective(SABR_params,vols,strikes,F,t):
   obj=0
   for i in range(len(strikes)):
       diff=math.sqrt(SABR_imp_var(SABR_params[0],SABR_params[1],SABR_params[2],F,
       strikes[i],t)/t) - vols[i]
       obj+=diff * diff 
   return math.sqrt(obj)
     
def calibrate_SABR(vols,strikes,F,t,params):    
    bnds=((0.00001,None),(-0.99999,0.99999),(0.00001,None))
    result=minimize(SABR_Objective,params,(vols,strikes,F,t),
                    method='L-BFGS-B',bounds=bnds)
    sabr_params=result.x
    error=0
    sabr_vols=[]
    for i in range(len(strikes)):
        sabr_vols.append(math.sqrt(SABR_imp_var(sabr_params[0],sabr_params[1],
                                                sabr_params[2],F,strikes[i],t) / t))
        vol_diff=sabr_vols[i]-vols[i]
        error+=vol_diff * vol_diff
    error=math.sqrt(error)
    return (sabr_params,sabr_vols,error)
        
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

def var_from_call(price_target,F,K,guess=0.01):
    MAX_ITER=200
    PRECISION=1.0e-6
    var=guess
    for i in range(0,MAX_ITER):
        price=BS_Call_Price(F,K,var)
        d2=(math.log(F/K)-0.5*var)/math.sqrt(var)
        deriv=K*stats.norm.pdf(d2)/2/math.sqrt(var)
        diff=price_target-price
        if (abs(diff)<PRECISION):
            return var
        var=var+diff/deriv
    return var
         
def var_from_put(price_target,F,K,guess=0.01):
    MAX_ITER=200
    PRECISION=1.0e-6
    var=guess
    for i in range(0,MAX_ITER):
        price=BS_Put_Price(F,K,var)
        d2=(math.log(F/K)-0.5*var)/math.sqrt(var)
        deriv=K*stats.norm.pdf(d2)/2/math.sqrt(var)
        diff=price_target-price
        if (abs(diff)<PRECISION):
            return var
        var=var+diff/deriv
    return var        

def pdf_BS(y,w):
    """Black-Scholes pdf at value y=logK/F. Working in log space means this 
    is a normal """
    d2=-(w**(-0.5)) * (y + 0.5*w)
    return (w**(-0.5)) * stats.norm.pdf(d2)

def pdf_SABR(x,F,params,vol_t):
    """SABR pdf at value y=logx/F. Calculated as second derivative of call price with respect
    to strikewith factor x to produce log space density"""
    dx=F/10000
    scnd_deriv=derivative(lambda x : BS_Call_Price(F,x,SABR_imp_var(params[0],params[1],params[2],F,x,vol_t)),x,dx,2)
    return scnd_deriv * x

def cdf_BS_int(y,w):
    """Black-Scholes cdf at value y (log space). Calculated as integral of BS pdf."""
    integral=integrate.quad(lambda t : pdf_BS(t,w),-inf,y)[0]
    return integral

def cdf_BS_exact(y,w):
    return stats.norm.cdf((w**(-0.5)) * (y + 0.5*w))

def cdf_SABR(x,F,params,vol_t):
    """SABR cdf at value x. Depends on forward F, SABR params params and
    expiry vol_t. Calculated as integral of SABR pdf."""
    integral=integrate.quad(lambda t : pdf_SABR(t,F,params,vol_t),-inf,math.log(x/F))[0]
    return integral    

def cdf_SSVI1(y,params):
    #integral=integrate.quad(lambda t : pdf_SSVI1(t,params),-5,y)[0]
    w=SSVI1_w(y,params[0],params[1],params[2], params[3])
    wp=SSVI1_wp(y,params[0],params[1],params[2], params[3])
    d2=-(w**(-0.5))*(y+0.5*w)
    Nd2 = stats.norm.cdf(d2)
    nd2 = stats.norm.pdf(d2)
    cdf=1.0 - Nd2 + 0.5 * (w**(-0.5)) * nd2 * wp
    return cdf

def diagnostic_fwd_bs(F,var):  
    """Diagnostic function which checks the error on the forward calculated by 
    integrating spot against the density"""
    fwd_error=F-F*integrate.quad(lambda t : math.exp(t)*pdf_BS(t,var),-inf,inf)[0]
    return fwd_error

def diagnostic_fwd_sabr(F,params,vol_t):  
    """Diagnostic function which checks the error on the forward calculated by 
    integrating spot against the density"""
    fwd_error=F-F*integrate.quad(lambda t : math.exp(t)*pdf_SABR(t,F,params,vol_t),
                               -inf,inf,limit=100)[0]
    return fwd_error

def pdf_gauss_copula_ssvi1(x1,x2,rho,params1,params2):
    uni_X=cdf_SSVI1(x1,params1)
    uni_Y=cdf_SSVI1(x2,params2)
    X=stats.norm.ppf(uni_X)
    Y=stats.norm.ppf(uni_Y)
    binorm=stats.multivariate_normal(mean=[0,0],cov=[[1,rho],[rho,1]])
    binormdens=binorm.pdf([X,Y]) 
    normdens1=stats.norm.pdf(X)
    normdens2=stats.norm.pdf(Y)
    cop_dens=binormdens / normdens1 / normdens2
    marg1=pdf_SSVI1(x1,params1)
    marg2=pdf_SSVI1(x2,params2)  
   
    result = cop_dens * marg1 * marg2
    return result

def pdf_cross_bs(k,rho,mean1,var1,mean2,var2):
    w = var1 + var2 - 2 * rho * math.sqrt(var1 * var2)
    int_halfrange=8 * math.sqrt(w)
    int_mid=-0.5*w
    lower=int_mid-int_halfrange
    upper=int_mid+int_halfrange
    binorm=stats.multivariate_normal(mean=[mean1,mean2],
                                     cov=[[var1,rho * math.sqrt(var1 * var2)],
                                          [rho * math.sqrt(var1 * var2),var2]])
    integral=integrate.quad(lambda t : math.exp(t) * binorm.pdf([k+t,t]),lower,upper)[0]
    return integral

def pdf_cross(k,rho,params1,params2):
    w=SSVI1_w(0,params2[0],params2[1],params2[2], params2[3])
    int_halfrange=8 * math.sqrt(w)
    int_mid=-0.5*w
    lower=int_mid-int_halfrange
    upper=int_mid+int_halfrange
    integral=integrate.quad(lambda t : math.exp(t) * pdf_gauss_copula_ssvi1(k+t,t,rho,params1,params2),lower,upper)[0]
    return integral

def main():
    today=date.today()
    qlDate = ql.Date(today.day,today.month,today.year)
    qlCalendar = ql.UnitedStates()
    qlDayCounter = ql.ActualActual()
    

    
    """Load in vol data and populate MktSmileData dictionary"""
    MktSmileData={}
    ccy_pairs=("EURUSD","GBPUSD","EURGBP")
    for ccypair in ccy_pairs:      
        df=pd.read_excel(ospath('~/Documents/Python/MktVolData.xlsm'),sheet_name=ccypair)
        for i in df.index:
            MktSmileData[(ccypair,df.at[i,'Tenor'])]=[((df.at[i,'Expiry'].date()
                                                         -today).days/365,),
                                                       (df.at[i,'Forward'],),
                                                       (df.at[i,'5P Strike'],df.at[i,'5P Vol']),
                                                       (df.at[i,'10P Strike'],df.at[i,'10P Vol']),
                                                       (df.at[i,'25P Strike'],df.at[i,'25P Vol']),
                                                       (df.at[i,'DN Strike'],df.at[i,'DN Vol']),
                                                       (df.at[i,'25C Strike'],df.at[i,'25C Vol']),                                                       
                                                       (df.at[i,'10C Strike'],df.at[i,'10C Vol']),                                                      
                                                       (df.at[i,'5C Strike'],df.at[i,'5C Vol'])]
            
    """For each ccypair and for each tenor calibrate SSVI params. There's some old 
    SABR code here --- to do: tidy up!"""
    params=[0.1,1.,0,0.25]#SSVI
    #params=[0.10,0.0,1.0]#SABR
    for key in MktSmileData:
        T=MktSmileData[key][0][0]
        F=MktSmileData[key][1][0]
        strikes=(MktSmileData[key][4][0],MktSmileData[key][5][0],MktSmileData[key][6][0])
        vols=(MktSmileData[key][4][1],MktSmileData[key][5][1],MktSmileData[key][6][1])
        #calibration=calibrate_SABR(vols,strikes,F,T,params)        
        calibration=calibrate_SSVI1(vols,strikes,F,T,params)
        params=calibration[0]
        vols=calibration[1]
        error=calibration[2]
        #fwd_error=diagnostic_fwd_sabr(MktSmileData[key][1][0],params,MktSmileData[key][0][0])
        MktSmileData[key].append(params)
        MktSmileData[key].append(vols)
        MktSmileData[key].append(error)
        #MktSmileData[key].append(fwd_error)   
    
    """Write calibration to file"""
    w = csv.writer(open("Calibration.csv", "w"))
    for key, val in MktSmileData.items():
        w.writerow([key, val])
    
    """Focus on specific tenor to analyse copula cross smile"""
    
    tenor = "1Y"
    cross = ccy_pairs[2]   
    key = (cross,tenor)
    t=MktSmileData[key][0][0]
    expiryDate = today + timedelta(t * 365)
    
    """Get forward of cross"""
    F=MktSmileData[key][1][0]
    
    """Get SSVI params and atms for each ccypair"""
    loc_params = []
    atms = []
    for ccypair in ccy_pairs:
        loc_params.append(MktSmileData[(ccypair,tenor)][9])
        atms.append(MktSmileData[(ccypair,tenor)][5][1])
        
    
        
    
   
        
    """Calculate 3 correlations to use in copula, mid constructed using triangle formula"""
    plus_minus = 0.05
    rho_mid = (atms[0]**2 + atms[1]**2 - atms[2]**2)/(2 * atms[0] * atms[1])
    rho_low = max(-0.999,rho_mid - plus_minus)
    rho_high = min(0.999,rho_mid + plus_minus)
    
    

    
    """Determine plus and minus infinity for numerical integrations"""
    """done using number of std_devs either side of mean"""
    num_stdevs = 8.0  
    var = SSVI1_w(0,loc_params[2][0],loc_params[2][1],loc_params[2][2], loc_params[2][3])
    int_halfrange = num_stdevs * math.sqrt(var)
    int_mid = -0.5*var
    lower = int_mid - int_halfrange
    upper = int_mid + int_halfrange
    

    
    """Construct dictionary of copula smile data. Keys will be strikes. With 
    the calculated vals, vars and vols for each of the correlations."""
    
    pks = []
    cks = []
    CopulaSmile = {}
    
    for i in range(2,6):
        pks.append(MktSmileData[key][i][0])
        CopulaSmile[MktSmileData[key][i][0]] = []
        
    for i in range(6,9):
        cks.append(MktSmileData[key][i][0])    
        CopulaSmile[MktSmileData[key][i][0]] = []
    
    """inital guess var"""
    w=atms[2]*atms[2]*t
    
    """Test using Black-Scholes joint density"""
    # strike=pks[-1]
    # lnKF=math.log(strike/F)
    # var1=atms[0]**2 * t
    # var2=atms[1]**2 * t

    # val = integrate.quad(lambda t : (strike-F*math.exp(t)) * 
    #                         pdf_cross_bs(t,rho_mid,-0.5 * var1,var1,-0.5 * var2,var2),lower,lnKF)[0]
    # var=var_from_put(val,F,strike,w)
    
    # check = w - var
    
    """Loop through correlations. In each case value option using copula density
    then solve for corresponding var and vol. (val,var,vol) tuple appended to 
    strike key of dictionary"""    
    timer_start=time.time()
    for rho in (rho_low,rho_mid,rho_high):
        for strike in CopulaSmile:
            lnKF=math.log(strike/F)
            if strike <= pks[-1]:
                val = integrate.quad(lambda t : (strike-F*math.exp(t)) * 
                            pdf_cross(t,rho,loc_params[0],loc_params[1]),lower,lnKF)[0]
                var=var_from_put(val,F,strike,w)
                
            else:
                 val = integrate.quad(lambda t : (F*math.exp(t)-strike) * 
                                      pdf_cross(t,rho,loc_params[0],loc_params[1]),lnKF,upper)[0]
                 var=var_from_call(val,F,strike,w)
                 
            vol=math.sqrt(var/t)
            CopulaSmile[strike].append((val,var,vol))

    """Just check the prices tie with solved vols - 25s and ats"""
    put_val_solved=BS_Put_Price(F, pks[2], CopulaSmile[pks[2]][1][1])
    atm_val_solved=BS_Put_Price(F, pks[3], CopulaSmile[pks[3]][1][1])
    call_val_solved=BS_Call_Price(F, cks[0], CopulaSmile[cks[0]][1][1])
    
    print("(Integrated value, BS from vol) for mid rho")
    print(CopulaSmile[pks[2]][1][0],put_val_solved)
    print(CopulaSmile[pks[3]][1][0],atm_val_solved)
    print(CopulaSmile[cks[0]][1][0],call_val_solved)

    """Compare copula vols with market vols"""
    print("")
    print("Bloomberg vols")    
    print(MktSmileData[key][2][1],MktSmileData[key][3][1],MktSmileData[key][4][1],
          MktSmileData[key][5][1],MktSmileData[key][6][1],MktSmileData[key][7][1],
          MktSmileData[key][8][1])
    print("")
    print("low rho",rho_low)
    for strike in CopulaSmile:
        print(CopulaSmile[strike][0][2], end=' ')
    print("")
    print("mid rho",rho_mid)
    for strike in CopulaSmile:
        print(CopulaSmile[strike][1][2], end=' ')
    print("")
    print("high rho",rho_high)
    for strike in CopulaSmile:
        print(CopulaSmile[strike][2][2], end=' ')
    
   
    """Use QuantLib to interpolate copula vols"""
    """Interpolation object is BlackVarianceSurface"""
    qlExpiry = [ql.Date(expiryDate.day,expiryDate.month,expiryDate.year)]
    qlVols = ql.Matrix(7,1)
    qlStrikes = []
    qlSmiles = []
    
    
    for strike in CopulaSmile:
        qlStrikes.append(strike)
    
    for j in range(3):
        for i, strike in enumerate(CopulaSmile):
            qlVols[i][0] = CopulaSmile[strike][j][2]
        
        smile = ql.BlackVarianceSurface(
            qlDate,qlCalendar,qlExpiry,qlStrikes,qlVols,qlDayCounter,
            ql.BlackVarianceSurface.ConstantExtrapolation,
            ql.BlackVarianceSurface.ConstantExtrapolation)
    
        smile.enableExtrapolation()
        smile.setInterpolation("bicubic")
        
        qlSmiles.append(smile)
    
    
    
    ks=np.linspace(qlStrikes[0],qlStrikes[-1],100)
    vols_SSVI=[]
    vols_copula1=[]
    vols_copula2=[]
    vols_copula3=[]
    
    for i in range(len(ks)):
        vols_SSVI.append(math.sqrt(SSVI1_imp_var(math.log(ks[i] / F),loc_params[2][0],loc_params[2][1],loc_params[2][2], loc_params[2][3])/t))
        vols_copula1.append(qlSmiles[0].blackVol(qlExpiry[0],ks[i]))
        vols_copula2.append(qlSmiles[1].blackVol(qlExpiry[0],ks[i]))
        vols_copula3.append(qlSmiles[2].blackVol(qlExpiry[0],ks[i]))
     
    plt.plot(ks,vols_SSVI)   
    plt.plot(qlStrikes,[MktSmileData[key][2][1],MktSmileData[key][3][1],MktSmileData[key][4][1],
          MktSmileData[key][5][1],MktSmileData[key][6][1],MktSmileData[key][7][1],
          MktSmileData[key][8][1]],'bo')
    plt.plot(ks,vols_copula1)
    plt.plot(ks,vols_copula2)
    plt.plot(ks,vols_copula3)
    
    plt.legend(["SSVI","BBG","Copula low","Copula mid","Copula high"])   
    plt.savefig('CopulaSmiles.pdf')
    plt.show()
    
    timer_taken = time.time()-timer_start    
    print("")
    print("This took", end=' ')
    print (timer_taken, end=' ')
    print(" to calculate.")
    
    #print(cop_int)
    #print(ssvi_int)
    #print(bs_int)
 
    
if __name__ == "__main__":
    main()
        