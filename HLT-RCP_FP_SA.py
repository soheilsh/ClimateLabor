import numpy as np
from scipy import optimize
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import Symbol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import bisect
import six 
from six.moves import zip
import csv
from numpy import genfromtxt
import xlsxwriter as xlwt
from matplotlib.ticker import FuncFormatter
import pandas as pd
# ========================================== parameters ========================================== #
global PMs, PMu, gammaN, mc
global tsN, tuN, N, Aa, Am, Da, Dm, Dr, Ar
# ========================================== parameters ========================================== #
T = 6                                               # Time horizon
nsc = 2                                             # number of scenarios
nrcp = 4                                            # number of RCPs
nreg = 10                                           # number of regions
alpha = 0.55/2                                      # Agricultural share in Consumption function
eps = 0.75                                          # Elasticity of substitution in Consumption function

N1980 = 17
Y2000 = [0] * 10
Iu0 = 4023.9
Is0 = 5431.7

coeffTmax = [12.4, 0.8798]
# ============================= RCP Scenarios, Periods, Regions ================================= # 
RCPName = ['RCP26', 'RCP45', 'RCP60', 'RCP85']
Tperiod = [2000, 2020, 2040, 2060, 2080, 2100]
RegName = ['Western Cape', 'Eastern Cape', 'Northern Cape', 'Free State', 'KwaZulu-Natal', 'North West', 'Gauteng', 'Mpumalanga', 'Limpopo', 'South Africa']
# ========================================== Damages =========================================== #
# D = g0 + g1 * T + g2 * T^2

# Agricultural parameters
g0a = -2.24
g1a = 0.308
g2a = -0.0073

# Manufacturing parameters
g0m = 0.3
g1m = 0.08
g2m = -0.0023

# ========================================== Variables =========================================== #                    

# == temperature == #
Temp = np.zeros((nreg, T, nrcp, nsc))                     # Mean Temperature
MaxTemp = np.zeros((nreg, T, nrcp, nsc))                  # Max Temperature

# == child-rearing time == #
gamma0 = 0.4                                              # Share of children's welbeing in Utility function of parents in 1980
gamma = np.zeros((nreg, T, nrcp, nsc))
gammaN = [0] * nreg

# == Age matrix == #
nu = np.zeros((nreg, T, nrcp, nsc))                       # number of unskilled children
ns = np.zeros((nreg, T, nrcp, nsc))                       # number of skilled children
L = np.zeros((nreg, T, nrcp, nsc))                        # Number of unskilled parents
H = np.zeros((nreg, T, nrcp, nsc))                        # Number of skilled parents
h = np.zeros((nreg, T, nrcp, nsc))                        # Ratio of skilled to unskilled labor h=H/L
hN = np.zeros((nreg, T, nrcp, nsc))                       # Ratio of skilled to unskilled children h=ns/nu
N = np.zeros((nreg, T, nrcp, nsc))                        # Adult population
Pop = np.zeros((nreg, T, nrcp, nsc))                      # total population
Pgr = np.zeros((nreg, T, nrcp, nsc))                      # population growth rate

# == Prices == #
pa = np.zeros((nreg, T, nrcp, nsc))                       # Pice of AgricuLtural good
pm = np.zeros((nreg, T, nrcp, nsc))                       # Pice of Manufacturing good
pr = np.zeros((nreg, T, nrcp, nsc))                       # Relative pice of Manufacturing to Agricultural goods

# == Wages == #
wu = np.zeros((nreg, T, nrcp, nsc))                       # Wage of unskilled labor
ws = np.zeros((nreg, T, nrcp, nsc))                       # Wage of skilled labor
wr = np.zeros((nreg, T, nrcp, nsc))                       # Wage ratio of skilled to unskilled labor

# == Technology == #
Aa = np.zeros((nreg, T, nrcp, nsc))                       # Technological growth function for Agriculture
Am = np.zeros((nreg, T, nrcp, nsc))                       # Technological growth function for Manufacurng
Ar = np.zeros((nreg, T, nrcp, nsc))                       # ratio of Technology in Manufacurng to Agriculture
Aag = np.zeros((nreg, nrcp))                              # growth rate of Agricultural productivity
Amg = np.zeros((nreg, nrcp))                              # growth rate of Manufacturing productivity
Amgr = 0.01                                               # annual growth rate of Manufacturing productivity

# == Output == #
Y = np.zeros((nreg, T, nrcp, nsc))                        # Total output
Ya = np.zeros((nreg, T, nrcp, nsc))                       # AgricuLtural output
Ym = np.zeros((nreg, T, nrcp, nsc))                       # Manufacturing output
Yr = np.zeros((nreg, T, nrcp, nsc))                       # Ratio of Manufacturing output to Agricultural output
Ypc = np.zeros((nreg, T, nrcp, nsc))                      # Output per adult

# == Output == #
Da = np.zeros((nreg, T, nrcp, nsc))                       # AgricuLtural damage
Dm = np.zeros((nreg, T, nrcp, nsc))                       # Manufacturing damage
Dr = np.zeros((nreg, T, nrcp, nsc))                       # Ratio of Manufacturing damages to Agricultural damages

# == Availability == #
Su = 1 + np.zeros((nreg, T, nrcp, nsc))                   # Availability of unskilled labor
Ss = 1 + np.zeros((nreg, T, nrcp, nsc))                   # Availability of skilled labor
Sr = np.zeros((nreg, T, nrcp, nsc))                       # Ratio of Availability of skilled to unskilled labor

# == Consumption == #
cau = np.zeros((nreg, T, nrcp, nsc))                      # consumption of agricultural good unskilled
cas = np.zeros((nreg, T, nrcp, nsc))                      # consumption of agricultural good skilled
cmu = np.zeros((nreg, T, nrcp, nsc))                      # consumption of manufacturing good unskilled
cms = np.zeros((nreg, T, nrcp, nsc))                      # consumption of manufacturing good skilled
cu = np.zeros((nreg, T, nrcp, nsc))                       # consumption of all goods unskilled
cs = np.zeros((nreg, T, nrcp, nsc))                       # consumption of all goods skilled

# ============================================== Country Calibration ============================================== #
# hx: Ratio of skilled to unskilled labor in 2100 hx = Hx/Lx
# popgx: population growth rate in 2100
# nux: number of unskilled children per parent in 2100
# nsx: number of skilled children per parent in 2100
# Arx: Ratio of technology in manufacturing to technology in agriculture in 2100
# H00: skilled labor in 1980 (milion people)
# L00: unskilled labor in 1980 (milion people)
# h0: Ratio of skilled to unskilled labor in 1980 h0 = H0/L0
# nu0: number of unskilled children per parent in 1980
# ns0: number of skilled children per parent in 1980
# Am0: technology in manufacturing in 1980
# Aa0: technology in agriculture in 1980
# Ar0: Ratio of technology in manufacturing to technology in agriculture in 1980
# N0: Total labor in 1980 (milion people)
# Y00: GDP in 1980 (bilion 1990 GK$)
# C0: population of children in 1980
# r0: ratio of ts/tu
# =========== Scenarios ============ #
Edu_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/education-ssp-SA.csv')
Pop_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/population-ssp-SA.csv')
Pro_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/productivity.csv')
Tavg_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/Tavg-reg-SA.csv')
Tmax_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/Tmax-reg-SA.csv')
Pop_reg_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/pop-reg-2000-SA.csv')
Temp_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/Temp-SA.csv')
wage_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/7-HLT_HealthEconomyClimate/Model/Input/wage-SA.csv')

Ndata = np.zeros((nreg, T, nrcp))
Ldata = np.zeros((nreg, T, nrcp))
Hdata = np.zeros((nreg, T, nrcp))
hdata = np.zeros((nreg, T, nrcp))
Tavgdata = np.zeros((nreg, T, nrcp))
Tmaxdata = np.zeros((nreg, T, nrcp))

for k in range(nreg):
    share = Pop_reg_data.loc[(Pop_reg_data['region'] == RegName[k])]['share'].values[0]
    # GDP constant 2010 USD (https://data.worldbank.org/indicator/ny.gdp.mktp.kd)
    Y2000[k] = 267e9 * share
    for i in range(T):
        for j in range(nrcp):
            noed = Edu_data.loc[(Edu_data['ssp'] == 'ssp2')&(Edu_data['educat'] == 'noeducation')&(Edu_data['t'] == Tperiod[i])]['value'].values[0]
            prim = Edu_data.loc[(Edu_data['ssp'] == 'ssp2')&(Edu_data['educat'] == 'primary')&(Edu_data['t'] == Tperiod[i])]['value'].values[0]
            seco = Edu_data.loc[(Edu_data['ssp'] == 'ssp2')&(Edu_data['educat'] == 'secondary')&(Edu_data['t'] == Tperiod[i])]['value'].values[0]
            tert = Edu_data.loc[(Edu_data['ssp'] == 'ssp2')&(Edu_data['educat'] == 'tertiary')&(Edu_data['t'] == Tperiod[i])]['value'].values[0]
            Ldata[k, i, j] = (noed + prim) * share
            Hdata[k, i, j] = (seco + tert) * share
            Ndata[k, i, j] = Ldata[k, i, j] + Hdata[k, i, j]
            hdata[k, i, j] = Hdata[k, i, j]/Ldata[k, i, j]
            if k < nreg - 1:
                Tavgdata[k, i, j] = Tavg_data.loc[(Tavg_data['region'] == RegName[k])&(Tavg_data['year'] == Tperiod[i])][RCPName[j]].values[0]
                Tmaxdata[k, i, j] = Tmax_data.loc[(Tmax_data['region'] == RegName[k])&(Tmax_data['year'] == Tperiod[i])][RCPName[j]].values[0]
            else:
                Tavgdata[k, i, j] = Temp_data.loc[(Temp_data['RCP'] == RCPName[j])&(Temp_data['year'] == Tperiod[i])]['temperature_mean'].values[0]
                Tmaxdata[k, i, j] = Temp_data.loc[(Temp_data['RCP'] == RCPName[j])&(Temp_data['year'] == Tperiod[i])]['temperature_max'].values[0]          
### ============================================== Model Calibration ============================================== #
h0 = hdata[nreg - 1, 0, 0]
popg0 = (Ndata[nreg - 1, 0, 0] - N1980)/N1980
nu0 = (1 + popg0) / (1 + h0)
ns0 = (1 + popg0) * h0 / (1 + h0)
    
tr = Is0/Iu0
tu = gamma0 / (tr * ns0 + nu0)
ts = tu * tr
    
hx = hdata[nreg - 1, T - 1, 0]

def Calib(RCPx, Regx, Temp0, MaxTemp0):
    MaxTemp0 = coeffTmax[0] + coeffTmax[1] * Temp0
    MaxTempIndex0 = int(round((MaxTemp0 - 12) * 5))
    scale_u0 = max(Pro_data.u_hours)
    scale_s0 = max(Pro_data.s_hours)
    Su0 = (Pro_data.u_hours[MaxTempIndex0])/(scale_u0)
    Ss0 = (Pro_data.s_hours[MaxTempIndex0])/(scale_s0)
    Sr0 = Ss0/Su0
    
    Da0 = max(0.001, g0a + g1a * Temp0 + g2a * Temp0**2)
    Dm0 = max(0.001, g0m + g1m * Temp0 + g2m * Temp0**2)
    Dr0 = Dm0/Da0
    
#    DIndex0 = int(round((Temp0 - 5) * 5))
#    scale_D0 = np.exp(max(wage_data.lnwage))
#    Da0 = np.exp(wage_data.lnwage[DIndex0])/(scale_D0)
#    Dm0 = 1
#    Dr0 = Dm0/Da0
    
    L0 = Ldata[Regx, 0, RCPx]
    H0 = Hdata[Regx, 0, RCPx]
    N0 = Ndata[Regx, 0, RCPx]
    
    Ar0 = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(tr) - np.log(h0)) /(1 - eps) - np.log(Dr0) - np.log(Sr0))
    Am0 = Y2000[Regx]/((alpha * (L0 * Su0 * Da0 / Ar0)**((eps - 1)/eps) + (1 - alpha) * (H0 * Ss0 * Dm0)**((eps - 1)/eps))**(eps/(eps - 1)))
    Aa0 = Am0/Ar0
    
    Arx = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(tr) - np.log(hx)) /(1 - eps) - np.log(Dr0) - np.log(Sr0))

    Arg = np.exp((np.log(Arx/Ar0))/((2100 - 2000)/20)) - 1
    
    Amgx = (1 + Amgr)**20 - 1
    Aagx = (1 + Amgx)/(1 + Arg) - 1
    
    Ya0 = Aa0 * L0 * Su0 * Da0
    Ym0 = Am0 * H0 * Ss0 * Dm0
    Yr0 = Ym0 / Ya0
    
    pr0 = (Yr0)**(-1/eps) * alpha / (1 - alpha)
    
    YA0 = Ya0 / N0
    cmu0 = Ym0 / (H0 * Ss0 * tr/Sr0 + L0)
    cms0 = cmu0 * tr/Sr0
    cau0 = Ya0 / (H0 * Ss0 * tr/Sr0 + L0)
    cas0 = cau0 * tr/Sr0
    cu0 = (alpha * cau0**((eps - 1)/eps) + (1 - alpha) * cmu0**((eps - 1)/eps))**(eps/(eps - 1))
    cs0 = (alpha * cas0**((eps - 1)/eps) + (1 - alpha) * cms0**((eps - 1)/eps))**(eps/(eps - 1))
    wu0 = cu0 / (1 - gamma0)
    ws0 = cs0 / (1 - gamma0)
    pa0 = wu0 / (Da0 * Aa0)
    pm0 = ws0 / (Dm0 * Am0)
    wr0 = ws0/wu0
    
    Outputx = [gamma0, N0, Temp0, MaxTemp0, h0, Da0, Dm0, Dr0, H0, L0, Y2000[Regx], YA0, Ya0, Ym0, Yr0, Aa0, Am0, cmu0, cms0, cau0, cas0, cu0, cs0, wu0, ws0, wr0, pa0, pm0, pr0, Su0, Ss0, Sr0]
    Ratex = [Aagx, Amgx]
    return (Outputx, Ratex)

# ============================================== Model Dynamics ============================================== #

for sc in range(nsc):
    for j in range(nrcp):    
        for k in range(nreg):
            [Output, Rate] = Calib(j, k, Tavgdata[k, 0, 0], Tmaxdata[k, 0, j])
            [Aag[k, j], Amg[k, j]] = Rate
            [gamma[k, 0, j, sc], N[k, 0, j, sc], Temp[k, 0, j, sc], MaxTemp[k, 0, j, sc], h[k, 0, j, sc], Da[k, 0, j, sc], Dm[k, 0, j, sc], Dr[k, 0, j, sc], H[k, 0, j, sc], L[k, 0, j, sc], Y[k, 0, j, sc], Ypc[k, 0, j, sc], Ya[k, 0, j, sc], Ym[k, 0, j, sc], Yr[k, 0, j, sc], Aa[k, 0, j, sc], Am[k, 0, j, sc], cmu[k, 0, j, sc], cms[k, 0, j, sc], cau[k, 0, j, sc], cas[k, 0, j, sc], cu[k, 0, j, sc], cs[k, 0, j, sc], wu[k, 0, j, sc], ws[k, 0, j, sc], wr[k, 0, j, sc], pa[k, 0, j, sc], pm[k, 0, j, sc], pr[k, 0, j, sc], Su[k, 0, j, sc], Ss[k, 0, j, sc], Sr[k, 0, j, sc]] = Output
                         
        for i in range(T - 1):             
            for k in range(nreg):
                if sc == 0:               # Basecase
                    Temp[k, i + 1, j, sc] = Temp[k, 0, j, sc]
                    MaxTemp[k, i + 1, j, sc] = MaxTemp[k, 0, j, sc]
                else:
                    Temp[k, i + 1, j, sc] = Tavgdata[k, i + 1, j]
                    MaxTemp[k, i + 1, j, sc] = coeffTmax[0] + coeffTmax[1] * Temp[k, i + 1, j, sc]
#                    MaxTemp[k, i + 1, j, sc] = Tmaxdata[k, i + 1, j]
                    
                Da[k, i + 1, j, sc] = max(0.001, g0a + g1a * Temp[k, i + 1, j, sc] + g2a * Temp[k, i + 1, j, sc]**2)
                Dm[k, i + 1, j, sc] = max(0.001, g0m + g1m * Temp[k, i + 1, j, sc] + g2m * Temp[k, i + 1, j, sc]**2)
                Dr[k, i + 1, j, sc] = Dm[k, i + 1, j, sc]/Da[k, i + 1, j, sc]
                DrN = Dr[k, i + 1, j, sc]

                # Fixed sectoral damages                
#                Da[k, i + 1, j, sc] = Da[k, 0, j, sc]
#                Dm[k, i + 1, j, sc] = Dm[k, 0, j, sc]
#                Dr[k, i + 1, j, sc] = Dm[k, 0, j, sc]/Da[k, 0, j, sc]
#                DrN = Dr[k, i + 1, j, sc]
                
                Aa[k, i + 1, j, sc] = Aa[k, i, j, sc] * (1 + Aag[k, j])
                Am[k, i + 1, j, sc] = Am[k, i, j, sc] * (1 + Amg[k, j])
                Ar[k, i + 1, j, sc] = Am[k, i + 1, j, sc]/Aa[k, i + 1, j, sc]
                ArN = Ar[k, i + 1, j, sc]
                
                MaxTempIndex = int(round((MaxTemp[k, i + 1, j, sc] - 12) * 5))
                scale_u = max(Pro_data.u_hours)
                scale_s = max(Pro_data.s_hours)
                Su[k, i + 1, j, sc] = (Pro_data.u_hours[MaxTempIndex])/(scale_u)
                Ss[k, i + 1, j, sc] = (Pro_data.s_hours[MaxTempIndex])/(scale_s)
                Sr[k, i + 1, j, sc] = Ss[k, i + 1, j, sc]/Su[k, i + 1, j, sc]
                SrN = Sr[k, i + 1, j, sc]
                
                # Fixed labor damages                
#                Su[k, i + 1, j, sc] = Su[k, 0, j, sc]
#                Ss[k, i + 1, j, sc] = Ss[k, 0, j, sc]
#                Sr[k, i + 1, j, sc] = Ss[k, 0, j, sc]/Su[k, 0, j, sc]
#                SrN = Sr[k, i + 1, j, sc]

                hN = np.exp(eps * np.log((1 - alpha)/alpha) - eps * np.log(tr) - (1 - eps) * (np.log(ArN) + np.log(DrN) + np.log(SrN)))
                
                h[k, i + 1, j, sc] = hN
# Exogenous Population
                N[k, i + 1, j, sc] = Ndata[k, i + 1, j]
                L[k, i + 1, j, sc] = N[k, i + 1, j, sc]/(1 + h[k, i + 1, j, sc])
                H[k, i + 1, j, sc] = L[k, i + 1, j, sc] * h[k, i + 1, j, sc]
   
# changing Fertilit             
#                L[k, i + 1, j, sc] = gamma0/(hN * ts + tu) * N[k, i, j, sc]
#                H[k, i + 1, j, sc] = L[k, i + 1, j, sc] * hN
#                N[k, i + 1, j, sc] = L[k, i + 1, j, sc] + H[k, i + 1, j, sc]
            
                gamma[k, i + 1, j, sc] = (ts * H[k, i + 1, j, sc] + tu * L[k, i + 1, j, sc])/N[k, i, j, sc]
                
                Ya[k, i + 1, j, sc] = Aa[k, i + 1, j, sc] * L[k, i + 1, j, sc] * Su[k, i + 1, j, sc]
                Ym[k, i + 1, j, sc] = Am[k, i + 1, j, sc] * H[k, i + 1, j, sc] * Ss[k, i + 1, j, sc]
                Yr[k, i + 1, j, sc] = Ym[k, i + 1, j, sc] / Ya[k, i + 1, j, sc]
                
                pr[k, i + 1, j, sc] = (Yr[k, i + 1, j, sc])**(-1/eps) * alpha / (1 - alpha)
                
                cmu[k, i + 1, j, sc] = Ym[k, i + 1, j, sc] / (H[k, i + 1, j, sc] * tr/SrN + L[k, i + 1, j, sc])
                cms[k, i + 1, j, sc] = cmu[k, i + 1, j, sc] * tr/SrN
                cau[k, i + 1, j, sc] = Ya[k, i + 1, j, sc] / (H[k, i + 1, j, sc] * tr/SrN + L[k, i + 1, j, sc])
                cas[k, i + 1, j, sc] = cau[k, i + 1, j, sc] * tr/SrN
                cu[k, i + 1, j, sc] = (alpha * cau[k, i + 1, j, sc]**((eps - 1)/eps) + (1 - alpha) * cmu[k, i + 1, j, sc]**((eps - 1)/eps))**(eps/(eps - 1))
                cs[k, i + 1, j, sc] = (alpha * cas[k, i + 1, j, sc]**((eps - 1)/eps) + (1 - alpha) * cms[k, i + 1, j, sc]**((eps - 1)/eps))**(eps/(eps - 1))
                wu[k, i + 1, j, sc] = cu[k, i + 1, j, sc] / (1 - gamma[k, i, j, sc])
                ws[k, i + 1, j, sc] = cs[k, i + 1, j, sc] / (1 - gamma[k, i, j, sc])
                pa[k, i + 1, j, sc] = wu[k, i + 1, j, sc] / (Aa[k, i + 1, j, sc])
                pm[k, i + 1, j, sc] = ws[k, i + 1, j, sc] / (Am[k, i + 1, j, sc])
                Y[k, i + 1, j, sc] = (pa[k, i + 1, j, sc] * Ya[k, i + 1, j, sc] + pm[k, i + 1, j, sc] * Ym[k, i + 1, j, sc]) * (1 - gamma[k, i + 1, j, sc])
                wr[k, i + 1, j, sc] = ws[k, i + 1, j, sc]/wu[k, i + 1, j, sc]        
                Ypc[k, i + 1, j, sc] = Y[k, i + 1, j, sc] / (N[k, i + 1, j, sc])

# ===================================================== Output ===================================================== #    
x = [2000, 2020, 2040, 2060, 2080, 2100]

for j in range(nrcp):
    for k in range(nreg):

#        plt.plot(x, Ndata[k, :, j], 'r:', label = "Data")
#        plt.plot(x, N[k, :, j, 0], 'b', label = "Baseline")
#        plt.plot(x, N[k, :, j, 1], 'g', label = "Tmax only")
#        plt.xlabel('Time')
#        plt.ylabel('millions')
#        plt.title('Adult population under ' + SSPName[j])
#        axes = plt.gca()
#        plt.xticks(np.arange(min(x), max(x) + 1, 20))
#        plt.legend(loc=2, prop={'size':8})
#        plt.show()
    
        plt.plot(x, hdata[k, :, j], 'r:', label = "Data")
        plt.plot(x, h[k, :, j, 0], 'b', label = "Baseline")
        plt.plot(x, h[k, :, j, 1], 'g', label = "Tmax only")
        plt.xlabel('Time')
        plt.ylabel('Ratio')
        plt.title('Skilled to unskilled labor ratio under ' + RCPName[j] + ' in region ' + RegName[k])
        axes = plt.gca()
        #axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        plt.xticks(np.arange(min(x), max(x) + 1, 20))
        plt.legend(loc=2, prop={'size':8})
        plt.show()

        plt.plot(x, ws[k, :, j, 0], 'b', label = "Baseline")
        plt.plot(x, ws[k, :, j, 1], 'g', label = "Tmax only")
        plt.xlabel('Time')
        plt.ylabel('Wages')
        plt.title('Skilled wage under ' + RCPName[j] + ' in region ' + RegName[k])
        axes = plt.gca()
        #axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        plt.xticks(np.arange(min(x), max(x) + 1, 20))
        plt.legend(loc=2, prop={'size':8})
        plt.show()
    
# =============================================== Export into Excel =============================================== #

def output(filename, sheet, list1, v):
    book = xlwt.Workbook(filename)
    sh = book.add_worksheet(sheet)

    v0_desc = 'Region Name'
    v1_desc = 'alpha'
    v2_desc = 'eps'
    v3_desc = 'gamma0'
    v4_desc = 'Tu'
    v5_desc = 'Ts'

    desc = [v0_desc, v1_desc, v2_desc, v3_desc, v4_desc, v5_desc]
    m = 0
    for v_desc, v_v in zip(desc, v):
        sh.write(m, 0, v_desc)
        sh.write(m, 1, v_v)
        m = m + 1
        
    varname = ['Time', 'Ya', 'Ym', 'Y', 'L', 'H', 'N', 'h', 'wu', 'ws', 'Temp', 'MaxTemp', 'Da', 'Dm', 'Su', 'Ss', 'Pa', 'Pm', 'Ya', 'Ym', 'Ypc']
    
    for n in range(nsc):
        m = 6 + 25 * n
        for j in range(nrcp):
            for indx , q in enumerate(range(2000, 2120, 20), 1):
                sh.write(m + 0, j * 10 + indx, q)
                sh.write(m + 0, j * 10, varname[0])
            for k in range(20):
                for indx in range(T):
                    sh.write(m + k + 1, j * 10 + indx + 1, list1[k][indx][j][n])
                    sh.write(m + k + 1, j * 10, varname[k + 1]) 
    book.close()
    
for k in range(9, nreg):
    output1 = [Ya[k], Ym[k], Y[k], L[k], H[k], N[k], h[k], wu[k], ws[k], Temp[k], MaxTemp[k], Da[k], Dm[k], Su[k], Ss[k], pa[k], pm[k], Ya[k], Ym[k], Ypc[k]]
    par = [RegName[k], alpha, eps, gamma0, tu, ts]
    fileName = 'HLT_Output_FP_' + str(k) + '.xlsx'
#    output(fileName, 'Sheet1', output1, par)