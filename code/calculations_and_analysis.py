# -*- coding: utf-8 -*-
"""

Analysis of results and plots

"""

#%% import packages

import pandas as pd
import scipy.stats as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
# import functions from backtest_funcs.py
from backtest_funcs import *

#%% load data from excel sheets

file = 'data.xlsx'

# FTSE 100 
index = pd.read_excel(file, sheet_name='TR Index')
# return of index
mon_ret_index = index.pct_change(fill_method=None)

# monthly libor
libor = pd.read_excel (file, sheet_name='Libor 1 month')
libor['date'] = pd.date_range(start='1/1/1996', periods=len(libor),freq= 'M')
libor.set_index(libor['date'], inplace= True)
libor = libor.drop(columns=['date'])

# monthly price per constituent in the index at the time
data_j = pd.read_excel (file, sheet_name='Data for J')
data_j.drop(['Unnamed: 0'], axis=1, inplace=True)
# return of constituents
mon_retj = data_j.pct_change(fill_method=None)

# monthly price per for all constituents ever in the 25 years
data_k = pd.read_excel (file, sheet_name='Data for K')
data_k.drop(['Unnamed: 0'], axis=1, inplace=True)
# return
mon_retk = data_k.pct_change(fill_method=None)

# average spread of all constituents over 25 years analyzed
spread = pd.read_excel(file, sheet_name='% spread')
spread.drop(['Unnamed: 0'], axis=1, inplace=True)
avgspr = spread.mean().mean()
#avgspr is 0.38% 

# market cap data 
marcap = pd.read_excel(file, sheet_name='MV Data')
marcap.drop(['Unnamed: 0'], axis=1, inplace=True)


#%% create df of 12 month rolling betas

# copy of all data
mon_ret = mon_retk.copy()
# empty df for results
data_b = []

# add benchmark
mon_ret['index'] = mon_ret_index.iloc[:,0]

# iterate over all cols
for c in mon_ret.columns:
    # rolling covariances over past 12 months of asset and index
    cov = mon_ret[c].rolling(12).cov(mon_ret.loc[:,'index'])
    # store in list
    data_b.append(cov)
    
df_b = pd.concat(data_b, axis = 1)
df_b.columns = mon_ret.columns

# divide by variance of index to get rolling betas per asset 
beta = df_b.iloc[:,0:].div(df_b['index'], axis=0)

#%% momentum strategies before transaction costs

# variables are named after the length of J following the length of K
# e.g. onethree is J = 1, K = 3

oneone = backtestMomentum(1,1, mon_retj, mon_retk)

# onethree = backtestMomentum(1,3)

# onesix = backtestMomentum(1,6)

# onetwelve = backtestMomentum(1,12)

# threeone = backtestMomentum(3,1)

# threethree = backtestMomentum(3,3)

# threesix = backtestMomentum(3,6)

# threetwelve = backtestMomentum(3,12)

# sixone = backtestMomentum(6,1)

# sixthree = backtestMomentum(6,3)

# sixsix = backtestMomentum(6,6)  

# sixtwelve = backtestMomentum(6,12)

# twelveone = backtestMomentum(12,1)

# twelvethree = backtestMomentum(12,3)

# twelvesix = backtestMomentum(12,6)

# twelvetwelve = backtestMomentum(12,12)

# #%% 

#J = 1
j1k1 = backtestMomentumZC(1, 1, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j1k3 = backtestMomentumZC(1, 3, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j1k6 = backtestMomentumZC(1, 6, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j1k12 = backtestMomentumZC(1, 12, mon_retj, mon_retk, libor, avgspr, beta, marcap)

#J = 3
j3k1 = backtestMomentumZC(3, 1, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j3k3 = backtestMomentumZC(3, 3, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j3k6 = backtestMomentumZC(3, 6, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j3k12 = backtestMomentumZC(3, 12, mon_retj, mon_retk, libor, avgspr, beta, marcap)

#J = 6
j6k1 = backtestMomentumZC(6, 1, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j6k3 = backtestMomentumZC(6, 3, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j6k6 = backtestMomentumZC(6, 6, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j6k12 = backtestMomentumZC(6, 12, mon_retj, mon_retk, libor, avgspr, beta, marcap)

#J = 12
j12k1 = backtestMomentumZC(12, 1, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j12k3 = backtestMomentumZC(12, 3, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j12k6 = backtestMomentumZC(12, 6, mon_retj, mon_retk, libor, avgspr, beta, marcap)

j12k12 = backtestMomentumZC(12, 12, mon_retj, mon_retk, libor, avgspr, beta, marcap)

#%% analyze momentum effect net of transaction costs

# this cell analyzes if there is a significant outperformance of winner
# portfolios vs loser portfolios

# values for J
Jset = [1,3,6,12]

# valeus for K
Kset = [1,3,6,12]

sig_mat = np.zeros((len(Jset), len(Kset)))
# iterate over each combination
for j in range(len(Jset)):
    
    J = Jset[j]
    
    for k in range(len(Kset)):
        
        K = Kset[k]
        
        # calculate the strategies return
        momeff = backtestMomentum(J, K, mon_retj, mon_retk)
        
        # avg return of winners and losers
        avgwin = np.nanmean(momeff[0])
        avglos = np.nanmean(momeff[1])
        
        # 2 sided t test with H0 that the difference is 0
        t, p = sc.stats.ttest_1samp((momeff[0] - momeff[1]), 0, nan_policy = 'omit')
        
        # store results
        sig_mat[j, k] = p
        
        # print results
        print(f'Eval {J}, Hold {K}: Average Winner return {round(avgwin, 3)}')
        print(f'Average Loser return {round(avglos, 3)}')
        print(f' t stat of diff is {round(t, 3)} and pval {round(p, 3)}')
        print('***********')
        
# create a heatmap to visualize the p values
ax = sns.heatmap(sig_mat, lw = 0.5, cmap = 'copper_r')
ax.set_yticklabels(Jset)
ax.set_xticklabels(Kset)
ax.set_ylabel('J')
ax.set_xlabel('K')
ax.set_title('P values of outperformance of winners over losers')
plt.show()

# it is obvious that for certain strategies the winners outperform the losers
# especially for longer holding periods (K) the outperformance is significant

#%% FF3F to explain returns 

# create market portfolios held for K months
K = 1  #3, 6, 12

# strategy to compare with
strategie = oneone # K needs to be equal to above

# cumulative products of market returns within holding period K
comp_ind = (mon_ret_index + 1).rolling(K).apply(np.prod) - 1

# transform to monthly return to compare with momentum strategy
comp_ind_mtl = (comp_ind + 1) ** (1/K) - 1

# add the correct index from arbitrary df
comp_ind_mtl.index = j6k6.index

# load the factors
ff3 = pd.read_csv('../data/monthlyfactors.csv')
# select data from january 1996
ff3 = ff3.iloc[183:]
# add correct index for available time horizon 
ff3.index = j6k6.loc['1996,31,01':'2017,31,12'].index

# return of momentum is winner - loser
ff3['Mmtm'] = (strategy[0] - strategy[1])[:264]
# subtract rf
ff3['Mmtm - RF'] = ff3['Mmtm'] - ff3['rf']

# add market returns
ff3['Rm'] = comp_ind_mtl.loc['1996,31,01':'2017,31,12']
# and subtract rf
ff3['Rm-Lib'] = ff3['Rm'] - ff3['rf']

# remove rows where we have no data of returns 
ff3 = ff3.dropna()

# run regression
y = ff3['Mmtm - RF']
X = ff3[['smb', 'hml', 'Rm-Lib']]
X_sm = sm.add_constant(X)

model = sm.OLS(y, X_sm)
result = model.fit()
print(result.summary())


#%% analyze actual returns of momentum strategies


strats = [j1k1, j1k3, j1k6, j1k12, j3k1, j3k3, j3k6, j3k12, 
          j6k1, j6k3, j6k6, j6k12, j12k1, j12k3, j12k6, j12k12]

for s in strats:
    # avg winner return
    win = s['Winner'].mean()
    # avg loser return
    loss = s['Loser'].mean()
    # return of long winners and short losers
    profit = s['Zero Cost'].mean()
    # t test that portfolio had return 0
    tp, pp = sc.stats.ttest_1samp(s['Zero Cost'], 0, nan_policy = 'omit')
    print(f'avg of winnerp is {win}')
    print(f'avg of loserp is {loss}')
    print(f'avg of portfolio p is {profit}; tstat {tp} and p val {pp}')
    print('**********************')
    
#%% test betas 

for i in strats:
    
    # portfolio betas
    bp = i['Zero Cost Beta']
    avgbp = bp.mean()
    
    # winner betas
    bw = i['Winner Beta']
    avgbw = bw.mean()
    
    # loser betas
    bl = i['Loser Beta']
    avgbl = bl.mean()
    
    # test whether portfolio beta is different from 0 
    tp, pp = sc.stats.ttest_1samp(bp, 0, nan_policy = 'omit')
    print(f'winb is {avgbw}')
    print(f'losb is {avgbl}')
    print(f'portb is {avgbp} with tstat {tp} and pval {pp}')
    print('********')
  
#%% test market caps (for size effect)

for i in strats:
    
    # winner market cap
    mcw = i['Winner Market Cap']
    
    # lsoer market cap
    mcl = i['Loser Market Cap']
    
    # test whether difference in market caps is different from 0
    tmc, pmc = sc.stats.ttest_1samp(mcw - mcl, 0, nan_policy = 'omit')
    print('Avg winner and loser market cap')
    print(mcw.mean(), mcl.mean())
    print(f'Market Cap Winners - Losers have a tstat {tmc}')
    print(f'and a pval of {pmc}')
    print('**********')

#%% analysis of a winner only long strategy

# rolling market returns for 6 months holding period
comp_ind = (mon_ret_index + 1).rolling(6).apply(np.prod) - 1
# monthly returns
comp_ind_mtl = (comp_ind+1) ** (1/6) - 1

# repeat procedure as before
comp_ind_mtl.index = j6k6.index

ff3 = pd.read_csv('monthlyfactors.csv')
ff3 = ff3.iloc[183:]
ff3.index = j6k6.loc['1996,31,01':'2017,31,12'].index
# only winners of j6k6 strategy
ff3['Mmtm'] = j6k6.loc['1996,31,01':'2017,31,12', 'Winner']
ff3['Mmtm - RF'] = ff3['Mmtm'] - ff3['rf']
ff3['Rm'] = comp_ind_mtl.loc['1996,31,01':'2017,31,12']
ff3['Rm-Lib'] = ff3['Rm'] - ff3['rf']
ff3 = ff3.dropna()


y = ff3['Mmtm - RF']
X = ff3[['smb', 'hml', 'Rm-Lib']]
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
result = model.fit()
print(result.summary())


#%% analysis of strategies' performance over sub intervals

#Anfang und Ende von Subperioden
startp1 = '1996,31,01'
endp1 = '2000,31,12'
startp2 = '2001,31,01'
endp2 = '2005,31,12'
startp3 = '2006,31,01'
endp3 = '2010,31,12'
startp4 = '2011,31,01'
endp4 = '2015,31,12'
startp5 = '2016,31,01'
endp5 = '2020,31,12'

# matrices to store avg return and pval per period
ret_mat = np.zeros((len(strats), 5))
sig_mat = np.zeros((len(strats), 5))

#Für alle Strategien
for s in range(len(strats)):
    
    strat = strats[s]
    
    #Unterteilung der Ergebnisse in Subperioden
    sub1 = strat.loc[startp1:endp1]
    sub2 = strat.loc[startp2:endp2]
    sub3 = strat.loc[startp3:endp3]
    sub4 = strat.loc[startp4:endp4]
    sub5 = strat.loc[startp5:endp5]
    
    # avg return of portfolio in sub 1
    mean1 = sub1['Zero Cost'].mean()
    # significance test 
    t1, p1 = sc.stats.ttest_1samp(sub1['Zero Cost'], 0, nan_policy = 'omit')
    # store results
    ret_mat[s, 0] = mean1
    sig_mat[s, 0] = p1
    
    mean2 = sub2['Zero Cost'].mean()
    t2, p2 = sc.stats.ttest_1samp(sub2['Zero Cost'], 0, nan_policy = 'omit')
    ret_mat[s, 1] = mean2
    sig_mat[s, 1] = p2
    
    mean3 = sub3['Zero Cost'].mean()
    t3, p3 = sc.stats.ttest_1samp(sub3['Zero Cost'], 0, nan_policy='omit')
    ret_mat[s, 2] = mean3
    sig_mat[s, 2] = p3
    
    #hier  Periode 4
    mean4 = sub4['Zero Cost'].mean()
    t4, p4 = sc.stats.ttest_1samp(sub4['Zero Cost'], 0, nan_policy='omit')
    ret_mat[s, 3] = mean4
    sig_mat[s, 3] = p4
    
    mean5 = sub5['Zero Cost'].mean()
    t5, p5 = sc.stats.ttest_1samp(sub5['Zero Cost'], 0, nan_policy='omit')
    ret_mat[s, 4] = mean5
    sig_mat[s, 4] = p5


# create plot    
fig, ax = plt.subplots(ncols = 2, figsize = (10, 7))

ax[0].plot(ret_mat.T)
ax[0].set_xticks(np.arange(5))
ax[0].set_xticklabels(np.arange(5))
ax[0].set_xlabel('Period')
ax[0].set_ylabel('Return')
ax[0].set_title('Average return of various momentum strategies over 5 periods')

ax[1].plot(sig_mat.T, 'o')
ax[1].set_xticks(np.arange(5))
ax[1].set_xticklabels(np.arange(5))
ax[1].set_xlabel('Period')
ax[1].set_ylabel('P value')
ax[1].set_title('P value of test that return is 0')



#%% plots

plt.rcParams["font.family"] = "Times New Roman"
sns.set_theme(style='whitegrid')

#%% index
pltind = pd.read_excel('data.xlsx', sheet_name='Index')
pltind.index = j6k6.index
plt.plot(pltind)
plt.ylabel('Price')
plt.grid(b=None)

#plt.savefig('Index-Price.png', dpi=300)

#%% Figure 2
plt.figure(figsize=(8,4))
mtl_returns = sns.lineplot(data=j6k6[['Winner', 'Loser', 'Zero Cost']])
mtl_returns.set(xlabel = 'Year', ylabel = 'Monthly Returns')
vals = mtl_returns.get_yticks()
mtl_returns.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

#plt.savefig('J6K6-hires.png', dpi=300)

#%% betas
plots, axes = plt.subplots(2,1)
sns.lineplot(ax=axes[0], data=j6k6['Zero Cost'])
sns.lineplot(ax = axes[1], data=j6k6['Zero Cost Beta'])


#%% hist of returns
plt.hist(j6k6['Zero Cost'], 20)
plt.xlabel('Return in %')
#plt.savefig('j6k6_dist.png', dpi=300)

plt.hist(j6k6['Winner'], 21)
plt.xlabel('Return in %')
#plt.savefig('j6k6winner_dist.png', dpi=300)

#%% betas over time
plt.plot(j12k3['Zero Cost Beta'])
plt.plot(j12k3['Winner Beta'])    
plt.plot(j12k3['Loser Beta'])

#%% Figure 3
cumu = (j6k6['Winner']+1).cumprod()
cumi = (mon_ret_index.iloc[:,0]+1).cumprod()
cumi.index = cumu.index
cumu =cumu * 1000
cumi = cumi * 1000
cump = (j6k6['Zero Cost']+1).cumprod()
cump = cump * 1000
cump.index = cumu.index

from matplotlib.ticker import PercentFormatter
plt.figure(figsize=(6,4))
plt.plot(cumu, label = 'Winner', linewidth = 2)
plt.plot(cumi, label = 'Market', color='purple')
plt.plot(cump, label='Zero Cost', color = 'green')
plt.xlabel('Year')
plt.ylabel('Wealth')
plt.ylim(0)
plt.legend()
#plt.savefig('CUM-RET_J6K6.png', dpi=300)




