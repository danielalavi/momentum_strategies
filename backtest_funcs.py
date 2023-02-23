# -*- coding: utf-8 -*-
"""
Algorithmns for Bachelorsthesis 2021
Momentum Strategies
Daniel Alavi
"""
# %% import packages

import pandas as pd
import numpy as np

#%% simple momentum effect


def backtestMomentum(J, K, mon_retj, mon_retk):

    """
    This function calculates winners and losers portfolios for a given 
    momentum strategy. The returns are then stored in array and are returned
    as output. 
    """
    
    # rolling returns to select stocks
    past_j = (mon_retj + 1).rolling(J).apply(np.prod) - 1

    # empty arrays to store returns of portfolios
    winnerret = np.full((J + K, 1), np.nan)
    loserret = np.full((J + K, 1), np.nan)

    # iterate through every month from J until T-K
    for t in range(J, len(mon_retj) - K):

        
        # FORMATION OF PORTFOLIOS

        # return in month t
        ret_j = past_j.iloc[t]

        # reformat
        ret_j = ret_j.reset_index()
        ret_j.dropna(axis = 0, inplace = True)
        Q = 10  # number of quantiles

        # sort returns into quantiles to select best and worst performers
        ret_j['quantiles'] = pd.qcut(ret_j.iloc[:, 1], Q, labels = False)

        winners = ret_j[ret_j.quantiles == max(ret_j['quantiles'])]
        
        # list of winners
        lstw = winners['index'].tolist()

        losers = ret_j[ret_j.quantiles == 0]
        # list of losers
        lstl = losers['index'].tolist()


        # WINNER RETURNS

        # get returns of following K months
        ret_win = mon_retk.loc[t + 1:t + K, lstw]

        # if no data available (due to bankruptcy) add 0
        for i in range(len(ret_win)):
            ret_win.iloc[i] = ret_win.iloc[i].fillna(0)

        # profit is product of returns
        past_k_win = (ret_win + 1).apply(np.prod) - 1

        # assuming simple equal weighted portfolio take mean
        winnerret = np.append(winnerret, past_k_win.mean())


        # LOSER RETURNS

        ret_los = mon_retk.loc[t + 1:t + K, lstl]

        for i in range(len(ret_los)):
            ret_los.iloc[i] = ret_los.iloc[i].fillna(0)

        past_k_los = (ret_los + 1).apply(np.prod) - 1
        
        loserret = np.append(loserret, past_k_los.mean())

    # convert to monthly returns from K months return
    loserret = (loserret + 1) ** (1 / K) - 1
    winnerret = (winnerret + 1) ** (1 / K) - 1

    return winnerret, loserret


# %% zero cost function


def backtestMomentumZC(J, K, mon_retj, mon_retk, rf, spr, beta, marcap):
    
    """
    This function is more sophisticated and will include investments in 
    risk free rates if the a company goes bankrupt. 
    It will also take spreads, trading costs, and fees for borrowing stock
    into account. 
    
    This function backtests a more realistic expectation of a momentum strategy
    if executed in real life by a PM.
    
    Returns a data frame with returns of winners, losers, actually traded 
    portfolio, betas, and market caps of winners and losers. 
    """    
    
    
    # cumulated returns of past J months
    past_j = (mon_retj + 1).rolling(J).apply(np.prod) - 1

    # transaction cost of 0.05%
    TCt = -0.0005
    
    # annual cost for borrowing stock 0.5%
    TCs = (1 - 0.005) ** (1 / 12) - 1

    # winner
    winnerret = np.full((J + K, 1), np.nan)
    # verlierer
    loserret = np.full((J + K, 1), np.nan)
    # winner beta
    betawin = np.full((J + K, 1), np.nan)
    # loser beta
    betalos = np.full((J + K, 1), np.nan)
    # winner market cap
    marketcapw = np.full((J + K, 1), np.nan)
    # loser market cap
    marketcapl = np.full((J + K, 1), np.nan)

    # iterate through time
    for t in range(J, len(mon_retj) - K):

        # FORMATION OF PORTFOLIOS

        # return of last J month in time t
        ret_j = past_j.iloc[t]

        # reformat
        ret_j = ret_j.reset_index()
        ret_j.dropna(axis = 0, inplace = True)

        # number of quantiles
        Q = 10

        # sort returns in quantiles
        ret_j['quantiles'] = pd.qcut(ret_j.iloc[:, 1], Q, labels = False)

        # select best and worst performers
        winners = ret_j[ret_j.quantiles == max(ret_j['quantiles'])]
        lstw = winners['index'].tolist()  #list of winners
        losers = ret_j[ret_j.quantiles == 0]
        lstl = losers['index'].tolist()  #list of losers


        # WINNER RETURNS

        # returns of following months
        ret_win = mon_retk.loc[t:t + K, lstw]
        net_win = []

        # iterate over every winner 
        for c in ret_win.columns:
            
            # index of col
            n = ret_win.columns.get_loc(c)

            # last available price data
            s = ret_win[c].index.get_loc(ret_win[c].last_valid_index())

            # if the stock is delisted after formation the money is invested 
            # in a risk free asset
            
            # if the last period is before we want to sell again
            if s < K:
                # calculate return until delisting
                pre = (ret_win.iloc[1:s + 1, n] + 1).prod() - 1
                # incorporate spread and trading cost
                pre = (1 + pre) * (1 - spr) * (1 + TCt)**2 - 1

                # return after delisting
                post = (rf.iloc[t + s + 1:t + K + 1, 0] + 1).prod() - 1
                # incorporate trading costs
                post = (1 + post) * (1 + TCt)**2 - 1

                # total return is product of the 2
                compl = (1 + pre) * (1 + post) - 1
                # store in list
                net_win.append(compl)
            else:
                # is data is available for whole K months
                long = (ret_win.iloc[1:, n] + 1).prod() - 1

                # spread and transaction costs
                netlong = (1 + long) * (1 - spr) * (1 + TCt)**2 - 1
                net_win.append(netlong)

        # take average of all returns as equally weighted
        winnerret = np.append(winnerret, np.mean(net_win))


        # WINNER BETA

        betasw = beta.loc[t, lstw]
        bwin = betasw.mean()
        betawin = np.append(betawin, bwin)


        # WINNER MARKET CAP
        
        marcapw = marcap.loc[t, lstw]
        mcw = marcapw.mean()
        marketcapw = np.append(marketcapw, mcw)


        # LOSER RETURNS

        ret_los = mon_retk.loc[t:t + K, lstl]
        net_los = []

        for c in ret_los.columns:

            n = ret_los.columns.get_loc(c)

            s = ret_los[c].index.get_loc(ret_los[c].last_valid_index())

            if s < K:
                # we buy the share back early

                # return while not delisted incorporating costs
                pre = (-(ret_los.iloc[1:s + 1, n]) + 1) * (1 + TCs) - 1
                pre = (1 + pre).prod() - 1
                # Abzug der Transaktionsgebühr und Spread
                pre = (1 + pre) * (1 - spr) * (1 + TCt)**2 - 1

                post = (rf.iloc[t + s + 1:t + K + 1] + 1).prod() - 1
                post = (1 + post) * (1 + TCt)**2 - 1
                compl = (1 + pre) * (1 + post) - 1
                net_los.append(compl)
            
            else:
          
                short = (-(ret_los.iloc[1:, n]) + 1) * (1 + TCs) - 1
                short = (1 + short).prod() - 1
                short = (1 + short) * (1 - spr) * (1 + TCt)**2 - 1
                net_los.append(short)

        loserret = np.append(loserret, np.mean(net_los))


        # LOSER BETA
        betasl = beta.loc[t, lstl]
        blos = betasl.mean()
        betalos = np.append(betalos, blos)


        # LOSER MARKET CAP
        marcapl = marcap.loc[t, lstl]
        mcl = marcapl.mean()
        marketcapl = np.append(marketcapl, mcl)

    # change to monthly return
    winnerret = (1 + winnerret) ** (1 / K) - 1
    loserret = (1 + loserret) ** (1 / K) - 1

    # store arrays in new df
    mom = pd.DataFrame()
    mom['Date'] = pd.date_range(
        start = '1/1/1996',
        periods = len(winnerret),
        freq = 'M')
    mom.set_index(mom['Date'], inplace = True)
    mom = mom.drop(columns = ['Date'])

    # Neue Kolonne für Winners und Losers
    mom['Winner'] = winnerret
    mom['Loser'] = loserret
    mom['Winner Beta'] = betawin
    mom['Loser Beta'] = betalos
    mom['Winner Market Cap'] = marketcapw
    mom['Loser Market Cap'] = marketcapl

    # now create the return of a long winner short loser portfolio by adding 
    # the returns
    mom['Zero Cost'] = mom['Winner'] + mom['Loser']
    # and the beta of the portfolio
    mom['Zero Cost Beta'] = betawin - betalos
    
    return mom
