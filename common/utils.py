import numpy as np
import re
import pandas as pd
import os
h = {
    'Date': 0,
    'Open': 1,
    'High': 2,
    'Low': 3,
    'Close': 4,
    'Volume': 5,
    'OpenInt': 6
}

# h = {
#     'Date': 'datetime64[s]',
#     'Open': 'f4',
#     'High': 'f4',
#     'Low': 'f4',
#     'Close': 'f4',
#     'Volume': 'i4',
#     'OpenInt': 'i4'
# }



def get_potential(infile, method='maxr', trans=1000):
    data = np.genfromtxt(infile, dtype='str', delimiter=',')
    if len(data) > 0:
        data = data[1:]
    else:
        print('Empty file: {}'.format(infile))
        return []
    
    # data = np.array(data, dtype={'names': list(h.keys()), 'formats': list(h.values())})
    if method == 'maxr':
        # data = pd.read_csv(infile, header=0, index_col='Date')
        # Now I need to find the max- min, where max_date > min_date
        lmin = np.zeros(len(data), float)
        rmax = np.zeros(len(data), float)
        lmin[0] = float(data[0, h['Low']])
        for i in np.arange(1, len(data)):
            lmin[i] = min(float(data[i, h['Low']]), lmin[i-1])

        rmax[-1] = float(data[-1][h['High']])
        for i in np.arange(len(data)-2, -1, -1):
            rmax[i] = max(float(data[i, h['High']]), rmax[i+1])
        rmax = rmax - lmin
        idx = np.argmax(rmax)
        return [data[idx, h['Date']], rmax[idx]]
    elif method == 'subd':
        # I need some cheap stocks, I need to know when it was less
        # than a dollar, and if after that it increased
        dollar_i = -1
        for i, d in enumerate(data):
            if float(d[h['Low']]) <= 1:
                dollar_i = i
                break
        if dollar_i < 0:
            # this means that it was never less than 1usd
            return []
        # else I need to find the max value, after this index
        max_val = np.array(data[dollar_i:, h['High']], float)
        max_val = np.max(max_val)
        return [data[dollar_i, h['Date']], max_val]
    elif method == 'dp':
        # limitations: transactions
        # daily volume
        # each day, I can either buy one or more stocks
        # sell one or more stocks
        # each day I can do one of the 6 trans types
        # for all stocks I own
        # dp[trans][date][stocks] = balance and I start with max (1usd, min low)
        # dp[0] = 0
        # lets say stocks 
        # I need to find dp[N][last date]
        dp = [[{}] * len(data)+1] * (trans + 1)
        return dp_potential(data, dp)


def dp_potential(data, dp):

        sbal = max(1, np.min(np.array(data[:, h['Low']], float)))
        
        dp = np.zeros((trans+1, len(data+1)), float)
        # This says we start with money enough to buy one stock
        dp[:, 0] = sbal

        # With no transactions, we only have sbal money 
        dp[0, :] = sbal

        for t in np.arange(1, trans+2):
            for i in np.arange(len(data)):
                max_so_far = -1
                # either I buy as many stocks as I can
                dp[t, i] = max()
                # or I do nothing
                # or I sell all my stocks

        return []


def parse_stock_potential(indir, method='maxr', numstocks=-1):
    # df = pd.DataFrame()
    stocks = []
    files = os.listdir(indir)
    if numstocks > 0:
        files = files[:numstocks]
    i = 0
    for stockfile in files:
        print('[{}/{}]'.format(i, len(files)))
        if 'us.txt' not in stockfile:
            continue
        name = stockfile.replace('.us.txt', '')
        potential = get_potential(os.path.join(indir, stockfile), method=method)
        if len(potential) > 0:
            stocks.append([name] + potential)
        i += 1
    stocks = sorted(stocks, key=lambda a: a[-1], reverse=True)
    return stocks
