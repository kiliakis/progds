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


def get_potential(infile, method='maxrange'):
    data = np.genfromtxt(infile, dtype='str', delimiter=',')
    if method == 'maxrange':
        if len(data) > 0:
            data = data[1:]
        else:
            print('Empty file: {}'.format(infile))
            return 0
        # data = pd.read_csv(infile, header=0, index_col='Date')
        # Now I need to find the max- min, where max_date > min_date
        lmin = np.zeros(len(data), float)
        rmax = np.zeros(len(data), float)
        lmin[0] = float(data[0][h['Low']])
        for i in np.arange(1, len(data)):
            lmin[i] = min(float(data[i][h['Low']]), lmin[i-1])

        rmax[-1] = float(data[-1][h['High']])
        for i in np.arange(len(data)-2, -1, -1):
            rmax[i] = max(float(data[i][h['High']]), rmax[i+1])

        return np.max(rmax - lmin)
    elif method == 'subdollar':
        if len(data) > 0:
            data = data[1:]
        else:
            print('Empty file: {}'.format(infile))
            return 0
        # I need some cheap stocks

def parse_stock_potential(indir):
    # df = pd.DataFrame()
    stocks = []
    files = os.listdir(indir)
    # files = ['flbr.us.txt']
    # sys.stdout.write("[0/{}]".format(len(files)))
    # sys.stdout.flush()
    # sys.stdout.write("\b" * )
    i = 0
    for stockfile in files:
        print('[{}/{}]'.format(i, len(files)))
        if 'us.txt' not in stockfile:
            continue
        name = stockfile.replace('.us.txt', '')
        potential = get_potential(os.path.join(indir, stockfile), method='maxrange')
        potential = get_potential(os.path.join(indir, stockfile), method='subdollar')
        stocks.append([name, potential])
        i+=1
    stocks = sorted(stocks, key=lambda a: a[1], reverse=True)
    return stocks
