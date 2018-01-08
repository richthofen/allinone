#!/usr/bin/python 
# coding=UTF-8
from pylab import figure, show
from numpy import NaN
from matplotlib.finance import quotes_historical_yahoo_ochl,quotes_historical_yahoo_ohlc
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
import talib
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import sys
import send
msg = ""


def norm(x):
    xmin = min(x)
    if xmin < 0:
        xmin = 0

    xmax = max(x)
    down = xmax - xmin
    biu = (x - xmin) / down
    return biu


def de_prio(x):
    global g_prio
    # print g_prio
    if x > 0 and g_prio > 0:
        x = g_prio * 0.96
    g_prio = x
    return x


def de_prio2(x, y):
	if x or y :
		if x and y :
			# print x,y
			y = x * y * 0.96
			x = x * y * 0.96
			return x * y * 0.96
		elif y == 1:
			return 1
		else:
			return 0
	else:
		return 0


def cal(symbol):
    # df = ts.get_k_data(symbol, index=True)
    df = ts.get_k_data(symbol)
    
    try:
        # print df
	df.to_csv(symbol + ".csv" ,columns=['open','close','high','low','volume'])
	df.to_csv(symbol + "_base.csv")
        open = df['open'].values
        dates = df['date'].values
        lows = df['low'].values
        highs = df['high'].values
        closes = df['close'].values
        macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        slowk, slowd = talib.STOCH(highs,
    	                                   lows,
    									   closes,
    	                                   fastk_period=9,
    	                                   slowk_period=3,
    	                                   slowk_matype=0,
    	                                   slowd_period=3,
    	                                   slowd_matype=0)
        slowd = np.nan_to_num(slowd)
        slowk = np.nan_to_num(slowk)
        kdj = slowk - slowd
        ncloses = norm(closes)
        nslowk = norm(slowk)
        nslowd = norm(slowd)

        kdj = norm(kdj)
        nkdj = np.array([ x>0 and 1 or 0 for x in kdj])
        macd1 = ( macd )
        xhist = np.array([ x>0 and x or 0 for x in macdhist])
        lolo = [ x>0 and 1 or 0 for x in macdhist]
        global g_prio 
        g_prio = 0
        xhist = np.array(map (de_prio ,lolo))
        g_prio = 0
        nkdj = np.array(map (de_prio ,nkdj))
        weight = (nkdj + xhist)/2
        weight = np.array([ x>0.95 and 1 or 0 for x in weight])
        op =  weight * closes
        if op[-1] > 0:
            # print "%s op %s" % (symbol, op[-1])
            return "%s op %s " % (symbol, op[-1])
        return ""
    except Exception as ex:
        return ""
        # return " ex %s " % (symbol)

# all = ts.get_hs300s()
# print all['code'].values

with open("s.log") as file:
    data = file.read().split("\r\n",113)
    # print data[0:-1]
    msg = ""
    # print ts.get_k_data('600000')
    for symbol in data[0:-1]:
        # print symbol
        ret = cal(symbol)
        if('' != ret.strip()):
            if('' != msg.strip()):
                ret = "\n" + ret
        msg += ret
    # msg = "22"
    # print "222"
    print msg
    if "" != msg :
        pass
        #send.send_msg(msg)
