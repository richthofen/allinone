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
predicts = {}
def cal(symbol):
    df = ts.get_k_data(symbol)
    try:
        closes = df['close'].values 
        if closes[-1] < predicts[symbol]:
            return ("%s:%2f")%(symbol, predicts[symbol])
        return ""
    except Exception as ex:
        return ""

with open("predict") as file:
    lines = file.readlines()
    print(lines)
    for l in lines:
        s = l.split(",")
        symbol = s[0] 
        buy = s[2].split(":")[1]
        predicts[symbol] = buy
with open("s.log") as file:
    data = file.read().split("\n",113)
    print (predicts)     
    msg = ""
    # print ts.get_k_data('600000')
    for symbol in data[0:-1]:
        print (symbol)
        ret = cal(symbol)
        if('' != ret.strip()):
            if('' != msg.strip()):
                ret = "\n" + ret
        msg += ret
    print (msg)
    if "" != msg :
        pass
        #send.send_msg(msg)
