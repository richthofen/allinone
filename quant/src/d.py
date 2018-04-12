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
import json
now = str(datetime.datetime.now().date())
msg = ""
predicts = {}
isend = {}
def cal(symbol):
    df = ts.get_k_data(symbol)
    try:
        closes = df['close'].values 
        ret = "%s:%s"%(symbol, predicts[symbol])
        if closes[-1] < float(predicts[symbol]):
            if isend.get(now).get(symbol) is  None:
                isend[now][symbol] = 1
                return ret
        return ""
    except Exception as ex:
        return ""

with open("predict") as file:
    lines = file.readlines()
    for l in lines:
        s = l.split(",")
        symbol = s[0] 
        buy = s[2].split(":")[1]
        predicts[symbol] = buy

with open('send.json', 'r') as f:
    try:
        isend = json.load(f)
    except:
        pass
    if isend.get(now) is  None:
        isend[now] = {}
	

with open("s.log") as file:
    data = file.read().split("\n",113)
    msg = ""
    for symbol in data[0:-1]:
        ret = cal(symbol)
        if('' != ret.strip()):
            if('' != msg.strip()):
                ret = "\n" + ret
        msg += ret
    if "" != msg :
        print(msg)
        send.send_msg(msg)

with open('send.json', 'w') as f:
    json.dump(isend, f, indent=2)

