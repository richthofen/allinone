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
import requests as re
import sys
import os
import send
msg = ""
pro = ts.pro_api('abf236e4493d991a1492271e8289f5952301750aa7f7345c9a6abd9e')
allSymbol = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

def get_latest_data(symbol):
    area=allSymbol.query('symbol==@symbol')['ts_code']
    if 0 == area.shape[0]:
        return None
    area = area.values[0] 
    tx_symbol=""
    if "SZ" == area[7:9]:
        tx_symbol= "sz" + symbol
    else:
        tx_symbol= "sh" + symbol
    tx_symbol = tx_symbol+'.js'
    res=re.get('http://data.gtimg.cn/flashdata/hushen/latest/daily/' + tx_symbol)
    ret = res.text.split('\\n\\\n')[2:-1]
    data=[]
    for i in range(len(ret)):
        d = ret[i]
        d = '20' + d[0:2] + '-' + d[2:4] + '-' + d[4:]
        d= d.split(' ')
        # d = np.array(d)
        data.append(d)
    data = np.array(data)
    return data
def get_real_data(symbol):
    area=allSymbol.query('symbol==@symbol')['ts_code']
    if 0 == area.shape[0]:
        return None
    area = area.values[0] 
    sina_symbol=""
    if "SZ" == area[7:9]:
        sina_symbol= "sz" + symbol
    else:
        sina_symbol= "sh" + symbol
    res=re.get('http://hq.sinajs.cn/?format=text&list=' + sina_symbol)
    ret = res.text.split(',')
    ob_data = ret[30:31]
    ob_data.extend(ret[1:2])
    ob_data.extend(ret[3:6])
    ob_data.extend(ret[8:9])
    ob_data.extend(['0'])
    ob_data = np.array([ob_data])
    return ob_data
def check(symbol):
    [symbol,price] = symbol.split(":")
    # print(symbol)
    symbol = symbol.strip()
    ts_code=allSymbol.query('symbol==@symbol')['ts_code'].values[0]
    data = get_real_data(symbol)
    price = float(price)
    now = float(data[-1, 4])
    if price > now:
        print(now)
        return symbol + " : " + str(price)

    return ""

with open("/Users/richt/workspace/github/allinone/yy.log") as file:
    data = file.read().split("\n")
    msg = ""
    for symbol in data[:-1]:
        # print (symbol)
        ret = check(symbol)
        if('' != ret.strip()):
            if('' != msg.strip()):
                cmd = "sed -i '' /" + symbol.split(":")[0].strip() + "/d /Users/richt/workspace/github/allinone/xx.log" 
                print(cmd)
                #os.popen(cmd)
                ret = "\n" + ret
        msg += ret
        
    print (msg)
    #if "" != msg :
    #    send.send_msg(msg)
