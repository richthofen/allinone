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

# def GetHistoricGoogle( symbol,startDate,endDate ):
# 	urlFmt = ('http://www.google.com/finance/historical?q=%s&startdate=%s&' +
#               'endDateStr=%s')

#     url = urlFmt % (symbol, startDate, endDate)
#     with urlopen(url) as a:
#     	prin a

# 一次性获取全部日k线数据
ts.get_hist_data('600848')


# 一次性获取当前交易所有股票的行情数据

# quotes = ts.get_today_all()

date1 = datetime.date(2017, 4, 10)
date2 = datetime.date(2017, 4, 14)

daysFmt = DateFormatter('%m-%d-%Y')
symbol = '000810'

# val =  ts.matplotlib.finance.candlestick2_ochl(ax, opens, closes, highs, lows, width=4, colorup='k', colordown='r', alpha=0.75)¶

# print val
# quotes = quotes_historical_yahoo_ochl('000810.sz', date1, date2)
quotes = ts.get_hist_data(symbol)
# print quotes
df=ts.get_k_data('600848')
open = df['open'].values
dates = df['date'].values
print dates
#print quotes
# data = ts.get_h_data(symbol)
#delta_s = np.array(data)
# print delta_s
# print type(qut)
#dates = quotes['date']
#print dates
#print quotes['high']
#print qut
#sys.exit()
# quotes = quotes_historical_yahoo_ochl('600000.ss', date1, date2) 
# quotes = quotes_historical_yahoo_ohlc('600000.ss', date1, date2)
# quotes = quotes_historical_yahoo_ohlc('000810.sz', date1, date2)
# quotes = quotes_historical_yahoo_ohlc('002415.sz', date1, date2)
# dates = np.array([q[0] for q in quotes])
# closes = np.array([q[2] for q in quotes])
# lows = np.array([q[4] for q in quotes])
# highs = np.array([q[3] for q in quotes])

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

# xclose = [ x/10 for x in close]


def norm(x):
	# print type(x)
	# xmin,xmax = x.min(),x.max()
	# numpy.nan_to_num(x)
	# xmin = min(x, key=lambda x: np.isnan(x) and 0 or 1)
	xmin = min(x)
	if xmin < 0 :
		xmin = 0
	xmax = max(x)
	# print xmin, xmax
	down = xmax - xmin
	biu = (x - xmin) / down	
	# print biu
	return biu
ncloses = norm(closes)

# for i in np.nan_to_num(slowk):
# 	print "%s type is %s" %( i,type(i))
# 	break
	
nslowk = norm(slowk)
# print nslowk
nslowd = norm(slowd)
# lolo = [ x>0 and 1 or 0 for x in macdhist]

kdj = norm(kdj)
nkdj = np.array([ x>0 and 1 or 0 for x in kdj])
# print nslowk
# print nclose

macd1 = ( macd )
# print "next"
# print macd1.next()

# print quotes
xhist = np.array([ x>0 and x or 0 for x in macdhist])
lolo = [ x>0 and 1 or 0 for x in macdhist]
global g_prio 

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


	
# print lolo
# reduce (de_prio ,lolo)
g_prio = 0
xhist = np.array(map (de_prio ,lolo))
# print nkdj
# xhist = [ x>0 and x or 0 for x in macdhist]

g_prio = 0
nkdj = np.array(map (de_prio ,nkdj))
weight = (nkdj + xhist)/2

weight = np.array([ x>0.95 and 1 or 0 for x in weight])
op =  weight * closes
print "op %s"%(op)
# print lolo
# print array_reduce(lolo, de_prio)
# print reduce (de_prio2 ,lolo )
# print lolo


# print xhist

if len(quotes) == 0: 
	raise SystemExit 

fig = plt.figure(figsize=[18,5])
fig.add_subplot(111)  
ax = fig.add_subplot(111)  
ax.plot_date(dates, ncloses, '-') 


# plt.plot(dates,macd,label='macd dif')
# plt.plot(dates,macdsignal,label='signal dea')
# plt.plot(dates,macdhist,label='hist bar')
plt.plot(dates,xhist,label='hist bar')


plt.plot(dates,weight,label='weight')
# plt.plot(dates,nslowd,label='d')
# plt.plot(dates,nkdj,label='kdj')
plt.plot(dates,kdj,label='kdj/11')



plt.legend(loc='best')
plt.show()
# dates = [q[0] for q in quotes] 
# opens = [q[1] for q in quotes] 

# fig = figure() 
# ax = fig.add_subplot(111) 
# ax.plot_date(dates, opens, '-') 

# # format the ticks 
# ax.xaxis.set_major_formatter(daysFmt) 
# ax.autoscale_view() 

# # format the coords message box 
# def price(x): return '$%1.2f'%x 
# ax.fmt_xdata = DateFormatter('%Y-%m-%d') 
# ax.fmt_ydata = price 
# ax.grid(True) 

# fig.autofmt_xdate() 
# show()
