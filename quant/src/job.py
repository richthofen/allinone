import numpy
import evaluate
import math
import os
import json
import datetime
from os import path
#from datetime import datetime
import datetime
import send
import sql


in_message = ""
out_message = ""
def checkIn():
  global in_message
  print("checkin")
  pre = sql.get_hold_pre_data()
  print (pre)
  symbols = [] if len(pre) == 0 else pre[:,0]
  for p in pre:
    symbol = p[0]
    print (symbol)
    fresh = sql.get_data (symbol)[-1:][0]
    print (fresh)
    if fresh[4]  < p[3]:
       # print ("in")
       in_message += symbol + " : " + p[3] + "\n"
       # update holding
       sql.update_hold_data(status="holding", symbol=symbol, dt=p[1])
   # send info
   
def checkOut():
  global out_message
  print("check out ")
  holding = sql.get_hold_holding_data()
  print (holding)
  for h in holding:
    symbol = h[0]
    print (symbol)
    fresh = sql.get_data (symbol)[-1:][0]
    print ('fresh  ' + str(fresh))
    if fresh[4]  > h[4]:
       # print ("in")
       out_message += symbol + " : " + h[3] + "\n"
       # update holding
       sql.update_hold_data(status="solding", symbol=symbol, dt=h[1])
   # send info
#checkOut()
checkIn()
msg=""
msg += in_message
msg += "- - -\n" 
msg += out_message
if in_message != "" or out_message != "":
    a = datetime.date.today()
    print (datetime.date.today())  
    print (msg)
    send.send_msg(msg)