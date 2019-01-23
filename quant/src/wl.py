import numpy
import evaluate
import math
import os
import json
from os import path
from datetime import datetime

import send
import sql
_MODULE_PATH = path.dirname(__file__)
import sys
def findMax(arr):
    ret = -1
    count = 0
    for i in arr:
        ret = max(ret, i[2])
        count += 1
    return ret,count
def findfirstMin(arr, xmin, buy):
    count = 0
    for i in arr:
        if i[3] < xmin:
            break
        count += 1
    ret = min(buy, count) 
    return ret
def findTime(arr):
    arr = numpy.array(arr).astype(float)
    # print (arr)
    # print (arr.shape)
    profit = 0
    buy = 0
    sell = 0
    curMin = arr[0,3]
    curMinIdx = 0
    for i in range(arr.shape[0] -1):
        if arr[i,3] < curMin:
            curMin = arr[i,3]
            curMinIdx = i
        if curMin < arr[i+1,2]:
            p = max(profit, (arr[i+1,2] - curMin)/curMin)
            if(p > profit):
                sell = i+1
                buy = min(curMinIdx,4)
                profit = p
    return profit,buy,sell
def cal_profit(ob, pre, obj,lastobj):
    # 1. 止盈 
    #    固定比例p=1% ; 同期国债收益3倍 dp = exp(pro*3 * i) - 1
    #    max(p,e)
    # 2. 止损
    #    固定比例p -4%;  预测偏差mse > 0.2
    profit = 0
    deviation = 0
    end = ob[-1,1]
    om = numpy.max(ob[1:,2])
    pday = 0.12 / 250
    preProfit,buy,sell=findTime(pre)
    pm = pre[sell,2]
    preBuy = pre[buy, 3]
    print("p:%f, buy:%d, sell:%d, prebuy:%f"%(preProfit, buy, sell, preBuy))
    buy = findfirstMin(ob, preBuy, buy)
    print("p:%f, buy:%d, sell:%d, prebuy:%f"%(preProfit, buy, sell, preBuy))
    p = 0.99
    want = 999999
    if preBuy > ob[buy,0]:
        want = min(want, ob[buy, 0] * 0.99)
    if pre[buy, 4] > ob[buy, 0]:
        want = min(want , ob[buy, 0] * 0.99)
    start = min (want, ob[buy, 0])
    # start = min(preBuy * 1.01, ob[buy,0] * p)
    # start = min(preBuy, ob[buy,0] * p)
    # target = start * 1.01
    target = start * (1 + min(0.2, preProfit))
    ob_low = ob[buy,3] 
    maxProfit,_,_ = findTime(ob)
    if preProfit < 0.03:
        print ("3terminal not ")
        return 0,maxProfit,False
    if ob[buy,3] > start :
        print ("terminal buy failed buy:%f, open:%f, low:%f xx%f " %(start, ob[buy,0],ob[buy,3], (ob_low - start)/ob_low ))
        return 0,maxProfit,False
    if buy != 0:
        if obj[buy-1] > 0.3:
            print ("mse out before at : e=%f, i=%d"%(obj[buy-1],buy))
            return 0,maxProfit,False
    i = buy
    inloss = 0
    for e in obj[buy:]:
        if e > 0.2:
            if pre[i, 2] > ob[i,0]:
                if i != buy:
                    end = ob[i,1]
                else:
                    end = ob[i+1,0]
                print ("mse out terminal at : e=%f, i=%d"%(e,i))
                # print (numpy.append(ob,pre, axis=1))
                break
            else:
                print ("testl at : e=%f, i=%d"%(pm,i))
                pass
        # elif e > 0.1 and i == 0:
        #     end = ob[1,0]
        #     print ("mse out terminal at first : e=%f, i=%d"%(e,i))
        #     break
        else:
            up = 0
            down = 0
            if lastobj.any():
                x = i + 4
                if lastobj[x-3:x].sum() < 0.6:
                    up = pm - (pm - start) /sell * i
            down = math.exp(i * pday) * start
            m = max(up,down) 
            target = max(target,m)
            if i != buy:
                #  get up bond
                if ob[i, 2] > target:
                    end = max(ob[i,0], target)
                    print ("terminal at : e=%f, i=%d , t=%f"%(e, i, target))
                    break
                # get low bond 
                if ob[i,3] < start * 0.98 and e > 0.3:
                    end = ob[i,1]
                    print ("terminal out loss : sold=%f, i=%d "%(ob[i,1], i))
                    break
                if ob[i,3] < start * 0.9:
                    end = ob[i,1]
                    print ("terminal out loss : sold=%f, i=%d "%(ob[i,1], i))
                    break
            elif i >= sell:
                end = ob[i,1]
                print ("1terminal at : e=%f, i=%d"%(e,i))
                break
            else:
                print("un caught at: %d"%(i))
        if ob[i, 3] < start:
            inloss += 1
        i += 1
    # print (numpy.append(ob,pre, axis=1))
    # loss in 
    if inloss:
        print ("loss day   %d    " % (inloss))
    print ("start %f, end %f"%(start, end))
    profit = end - start
    profit = profit / start
    profit = max(-0.03, profit)
    deviation = om - start
    deviation = deviation / start
    return profit,maxProfit - profit,True
# def checkOut(symbol, dt, start):
def checkOut(symbol, dt = None, start=None):
    hold = sql.get_holding_data(symbol)
    print (hold)
    start = float(hold[0,3])
    dt = hold[0,1]
    observe = sql.get_data_from_date(symbol, dt)
    lastpre = sql.get_predict(symbol, dt)
    # print(lastpre)
    # print(observe)
    ob = observe[:,[1,2,3,4,6]]
    ob = ob.astype(numpy.float)
    dt = observe[-1,0]
    pre =  sql.get_predict(symbol, dt)[:,1:6].astype(float)
    print (pre)

    profit = 0
    deviation = 0
    end = ob[-1,1]
    om = numpy.max(ob[0:,2])
    pday = 0.12 / 250
    preProfit,buy,sell=findTime(pre)
    i = ob.shape[0] - 1
    # pm = pre[sell,2]
    p = 0.99
    want = 999999
    preBuy = pre[buy, 3]
    # if preBuy > ob[buy,0]:
    #     want = min(want, ob[buy, 0] * 0.99)
    # if pre[buy, 4] > ob[buy, 0]:
    #     want = min(want , ob[buy, 0] * 0.99)
    # start = float(observe[line][2])
    # target = start * 1.01
    target = start * (1 + min(0.2, preProfit))
    # ob_low = ob[buy,3] 
    maxProfit,_,_ = findTime(ob)
    # start = min (want, ob[i, 0])
    lastobj = evaluate.mse(ob, pre)
    pm = pre[sell,2]
    # for e in obj[buy:]:
    e = evaluate.mse(ob, pre)[-1]
    # 4 dbg
    if e > 0.2:
    # if e > 9999:
        if pre[i, 2] > start:
            end = ob[-1,1]
            print ("mse out terminal at : e=%f, i=%d"%(e,i))
        else:
            print ("testl at : e=%f, i=%d"%(pm,i))
            # pass
        # return symbol
    else:
        up = 0
        down = 0
        if lastobj.any():
            x = i + 4
            if lastobj[x-3:x].sum() < 0.6:
                up = pm - (pm - start) /sell * i
        down = math.exp(i * pday) * start
        m = max(up,down) 
        target = max(target,m)
        #  get up bond
        if ob[i, 2] > target:
            end = max(ob[i,0], target)
            print ("terminal at : e=%f, i=%d , t=%f"%(e, i, target))
            return symbol
            # break
        # get low bond 
        if ob[i,3] < start * 0.98 and e > 0.3:
            end = ob[i,1]
            print ("terminal out loss : sold=%f, i=%d "%(ob[i,1], i))
            return symbol
            # return
            # break
        if ob[i,3] < start * 0.9:
            end = ob[i,1]
            print ("terminal out loss : sold=%f, i=%d "%(ob[i,1], i))
            return symbol
            # return
            # break
        if i >= sell:
            end = ob[i,1]
            print ("1terminal at : e=%f, i=%d"%(e,i))
            return symbol
            # return
            # break
        print("un caught at: %d"%(i))
    print ("start %f, end %f"%(start, end))
    profit = end - start
    profit = profit / start
    profit = max(-0.03, profit)
    deviation = om - start
    deviation = deviation / start
    
    status = "sold"
    sql.update_hold_end(status, end, profit, dt, hold[0,1], symbol) 
    return None
def checkIn(symbol):
    ob_data = numpy.array(sql.get_data_lastest(symbol))
    # print (ob_data)
    if ob_data.shape[0] == 0 :
        print ("not observe data")
        return None,None,None,None 
    dt = ob_data[0,0]
    ob = ob_data[1:,[1,2,3,4]].astype(float)
    lastpre = sql.get_predict_from_date(symbol, dt)
    if lastpre.shape[0] == 0 :
        return None,None,None,None 
    lastpre = lastpre[:,[1,2,3,4,6]]
    if len(lastpre) == 0:
        return None,None,None,None 
    # dt = datetime.now().strftime("%Y-%m-%d")
    dt = ob_data[-1,0] 
    pre = sql.get_predict_from_date(symbol, dt)
    # print (pre)
    if len(pre) == 0:
        print("no predict data")
        return None,None,None,None 


    pre = pre[:,[1,2,3,4,6]].astype(float)
    lastobj = evaluate.mse(ob, lastpre)
    if lastobj.any():
        lo = lastobj[:5].sum()
        if lo > 2.5:
        # if lastobj[:5].sum() < 2.5:
            print("last obj gt 2.5 %f" % lo)
            # pass
            return None,None,None,None
    # print("%d -----"%(i))
    
    profit = 0
    deviation = 0
    end = ob[-1,1]
    om = numpy.max(ob[1:,2])
    pday = 0.12 / 250
    preProfit,buy,sell=findTime(pre)
    pm = pre[sell,2]
    preBuy = pre[buy, 3]
    want = 999999
    close = ob[-1, 1] 
    if preBuy > close:
        want = min(want, close * 0.99)
    if pre[buy, 4] > close:
        want = min(want , close * 0.99)
    start = min (want, preBuy) 
    print (preProfit, buy, sell)
    if buy > 1:
        print ("not today")
        return None, None, None, None
    # if preProfit < 0.03:
    if preProfit > 0.03:
    # statement = """INSERT or ignore INTO hold (symbol, date, status, checkin, checkout, profit, original)  VALUES(?,?,?,?,?,?,?)""" 
        data=[symbol, dt, "pre", start, 0, preProfit, "ori", dt, sell]
        data = numpy.array(data).reshape(1,9)
        sql.add_hold(data)
        return symbol, dt, dt, start
    return None,None,None,None
import os.path
if __name__ == "__main__":
    abs_path =  path.join(_MODULE_PATH, 'back')
    with open(abs_path) as file:
    # with open('back') as file:
        data = file.read().split("\n")
        in_path =  path.join(_MODULE_PATH, 'in')
        print (in_path)
        # record buy symbol
        # hold = numpy.loadtxt(in_path, dtype=numpy.str,  delimiter=" ", ndmin=2)
        sym_map = {}
        in_map = {}
        out_map = {}
        holding = sql.get_holding()  
        print ("holding")
        print (holding)
        # symbols = [] 
        symbols = [] if len(holding) == 0 else holding[:,0]
        xsum = 0
        xcount = 0
        msg = "" 
        for s in data:
            if '' != s:
                print("%s -----abc"%(s))
                # if s not in symbols:
                if s  not in symbols:
                    # try:
                    symb, line, dt, start = checkIn(s)
                    if symb is not None: 
                      msg += symb + " " + str(start) + "\n"
                else:
                    # checkOut
                    out = checkOut(s)
                    if out is not None:
                      msg += symb + " \n"
            # sys.exit()
        # send msg
        for i in in_map.keys():
            msg += i + " " + str(in_map[i]["start"]) + " \n"
        for i in out_map.keys():
            msg += "out " + i + " \n"
        print (msg) 
        # if msg != "":
            # send.send_msg(msg)
        
    # 更新 last 5 day pre
    
