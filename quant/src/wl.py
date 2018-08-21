import numpy
import evaluate
import math
import os
import json
from os import path
import send
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
    print (arr)
    print (arr.shape)
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
def checkOut(symbol, line, start):
    start = float(start)
    abs_path =  path.join(_MODULE_PATH, 'result/' + symbol + '*')
    print (abs_path)
    flist = os.popen('ls ' + abs_path).readlines()
    file = flist[-1].strip()
    arr = file.split('-')
    no = int(line) - 5
    last = arr[0] + '-' + str(no) 
    last = os.popen('ls ' + last + '*').readlines()[0].strip()
    lastpre = numpy.loadtxt(last).astype('float64')
    ob_path = path.join(_MODULE_PATH, 'data/' + symbol + '.csv1') 
    # print(ob_path)
    observe = numpy.loadtxt(ob_path, dtype=numpy.str )
    # line = > index
    line = int(line) - 1
    print(observe.shape)
    print(line)
    ob = observe[line:,[2,3,4,5,8]]
    ob = ob.astype(numpy.float)
    # print (ob)
    # print (ob.shape)
    # print (lastpre)
    print(lastpre)
    pre = numpy.loadtxt(file).astype('float64')

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
        return symbol
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
    # if ob[i, 3] < start:
    #     inloss += 1
    # i += 1
    # print (numpy.append(ob,pre, axis=1))
    # loss in 
    # if inloss:
    #     print ("loss day   %d    " % (inloss))
    print ("start %f, end %f"%(start, end))
    profit = end - start
    profit = profit / start
    profit = max(-0.03, profit)
    deviation = om - start
    deviation = deviation / start
    # print (pre)
    # print (lastpre)
    # sum = 0
    # ret = cal_profit(ob, pre, obj, lastobj)
    return None
def checkIn(symbol):
    abs_path =  path.join(_MODULE_PATH, 'result/' + symbol + '*')
    print (abs_path)
    flist = os.popen('ls ' + abs_path).readlines()
    file = flist[-1].strip()
    arr = file.split('-')
    no = int(arr[1]) - 5
    last = arr[0] + '-' + str(no) 
    last = os.popen('ls ' + last + '*').readlines()[0].strip()
    lastpre = numpy.loadtxt(last).astype('float64')
    ob_path = path.join(_MODULE_PATH, 'data/' + symbol + '.csv1') 
    # print(ob_path)
    ob = numpy.loadtxt(ob_path, dtype=numpy.str )
    ob = ob[no:,[2,3,4,5,8]]
    ob = ob.astype(numpy.float)
    print (ob)
    print (ob.shape)
    # print (lastpre)
    print(lastpre)
    pre = numpy.loadtxt(file).astype('float64')
    # print (pre)
    # print (lastpre)
    # sum = 0
    lastobj = evaluate.mse(ob, lastpre)
    if lastobj.any():
        if lastobj[:5].sum() > 2.5:
            # print(lastobj)
            pass
            # return
    # print("%d -----"%(i))
    
    profit = 0
    deviation = 0
    end = ob[-1,1]
    om = numpy.max(ob[1:,2])
    pday = 0.12 / 250
    preProfit,buy,sell=findTime(pre)
    pm = pre[sell,2]
    preBuy = pre[buy, 3]
    # buy = findfirstMin(ob, preBuy, buy)
    want = 999999
    close = ob[-1, 1] 
    if preBuy > close:
        want = min(want, close * 0.99)
    if pre[buy, 4] > close:
        want = min(want , close * 0.99)
    start = min (want, ob[buy, 0]) 
    print (preProfit, buy, sell)
    if preProfit > 0.03:
        return symbol, str(arr[1]), str(arr[2]), start
    # ret = cal_profit(ob, pre, obj, lastobj)
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
        with open(in_path, 'r') as hold_file:
            tmp =  hold_file.read()
            # print(tmp)
            if tmp != "":
                sym_map = json.loads(tmp)
        print (sym_map)
        symbols = sym_map.keys()
        # mark in symbol
        # hint out 
        out_map = {}
        # hint in
        in_map = {}
        xsum = 0
        xcount = 0
        for s in data:
            if '' != s:
                # target_file = 'result/' + s + 'pre.csv'
                # if os.path.isfile(target_file) :
                print("%s -----abc"%(s))
                # if s not in symbols:
                if s  not in symbols:
                    print("%s -----abc"%(s))
                    symb, line, dt, start = checkIn(s)
                    if symb is not None:
                        # with open(in_path, 'b') as f:
                            # print(symb + " " + line + " " + dt)
                            # f.write(("%s %s %s " %(symb, line, dt)).encode())
                        in_map[symb] = {"line":line, "dt":dt, "start": start}
                        sym_map[symb] = {"line":line, "dt":dt, "start": start}
                else:
                    # checkOut
                    out = checkOut(s, sym_map[s]["line"], sym_map[s]["start"])
                    if out is not None:
                        out_map[s] = {}
        # send msg
        msg = "" 
        for i in in_map.keys():
            msg += i + " " + str(in_map[i]["start"]) + " \n"
        for i in out_map.keys():
            msg += "out " + i + " \n"
        print (msg) 
        if msg != "":
            send.send_msg(msg)
        # save info 
        with open(in_path, 'w') as hold_file:
            dump = json.dumps(sym_map)
            print(dump)
            hold_file.write(dump)