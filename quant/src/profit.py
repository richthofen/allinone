import numpy
import evaluate
import math
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
    p = 1
    start = min(preBuy * 1.01, ob[buy,0] * p)
    # start = min(preBuy, ob[buy,0] * p)
    target = start * 1.01
    ob_low = ob[buy,3] 
    maxProfit,_,_ = findTime(ob)
    if preProfit < 0.06:
        print ("3terminal not ")
        return 0,maxProfit,False
    if ob[buy,3] > start :
        print ("terminal buy failed buy:%f, open:%f, low:%f xx%f " %(start, ob[buy,0],ob[buy,3], (ob_low - start)/ob_low ))
        return 0,maxProfit,False
    if buy != 0:
        if obj[buy-1] > 0.2:
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
                if ob[i,3] < start * 0.99 and e > 0.1:
                    end = start * 0.99
                    print ("terminal out loss : sold=%f, i=%d "%(ob[i,1], i))
                    break
                if ob[i,3] < start * 0.98:
                    end = start * 0.98
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
def anylayzeOne(file):
    data = numpy.loadtxt(file)
    sum = 0
    obj = numpy.array([])
    count = 0
    rate = 0
    for i in range(10, data.shape[0], 10):
        one = data[i:i+10]
        ob = one[:,[0,1,2,3]]
        pre = one[:,[4,5,6,7]]
        lastobj = obj
        obj = evaluate.mse(ob, pre)
        if lastobj.any():
            if lastobj[:5].sum() > 2.5:
                continue
        print("%d -----"%(i))
        ret = cal_profit(ob, pre, obj, lastobj)
        if ret[1] > 0.1:
            print("one dev -----")
        if ret[2] != False:
            if ret[0] > 0:
                rate += 1
            print(ret)
            print(obj)
            print(lastobj)
            sum += ret[0]
            count += 1
        else:
            print(ret)
        
            
    if count != 0:
        print("abc sum:%f, count:%d, avg:%f, rate:%f"%(sum,count,sum/count, rate /count))
    else:
        print("not operate")
    return sum,count
import os.path
if __name__ == "__main__":
    with open('back') as file:
        data = file.read().split("\n")
        xsum = 0
        xcount = 0
        for s in data:
            if '' != s:
                target_file = 'result/' + s + 'compare.csv'
                if os.path.isfile(target_file) :
                    print("%s -----abc"%(s))
                    a,b = anylayzeOne(target_file)
                    xsum += a
                    xcount += b
        print("abc final sum:%f, count: %d, avg: %f"%(xsum, xcount, xsum/xcount))
