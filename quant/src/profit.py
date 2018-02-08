import numpy
import evaluate
import math
def findMax(arr):
    max = -1
    count = 0
    for i in arr:
        if max < i[2]:
            max = i[2] 
        count += 1
    return max,count
def cal_profit(ob, pre, obj):
    # 1. 止盈 
    #    固定比例p=1% ; 同期国债收益3倍 dp = exp(pro*3 * i) - 1
    #    max(p,e)
    # 2. 止损
    #    固定比例p -4%;  预测偏差mse > 0.2
    profit = 0
    deviation = 0
    start = ob[0,0]
    end = ob[-1,1]
    om = numpy.max(ob)
    pm,index = findMax(pre)
    pday = 0.12/250
    if pm > (start * 1.01):
        i = 0
        for e in obj:
            if e > 0.2:
                if i != 0:
                    end = ob[i,1]
                else:
                    end = ob[1,0]
                
                # print ("terminal at : e=%f, i=%d"%(e,i))
                break
            else:
                target = math.exp(i*pday) * end
                if ob[i, 2] > target:
                    end = target
                    break
                    # print ("5terminal at : e=%f, i=%d"%(e,i))
                if i >= index:
                    end = ob[i,1]
                    # print ("1terminal at : e=%f, i=%d"%(e,i))
                    break
                if pm < om:
                    end = pm
                    # print ("2terminal at : e=%f, i=%d"%(e,i))
                    break
        # print (numpy.append(ob,pre, axis=1))
        print ("start %f, end %f"%(start, end))
    else:
        # print ("3terminal at : end ")
        end = start
    profit = end - start
    profit = max(-0.4, profit)
    profit = profit / start
    deviation = end - pm
    deviation = deviation / start
    return profit,deviation
if __name__ == "__main__":
    symbol = '000725'
    data = numpy.loadtxt('result/' + symbol + 'compare.csv', skiprows=1)
    sum = 0
    for i in range(0, data.shape[0], 10):
        one = data[i:i+10]
        print("%d-----"%(i))
        ob = one[:,[0,1,2,3]]
        pre = one[:,[4,5,5,6]]
        obj = evaluate.mse(ob, pre)
        ret = cal_profit(ob, pre, obj)
        if ret[0] > 0:
            print(obj)
            print(ret)
        sum += ret[0]
    print (sum)
