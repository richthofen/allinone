import numpy
def mse(ob, pri):
    ret = numpy.zeros(10)
    s = 0
    num = 0
    count = 0
    for o, p in zip(ob, pri):
        maxt = max(pri[num, 0], pri[num,1])
        mint = min(pri[num, 0], pri[num,1])
        maxn = pri[num, 2] 
        minn = pri[num, 3] 
        num += 1
        for i,j in zip(o,p):
            s += abs(i - j) /i
        if maxt > maxn:
            count += 1
            s += (count * abs(maxt - maxn))/maxn
        if minn < mint:
            count += 1
            s += (count * abs(minn - mint))/minn
        ret[num - 1] =  s / num
    return ret