import numpy
def mse(ob, pri):
    ret = numpy.zeros(10)
    s = 0
    num = 0
    for o, p in zip(ob, pri):
        num += 1
        for i,j in zip(o,p):
            s += abs(i - j) /i
            ret[num - 1] =  s / num
    return ret