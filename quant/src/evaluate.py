import numpy
def mse(ob, pre):
    pre = numpy.array(pre).astype(numpy.float)
    ob = numpy.array(ob).astype(numpy.float)
    ret = numpy.zeros(ob.shape[0])
    s = 0
    num = 0
    for o, p in zip(ob, pre):
        num += 1
        for i,j in zip(o,p):
            # print (i)
            # print (i.dtype)
            # print (j.dtype)
            s += abs(i - j) / i
            ret[num - 1] =  s / num
    return ret
