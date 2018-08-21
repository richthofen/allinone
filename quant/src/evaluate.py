import numpy
def mse(ob, pre):
    print (ob.shape)
    ret = numpy.zeros(ob.shape[0])
    s = 0
    num = 0
    for o, p in zip(ob, pre):
        num += 1
        for i,j in zip(o,p):
            # print (i.dtype)
            # print (j.dtype)
            # print (s)
            s += abs(i - j) / i
            ret[num - 1] =  s / num
    return ret
