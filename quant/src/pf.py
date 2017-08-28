# coding=gbk
import os
import datetime
import re
import tushare as ts
try:
    # py3
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
except ImportError:
    # py2
    from urllib2 import Request, urlopen
    from urllib import urlencode


def _request(symbol):
    try:
        url = 'http://doctor.10jqka.com.cn/%s/#nav_basic' % (symbol)
        print(url)
        req = Request(url)
        resp = urlopen(req)
        content = resp.read().decode("gbk").strip()
        #print(content);
    except Exception as ex:
        print('except:%s'%ex)
        content = 'null'
    return content
def get_stock(symbol,filen):
    content = _request(symbol)
    #content = retstring().decode("gb2312").strip();
    #print content;
    pattern = re.compile(r'<div class=\"stocktotal\">.*(\d\.\d)')
    match = pattern.findall(content)
    #print len(match)
    print match
    if len(match)>=1:
        print(symbol+ "-" +match[0])
        filen.write(symbol+ "-" +match[0]+"\n")
    return 0
def get_all_stock():
    filew = open("pf.txt",'w')
    all_num = ('6','0','3')
    count = 0
    all = ts.get_hs300s()
    for symbol in all['code'].values:
        get_stock(symbol,filew)
    return 0
    
get_all_stock()
