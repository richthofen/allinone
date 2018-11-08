# coding=gbk
import os
import datetime
import re
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
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER')
        resp = urlopen(req)
        content = resp.read().decode("gb2312").strip()
        #print(content);
    except Exception as ex:
        print('except:%s'%ex)
        content = 'null'
    return content
def retstring():
    return "<div class=\"stocktotal\">综合诊断：6.8分 打败了97%的股票！</div>";
def get_stock(symbol,filen):
    content = _request(symbol)
    #content = retstring().decode("gb2312").strip();
    #print content;
    pattern = re.compile(r'<div class=\"stocktotal\">.*(\d\.\d)')
    match = pattern.findall(content)
    #print len(match)
    if len(match)>=1:
        print(symbol+ "-" +match[0])
        filen.write(symbol+ "-" +match[0]+"\n")
    return 0
def get_all_stock():
    filew = open("pf.txt",'w')
    all_num = ('6','0','3')
    count = 0
    while (count < 3):
        cc = 0
        cc_max = 4000
        while ( cc < cc_max):
            stock_index = '%s0%04d' %(all_num[count],cc)
            print ('name = %s' %(stock_index))
            get_stock(stock_index,filew)
            cc = cc + 1
        count = count + 1
    return 0
	
#get_stock("000661",open("pf.txt",'w'))
get_all_stock()
