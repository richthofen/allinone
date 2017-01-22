#!/bin/bash
# string s_cmd = "curl --connect-timeout 10 -m 20 -o " + s_file_name + " \"http://market.finance.sina.com.cn/downxls.php?date=" + sDate + "&symbol=" + symbol + "\"";
# string s_cmd = "C:\\Users\\Administrator\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe - c \"http://market.finance.sina.com.cn/downxls.php?date=" + sDate + "&symbol=" + symbol + "\"";
# string s_url = "http://market.finance.sina.com.cn/downxls.php?date=" + sDate + "&symbol=" + symbol;

cd /data/workwork/
sDate=`date +"%F"`

trans(){
for i in `ls tmp`
do
s_file_name=tmp/$i
grep alert $s_file_name && continue
sed -i '1d' $s_file_name
cat $s_file_name | iconv -f GBK -t utf-8  | awk -v date="$sDate" '{print date " " $1 "," $2 "," $3 "," $4 "," $5 "," $6}' >  tran/$i
done
}
para()
{
while [ `jobs |wc -l` -eq 20 ] ;do  #判断后台下载任务数量是否在20个，如果是则等待一段时间，否就新增一个下载任务
  echo 'waitting...'
  sleep 1;
done
}
rm -f tmp/*
rm -f tran/* 

[ -n $1 ] && sDate="$1"
for i in `seq 300000 300400`
do
para
symbol=sh`printf "%06d\n" $i`
s_file_name="/data/workwork/tmp/"$symbol
curl --connect-timeout 10 -m 20 -o $s_file_name "http://market.finance.sina.com.cn/downxls.php?date=$sDate&symbol=$symbol" &
done
for i in `seq 300000 300600`
do
para
symbol=sz`printf "%06d\n" $i`
s_file_name="/data/workwork/tmp/"$symbol
curl --connect-timeout 10 -m 20 -o $s_file_name "http://market.finance.sina.com.cn/downxls.php?date=$sDate&symbol=$symbol" &
done
for i in `seq 600000 604000`
do
para
symbol=sh`printf "%06d\n" $i`
s_file_name="/data/workwork/tmp/"$symbol
curl --connect-timeout 10 -m 20 -o $s_file_name "http://market.finance.sina.com.cn/downxls.php?date=$sDate&symbol=$symbol" &
done
for i in `seq 1 003000`
do
para
symbol=sz`printf "%06d\n" $i`
s_file_name="/data/workwork/tmp/"$symbol
curl --connect-timeout 10 -m 20 -o $s_file_name "http://market.finance.sina.com.cn/downxls.php?date=$sDate&symbol=$symbol" &
done

trans

