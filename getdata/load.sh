#!/bin/bash
d=$1
cd /data/workwork
rm -f tran/load.sql 
for i in `ls tran`
do
echo  "LOAD DATA LOCAL INPATH '/data/workwork/tran/$i' into table dw.detail PARTITION (date_time='$d', symbol='$i');"  >> tran/load.sql
done
hive -f tran/load.sql

