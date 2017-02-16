#!/bin/bash
i=0
v=`date --date="-$i day" +"%F"`
while [ $v != "1999-01-01" ]
do
i=$(($i + 1))
v=`date --date="-$i day" +"%F"`
sh dump.sh  $v
sh load.sh  $v
done
