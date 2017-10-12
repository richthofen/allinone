#!/bin/sh
gettokenURL="https://qyapi.weixin.qq.com/cgi-bin/gettoken"
sendmsgURL="https://qyapi.weixin.qq.com/cgi-bin/message/send"
CORPID="wxb6bffdb6811c3034"
CORPSECRET="vxc4fROTxzTmPJEfsYQCOrMGMNi66EYy6wk36D_zmAqY_3BFIL6Sy2BKFAvRSqUO"


line=`echo $3`
if [[ "$line" != "Priority: 1" ]]; then
    ret=`curl -s "https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=$CORPID&corpsecret=$CORPSECRET" | python -c 'import sys,json;print json.load(sys.stdin)["access_token"]'`
    if [ $? != 0 ];then
        echo "getAccessToken failed" >> /tmp/get-token-fail.log
        exit
 fi
     curl  -H "Accept: application/json" -H "Content-type: application/json" -X POST -d "{\"touser\":\"$4\",\"msgtype\":\"text\",\"agentid\":\"$1\",\"text\":{\"content\":\"$line\"},\"safe\":\"0\"}" $sendmsgURL?access_token=$ret
fi
