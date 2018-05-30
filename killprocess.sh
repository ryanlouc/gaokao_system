#!/bin/sh

NAME=$1  
echo $NAME  
ID=`ps -ef | grep "$NAME" | grep -v "grep" | awk '{print $2}'`
for id in $ID  
do  
kill -9 $id  
echo "killed $id"  
done    
