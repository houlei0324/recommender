#!/bin/sh
if [ $# == 0 ];then
  scp -r ~/houlei/RecommendSystem/ ubuntu@bd5:~/houlei/
  scp -r ~/houlei/RecommendSystem/ ubuntu@bd4:~/houlei/
else
  file=$1
    scp  -r ~/houlei/RecommendSystem/$file ubuntu@bd5:~/houlei/RecommendSystem
    scp  -r ~/houlei/RecommendSystem/$file ubuntu@bd4:~/houlei/RecommendSystem
fi

