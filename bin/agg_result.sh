#!/bin/sh
RSHOME=~/houlei/RecommendSystem
DATASET=$1
NFRAG=$2

RESULT_DIR=$RSHOME/results/$DATASET"_"$NFRAG
scp ubuntu@bd4:$RESULT_DIR/* $RESULT_DIR
scp ubuntu@bd5:$RESULT_DIR/* $RESULT_DIR

for result in $RESULT_DIR/*
do
  if test -f $result
  then
    grep "m" $result >> $RESULT_DIR/../result_movie
    grep "u" $result >> $RESULT_DIR/../result_user
  fi
done


