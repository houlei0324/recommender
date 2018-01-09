#!/bin/sh
RSHOME=~/houlei/RecommendSystem
num_worker=$1
num_frag=$2
dataset=ml-20m
RESULT_DIR=$RSHOME/results/$dataset"_"$num_frag

if [ ! -d $RESULT_DIR ]; then
  mkdir $RESULT_DIR
else
  rm $RESULT_DIR/*
fi

scp -r $RESULT_DIR ubuntu@bd4:$RESULT_DIR
scp -r $RESULT_DIR ubuntu@bd5:$RESULT_DIR

mpirun -np $num_worker -hostfile ../conf/host python3 start.py\
  --nfrag $num_frag \
  --tolerance 0 \
  --dim 50 \
  --init_step_size  0.005 \
  --step_size_pow  0 \
  --lambd  0.01 \
  --max_iteration 20 \
  --result_dir $RESULT_DIR
