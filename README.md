An Application of the DSGD using Python and mpi

Authors: Lei Hou, Dongze Li
ToDo: What we are going to do is an recommend system based on MF, we are going to complete an application of the DSGD using mpi in python, then try to use the results to do some recommend based on the calculation of similation.
Dataset: ml-20m, from MovieLens, which contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 9, 1995 and March 31, 2015. This dataset was generated on October 17, 2016. http://grouplens.org/datasets/movielens/

Dependence;

-- python3
   -- numpy
   -- mpi4py
   -- gflags

Install:



Run:

-- DSGD
   
cd bin
python3 ./partition.py  $nfrag  # to partition dataset
./sync.sh db
./run $nworker $nfrag
./agg_result.sh
python3 ./clean.py

-- Recommender

python3 ./recommender.py --query --query_class  and other params
