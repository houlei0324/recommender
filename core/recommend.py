__author__ = 'HouLei'
__date__ = '12/15/2017'

import sys
import os
import re
import time
import math
import gflags
import numpy as np

import core.rs_log as log

FLAGS = gflags.FLAGS

gflags.DEFINE_string('query', '0', 'the query id')
gflags.DEFINE_string('query_class', 'user', "the class of query, to be one of \
user and movie")
gflags.DEFINE_string('sim_method', 'cos', 'the method of calculate the sim of \
users or items, including\n \
                    cos -- Cosine sim\n \
                    tan -- Tanimoto (Jaccard) Coefficient sim\n \
                    pcc -- Pearson Correlation Coefficient sim\n \
                    ed  -- Euclidean Distance sim')
gflags.DEFINE_string('choose_method', 'knn', 'the method of choose neighbors, \
                    the following methods can be choosed\n \
                    knn -- k nearest neighbors\n \
                    thd -- threshold based neighbors')
gflags.DEFINE_integer('k', 5, 'the number choosed when using knn')
gflags.DEFINE_float('threshold', 0.7, 'the threshold when using thd')

INFO = log.logger
log.init_log(INFO,'../logs/recommender.log')

class Recommend:
    query = '0'
    __vector_file = ''
    __sim_method = ''
    __choose_method = ''
    __threshold = 0
    __k = 0
    __neighbors = []
    __sim_value = {}
    __file_list = {
            'user' : '../results/result_user',
            'movie' : '../results/result_movie_clean'
            }

    def __init__(self, query_class, sim_method, choose_method, threshold, k):
        self.__vector_file = self.__file_list[query_class]
        self.__sim_method = sim_method
        self.__choose_method = choose_method
        self.__threshold = threshold
        self.__k = k
    
    def getQuery(self, query):
        self.query = query

    def cos_sim(self, src_vector, obj_vector):
        sim = sum(src_vector[i] * obj_vector[i] for i in range(0, len(src_vector)))
        tmp1 = math.sqrt(sum(pow(src_vector[i], 2) for i in range(0, len(src_vector))))
        tmp2 = math.sqrt(sum(pow(obj_vector[i], 2) for i in range(0, len(src_vector))))
        return sim / (tmp1 * tmp2)

    def tan_sim(self, src_vector, obj_vector):
        sim = sum(src_vector[i] * obj_vector[i] for i in range(0, len(src_vector)))
        tmp1 = math.sqrt(sum(pow(src_vector[i], 2) for i in range(0, len(src_vector))))
        tmp2 = math.sqrt(sum(pow(obj_vector[i], 2) for i in range(0, len(src_vector))))
        return sim / (tmp1 * tmp2 - sim)

    def pcc_sim(self, src_vector, obj_vector):
        return np.cov(src_vector, obj_vector) / (np.var(src_vector) * np.var(obj_vector))

    def ed_sim(self, src_vector, obj_vector):
        sim  = math.sqrt(sum(pow(src_vector[i] - obj_vector[i], 2) for i in range(0, len(src_vector))))
        return 1.0 / (1.0 + sim)

    def calculate_sim(self, src_vector, obj_vector):
        return {
            'cos' : self.cos_sim(src_vector, obj_vector),
            'tan' : self.tan_sim(src_vector, obj_vector),
            'pcc' : self.pcc_sim(src_vector, obj_vector),
            'ed' : self.ed_sim(src_vector, obj_vector)
        }.get(self.__sim_method, self.cos_sim(src_vector, obj_vector))

    def sort_sim(self):
         return sorted(self.__sim_value.items(), key = lambda x:x[1], reverse = True)

    def choose_neighbors(self):
        if self.__choose_method == 'knn':
            INFO.info("[Recommender] choose neighbors using knn")
            return self.sort_sim()[0 : self.__k]
        elif self.__choose_method == 'thd':
            INFO.info("[Recommender] choose neighbors using threshold based method")
            num = 0
            sim_sorted = self.sort_sim()
            while sim_sorted[num][1] > self.__threshold:
                num  = num + 1
            return sim_sorted[0 : num]
        else:
            INFO.info("[Recommender] choose neighbors using knn  and k = 5 in default")
            self.__k = 5
            return self.sort_sim()[0 : self.__k]

    def open_resultfile(self):
        return open(self.__vector_file, "r")

    def set_simvalue(self, pid, value):
        self.__sim_value[pid] = value
        

def run_recommender(argv):
    FLAGS(argv)
    query_time = time.time()
    recommend = Recommend(FLAGS.query_class, FLAGS.sim_method, FLAGS.choose_method, FLAGS.threshold, FLAGS.k)
    INFO.info('[Recommender] qet query : ' + FLAGS.query_class + '-' + FLAGS.query)
    recommend.getQuery(FLAGS.query)
    infile = recommend.open_resultfile()
    query_vector = []
    for line in infile.readlines():
        new_line = line.split()
        if new_line[1] == recommend.query:
            for i in range(2, len(new_line)):
                query_vector.append(float(new_line[i]))
            query_vector = np.array(query_vector)
            break
    infile.close()
    if len(query_vector) == 0:
        INFO.info("[Recommender] new user or movie, can not do recommend!")
        return 0


    infile = recommend.open_resultfile()
    num = 0
    for line in infile.readlines():
        line = line.split()
        if line[1] == recommend.query:
            continue
        param_vector = []
        for i in range(2, len(line)):
            param_vector.append(float(line[i]))
        param_vector = np.array(param_vector)
        recommend.set_simvalue(line[1], recommend.calculate_sim(query_vector, param_vector))
        num = num + 1
        if num % 10000 == 0:
            print(num)

    INFO.info(recommend.choose_neighbors())
    INFO.info('[Recommender] query time: ' + str(query_time - time.time() + 's')


