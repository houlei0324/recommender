__author__ = 'HouLei'
__date__ = '12/12/2017'

import sys
import os
import gflags
import time
import mpi4py.MPI as MPI
import numpy as np

import core.rs_log as log
import core.load_data as LOAD
import core.dsgd as dsgd

FLAGS = gflags.FLAGS

gflags.DEFINE_string('dataset', '../db/ml-20m/ratings.e', 'the input dataset')
gflags.DEFINE_string('result_dir', '../results', 'the dir for results')
gflags.DEFINE_integer('nfrag', 4, "the number of fragments")
gflags.DEFINE_integer('dim', 50, 'the dim of parameter vectors')
gflags.DEFINE_integer('max_iteration', '100', 'the max iteration to run')
gflags.DEFINE_float('init_step_size', 0.5, 'init learning rate')
gflags.DEFINE_float('step_size_offset', 100.0, 'sgd step size at iteration t is \
                    init_step_size * (step_size_offset + t)^(-step_size_pow)')
gflags.DEFINE_float('step_size_pow', 0.5, 'sgd step size at iteration t is \
                    init_step_size * (step_size_offset + t)^(-step_size_pow)')
gflags.DEFINE_float('lambd', 0.01, 'regulatization, (lambda)')
gflags.DEFINE_float('tolerance', 1.0, 'tolerance')

INFO = log.logger
log.init_log(INFO,'../logs/dsgd.log')
# instance for invoking MPI related functions
comm = MPI.COMM_WORLD
# the node rank in the whole community
comm_rank = comm.Get_rank()
# the size of the whole community (the total number of working nodes)
comm_size = comm.Get_size()

def run(argv):
    FLAGS(argv)
    file_size = 20000001

    data_dir = os.path.dirname(os.path.abspath(FLAGS.dataset)) + "/parts"
    load_time = 0
    run_time = 0
    if comm_rank == 0:
        load_time = time.time()
    user_movie = LOAD.data_loading(data_dir, comm, comm_rank, comm_size)
    comm.barrier()
    if comm_rank == 0:
        load_time = time.time() - load_time

    mf_dsgd = dsgd.DSGD(user_movie, comm, comm_rank, FLAGS.dim, FLAGS.nfrag, file_size,
                        FLAGS.init_step_size, FLAGS.step_size_offset, FLAGS.step_size_pow,
                        FLAGS.lambd, FLAGS.tolerance, FLAGS.max_iteration, FLAGS.result_dir)
    if comm_rank == 0:
        run_time = time.time()
    mf_dsgd.run_dsgd()
    if comm_rank == 0:
        run_time = time.time() - run_time
        INFO.info('[DSGD] load data time: ' + str(load_time) + 's')
        INFO.info('[DSGD] run time: ' + str(run_time) + 's')
