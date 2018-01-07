__author__ = 'HouLei'
__date__ = '12/12/2017'

import sys
import os
import re
import gflags
import mpi4py.MPI as MPI
import numpy as np

import core.rs_log as log

INFO = log.logger
FLAGS = gflags.FLAGS

gflags.DEFINE_string('pdataset', '../db/ml-20m/ratings.e', 'the input dataset')
gflags.DEFINE_integer('pnfrag', 4, "the number of fragments")


def data_partition(argv):
    FLAGS(argv)
    filename = FLAGS.pdataset
    nfrag = FLAGS.pnfrag
    infile = open(filename, "r")
    dirname = os.path.dirname(os.path.abspath(filename))
    parts_dir = dirname + "/parts"
    if os.path.exists(parts_dir) == False:
        os.mkdir(parts_dir)
    else:
        file_list = os.listdir(parts_dir)
        if len(file_list) == nfrag:
            infile.close()
            return parts_dir
        else:
            shutil.rmtree(parts_dir)
            os.mkdir(parts_dir)
    part_file = []
    rank = 1
    origin_file = re.split(r'[./]', filename)
    while rank <= nfrag:
        outfile = open(parts_dir + "/" + origin_file[-2] +
                        ".part_" + str(rank), "w")
        part_file.append(outfile)
        rank = rank + 1
    INFO.info("Partition %s into %d parts..." %(filename, nfrag))
    line_num = 0
    for line in infile.readlines():
        rating = re.split(r'[\s]', line)
        part_file[int(rating[0]) % nfrag].write(line)
        line_num = line_num +1
        if (line_num % 1000000 == 0):
            INFO.info(line_num)
    for outfile in part_file:
      outfile.close()
    infile.close()

    return line_num

def data_loading(filename, comm, comm_rank, comm_size):
    user_movie = []
    path = filename

    if comm_rank == 0:
        file_list = os.listdir(path)
        INFO.info("[Processor 0] %d files" % len(file_list))
    file_list = comm.bcast(file_list if comm_rank == 0 else None, root = 0)
    num_files = len(file_list)
    #local_files_offset = np.linspace(0, num_files, comm_size +1).astype('int')
    #local_files = file_list[local_files_offset[comm_rank] 
    #                        :local_files_offset[comm_rank + 1]]
    #INFO.info("[Processor %d] gets %d/%d data "
    #            %(comm_rank, len(local_files), num_files))
    cnt = 0
    if comm_rank > 0:
        file_name = ''
        for local_file in file_list:
            local_list = local_file.split('_')
            if local_list[1] == str(comm_rank):
                file_name = local_file
                break
        INFO.info("[Processor %d] gets %s" %(comm_rank, file_name))
        hd = open(os.path.join(path, file_name))
        num = 0
        for line in hd:
            user_movie.append(line.split())
        #    if num % 1000000 == 0:
        #        INFO.info("[Processor %d] %d" %(comm_rank, num))
            num = num + 1
        user_movie = np.array(user_movie)
        user_movie = user_movie.astype(np.float64)
        cnt += 1
        INFO.info("[Processor %d] has processed %s files" %(comm_rank, file_name))
        hd.close()

    return user_movie
