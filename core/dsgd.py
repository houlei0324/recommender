__author__ = 'HouLei'
__date__ = '12/13/2017'

import sys
import os
import re
import math
import mpi4py.MPI as MPI
import numpy as np

import core.rs_log as log
INFO = log.logger

class DSGD:
    __nfrag = 0
    __comm = 0
    __comm_rank = 0
    __loss = 0.0
    __init_step_size = 0.5
    __step_size_offset = 0.0
    __step_size_pow = 0.0
    __lambda = 0.01
    __tolerance = 1.0
    __size = 0
    __shut_down = 0
    __max_iter = 0
    __matrix = np.array([])
    __user = {}
    __movie = {}
    __result_dir = ''

    superstep = 0
    iteration = 0

    def __init__(self, matrix, comm, rank, k, nfrag, size, init_step_size, \
                step_size_offset, step_size_pow, lambd, tolerance, max_iter, result_dir):
        self.__comm_rank = rank
        self.__matrix = matrix
        self.__comm = comm
        self.__nfrag = nfrag
        self.__size = size
        self.__init_step_size = init_step_size
        self.__step_size_offset = step_size_offset
        self.__step_size_pow = step_size_pow
        self.__lambda = lambd
        self.__max_iter = max_iter
        self.__tolerance = tolerance
        self.__result_dir = result_dir

        if self.__comm_rank != 0:
            for line in self.__matrix:
                userID = int(line[0])
                movieID = int(line[1])
                if userID not in self.__user:
                    self.__user[userID] = 1 - 2 * np.random.rand(k)
                if movieID not in self.__movie:
                    self.__movie[movieID] = 1 - 2 * np.random.rand(k)
        INFO.info("[Processor %d] Init finished!" %self.__comm_rank)

    def message_send(self, params, dst):
        self.__comm.send(params, dest = dst)

    def message_recv(self, src):
        data_recv = self.__comm.recv(source = src)
        if isinstance(data_recv, dict):
            for key in data_recv:
                if key in self.__movie:
                    self.__movie[key] = data_recv[key]
    
    def barrier(self):
        self.__comm.barrier()

    def sgd(self):
        step_size = self.__init_step_size * pow(self.__step_size_offset + \
                    self.superstep, -self.__step_size_pow)
        movie_params_updated = {}
        summary = 0
        item = -1
        for line in self.__matrix:
            movieID = int(line[1])
            item = movieID
            if (movieID % self.__nfrag == (self.iteration + self.__comm_rank) \
                                            % self.__nfrag):
                userID = int(line[0])
                old_rating = line[2]
                old_user = self.__user[userID]
                rating = np.dot(self.__user[userID], self.__movie[movieID])
                tmp = rating - old_rating

                self.__user[userID] = self.__user[userID] - \
                        (self.__movie[movieID] * tmp + self.__user[userID] * \
                        self.__lambda) * step_size
                self.__movie[movieID] = self.__movie[movieID] - \
                        (old_user * tmp + self.__movie[movieID] * \
                        self.__lambda) * step_size
                movie_params_updated[movieID] = self.__movie[movieID]
                self.__loss = self.__loss + pow((np.dot(self.__user[userID], \
                              self.__movie[movieID]) - old_rating), 2)
            summary = summary + 1
        #INFO.info("[Processor %d] process %d lines" %(self.__comm_rank, summary))
        example = '0'
        movie_params_updated[example] = self.__movie[item]
        
        send_to = (self.iteration + self.__comm_rank) % self.__nfrag + 1
        offset = self.__comm_rank - (self.iteration + 1) % self.__nfrag
        if offset <= 0:
            offset = offset + self.__nfrag

        self.message_send(movie_params_updated, 0)
        #INFO.info("[Processor %d] Send messages to %d" %(self.__comm_rank, send_to))
        self.barrier()
        self.message_recv(0)
        #INFO.info("[Processor %d] Recvied messages form %d" %(self.__comm_rank, offset))
        self.barrier()

        self.__comm.send(self.__loss, dest = 0)
        self.__loss = 0.0
        self.barrier()

    def step_up(self):
        self.iteration = self.iteration + 1
        self.superstep = self.iteration / self.__nfrag

    def result_output(self):
        if self.__comm_rank > 0:
            outfile = open(self.__result_dir + "/result_" + str(self.__comm_rank), "w")
            for user in self.__user:
                out = "u " + str(user)
                for val in self.__user[user]:
                    out = out + " " + str(val)
                out = out + "\n"
                outfile.write(out)
                #outfile.write("u " + str(user) + " "  + np.array2string \
                #        (self.__user[user], precision=6, separator=',', \
                #        suppress_small=True) + "\n")
            for movie in self.__movie:
                out = "m " + str(movie) 
                for val in self.__movie[movie]:
                    out = out + " " + str(val)
                out = out + "\n"
                outfile.write(out)
                #outfile.write("m " + str(movie) + " " + np.array2string \
                #        (self.__movie[movie], precision=6, separator=',', \
                #        suppress_small=True)+ "\n")
            outfile.close()

    def run_dsgd(self):
        self.barrier()
        while self.__shut_down == 0:
            if self.__comm_rank == 0:
                INFO.info("[Processor %d] Start %d iteration ..." \
                        %(self.__comm_rank, self.iteration))
                recv_matrix = []
                for i in range(1, self.__nfrag + 1):
                    recv_matrix.append(self.__comm.recv(source = i))
                self.barrier()
                for i in range(1, self.__nfrag + 1):
                    send_to = (self.iteration + i) % self.__nfrag + 1
                    self.message_send(recv_matrix[i-1], send_to)
                self.barrier()
                for i in range(1, self.__nfrag + 1):
                    self.__loss = self.__loss + self.__comm.recv(source = i)
                #INFO.info("[Processor 0] loss is %f" %self.__loss)
                if self.iteration % self.__nfrag == self.__nfrag - 1:
                    self.__loss = math.sqrt(self.__loss / self.__size)
                    INFO.info("[Processor 0] Superstep %d loss is %f" \
                            %(self.superstep, self.__loss))
                    if self.__loss < self.__tolerance or \
                       self.superstep >= self.__max_iter :
                        self.__shut_down = 1
                    self.__loss = 0.0
                self.barrier()
                INFO.info("[Processor %d] Finished %d iteration ..." \
                        %(self.__comm_rank, self.iteration))
            else:
                self.sgd()
            self.__shut_down = self.__comm.bcast(self.__shut_down \
                    if self.__comm_rank == 0 else None, root = 0)
            self.barrier()
            self.step_up()
        if self.__comm_rank == 0:
            INFO.info("[Processor 0] Finished all iterations and Start to save results.")
        INFO.info("[Processor %d] Start to save results into files" \
                %(self.__comm_rank))
        self.result_output()
        INFO.info("[Processor %d] Finish to save results into files" \
                %(self.__comm_rank))
