__author__ = 'HouLei'
__date__ = '12/12/2017'

import logging
  
# create a logger  
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_log(logger, logfile):
    # create a handler for log into files
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    # create a handler for log on console 
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # the format of handler
    formatter = logging.Formatter("%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
