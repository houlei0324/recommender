__author__ = 'HouLei'
__date__ = '12/12/2017'

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import main

if __name__ == '__main__':
#    partition.data_partition("../db/ml-20m/ratings.e", 3)
    main.run(sys.argv)
