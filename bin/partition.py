__author__ = 'HouLei'
__date__ = '12/15/2017'

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_data

if __name__ == '__main__':
    load_data.data_partition(sys.argv)
