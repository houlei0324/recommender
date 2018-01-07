__author__ = 'HouLei'
__date__ = '12/17/2017'

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import recommend

if __name__ == '__main__':
    recommend.run_recommender(sys.argv)
