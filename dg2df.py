# DLSH dg (dynamic group) to python pandas df (DataFrame)
# Shaobo Guan, 2016-0517 TUE

import os
import sys
cur_path = os.path.dirname(__file__)  # get the path of this file
sys.path.append( os.path.join(cur_path, 'dgread'))

import dgread
import pandas as pd
from scipy import stats