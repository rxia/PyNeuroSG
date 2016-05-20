# test DLSH dgread function
# from test_dgread import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

# import the customize module
cur_path = os.path.dirname(__file__)  # get the path of this file
sys.path.append(cur_path)
import dg2df


# set path of datafile
dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'
file_dg = 'd_MTS_bin_051916009.dg'
path_dg = os.path.join(dir_dg, file_dg)

# load file
data_df = dg2df.dg2df(path_dg)

# save df to disk
# data_df.to_hdf('test_hdf_store.h5', 'stimdg', mode='w')
# data_df.to_json('test_json_store.json')

# analyze
data_df.groupby('NoiseOpacity')[['status']].agg(np.mean).plot()



print("finish")