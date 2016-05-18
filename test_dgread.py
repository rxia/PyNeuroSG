# test DLSH dgread function
# from test_dgread import *

# set path
import os
import sys
cur_path = os.path.dirname(__file__)  # get the path of this file
sys.path.append( os.path.join(cur_path, 'dgread'))

import dgread
import pandas as pd
from scipy import stats

data_dg = dgread.dgread('/Volumes/Labfiles/tmp/d_MTS_bin_051116.dgz')

dg_keys = data_dg.keys()


def remove_short_columns(data_dg, f_verbose=False):

    # ----- get number of rows by majority vote of all columns -----
    num_rows = []  # a list of number of rows for every column
    for dg_key in dg_keys:
        num_rows.append(len(data_dg[dg_key]))
        if f_verbose:
            print('filed {} has {} rows'.format(dg_key, len(data_dg[dg_key]) ))
    N = stats.mode(num_rows)[0][0]  # get number of rows by majority vote

    # ----- remove columns with different numbers of rows -----
    for dg_key in dg_keys:
        if len(data_dg[dg_key]) != N:
            if f_verbose:
                print('remove dg column: {} that contains {} rows'.format(dg_key, len(data_dg[dg_key]) ))
            del data_dg[dg_key]


remove_short_columns(data_dg)
data_df = pd.DataFrame.from_dict(data_dg)
# data_df.to_hdf('test_hdf_store.h5', 'stimdg', mode='w')
data_df.to_json('test_hdf_store.json')

print("finish")