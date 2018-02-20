# convert DLSH dg (dynamic group) to python pandas df (DataFrame)
# Shaobo Guan, 2016-0517 TUE

import os
import sys
# cur_path = os.path.dirname(__file__)  # get the path of this file
# sys.path.append(os.path.join(cur_path, 'dgread'))

import pandas as pd
from scipy import stats
import dgread

def dg2df(dgfile):
    data_dg = dgread.dgread(dgfile)

    remove_short_columns(data_dg)

    data_df = pd.DataFrame.from_dict(data_dg)

    return data_df


def remove_short_columns(data_dg, f_verbose=False):
    # ===== remove irregular columns of dg, make it ready to be converted to pandas df =====
    dg_keys = data_dg.keys()
    if 'ids' in data_dg:
        # ----- get number of rows by key "ids" -----
        N = len(data_dg['ids'])
    else:
        # ----- get number of rows by majority vote of all columns -----
        num_rows = []  # a list of number of rows for every column
        for dg_key in dg_keys:
            num_rows.append(len(data_dg[dg_key]))
            if f_verbose:
                print('filed {} has {} rows'.format(dg_key, len(data_dg[dg_key]) ))
        N = stats.mode(num_rows)[0][0]  # get number of rows by majority vote

    # ----- remove columns with different numbers of rows -----
    for dg_key in list(dg_keys):
        if len(data_dg[dg_key]) != N:
            if f_verbose:
                print('remove dg column: {} that contains {} rows'.format(dg_key, len(data_dg[dg_key]) ))
            del data_dg[dg_key]

