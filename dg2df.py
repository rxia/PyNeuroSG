# convert DLSH dg (dynamic group) to python pandas df (DataFrame)
# Shaobo Guan, 2016-0517 TUE

import pandas as pd
from scipy import stats
import dgread

def dg2df(dgfile):
    """
    convert DLSH stimdg file to Pandas DataFrame by using dgread module

    :param dgfile: path to stimdg fle
    :return:       Pandas DataDf
    """

    data_dg = dgread.dgread(dgfile)

    remove_short_columns(data_dg)

    data_df = pd.DataFrame.from_dict(data_dg)

    return data_df


def remove_short_columns(data_dg, f_verbose=False):
    """
    some columns in stimdg is shorter (e.g. contain only one row) than the majority of columns,
    but Panda DataFrame requires every column has the same length.  So we remove the short columns

    :param data_dg:   stimdg object
    :param f_verbose: if print out log
    :return:          stimdg object with every column of the same length
    """

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

