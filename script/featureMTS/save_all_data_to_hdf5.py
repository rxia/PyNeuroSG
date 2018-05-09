""" script to load all dataset and get the six conditions, designed to run on Pogo """

import os
import numpy as np
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import re                   # regular expression
import time                 # time code execution
import pickle
import warnings
import h5py

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import signal_align         # in this package: align neural data according to task
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

""" ========== set parameters ========== """

# ----- data directory
dir_data_save = '/shared/homes/rxia/data'
dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT/'
dir_dg = '/shared/lab/projects/analysis/ruobing/data_dg'

# ----- tank names to use
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*Dexter.*GM32'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks = sorted(list_name_tanks)

# ----- filename (blockname) to use
block_type = 'image'
if block_type == 'featureMTS':
    t_plot = [-0.600, 1.600]
elif block_type == 'image':
    t_plot = [-0.500, 0.800]
elif block_type == 'spot':
    t_plot = [-0.300, 0.500]
block_name_filter = 'x_.*{}.*'.format(block_type)

h5filepath = '{}/all_data_dexter_{}.hdf5'.format(dir_data_save, block_type)

""" ========== define functions ========== """
def LoadDataOneDay(tankname, block_name_filter=block_name_filter):

    date_code = re.match('.*?-(\d{6}).*', tankname).group(1)

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(block_name_filter, tankname,
                                                               tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank,
                                                               dir_dg=dir_dg)

    filename_common = misc_tools.str_common(name_tdt_blocks)
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    data_neuro_lfp = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*',
                                                   name_filter='LFPs.*')

    data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                       name_filter='.*Code[1-9]$', spike_bin_rate=data_neuro_lfp['signal_info'][0]['sampling_rate'])


    return [date_code, data_neuro_spk, data_neuro_lfp, data_df]


def get_signal_id(signal_info, date_code):
    signal_info = pd.DataFrame(signal_info)
    signal_info['date'] = [date_code]*len(signal_info)
    signal_info['signal_id'] = signal_info['date']
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{:0>6}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return np.array(signal_info['signal_id']).astype('S16')


def get_df_valid_for_hdf5(data_df):
    keep_col = []
    for col in data_df.keys():
        if col=='':
            pass
        elif data_df[col].dtype != np.dtype('O'):
            keep_col.append(col)
        elif type(data_df[col][0]) is str:
            keep_col.append(col)
        else:
            pass
    return data_df[keep_col]


def SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, data_df, h5filepath=h5filepath):

    signal_id_spk = get_signal_id(data_neuro_spk['signal_info'], date_code)
    signal_id_lfp = get_signal_id(data_neuro_lfp['signal_info'], date_code)
    data_df_str = data_df.to_json(path_or_buf=None)
    data_df_valid_for_h5 = get_df_valid_for_hdf5(data_df)

    with h5py.File(h5filepath, 'a') as hf:

        if date_code not in hf.keys():
            h_data_day = hf.create_group(date_code)
            hf[date_code].create_dataset('trial_info_json', data=data_df_str)
        if 'spk' not in hf[date_code]:
            hf[date_code].create_group('spk')
            hf[date_code]['spk'].create_dataset('data', data=data_neuro_spk['data'])
            hf[date_code]['spk'].create_dataset('ts', data=data_neuro_spk['ts'])
            hf[date_code]['spk'].create_dataset('signal_id', data=signal_id_spk)
        if 'lfp' not in hf[date_code]:
            hf[date_code].create_group('lfp')
            hf[date_code]['lfp'].create_dataset('data', data=data_neuro_lfp['data'])
            hf[date_code]['lfp'].create_dataset('ts', data=data_neuro_lfp['ts'])
            hf[date_code]['lfp'].create_dataset('signal_id', data=signal_id_lfp)

    if True:
        with pd.HDFStore(h5filepath) as hf_pandas:
            hf_pandas.put(key='{}/trial_info'.format(date_code), value=data_df, data_columns=True)

n = 0
for tankname in list_name_tanks:
    try:
        [date_code, data_neuro_spk, data_neuro_lfp, data_df] = LoadDataOneDay(tankname)
        SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, data_df)
        n = n+1
        print(n)
    except:
        warnings.warn('tank {} cannot be processed'.format(tankname))




""" ========== below: temp script for testing ========== """
