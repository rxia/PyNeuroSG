""" script to load all dataset and get the six conditions, designed to run on Pogo """
import os
import numpy as np
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import re                   # regular expression
import time                 # time code execution
import pickle
import warnings

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import PyNeuroData as pnd   # in this package: align neural data according to task
import misc_tools           # in this package: misc
import store_hdf5           # save/load data using hdf5

import data_load_DLSH       # package specific for DLSH lab data

""" ========== set parameters ========== """

# ----- data directory
dir_data_save = '/shared/homes/rxia/data'
dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT/'
dir_dg = '/shared/lab/projects/analysis/ruobing/data_dg'

# ----- tank names to use
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*Thor.*GM32'
# keyword_tank = '.*Dexter.*GM32'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks = sorted(list_name_tanks)

# ----- filename (blockname) to use
block_type = 'srv_mask'

if block_type == 'featureMTS' or block_type == 'feature_reverse':
    t_plot = [-0.600, 1.600]
elif block_type == 'image':
    t_plot = [-0.500, 0.800]
elif block_type == 'spot':
    t_plot = [-0.300, 0.500]
elif block_type == 'movies':
    t_plot = [-0.300, 2.300]
elif block_type == 'srv_mask':
    t_plot = [-0.200, 0.600]
block_name_filter = '.*_.*{}.*'.format(block_type)

h5_filepath = '{}/all_data_thor_{}.hdf5'.format(dir_data_save, block_type)


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

    data_neuro_lfp = pnd.blk_align_to_evt(blk, ts_StimOn, t_plot,
                                          type_filter='ana.*', name_filter='LFPs.*')

    data_neuro_spk = pnd.blk_align_to_evt(blk, ts_StimOn, t_plot,
                                          type_filter='spiketrains.*', name_filter='.*Code[1-9]$',
                                          spike_bin_rate=data_neuro_lfp['signal_info']['sampling_rate'][0])

    pnd.include_trial_info(data_neuro_lfp, data_df=data_df)
    pnd.include_trial_info(data_neuro_spk, data_df=data_df)

    return [date_code, data_neuro_spk, data_neuro_lfp, data_df]


def SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, h5_filepath=h5_filepath):

    store_hdf5.SaveToH5(data_neuro_spk, h5_filepath=h5_filepath, h5_groups=[date_code, block_type, 'spk'])
    store_hdf5.SaveToH5(data_neuro_lfp, h5_filepath=h5_filepath, h5_groups=[date_code, block_type, 'lfp'])


for tankname in list_name_tanks:
    try:
        [date_code, data_neuro_spk, data_neuro_lfp, data_df] = LoadDataOneDay(tankname,block_name_filter)
        SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, h5_filepath)
    except:
        warnings.warn('tank {} cannot be processed'.format(tankname))


## Single days
[date_code, data_neuro_spk, data_neuro_lfp, data_df] = LoadDataOneDay('Thor_U16-180424','.*_.*srv.*')
SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, '/shared/homes/rxia/data/all_data_thor_srv.hdf5')
