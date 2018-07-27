import store_hdf5
import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp
import math
import time
import mne
import copy
import misc_tools
import signal_align
import data_load_DLSH
import matplotlib as mpl
import matplotlib.pyplot as plt
import df_ana
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import data_load_DLSH
import GM32_layout
mpl.style.use('ggplot')

dates_good = ['180715','180718','180721','180723']
dates_bad = ['180713','180714','180717','180725','180726']

def load_data_from_h5(dir_data_save='/shared/homes/rxia/data', name='', block_type='', signal_type='', dates=[]):
    hdf_file_path = '{}/all_data_{}_{}.hdf5'.format(dir_data_save, name, block_type)
    print(hdf_file_path)
    dict_data = dict()
    for date in dates:
        dict_data[date] = store_hdf5.LoadFromH5(hdf_file_path, h5_groups=[date, block_type, signal_type])
    return(dict_data)

dir_data_save = '/shared/homes/rxia/data'
data_lfp = load_data_from_h5(dir_data_save,'thor','featureMTS','lfp',dates_good)

channels = np.array([1,2,3,5,6,8,11,17,19,22,23,27,28,31])
days = data_lfp.keys()
fs = 1/np.mean(np.diff(data_lfp[Dates_good[0]]['ts']))