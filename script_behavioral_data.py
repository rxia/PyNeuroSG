import os
import sys

import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import warnings

# costom packages
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')
import dg2df  # for reading behavioral data
import PyNeuroPlot as pnp

""" get dg files and sort by time """

dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'
list_name_dg_all = os.listdir(dir_dg)

time_start = ('16','10','16','000')
time_end   = ('17','12','01','000')
task_name  = 'd_matchnot'

list_dg = []
list_dg_time = []
for name in list_name_dg_all:
    obj_match = re.match('.*_(\d\d)(\d\d)(\d\d)(\d\d\d)\.dg', name)
    try:
        time = obj_match.group(3,1,2,4)
        if time > time_start and time < time_end and re.match('.*{}.*'.format(task_name), name):
            list_dg_time.append( time )
            list_dg.append(name)
    except:
        warnings.warn('filename {} can not be parsed'.format(name))

[time_dgs, file_dgs] = zip(*sorted(zip(list_dg_time, list_dg)))



""" load stim_dg files """
data_dfs = []          # list containing multiple pandas dataframes, each represents data form one file
for i, file_dg in enumerate(file_dgs):  # for every data file, read as a segment of block
    if True:
        print('loading dg: {}'.format(file_dg))
    path_dg = os.path.join(dir_dg, file_dg)
    data_df = dg2df.dg2df(path_dg)  # read using package dg2df, returns a pandas dataframe
    data_df['filename'] = [file_dg] * len(data_df)  # add a column for filename
    data_df['fileindex'] = [i] * len(data_df)  # add a column for file id (index from zero to n-1)
    data_dfs.append(data_df)

if len(data_dfs) > 0:
    data_df = pd.concat(data_dfs)  # concatenate in to one single data frame
    data_df = data_df.reset_index(range(len(data_df)))
else:
    data_df = pd.DataFrame([])

data_df[''] = [''] * len(data_df)  # empty column, make some default condition easy


""" check experimental conditions """
data_df['stim_names'] = data_df.SampleFilename
data_df['stim_familiarized'] = data_df.SampleFamiliarized
data_df['mask_opacity'] = data_df['MaskOpacity']
# make mask opacity a int, better for printing
data_df['mask_opacity_int'] = np.round(data_df['mask_opacity'] * 100).astype(int)
# make short name for stim, e.g. face_fam_6016
data_df['stim_sname'] = map((lambda (a, b): re.sub('_\w*_', '_fam_', a) if b == 1 else re.sub('_\w*_', '_nov_', a)),
                            zip(data_df['stim_names'].tolist(), data_df['stim_familiarized'].tolist(), ))


""" plot """
pnp.DfPlot(data_df, 'status','mask_opacity_int','stim_familiarized')
plt.ylim([0.5,1.0])
plt.savefig('./temp_figs/behavior_correct_rate.png')

data_df['reaction_time'] = data_df['rts']- data_df['delay']
pnp.DfPlot(data_df, 'reaction_time','mask_opacity_int','stim_familiarized')
plt.savefig('./temp_figs/behavior_rt.png')
