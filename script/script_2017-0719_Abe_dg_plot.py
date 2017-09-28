
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
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import data_load_DLSH

""" get dg files and sort by time """

# dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'
dir_dg = '/shared/lab/projects/analysis/ryan/data_dg'
list_name_dg_all = os.listdir(dir_dg)

keyword_dg = 'a.*_071917.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=False, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)

data_df['truert'] = data_df['rts']-data_df['TargetOnset']

# run up to here



""" distinct plot type comparison, do not use panel """
_, h_ax = plt.subplots(2,2, figsize=[10,8])
h_ax = np.ravel(h_ax)
plt.suptitle('show case of distinct plot type/style for GroupPlot')
# conitnuous x
plt.axes(h_ax[0])
pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'])
# descrete x, default box,
plt.axes(h_ax[1])
pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'])
# descrete x, bar, more labels
plt.axes(h_ax[2])
pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='bar', errbar='se',
              values_name='label_of_value', x_name='label_x', c_name='label_c', title_text='add text')
# descrete x, violin, do not use condition c, no legend, no count
plt.axes(h_ax[3])
pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], plot_type='violin', tf_legend=False, tf_count=False)







resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
reload(pnp); pnp.DfPlot(data_df[data_df['side']<0.5], values='response', x='TargetOnset', plot_type='box')

reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], p=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['filename'], plot_type='dot', tf_legend=True, values_name='rt', c_name='TargetOnset', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='bar', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='violin', tf_legend=True, values_name='rt', c_name='side')
reload(pnp); pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'],  plot_type='box', tf_legend=True, values_name='rt', c_name='side')
reload(pnp); pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'],p=data_df['filename'], limit=(data_df['status']==1), plot_type='bar', tf_legend=True, values_name='rt', c_name='side')

