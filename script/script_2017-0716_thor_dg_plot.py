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



""" ===== test code for GroupPlot ===== """
reload(pnp)
N=600
values = np.random.randn(N)
x_continuous = np.random.randn(N)
x_discrete = np.random.randint(0,4, size=N)
c = ['a','b','c']*(N/3)
p = ['panel 1','panel 2']*(N/2)

# distinct plot type comparison, do not use panel
_, h_ax = plt.subplots(2,2)
h_ax = np.ravel(h_ax)
# conitnuous x
plt.axes(h_ax[0])
pnp.GroupPlot(values=values, x=x_continuous, c=c)
# descrete x, default box,
plt.axes(h_ax[1])
pnp.GroupPlot(values=values, x=x_discrete, c=c)
# descrete x, bar, more labels
plt.axes(h_ax[2])
pnp.GroupPlot(values=values, x=x_discrete, c=c, plot_type='bar', errbar='se', x_name='label_x', c_name='label_c', title_text='add text')
# descrete x, violin, do not use condition c, no legend, no count
plt.axes(h_ax[3])
pnp.GroupPlot(values=values, x=x_discrete, plot_type='violin', values_name='label_of_value', tf_legend=False, tf_count=False)


# use seprate data by panel
pnp.GroupPlot(values=values, x=x_discrete, c=c, p=p, plot_type='box', values_name='label_of_value', x_name='label_x', c_name='label_c')




""" get dg files and sort by time """

# dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'
list_name_dg_all = os.listdir(dir_dg)

keyword_dg = 'h.*_071717.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)

data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
reload(pnp); pnp.DfPlot(data_df[data_df['side']<0.5], values='response', x='TargetOnset', plot_type='box')

reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], p=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['file'], plot_type='dot', tf_legend=True, values_name='rt', c_name='TargetOnset', p_name='file')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='bar', tf_legend=True, values_name='rt', c_name='side', p_name='filename', errbar='se')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='violin', tf_legend=True, values_name='rt', c_name='side')



reload(pnp); pnp.GroupPlot(values=data_df['RT'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['file'], plot_type='bar', tf_legend=True, values_name='rt', x_name='TargetOnset', c_name='side', p_name='file')
reload(pnp); pnp.GroupPlot(values=data_df['RT'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='box', tf_legend=True, values_name='rt',  x_name='TargetOnset', c_name='side')
reload(pnp); pnp.GroupPlot(values=data_df['RT'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='violin', tf_legend=True, values_name='rt',  x_name='TargetOnset', c_name='side')

""" use DfPlot """
reload(pnp); pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', p='fileindex', plot_type='bar', errbar='se')