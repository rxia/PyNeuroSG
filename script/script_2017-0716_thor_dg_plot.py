import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import warnings
import misc_tools


# costom packages
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')
import dg2df  # for reading behavioral data
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import data_load_DLSH


""" get dg files and sort by time """

# dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'
list_name_dg_all = os.listdir(dir_dg)

keyword_dg = 'h.*_071717.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]

""" use DfPlot """
# pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', p='fileindex', plot_type='bar', errbar='se')
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', plot_type='box', title_text='{}, resp_rate={:.2f}'.format(filename_common, resp_rate))
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))



""" train left/right switch """
keyword_dg = 'h.*_071917.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)
pnp.DfPlot(data_df, values='status', x='file', c='side', plot_type='bar', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='file', c='side', plot_type='box', title_text='{}, resp_rate={:.2f}'.format(filename_common, resp_rate))
pnp.DfPlot(data_df, values='RT', x='side', c='status', plot_type='violin', title_text='{}, resp_rate={:.2f}'.format(filename_common, resp_rate))


reload(pnp); pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='RT', x='file', c='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='status', p='side', title_text=title_text)



""" train left/right switch """
keyword_dg = 'h.*_072017.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

reload(pnp); pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='status', p='side', title_text=title_text)
