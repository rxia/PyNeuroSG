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


def order_consecutive(x):
    x = np.array(x)
    result = [0]
    for i in range(1, len(x)):
        cur = x[i]
        pre = x[i-1]
        if cur==pre:
            result.append(result[-1]+1)
        else:
            result.append(0)
    return np.array(result)


# dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'
list_name_dg_all = os.listdir(dir_dg)




""" get dg files and sort by time """
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

print "haha"

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


""" train left/right switch """
keyword_dg = 'h.*_072117.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

reload(pnp); pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='status', p='side', title_text=title_text)
reload(pnp); pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)




keyword_dg = 'h.*_072517.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)


pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.gcf().set_size_inches(10,5, forward=True)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='file', c='side', title_text=title_text)
plt.gcf().set_size_inches(10,5, forward=True)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))

pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='status', p='side', title_text=title_text)




""" 0726 """
keyword_dg = 'h.*_072617.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)
data_df['RT'] = data_df['rts'] - data_df['TargetOnset']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)


pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text, plot_type='bar', errbar='binom')
plt.gcf().set_size_inches(10,5, forward=True)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='file', c='side', title_text=title_text, plot_type='box')
plt.gcf().set_size_inches(10,5, forward=True)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))

pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='resp', p='status', title_text=title_text)


pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<6)*(data_df['file']>=30), title_text=title_text)
plt.savefig('./temp_figs/status_consecutive_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<6)*(data_df['file']>=30), title_text=title_text)
plt.savefig('./temp_figs/RT_consecutive_{}.png'.format(filename_common))



""" 0727 """
keyword_dg = 'h.*_072717.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df = data_df[ data_df['file']>=48 ]
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['TargetOnset']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='file', c='side', p='status', title_text=title_text)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))

pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', p='status', title_text=title_text)


pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/status_consecutive_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/RT_consecutive_{}.png'.format(filename_common))




""" 0728 """
keyword_dg = 'h.*_072817.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['TargetOnset']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='file', c='side', p='status', title_text=title_text)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))

pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', p='status', title_text=title_text)


pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/status_consecutive_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/RT_consecutive_{}.png'.format(filename_common))


""" 0809 """
keyword_dg = 'h.*_080917.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['TargetOnset']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='file', c='side', p='status', title_text=title_text)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))

pnp.DfPlot(data_df, values='status', x='TargetOnset', c='side', title_text=title_text)
pnp.DfPlot(data_df, values='RT', x='TargetOnset', c='side', p='status', title_text=title_text)


pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/status_consecutive_{}.png'.format(filename_common))
pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', p='status', limit=(data_df['order_consecutive']<5), title_text=title_text)
plt.savefig('./temp_figs/RT_consecutive_{}.png'.format(filename_common))



""" 0810, matchnot """
keyword_dg = 'h.*_081017.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['delay']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

h_fig, h_ax = plt.subplots(3,1, figsize=[6,8])
plt.axes(h_ax[0]);  pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.axes(h_ax[1]);  pnp.DfPlot(data_df, values='status', x='file', c='ImageOpacity', title_text=title_text)
plt.axes(h_ax[2]);  pnp.DfPlot(data_df, values='status', x='file', c='HelperOpacity', title_text=title_text)
plt.savefig('./temp_figs/stasus_{}.png'.format(filename_common))

h_fig, h_ax = plt.subplots(3,1, figsize=[6,8])
plt.axes(h_ax[0]);  pnp.DfPlot(data_df, values='RT', x='file', c='side', title_text=title_text)
plt.axes(h_ax[1]);  pnp.DfPlot(data_df, values='RT', x='file', c='ImageOpacity', title_text=title_text)
plt.axes(h_ax[2]);  pnp.DfPlot(data_df, values='RT', x='file', c='HelperOpacity', title_text=title_text)
plt.savefig('./temp_figs/RT_{}.png'.format(filename_common))


""" 0914 """

keyword_dg = 'h_matchnot.*_091417.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['delay']
data_df['order_consecutive'] = order_consecutive(data_df['side'])

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

h_fig, h_ax = plt.subplots(2,1, figsize=[6,8])
plt.axes(h_ax[0]);  pnp.DfPlot(data_df, values='status', x='file', c='side', title_text=title_text)
plt.axes(h_ax[1]);  pnp.DfPlot(data_df, values='RT', x='file', c='side', title_text=title_text)
plt.savefig('./temp_figs/training_performance_by_file_{}.png'.format(filename_common))
h_fig, h_ax = plt.subplots(2,1, figsize=[6,8])
plt.axes(h_ax[0]); pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', title_text=title_text, limit=(data_df['order_consecutive']<=24))
plt.axes(h_ax[1]); pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', title_text=title_text, limit=(data_df['order_consecutive']<=24))
plt.savefig('./temp_figs/training_performance_by_switch_{}.png'.format(filename_common))


""" 0915 """

keyword_dg = 'h_matchnot.*_091517.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=True, dir_dg=dir_dg, mode='dg')

# select a subset of files
data_df.reset_index(inplace=True, drop=True)

data_df = data_load_DLSH.standardize_data_df(data_df)
filename_common = misc_tools.str_common(name_datafiles)

data_df['RT'] = data_df['rts'] - data_df['delay']
data_df['order_consecutive'] = order_consecutive(data_df['side'])
data_df['order_consecutive'] = np.clip(data_df['order_consecutive'], 0, 5)

resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
title_text = '{}, resp_rate={:.2f}'.format(filename_common, resp_rate)

h_fig, h_ax = plt.subplots(2,1, figsize=[6,8])
data_df['file_coarse'] = data_df['file']//5*5
plt.axes(h_ax[0]);  pnp.DfPlot(data_df, values='status', x='file_coarse', c='side', title_text=title_text)
plt.axes(h_ax[1]);  pnp.DfPlot(data_df, values='RT', x='file_coarse', c='side', title_text=title_text)
plt.savefig('./temp_figs/training_performance_by_file_{}.png'.format(filename_common))
h_fig, h_ax = plt.subplots(2,1, figsize=[6,8], sharex=True)
plt.axes(h_ax[0]); pnp.DfPlot(data_df, values='status', x='order_consecutive', c='side', title_text=title_text)
plt.axes(h_ax[1]); pnp.DfPlot(data_df, values='RT', x='order_consecutive', c='side', title_text=title_text)
plt.savefig('./temp_figs/training_performance_by_switch_{}.png'.format(filename_common))