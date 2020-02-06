""" behavioral data analysise """

##
import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp

import misc_tools
import signal_align
import data_load_DLSH
import matplotlib as mpl
import matplotlib.pyplot as plt
import df_ana
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import data_load_DLSH
mpl.style.use('ggplot')


##
""" Dante  """

dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'

keyword_tank='Dante_U16-180603'
keyword_blk='retract'

# name_tdt_block = 'd_MTS_bin_072216.*'
name_tdt_block = 'd_MTS_bin_(0722|0803)16.*'
# name_tdt_block = 'd_MTS_bin_080316.*'


[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
    name_tdt_block, keyword_tank, mode='dg',
    dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg,
    tf_interactive=False, tf_verbose=False)


bwmaskonset_update = np.array(data_df['BwMaskOnset'])
bwmaskonset_update = (bwmaskonset_update - 50* (np.abs(bwmaskonset_update-150)<=0.01))
data_df['BwMaskOnset'] = bwmaskonset_update

data_df['fam_nov'] = ['fam' if item>0 else 'nov' for item in data_df['SampleFamiliarized']]


data_df['accuracy'] = data_df['status']
data_df['BwMask onset'] = data_df['BwMaskOnset']
data_df['occluding noise'] = data_df['NoiseOpacity']


# df_ana.DfPlot(data_df, 'status', x='MaskOpacity', c='SampleFamiliarized', p='', limit=None, plot_type=None)
# df_ana.DfPlot(data_df, 'status', x='NoiseOpacity', c='BwMaskOnset', p='SampleFamiliarized',
#               binom_alpha=0.1, tf_count=False)
df_ana.DfPlot(data_df, 'accuracy', x='occluding noise', c='BwMask onset', p='fam_nov',
              binom_alpha=0.1, tf_count=False)
plt.ylim(0.48, 1.02)
plt.gcf().set_size_inches(8,4)
plt.savefig('./temp_figs/final_Dante_Thor/behavior_bwmask_Dante.pdf')
plt.savefig('./temp_figs/final_Dante_Thor/behavior_bwmask_Dante.png')


##
""" Thor  """

dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'

keyword_dg = 'h_matchnot.*_052718.*'
# keyword_dg = 'h_matchnot.*_05..18.*'

if False:
    for day in range(31):
        keyword_dg = 'h_matchnot.*_05{}18.*'.format(day)
        try:
            [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
                keyword=keyword_dg, mode='dg', dir_dg=dir_dg,
                tf_interactive=False, tf_verbose=False)
            print(keyword_dg, data_df['BwMaskOnset'].unique())
        except:
            pass

keyword_dg = 'h_matchnot.*_052718.*'
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
    keyword=keyword_dg, mode='dg', dir_dg=dir_dg,
    tf_interactive=False, tf_verbose=False)


mask_opacity_update = np.array(data_df['MaskOpacity'])
mask_opacity_update = np.round((mask_opacity_update + 0.1* (np.abs(mask_opacity_update-0.5)<=0.01)) *100)/100
mask_opacity_update = np.round((mask_opacity_update - 0.1* (np.abs(mask_opacity_update-0.7)<=0.01)) *100)/100
data_df['MaskOpacity'] = mask_opacity_update

data_df['fam_nov'] = ['fam' if item>0 else 'nov' for item in data_df['SampleFamiliarized']]
data_df['accuracy'] = data_df['status']
data_df['BwMask onset'] = data_df['BwMaskOnset']
data_df['occluding noise'] = data_df['MaskOpacity']


# df_ana.DfPlot(data_df, 'status', x='MaskOpacity', c='SampleFamiliarized', p='', limit=None, plot_type=None)
# df_ana.DfPlot(data_df, values='status', x='MaskOpacity', c='BwMaskOnset', p='SampleFamiliarized',
#               binom_alpha=0.1, tf_count=False)
df_ana.DfPlot(data_df, 'accuracy', x='occluding noise', c='BwMask onset', p='fam_nov',
              binom_alpha=0.1, tf_count=False)


plt.ylim(0.48, 1.02)
plt.gcf().set_size_inches(8,4)
plt.savefig('./temp_figs/final_Dante_Thor/behavior_bwmask_Thor.pdf')
plt.savefig('./temp_figs/final_Dante_Thor/behavior_bwmask_Thor.png')

##
