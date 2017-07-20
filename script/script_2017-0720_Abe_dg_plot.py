import io
import matplotlib.pyplot as plt
from PySide.QtGui import QApplication, QImage

import sys
app = QApplication(sys.argv)

def add_clipboard_to_figures():
    # use monkey-patching to replace the original plt.figure() function with
    # our own, which supports clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)
        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf)

                blah = QImage.fromData(buf.getvalue())

                QApplication.clipboard().setImage(blah)
                buf.close()

        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig

    plt.figure = newfig

add_clipboard_to_figures()













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
dir_dg = 'L:/projects/analysis/ryan/data_dg'
list_name_dg_all = os.listdir(dir_dg)

keyword_dg = 'a.*_072017.*'

_, data_df, name_datafiles = data_load_DLSH.load_data(keyword=keyword_dg, tf_interactive=False, dir_dg=dir_dg, mode='dg')
data_df = data_load_DLSH.standardize_data_df(data_df)

data_df['truert'] = data_df['rts']-data_df['TargetOnset']


#run up to here, then choose a plotting command


resp_rate = 1.0*len(data_df)/data_df['obs_total'][0]
reload(pnp); pnp.DfPlot(data_df[data_df['side']<0.5], values='response', x='TargetOnset', plot_type='box')

reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], p=data_df['filename'], plot_type='dot')reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['filename'], plot_type='dot')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['obsid'], c=data_df['TargetOnset'], p=data_df['filename'], plot_type='dot', tf_legend=True, values_name='rt', c_name='TargetOnset', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='bar', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], p=data_df['filename'], plot_type='box', tf_legend=True, values_name='rt', c_name='side', p_name='filename')
reload(pnp); pnp.GroupPlot(values=data_df['rts'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='violin', tf_legend=True, values_name='rt', c_name='side')



reload(pnp); pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'], plot_type='box', tf_legend=True, values_name='rt', c_name='side')
plt.axis([-1,4,200,1000])

reload(pnp); pnp.GroupPlot(values=data_df['truert'], plot_type='box', tf_legend=True, values_name='rt', c_name='side')
plt.axis([-1,1,200,1000])


reload(pnp); pnp.GroupPlot(values=data_df['truert'], x=data_df['TargetOnset'], c=data_df['side'],p=data_df['filename'], limit=(data_df['status']==1), plot_type='bar', tf_legend=True, values_name='rt', c_name='side')

plt.axis([-1,12,200,1000])
