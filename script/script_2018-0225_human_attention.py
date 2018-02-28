import os
import re
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import dg2df

""" load data """

path_to_dg = '/shared/lab/projects/analysis/ruobing/featureIOR/data/feature'
dg_files = os.listdir(path_to_dg)
dg_files = [dg_file for dg_file in dg_files if dg_file[-3:] == '.dg']

list_dgs = [dg2df.dg2df(os.path.join(path_to_dg, dg_file)) for dg_file in dg_files]
df = pd.concat(list_dgs)
df.reset_index(inplace=True)

""" get grouped indices """
groupby = ['FeatureMatchNot', 'SpatialMatchNot']
idx_grp = df.groupby(groupby).indices

""" smooth time """
t = np.array(df['delay']-df['SampleOnset'], dtype='float')
hit = np.array(df['status'], dtype='float')
rt = np.array(df['rts']-df['delay'], dtype='float')
fa = (rt<250).astype('float')

std = 10
tt = np.arange(150, 1300, 10)

def smooth_y(x, y, std=10, xx=tt, limit=None):
    if limit is not None:
        x = x[limit]
        y = y[limit]
    weight = np.exp(-(x[None, :] - xx[:, None])**2/(2*std**2))
    yy = np.array([np.average(y, weights=weight[i, :]) for i in range(len(xx))])
    return yy

def rand_select(a, b):
    return np.random.choice(a, min(len(b), len(a)), replace=False)

# limit = fa<0.5
# yy = smooth_y(t, hit)
# plt.plot(tt, yy)
# yy = smooth_y(t, hit, limit=limit)
# plt.plot(tt, yy)

std = 30




h_fig, h_axes = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[12,9])

plt.axes(h_axes[0, 0])
yy_00 = smooth_y(t, hit, std=std, limit=idx_grp[(0,0)])
yy_10 = smooth_y(t, hit, std=std, limit=idx_grp[(1,0)])
# yy_10 = smooth_y(t, hit, std=std, limit=rand_select(idx_grp[(1,0)], idx_grp[(0,0)]))
plt.plot(tt, yy_00, '--', c='k', lw=2, label='feature nonmatch')
plt.plot(tt, yy_10,  '-', c='k', lw=2, label='feature match')
plt.fill_between(tt, yy_00, yy_10, where=yy_00>yy_10)
plt.fill_between(tt, yy_00, yy_10, where=yy_00<yy_10)
plt.legend()
plt.title('spatial nonmatch')

plt.axes(h_axes[0, 1])
yy_01 = smooth_y(t, hit, std=std, limit=idx_grp[(0,1)])
yy_11 = smooth_y(t, hit, std=std, limit=idx_grp[(1,1)])
plt.plot(tt, yy_01, '--', c='k', lw=2, label='feature nonmatch')
plt.plot(tt, yy_11,  '-', c='k', lw=2, label='feature match')
plt.fill_between(tt, yy_01, yy_11, where=yy_01>yy_11)
plt.fill_between(tt, yy_01, yy_11, where=yy_01<yy_11)
plt.legend()
plt.title('spatial match')

plt.axes(h_axes[1, 0])
yy_00 = smooth_y(t, hit, std=std, limit=idx_grp[(0,0)])
yy_01 = smooth_y(t, hit, std=std, limit=idx_grp[(0,1)])
# yy_10 = smooth_y(t, hit, std=std, limit=rand_select(idx_grp[(1,0)], idx_grp[(0,0)]))
plt.plot(tt, yy_00, '--', c='k', lw=2, label='Spatial nonmatch')
plt.plot(tt, yy_01,  '-', c='k', lw=2, label='Spatial match')
plt.fill_between(tt, yy_00, yy_01, where=yy_00>yy_01)
plt.fill_between(tt, yy_00, yy_01, where=yy_00<yy_01)
plt.legend()
plt.title('Feature nonmatch')

plt.axes(h_axes[1, 1])
yy_10 = smooth_y(t, hit, std=std, limit=idx_grp[(1,0)])
yy_11 = smooth_y(t, hit, std=std, limit=idx_grp[(1,1)])
plt.plot(tt, yy_10, '--', c='k', lw=2, label='spatial nonmatch')
plt.plot(tt, yy_11,  '-', c='k', lw=2, label='spatial match')
plt.fill_between(tt, yy_10, yy_11, where=yy_10>yy_11)
plt.fill_between(tt, yy_10, yy_11, where=yy_10<yy_11)
plt.legend()
plt.title('Feature match')

plt.suptitle('Feature 3:1, Spatial 1:1')
plt.savefig('./temp_figs/RX_attention_spatial.png')