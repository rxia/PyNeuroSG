import numpy as np
import os
import re
import dg2df
import pandas as pd
import matplotlib.pyplot as plt

def load_dgs(path_data = '/shared/lab/projects/analysis/ruobing/featureIOR/data/', recode_subj = False):
    dg_folders = os.listdir(path_data)
    list_dgs = []
    for dg_folder in dg_folders:
        cur_dir = os.path.join(path_data, dg_folder)
        if not os.path.isdir(cur_dir):
            continue
        print('loading {}'.format(dg_folder))
        name_subj, name_task_type = re.match('(.*)_(.*)', dg_folder).group(1, 2)
        dg_files = os.listdir(cur_dir)
        list_dgs_subj = [dg2df.dg2df(os.path.join(cur_dir, dg_file))
                         for dg_file in dg_files if dg_file[-3:] == '.dg']
        df_subj = pd.concat(list_dgs_subj)
        df_subj['subject'] = name_subj
        df_subj['task_type'] = name_task_type
        list_dgs.append(df_subj)

    df_orig = pd.concat(list_dgs)
    df_orig.reset_index(inplace=True)
    if recode_subj:
        temp = df_orig.subject.unique()
        temp1 = dict(zip(temp, range(len(temp))))
        df_orig.subject = ['subj{:02d}'.format(temp1[sub_ini]) for sub_ini in df_orig.subject]
    return(df_orig)


def smooth_y(x, y, std=50, xx=np.arange(150, 1300, 10), limit=None, shuffle = False):
    if limit is not None:
        x = x[limit]
        if shuffle:
            np.random.shuffle(limit)
        y = y[limit]
    weight = np.exp(-(x[None, :] - xx[:, None]) ** 2 / (2 * std ** 2))
    yy = np.array([np.average(y, weights=weight[i, :]) for i in range(len(xx))])
    return yy

def process_saccade(df, mic_sac_amp, tt_sac_long = np.arange(-200, 2000, 10)):
    t_sac_hist, dir_sac_hist= [], []
    sampleon = list(df['SampleOnset'])[0]
    for j in range(len(df['sacamps'])):
        sactime = list(df['sactimes'])[j]
        sacamp = list(df['sacamps'])[j]
        stimon = list(df['stimon'])[j]
        resp_time = list(df['rts'] - sampleon)[j]
        tt_sac = np.arange(-sampleon, resp_time, 10)
        t_sac_hist_j = np.empty(len(tt_sac_long))
        t_sac_hist_j[:] = np.nan
        t_sac_hist_j[tt_sac_long < tt_sac[-1]] = 0
        if isinstance(sactime, float) and sactime != sactime:
            t_sac_hist.append(t_sac_hist_j)
            continue
        sactime = sactime - stimon - sampleon
        t_sac_j = np.array(sactime[(sacamp < mic_sac_amp) & (sactime > -sampleon) & (sactime < resp_time)])
        if t_sac_j.shape[0] != 0:
            for m in range(len(t_sac_j)):
                t_sac_hist_j[np.argmin(np.abs(tt_sac - t_sac_j[m]))] = 1
        t_sac_hist.append(t_sac_hist_j)
    return(t_sac_hist)

def plot_avg(setting=None, value=None, t=None, title=None, idx_grp=None, h_axes=None, tt=np.arange(150, 1300, 10)):

    if title == 'spatial nonmatch':
        ax = (0, 0)
        idx_grp_0 = (0, 0)
        idx_grp_1 = (1, 0)
    elif title == 'spatial match':
        ax = (0, 1)
        idx_grp_0 = (0, 1)
        idx_grp_1 = (1, 1)
    elif title == 'feature nonmatch':
        ax = (1, 0)
        idx_grp_0 = (0, 0)
        idx_grp_1 = (0, 1)
    elif title == 'feature match':
        ax = (1, 1)
        idx_grp_0 = (1, 0)
        idx_grp_1 = (1, 1)
    idx0 = idx_grp[idx_grp_0]
    idx1 = idx_grp[idx_grp_1]
    yy_0 = smooth_y(t, value, std=setting['std'], limit=idx0, shuffle=setting['shuffleYN'])
    yy_1 = smooth_y(t, value, std=setting['std'], limit=idx1, shuffle=setting['shuffleYN'])
    if setting['plot_individual']:
        plt.axes(h_axes[ax])
        plt.plot(tt, yy_0, '--', c='k', lw=2, label='feature nonmatch' if title[:7] == 'spatial' else 'spatial nonmatch')
        plt.plot(tt, yy_1, '-', c='k', lw=2, label='feature match' if title[:7] == 'spatial' else 'spatial match')
        if title[:7] == 'spatial':
            color_inhibit = 'blue'
            color_facilitate = 'orange'
        else:
            color_inhibit = 'green'
            color_facilitate = 'red'
        if setting['plotting'] == 'hit':
            plt.fill_between(tt, yy_0, yy_1, where=yy_0 > yy_1, facecolor=color_inhibit)
            plt.fill_between(tt, yy_0, yy_1, where=yy_0 < yy_1, facecolor=color_facilitate)
        elif setting['plotting'] == 'rt':
            plt.fill_between(tt, yy_0, yy_1, where=yy_0 > yy_1, facecolor=color_facilitate)
            plt.fill_between(tt, yy_0, yy_1, where=yy_0 < yy_1, facecolor=color_inhibit)
        plt.legend()
        plt.title(title)
    return yy_0, yy_1, yy_1-yy_0