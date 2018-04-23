##

import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import sys
sys.path.append('./script/human_attention')
from utils_for_human_attention import *


""" load data """
path_data = '/shared/lab/projects/analysis/ruobing/featureIOR/data/horizontal'
df_orig = load_dgs(path_data = path_data, recode_subj = False)

##
""" Settings """
setting = {}
setting['task_type'] = 'feature'
setting['std'] = 10
setting['plotting'] = 'rt'
setting['select_trials'] = 0
setting['plot_individual'] = 0
setting['plot_diff'] = 0
setting['plot_fft'] = 1
setting['detrend'] = 1
setting['shuffleYN'] = 0
setting['mic_sac_amp'] = 2
tt = np.arange(150, 1300, 10)
tt_sac = np.arange(-200, 2000, 10)
df_selected = df_orig[(df_orig.side != 4) & (df_orig.subject == 'SG') & (df_orig.task_type == setting['task_type'])]


uniq_subj = np.unique(df_selected.subject)
yy_list, diff_list= [], []
for i,subj in enumerate(uniq_subj):
    df = df_selected[df_selected.subject == subj]

    """ smooth time """
    rt = np.array(df['rts']-df['delay'], dtype='float')
    fa = (rt<250).astype('float')
    df = df[fa == 0]
    df.reset_index()
    t = np.array(df['delay']-df['SampleOnset'], dtype='float')
    hit = np.array(df['status'], dtype='float')

    """ get grouped indices """
    groupby = ['FeatureMatchNot', 'SpatialMatchNot']
    idx_grp = df.groupby(groupby).indices
    if setting['select_trials']:
        if setting['task_type']=='feature':
            idx_grp[1,0] = np.random.choice(idx_grp[1,0],len(idx_grp[0,0]),replace = False)
            idx_grp[1,1] = np.random.choice(idx_grp[1,1],len(idx_grp[0,1]),replace = False)
        if setting['task_type']=='spatial':
            idx_grp[0,1] = np.random.choice(idx_grp[0,1],len(idx_grp[0,0]),replace = False)
            idx_grp[1,1] = np.random.choice(idx_grp[1,1],len(idx_grp[1,0]),replace = False)


    if setting['plotting'] == 'hit':
        value = hit
    elif setting['plotting'] == 'rt':
        value = rt

    if setting['plot_individual']:
        if setting['plotting'] == 'saccade':
            t_sac_hist = process_saccade(df=df, tt_sac_long=tt_sac, mic_sac_amp=setting['mic_sac_amp'])
            to_plot = np.nanmean(t_sac_hist, axis=0) * 100
            # to_plot = np.convolve(to_plot,np.exp(-(np.arange(-100,100,10) - 0) ** 2 / (2 * 50 ** 2)),'same')
            linetype = '--' if setting['task_type'] == 'feature' else '-'
            plt.plot(tt_sac, to_plot, linetype)
            plt.ylim([0, 2])
        else:
            h_fig, h_axes = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[12,9])
            if df.subject.unique().size > 1:
                subject = 'all'
            else:
                subject = df.subject.unique()[0]
            plt.suptitle('{} blocks, subject = {}'.format(setting['task_type'],subject))

    else:
        h_axes = []
    if setting['shuffleYN']:
        for i in range(1000):
            print(i)
            yy_0_s_n, yy_1_s_n, diff_s_n = plot_avg(setting, value, t, 'spatial nonmatch', idx_grp, h_axes, tt)
            yy_0_s_m, yy_1_s_m, diff_s_m = plot_avg(setting, value, t, 'spatial match', idx_grp, h_axes, tt)
            yy_0_f_n, yy_1_f_n, diff_f_n = plot_avg(setting, value, t, 'feature nonmatch', idx_grp, h_axes, tt)
            yy_0_f_m, yy_1_f_m, diff_f_m = plot_avg(setting, value, t, 'feature match', idx_grp, h_axes, tt)
            # f-n s-n; f-n s-m; f-m,s-n; f-m s-m
            yy_list.append(np.array([yy_0_s_n,yy_0_s_m,yy_1_s_n,yy_1_s_m]))
            diff_list.append(np.array([diff_s_n,diff_s_m,diff_f_n,diff_f_m]))
    else:
        yy_0_s_n, yy_1_s_n, diff_s_n = plot_avg(setting, value, t, 'spatial nonmatch', idx_grp, h_axes, tt)
        yy_0_s_m, yy_1_s_m, diff_s_m = plot_avg(setting, value, t, 'spatial match', idx_grp, h_axes, tt)
        yy_0_f_n, yy_1_f_n, diff_f_n = plot_avg(setting, value, t, 'feature nonmatch', idx_grp, h_axes, tt)
        yy_0_f_m, yy_1_f_m, diff_f_m = plot_avg(setting, value, t, 'feature match', idx_grp, h_axes, tt)
        # f-n s-n; f-n s-m; f-m,s-n; f-m s-m
        yy_list.append(np.array([yy_0_s_n,yy_0_s_m,yy_1_s_n,yy_1_s_m]))
        diff_list.append(np.array([diff_s_n,diff_s_m,diff_f_n,diff_f_m]))


if setting['plot_individual'] == 0:
    if setting['plot_diff']:
        to_plot = np.stack(diff_list,axis = 2)
        x = tt
        xlim = [tt.min(),tt.max()]
        ylim = [-0.3,0.5] if setting['plotting']=='hit' else [-200,100]
        titles = ['spatial nonmatch','spatial match','feature nonmatch','feature match']
    else:
        to_plot = np.stack(yy_list,axis = 2)
        x = tt
        xlim = [tt.min(),tt.max()]
        ylim = [0.3,1.] if setting['plotting']=='hit' else [200,600]
        titles = ['spatial nonmatch feature nonmatch','spatial match feature nonmatch','spatial nonmatch feature match','spatial match feature match']

    if setting['detrend'] == 1:
        for i in range(4):
            for j in range(len(yy_list)):
                model = np.polyfit(tt, to_plot[i, :, j], 2)
                predicted = np.polyval(model, tt)
                to_plot[i, :, j] = to_plot[i, :, j] - predicted
                ylim = [-0.4,0.4] if setting['plot_fft'] == 1 else [-0.2,0.2]

    if setting['plot_fft'] == 1:
        temp = []
        for i in range(4):
            fft_i = []
            for j in range(len(yy_list)):
                f, t, yf = signal.stft(to_plot[i, :, j], 100, window='hann', nperseg=len(tt))
                fft_i.append(np.abs(yf[:,1]))
            temp.append(np.stack(fft_i,axis = 1))
        to_plot = np.stack(temp)
        x = f
        xlim = [0,15]
        ylim = [0,10] if setting['plotting'] == 'rt' else [0,0.1]

    def plot_err_shade(x,Y,type='std'):
        if setting['shuffleYN']:
            Y = np.reshape(Y,[Y.shape[0],len(uniq_subj),1000])
            Y = np.mean(Y,axis=1)
        y = np.mean(Y,axis=1)
        if type=='std':
            error = 1.96*np.std(Y,axis=1)
        elif type=='se':
            error = 1.96*np.std(Y,axis=1)/np.sqrt(Y.shape[1])
        plt.plot(x, y, 'k-',linewidth=4)
        plt.fill_between(x, y - error, y + error, facecolor='black', alpha=0.4)
        plt.plot(x, y - error, 'k-')
        plt.plot(x, y + error, 'k-')

    h_fig_all, h_axes_all = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[12,9])
    plt.suptitle('{} blocks'.format(setting['task_type']))

    plt.axes(h_axes_all[0,0])
    plot_err_shade(x,to_plot[0,:,:],type='se')
    if setting['shuffleYN'] == 0:
        plots = plt.plot(x,to_plot[0,:,:])
    if setting['plot_diff']:
        plt.plot([x.min(),x.max()],[0,0],'--',color = 'gray')
    # plt.legend(iter(plots), (subj for subj in uniq_subj))
    plt.title(titles[0])

    plt.axes(h_axes_all[0,1])
    plot_err_shade(x,to_plot[1,:,:],type='se')
    if setting['shuffleYN'] == 0:
        plots = plt.plot(x,to_plot[1,:,:])
    if setting['plot_diff']:
        plt.plot([x.min(),x.max()],[0,0],'--',color = 'gray')
    # plt.legend(iter(plots), (subj for subj in uniq_subj))
    plt.title(titles[1])

    plt.axes(h_axes_all[1,0])
    plot_err_shade(x,to_plot[2,:,:],type='se')
    if setting['shuffleYN'] == 0:
        plots = plt.plot(x,to_plot[2,:,:])
    if setting['plot_diff']:
        plt.plot([x.min(),x.max()],[0,0],'--',color = 'gray')
    # plt.legend(iter(plots), (subj for subj in uniq_subj))
    plt.title(titles[2])

    plt.axes(h_axes_all[1,1])
    plot_err_shade(x,to_plot[3,:,:],type='se')
    if setting['shuffleYN'] == 0:
        plots = plt.plot(x,to_plot[3,:,:])
    if setting['plot_diff']:
        plt.plot([x.min(),x.max()],[0,0],'--',color = 'gray')
    # plt.legend(iter(plots), (subj for subj in uniq_subj))
    plt.title(titles[3])
    plt.xlim(xlim)
    plt.ylim(ylim)


## Plot true & shuffle data
shuffles = np.load('shuffle_{}_{}_SG.npy'.format(setting['plotting'],setting['task_type']))
shuffles = np.reshape(shuffles,[shuffles.shape[0],shuffles.shape[1],len(uniq_subj),1000])
xlim = [0,15]
ylim = [0,20] if setting['plotting'] == 'rt' else [0,0.1]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
def plot_err_shade_shuffle(x,Y,color):
    if np.ndim(Y)==3:
        Y = np.mean(Y,axis=1)
    y = np.mean(Y,axis=1)
    error = 1.96*np.std(Y,axis=1)
    plt.plot(x, y, '--', color = color, linewidth=4)
    plt.fill_between(x, y - error, y + error, facecolor=color, alpha=0.6)
    plt.plot(x, y - error, '--', color = color)
    plt.plot(x, y + error, '--', color = color)

def get_significance(true_data,shuffled_data):
    if true_data.ndim==2:
        diff_true_shuffle = true_data[:, :, None] - shuffled_data
        mean_diff = np.mean(diff_true_shuffle,axis=(1,2))
        std_diff = np.std(diff_true_shuffle,axis=(1,2))
        t = mean_diff/std_diff
        pval = [stats.t.sf(np.abs(tt), diff_true_shuffle.shape[1]*diff_true_shuffle.shape[2] - 1) * 2 for tt in t]
    elif true_data.ndim==1:
        diff_true_shuffle = true_data[:, None] - shuffled_data
        mean_diff = np.mean(diff_true_shuffle,axis=1)
        std_diff = np.std(diff_true_shuffle,axis=1)
        t = mean_diff/std_diff
        pval = [stats.t.sf(np.abs(tt), diff_true_shuffle.shape[1] - 1) * 2 for tt in t]
    return pval

# p = get_significance(to_plot[1,:,:],shuffles[1,:,:,:])


h_fig_all, h_axes_all = plt.subplots(2, 3, sharex='all', sharey='all', figsize=[12,9])
h_axes_all = [y for x in h_axes_all for y in x]
condition_to_plot = 1
plt.suptitle('{} blocks, {}, individuals'.format(setting['task_type'],titles[condition_to_plot]))
for i in range(len(uniq_subj)):
    plt.axes(h_axes_all[i])
    p = get_significance(to_plot[condition_to_plot, :, i], shuffles[condition_to_plot, :, i, :])
    print(p)
    plot_err_shade_shuffle(x,shuffles[condition_to_plot,:,i,:],color = colors[i])
    plt.plot(x,to_plot[condition_to_plot,:,i], color = colors[i], linewidth=4)
    plt.plot(np.array(x)[np.array(p)<0.05],(ylim[1]-0.02)*np.ones(sum(np.array(p)<0.05)),'k*')
    plt.title(uniq_subj[i])
    plt.xlim(xlim)
    plt.ylim(ylim)


h_fig_all, h_axes_all = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[12,9])
plt.suptitle('{} blocks, cross-subject mean'.format(setting['task_type']))

plt.axes(h_axes_all[0, 0])
plt.plot(x, to_plot[0, :, :], linewidth=2)
plot_err_shade_shuffle(x, shuffles[0, :, :, :], color='gray')
# plt.plot(x, np.mean(to_plot[0, :, :],axis=1), 'k-', linewidth=4)
plot_err_shade(x, to_plot[0, :, :],type='se')
plt.title(titles[0])

plt.axes(h_axes_all[0, 1])
plt.plot(x, to_plot[1, :, :], linewidth=2)
plot_err_shade_shuffle(x, shuffles[1, :, :, :], color='gray')
# plt.plot(x, np.mean(to_plot[1, :, :],axis=1), 'k-', linewidth=4)
plot_err_shade(x, to_plot[1, :, :],type='se')
plt.title(titles[1])

plt.axes(h_axes_all[1, 0])
plt.plot(x, to_plot[2, :, :], linewidth=2)
plot_err_shade_shuffle(x, shuffles[2, :, :, :], color='gray')
# plt.plot(x, np.mean(to_plot[2, :, :],axis=1), 'k-', linewidth=4)
plot_err_shade(x, to_plot[2, :, :],type='se')
plt.title(titles[2])

plt.axes(h_axes_all[1, 1])
plt.plot(x, to_plot[3, :, :], linewidth=2)
plot_err_shade_shuffle(x, shuffles[3, :, :, :], color='gray')
# plt.plot(x, np.mean(to_plot[3, :, :],axis=1), 'k-', linewidth=4)
plot_err_shade(x, to_plot[3, :, :],type='se')
plt.title(titles[3])
plt.xlim(xlim)
plt.ylim(ylim)

