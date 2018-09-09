""" script to generate the final result figures """

import importlib
import os
import warnings
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib as mlp
import matplotlib.pyplot as plt
import pickle

import PyNeuroData as pnd
import PyNeuroAna  as pna
import PyNeuroPlot as pnp
import store_hdf5
import df_ana
import misc_tools


# set path to data
path_to_hdf5_data_Dante = '../support_data/data_neuro_Dante_GM32_U16_all.hdf5'
path_to_hdf5_data_Thor = '../support_data/data_neuro_Thor_U16_all.hdf5'
path_to_fig = './temp_figs/final_Dante_Thor'


path_to_U_probe_loc = './script/U16_location_Dante_Thor.csv'

##
""" get U-probe location """
# granular layer location is zero-indexed
loc_U16 = pd.read_csv(path_to_U_probe_loc)
loc_U16['date'] = loc_U16['date'].astype('str')

def sinal_extend_loc_info(signal_all):
    signal_extend = pd.merge(signal_all, loc_U16, 'left', on='date')

    signal_extend['channel_index_U16'] = signal_extend['channel_index'] - 32*(signal_extend['animal']=='Dante')
    signal_extend['area'] = np.where((signal_extend['animal']=='Dante') & (signal_extend['channel_index']<=32),
                                     ['V4']*len(signal_extend), signal_extend['area'])

    signal_extend['depth'] = 0
    signal_extend['depth'] = (signal_extend['channel_index_U16']-1-signal_extend['granular']) * (signal_extend['area']=='TEd') \
                             +(16-signal_extend['channel_index_U16']-signal_extend['granular']) * (signal_extend['area']=='TEm')
    signal_extend['depth'] = np.where(signal_extend['area']=='V4',
                                     np.zeros(len(signal_extend))*np.nan, signal_extend['depth'])

    return signal_extend


def load_data_of_day(datecode):
    if datecode <= '180000':
        animal = 'Dante'
        path_to_hdf5_data = path_to_hdf5_data_Dante
    else:
        animal = 'Thor'
        path_to_hdf5_data = path_to_hdf5_data_Thor
    data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'spk'])
    return data_neuro


##
"""========== get PSTH for every day =========="""

# shwo data file stucture

tree_struct_hdf5_Dante = store_hdf5.ShowH5(path_to_hdf5_data_Dante, yn_print=False)
tree_struct_hdf5_Thor  = store_hdf5.ShowH5(path_to_hdf5_data_Thor , yn_print=False)
all_dates = list(tree_struct_hdf5_Dante.keys()) + list(tree_struct_hdf5_Thor.keys())
list_psth_group_mean = []
list_psth_group_std = []
list_signal = []
sk_std = 0.010

""" get every day's data """
for datecode in all_dates:
    print(datecode)
    try:
        # get data
        data_neuro = load_data_of_day(datecode)
        df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                              groupby=['stim_familiarized', 'mask_opacity_int'])

        # smooth trace over time
        data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], ts=data_neuro['ts'], sk_std=sk_std)

        # get mean and std
        psth_group_mean = pna.GroupStat(data_neuro)
        psth_group_std = pna.GroupStat(data_neuro, statfun='std')

        # get signal info
        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        list_psth_group_mean.append(psth_group_mean)
        list_psth_group_std.append(psth_group_std)
        list_signal.append(signal_info)
    except:
        print('date {} can not be processed'.format(datecode))

##
""" concatenate together """
data_group_mean = np.concatenate(list_psth_group_mean, axis=2)
data_group_std = np.concatenate(list_psth_group_std, axis=2)
signal_all = pd.concat(list_signal).reset_index()
ts = data_neuro['ts']
cdtn = data_neuro['cdtn']

def set_signal_id(signal_info):
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return signal_info
signal_all = set_signal_id(signal_all)

signal_all_spk = signal_all

signal_all_spk = sinal_extend_loc_info(signal_all_spk)

data_group_mean_norm = pna.normalize_across_signals(data_group_mean, ts=ts, t_window=[0.050, 0.300])


##
""" count neuorns """
temp = (signal_all_spk['valid']==1) & (signal_all_spk['animal']=='Dante') & (signal_all_spk['area']=='V4')
print(np.sum(temp & (signal_all_spk['sort_code']==1)))

temp = (signal_all_spk['valid']==1) & (signal_all_spk['animal']=='Dante') & (signal_all_spk['area']!='V4')
print(np.sum(temp))
print(np.sum(temp & (signal_all_spk['sort_code']==1)))

temp = (signal_all_spk['valid']==1) & (signal_all_spk['animal']=='Thor') & (signal_all_spk['area']!='V4')
print(np.sum(temp))
print(np.sum(temp & (signal_all_spk['sort_code']>1)))

##
""" plot overall data """
colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                    pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
cdtn_name = ['nov, 00', 'nov, 50', 'nov, 70', 'fam, 00', 'fam, 50', 'fam, 70']

plot_highlight = dict()
plot_highlight['all'] = {'trace': [1,1,1,1,1,1], 'compare': [0,0,0,1,0]}
plot_highlight['nov'] = {'trace': [1,1,1,0,0,0], 'compare': [0,0,0,1,0]}
plot_highlight['fam'] = {'trace': [0,0,0,1,1,1], 'compare': [0,0,0,0,1]}
plot_highlight['00']  = {'trace': [1,0,0,1,0,0], 'compare': [1,0,0,0,0]}
plot_highlight['50']  = {'trace': [0,1,0,0,1,0], 'compare': [0,1,0,0,0]}
plot_highlight['70']  = {'trace': [0,0,1,0,0,1], 'compare': [0,0,1,0,0]}

# yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')
yn_keep_signal = signal_all_spk['area'] == 'V4'
h_fig, h_axes = plt.subplots(2,3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
plt.xlim([-0.050, 0.400])
plt.savefig(os.path.join(path_to_fig, 'PSTH_V4_before_select.png'))
plt.savefig(os.path.join(path_to_fig, 'PSTH_V4_before_select.pdf'))


yn_keep_signal = (signal_all_spk['area'] != 'V4') & (signal_all_spk['valid']==1)
h_fig, h_axes = plt.subplots(2,3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
plt.xlim([-0.050, 0.400])
plt.savefig(os.path.join(path_to_fig, 'PSTH_IT_before_select.png'))
plt.savefig(os.path.join(path_to_fig, 'PSTH_IT_before_select.pdf'))


##
""" test whether a neuron is visual or not """

ts_baseline = np.logical_and(ts>=-0.050, ts<0.050)
ts_visual = np.logical_and(ts>=0.050, ts<0.200)
threhold_cohen_d = 0.5

if False:    # old, not sepratation nov and fam
    psth_mean_00_noise = data_group_mean[[0,3]].mean(axis=0)
    psth_var_00_noise = data_group_mean[[0,3]].mean(axis=0)

    psth_mean_70_noise = data_group_mean[[2,5]].mean(axis=0)
    psth_var_70_noise = data_group_mean[[2,5]].mean(axis=0)


    mean_baseline = psth_mean_00_noise[ts_baseline].mean(axis=0)
    std_baseline = psth_mean_00_noise[ts_baseline].mean(axis=0)
    mean_visual_00_noise = psth_mean_00_noise[ts_visual].mean(axis=0)
    std_visual_00_noise = psth_mean_00_noise[ts_visual].mean(axis=0)
    mean_visual_70_noise = psth_mean_70_noise[ts_visual].mean(axis=0)
    std_visual_70_noise = psth_var_70_noise[ts_visual].mean(axis=0)

    cohen_d_visual = (mean_visual_00_noise - mean_baseline) \
                     / np.sqrt((std_baseline**2 + std_visual_00_noise**2)/2)

    keep_visual = cohen_d_visual > threhold_cohen_d

    cohen_d_noise_effect = (mean_visual_00_noise - mean_visual_70_noise) \
                           / np.sqrt((std_visual_00_noise**2 + std_visual_70_noise**2)/2)
    keep_noise_effect = cohen_d_noise_effect > 0.2


mean_baseline = data_group_mean.mean(axis=0)[ts_baseline].mean(axis=0)
std_baseline  = data_group_mean.mean(axis=0)[ts_baseline].std(axis=0)
mean_nov_00   = data_group_mean[0][ts_visual].mean(axis=0)
std_nov_00    = data_group_mean[0][ts_visual].std(axis=0)
mean_fam_00   = data_group_mean[3][ts_visual].mean(axis=0)
std_fam_00    = data_group_mean[3][ts_visual].std(axis=0)
mean_nov_70   = data_group_mean[2][ts_visual].mean(axis=0)
std_nov_70    = data_group_mean[2][ts_visual].std(axis=0)
mean_fam_70   = data_group_mean[5][ts_visual].mean(axis=0)
std_fam_70    = data_group_mean[5][ts_visual].std(axis=0)

cohen_d_visual_nov = (mean_nov_00 - mean_baseline) \
                 / np.sqrt((std_nov_00 ** 2 + std_baseline ** 2) / 2)
cohen_d_visual_fam = (mean_fam_00 - mean_baseline) \
                 / np.sqrt((std_fam_00 ** 2 + std_baseline ** 2) / 2)

cohen_d_visual = np.where(cohen_d_visual_nov > cohen_d_visual_fam, cohen_d_visual_fam, cohen_d_visual_nov)

keep_visual = cohen_d_visual > threhold_cohen_d
signal_all_spk['is_visual'] = keep_visual

is_signal_V4_valid = signal_all_spk['valid'].astype('bool') & (signal_all_spk['area']=='V4')
is_signal_IT_valid = signal_all_spk['valid'].astype('bool') & ((signal_all_spk['area']=='TEm') | (signal_all_spk['area']=='TEd'))
is_signal_IT_valid_Dante = is_signal_IT_valid & (signal_all_spk['animal'] == 'Dante')
is_signal_IT_valid_Thor  = is_signal_IT_valid & (signal_all_spk['animal'] == 'Thor')

is_signal_V4_visual = is_signal_V4_valid & signal_all_spk['is_visual']
is_signal_IT_visual = is_signal_IT_valid & signal_all_spk['is_visual']
is_signal_IT_visual_Dante = is_signal_IT_visual & (signal_all_spk['animal'] == 'Dante')
is_signal_IT_visual_Thor  = is_signal_IT_visual & (signal_all_spk['animal'] == 'Thor')

print('number of V4 cells: {} visual / {} all'.format(np.sum(is_signal_V4_visual), np.sum(is_signal_V4_valid)))
print('number of IT cells: {} visual / {} all'.format(np.sum(is_signal_IT_visual), np.sum(is_signal_IT_valid)))


h_fig, h_axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=[12,4])
h_fig.patch.set_facecolor([0.9]*3)


plt.axes(h_axes[0])
plt.scatter(mean_baseline[is_signal_V4_visual], mean_nov_00[is_signal_V4_visual],
            c=[0.1]*3, s=10)
plt.scatter(mean_baseline[is_signal_V4_valid & ~is_signal_V4_visual], mean_nov_00[is_signal_V4_valid & ~is_signal_V4_visual],
            c=[0.5]*3, s=10)
plt.legend(['visual', 'non-visual'])
plt.plot([0, 50], [0, 50], color='gray', linestyle='--', alpha=0.5)
plt.axis('equal')
plt.xlabel('baseline')
plt.ylabel('visual')
plt.title('V4: {} visual / {} all'.format(np.sum(is_signal_V4_visual), np.sum(is_signal_V4_valid)))

plt.axes(h_axes[1])
signal_cur = is_signal_IT_visual & (signal_all_spk['animal']=='Dante')
plt.scatter(mean_baseline[signal_cur], mean_nov_00[signal_cur],
            c=[0.1] * 3, s=10)
signal_cur = is_signal_IT_valid &  ~is_signal_IT_visual & (signal_all_spk['animal']=='Dante')
plt.scatter(mean_baseline[signal_cur], mean_nov_00[signal_cur],
            c=[0.5] * 3, s=10)
plt.plot([0, 50], [0, 50], color='gray', linestyle='--', alpha=0.5)
plt.axis('equal')
plt.xlabel('baseline')
plt.ylabel('visual')
plt.title('Dante IT: {} visual / {} all'.format(np.sum(is_signal_IT_visual_Dante), np.sum(is_signal_IT_valid_Dante)))


plt.axes(h_axes[2])
signal_cur = is_signal_IT_visual & (signal_all_spk['animal']=='Thor')
plt.scatter(mean_baseline[signal_cur], mean_nov_00[signal_cur],
            c=[0.1] * 3, s=10)
signal_cur = is_signal_IT_valid &  ~is_signal_IT_visual & (signal_all_spk['animal']=='Thor')
plt.scatter(mean_baseline[signal_cur], mean_nov_00[signal_cur],
            c=[0.5] * 3, s=10)
plt.plot([0, 50], [0, 50], color='gray', linestyle='--', alpha=0.5)
plt.axis('equal')
plt.xlabel('baseline')
plt.ylabel('visual')
plt.title('Dante IT: {} visual / {} all'.format(np.sum(is_signal_IT_visual_Thor), np.sum(is_signal_IT_valid_Thor)))

plt.savefig(os.path.join(path_to_fig, 'PSTH_visual_criterion.png'))
plt.savefig(os.path.join(path_to_fig, 'PSTH_visual_criterion.pdf'))

if False:
    path_to_U_probe_manual_selection = './script/Thor_signal_manual_select.csv'
    keep_final_manual = pd.read_csv(path_to_U_probe_manual_selection)
    keep_final_manual = keep_final_manual['good_noise_effect'] == '1'


##
""" ========== signal_filter_function ========== """

def signal_filter(keep_mode=None, signal_all_spk=signal_all_spk):


    # keep_mode = 'V4'
    # keep_mode = 'IT'
    # keep_mode = 'IT_Dante'
    # keep_mode = 'IT_Thor'
    # keep_mode = 'IT_Gr'
    # keep_mode = 'IT_sGr'
    # keep_mode = 'IT_iGr'

    if keep_mode=='V4':
        yn_keep_signal = (signal_all_spk['area'] == 'V4') \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1)
    elif keep_mode=='IT':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1)
    elif keep_mode=='IT_Dante':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1) \
                         & (signal_all_spk['animal']=='Dante')
    elif keep_mode=='IT_Thor':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1) \
                         & (signal_all_spk['animal']=='Thor')
    elif keep_mode=='IT_Gr':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1) \
                         & (signal_all_spk['depth']>=-2) & (signal_all_spk['depth']<=2)
    elif keep_mode=='IT_sGr':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1) \
                         & (signal_all_spk['depth']>3)
    elif keep_mode=='IT_iGr':
        yn_keep_signal = ((signal_all_spk['area'] == 'TEd') | (signal_all_spk['area'] == 'TEm'))  \
                         & (signal_all_spk['valid']==1) & (signal_all_spk['is_visual']==1) \
                         & (signal_all_spk['depth']<-3)
    else:
        raise Exception('keep mode not correct')

    return yn_keep_signal



##
"""========== population decoding accuarcy =========="""
""" very slow """

count_mode = 'slide'

dict_decoding = dict()

window_size = 0.1
t_start = 0.0

ts_sample_guide = np.arange(-0.050, 0.551, 0.020)
def get_i_ts_sample(ts_sample, ts):
    return np.array([np.flatnonzero(ts>t)[0] for t in ts_sample_guide])


def compute_decoding(data_neuro, label_name='stim_sname', limit_ch=None, cdtn=(1,70)):
    data_df = data_neuro['trial_info']
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']

    if limit_ch is None:
        limit_ch = np.ones(data_neuro['data'].shape[-1])

    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    M = len(list_image_name)
    all_clf_score = np.zeros([M, M, len(ts)])*np.nan
    for indx_img1 in range(len(list_image_name)):
        for indx_img2 in range(indx_img1-1):
            fltr_tr = (data_df['stim_sname']==list_image_name[indx_img1]) | (data_df['stim_sname']==list_image_name[indx_img2])
            clf_score = pna.decode_over_time(data_neuro['data'], label=label, limit_tr=limit_tr & fltr_tr, limit_ch=limit_ch)
            all_clf_score[indx_img1, indx_img2, :] =  clf_score
    mean_clf_score = np.nanmean(all_clf_score, axis=(0,1))

    return mean_clf_score


def compute_decoding_multiclass(data_neuro, label_name='stim_sname', limit_ch=None, cdtn=(1,70)):
    data_df = data_neuro['trial_info']
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']

    if limit_ch is None:
        limit_ch = np.ones(data_neuro['data'].shape[-1])

    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    clf_score = pna.decode_over_time(data_neuro['data'], label=label, limit_tr=limit_tr, limit_ch=limit_ch)

    return clf_score


if False:   # very slow, may take several hours

    for datecode in all_dates:
        print(datecode)
        try:
            # get data
            data_neuro = load_data_of_day(datecode)

            # work on time
            ts = data_neuro['ts']

            if count_mode == 'cum':
                data_neuro['data'] = pna.SpikeCountCumulative(data_neuro['data'], ts=ts, t_start=t_start)
            elif count_mode == 'slide':
                data_neuro['data'] = pna.SpikeCountInWindow(data_neuro['data'], window_size, ts=ts)
            else:
                raise Exception('count_mode no valid')

            df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                                  groupby=['stim_familiarized', 'mask_opacity_int'])

            i_ts_sample = get_i_ts_sample(ts_sample_guide, ts)
            data_neuro['ts'] = ts[i_ts_sample]
            data_neuro['data'] = data_neuro['data'][:, i_ts_sample, :]

            # get signal info
            signal_info = data_neuro['signal_info']
            signal_info['date'] = datecode

            signal_info_more = sinal_extend_loc_info(signal_info)

            for area in ['V4', 'IT']:
                if area == 'V4':
                    limit_ch = (signal_info_more['area'] == 'V4')
                else:
                    limit_ch = (signal_info_more['area'] != 'V4')
                if np.sum(limit_ch) < 5:
                    continue

                print(area)
                list_decoding_image = []
                list_decoding_noise = []

                for cdtn in data_neuro['cdtn']:
                    print(cdtn)
                    decoding_image_cur = compute_decoding(data_neuro, label_name='stim_sname', limit_ch=limit_ch, cdtn=cdtn)
                    list_decoding_image.append(decoding_image_cur)

                    decoding_noise_cur = compute_decoding(data_neuro, label_name='mask_orientation', limit_ch=limit_ch, cdtn=cdtn)
                    list_decoding_noise.append(decoding_noise_cur)
                decoding_image = np.vstack(list_decoding_image)
                decoding_noise = np.vstack(list_decoding_noise)

                dict_decoding[(datecode, area, 'image')] = decoding_image
                dict_decoding[(datecode, area, 'noise')] = decoding_noise

        except:
            print('date {} can not be processed'.format(datecode))


    # save to hard disk
    dict_decoding['ts'] = ts_sample_guide
    with open('../support_data/population_decoding_Dante_Thor_accuracy.pickle', 'wb') as f:
        pickle.dump(dict_decoding, f)



##
"""========== population decoding probability =========="""
""" very slow """

count_mode = 'slide'

dict_decoding = dict()

window_size = 0.1
t_start = 0.0

ts_sample_guide = np.arange(-0.050, 0.551, 0.020)
def get_i_ts_sample(ts_sample, ts):
    return np.array([np.flatnonzero(ts>t)[0] for t in ts_sample_guide])


def compute_decoding(data_neuro, label_name='stim_sname', limit_ch=None, cdtn=(1,70)):
    data_df = data_neuro['trial_info']
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']

    if limit_ch is None:
        limit_ch = np.ones(data_neuro['data'].shape[-1])

    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    M = len(list_image_name)
    all_clf_score = np.zeros([M, M, len(ts)])*np.nan
    for indx_img1 in range(len(list_image_name)):
        for indx_img2 in range(indx_img1-1):
            fltr_tr = (data_df['stim_sname']==list_image_name[indx_img1]) | (data_df['stim_sname']==list_image_name[indx_img2])
            clf_score = pna.decode_over_time(data_neuro['data'], label=label,
                                             limit_tr=limit_tr & fltr_tr, limit_ch=limit_ch, return_p=True)
            all_clf_score[indx_img1, indx_img2, :] =  clf_score
    mean_clf_score = np.nanmean(all_clf_score, axis=(0,1))

    return mean_clf_score


def compute_decoding_multiclass(data_neuro, label_name='stim_sname', limit_ch=None, cdtn=(1,70)):
    data_df = data_neuro['trial_info']
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']

    if limit_ch is None:
        limit_ch = np.ones(data_neuro['data'].shape[-1])

    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    clf_score = pna.decode_over_time(data_neuro['data'], label=label,
                                     limit_tr=limit_tr, limit_ch=limit_ch, return_p=True)

    return clf_score


if False:   # very slow, may take several hours

    for datecode in all_dates:
        print(datecode)
        try:
            # get data
            data_neuro = load_data_of_day(datecode)

            # work on time
            ts = data_neuro['ts']

            if count_mode == 'cum':
                data_neuro['data'] = pna.SpikeCountCumulative(data_neuro['data'], ts=ts, t_start=t_start)
            elif count_mode == 'slide':
                data_neuro['data'] = pna.SpikeCountInWindow(data_neuro['data'], window_size, ts=ts)
            else:
                raise Exception('count_mode no valid')

            df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                                  groupby=['stim_familiarized', 'mask_opacity_int'])

            i_ts_sample = get_i_ts_sample(ts_sample_guide, ts)
            data_neuro['ts'] = ts[i_ts_sample]
            data_neuro['data'] = data_neuro['data'][:, i_ts_sample, :]

            # get signal info
            signal_info = data_neuro['signal_info']
            signal_info['date'] = datecode

            signal_info_more = sinal_extend_loc_info(signal_info)

            for area in ['V4', 'IT']:
                if area == 'V4':
                    limit_ch = (signal_info_more['area'] == 'V4')
                else:
                    limit_ch = (signal_info_more['area'] != 'V4')
                if np.sum(limit_ch) < 5:
                    continue

                print(area)
                list_decoding_image = []
                list_decoding_noise = []

                for cdtn in data_neuro['cdtn']:
                    print(cdtn)
                    decoding_image_cur = compute_decoding_multiclass(data_neuro, label_name='stim_sname', limit_ch=limit_ch, cdtn=cdtn)
                    list_decoding_image.append(decoding_image_cur)

                    decoding_noise_cur = compute_decoding_multiclass(data_neuro, label_name='mask_orientation', limit_ch=limit_ch, cdtn=cdtn)
                    list_decoding_noise.append(decoding_noise_cur)
                decoding_image = np.vstack(list_decoding_image)
                decoding_noise = np.vstack(list_decoding_noise)

                dict_decoding[(datecode, area, 'image')] = decoding_image
                dict_decoding[(datecode, area, 'noise')] = decoding_noise

        except:
            print('date {} can not be processed'.format(datecode))


    # save to hard disk
    dict_decoding['ts'] = ts_sample_guide
    with open('../support_data/population_decoding_Dante_Thor_prob_multiclass.pickle', 'wb') as f:
        pickle.dump(dict_decoding, f)



##
with open('../support_data/population_decoding_Dante_Thor_prob_multiclass.pickle', 'rb') as f:
    dict_decoding = pickle.load(f)



decode_keywords = ('V4', 'image')
# decode_keywords = ('IT', 'image')
# decode_keywords = ('V4', 'noise')
# decode_keywords = ('IT', 'noise')

list_valid_date = np.unique(loc_U16['date'][(loc_U16['valid']>0)])
list_decocde_result= []
for key in dict_decoding:
    if set(decode_keywords) < set(key) and key[0] in list_valid_date:
        list_decocde_result.append(dict_decoding[key])


decode_area = np.stack(list_decocde_result, axis=2)


colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                    pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
cdtn_name = ['nov, 00', 'nov, 50', 'nov, 70', 'fam, 00', 'fam, 50', 'fam, 70']

plot_highlight = dict()
plot_highlight['all'] = {'trace': [1,1,1,1,1,1], 'compare': [0,0,0,1,0]}
plot_highlight['nov'] = {'trace': [1,1,1,0,0,0], 'compare': [0,0,0,1,0]}
plot_highlight['fam'] = {'trace': [0,0,0,1,1,1], 'compare': [0,0,0,0,1]}
plot_highlight['00']  = {'trace': [1,0,0,1,0,0], 'compare': [1,0,0,0,0]}
plot_highlight['50']  = {'trace': [0,1,0,0,1,0], 'compare': [0,1,0,0,0]}
plot_highlight['70']  = {'trace': [0,0,1,0,0,1], 'compare': [0,0,1,0,0]}

h_fig, h_axes = plt.subplots(2, 3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts_sample_guide, np.mean(decode_area[c], axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
# plt.ylim([0.4, 1.0])
plt.xlim([-0.050, 0.450])
plt.savefig(os.path.join(path_to_fig, 'population_decoding_prop_{}_{}.pdf'.format(*decode_keywords)))
plt.savefig(os.path.join(path_to_fig, 'population_decoding_prop_{}_{}.png'.format(*decode_keywords)))


##
"""========== CSD analysis =========="""

dict_lfp_profile = dict()
""" get every day's ERP """
for datecode in tree_struct_hdf5.keys():
    print(datecode)
    try:
        # get data
        data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'lfp'])

        # lfp profiel: [number_channels, num_ts]
        lfp_profile = np.mean(data_neuro['data'][data_neuro['trial_info']['mask_opacity_int']==0], axis=0).transpose()

        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        dict_lfp_profile[datecode] = lfp_profile
    except:
        print('date {} can not be processed'.format(datecode))

##
yn_plot_spike_quality = True
yn_plot_granular_loc = True


""" analyze CSD """
chan_bad_all = dict()
chan_bad_all['180224'] = [2, 8]
chan_bad_all['180317'] = [0, 1, 2, 12]
chan_bad_all['180325'] = [2, 12]
chan_bad_all['180407'] = [2, 12, 13]
chan_bad_all['180411'] = [2, 8, 12, 13]
chan_bad_all['180413'] = [2, 8, 12, 13]
chan_bad_all['180418'] = [2, 8, 12, 13]
chan_bad_all['180420'] = [2, 8, 12, 13]
chan_bad_all['180424'] = [2, 8, 12, 13]
chan_bad_all['180501'] = []
chan_bad_all['180515'] = [13]
chan_bad_all['180520'] = [7, 13]
chan_bad_all['180523'] = [2, 7]
chan_bad_all['180530'] = [2, 5, 7, 13]
chan_bad_all['180603'] = [2, 5, 7, 13]
chan_bad_all['180606'] = [2, 5, 7, 13]
chan_bad_all['180610'] = [2, 5, 7, 13]
chan_bad_all['180614'] = [2, 5, 7, 13]
chan_bad_all['180617'] = [2, 5, 7, 13]
chan_bad_all['180620'] = [2, 5, 7, 13]
chan_bad_all['180622'] = [2, 5, 7, 13]
chan_bad_all['180624'] = [2, 5, 7, 13]


plt.ioff()

if yn_plot_spike_quality:
    # get spike info for channels
    signal_all_best_spk = signal_all_spk.groupby(['date', 'channel_index']).last()\
        .reset_index()[['date', 'channel_index', 'sort_code']]

    N_chan_total = 16

    def get_spike_quality_for_channel(datecode):
        """ get the signal quality, i.e, largest sortcode for every channel for a day """
        signal_quality = pd.merge(pd.DataFrame({'channel_index': range(N_chan_total)}),
                                   signal_all_best_spk[signal_all_best_spk['date']==datecode], 'left', on='channel_index')
        signal_quality = np.nan_to_num(np.array(signal_quality['sort_code']), 0).astype('int')
        return signal_quality

if yn_plot_granular_loc:
    path_to_U_probe_loc_Thor = './script/Thor_U16_location.csv'
    loc_U16 = pd.read_csv(path_to_U_probe_loc_Thor)
    loc_U16['date'] = loc_U16['date'].astype('str')
    def get_granular_loc(datecode):
        return loc_U16['granular'][loc_U16['date']==datecode]


""" plot CSD for every day """
for datecode in dict_lfp_profile:
    print(datecode)

    lfp = dict_lfp_profile[datecode]
    if datecode in chan_bad_all:
        chan_bad = chan_bad_all[datecode]
    else:
        chan_bad = []
    lambda_dev = np.ones(16)
    lambda_dev[chan_bad]=0

    _, h_axes = plt.subplots(2,3, figsize=[12,8], sharex='all', sharey='all')
    plt.axes(h_axes[0,0])
    pnp.ErpPlot_singlePanel(lfp, ts)
    plt.plot([0]*len(chan_bad), chan_bad, 'ok')
    plt.title('LFP original')
    plt.xlabel('time (s)')
    plt.ylabel('chan')
    plt.xticks(np.arange(-0.1,0.51,0.1))


    lfp_na = lfp
    lfp_sm = pna.lfp_cross_chan_smooth(lfp, method='der', lambda_dev=lambda_dev, lambda_der=5, sigma_t=0.5)
    lfp_nr = lfp_sm / np.std(lfp_sm, axis=1, keepdims=True)

    csd_na = pna.cal_1dCSD(lfp_na, axis_ch=0, tf_edge=True)
    csd_sm = pna.cal_1dCSD(lfp_sm, axis_ch=0, tf_edge=True)
    csd_nr = pna.cal_1dCSD(lfp_nr, axis_ch=0, tf_edge=True)

    plt.axes(h_axes[0, 1])
    pnp.ErpPlot_singlePanel(lfp_sm, ts)
    plt.title('LFP smoothed')

    if yn_plot_spike_quality:
        spike_quanlity = get_spike_quality_for_channel(datecode)
        plt.scatter(np.repeat(ts[0],N_chan_total), np.arange(N_chan_total),
                    c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral', edgecolors='k', s=100)
    if yn_plot_granular_loc:
        plt.plot(ts[0], get_granular_loc(datecode), 'k+')

    plt.axes(h_axes[0, 2])
    pnp.ErpPlot_singlePanel(lfp_nr, ts)
    plt.title('LFP normalized')

    plt.axes(h_axes[1, 0])
    pnp.ErpPlot_singlePanel(csd_na, ts, tf_inverse_color=True)
    plt.title('CSD native')
    plt.xlabel('time (s)')
    plt.ylabel('chan')

    plt.axes(h_axes[1, 1])
    pnp.ErpPlot_singlePanel(csd_sm, ts, tf_inverse_color=True)
    plt.title('CSD smoothed')

    if yn_plot_spike_quality:
        spike_quanlity = get_spike_quality_for_channel(datecode)
        plt.scatter(np.repeat(ts[0],N_chan_total), np.arange(N_chan_total),
                    c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral', edgecolors='k', s=100)
    if yn_plot_granular_loc:
        plt.plot(ts[0], get_granular_loc(datecode), 'k+')

    plt.axes(h_axes[1, 2])
    pnp.ErpPlot_singlePanel(csd_nr, ts, tf_inverse_color=True)
    plt.title('CSD normalized')

    plt.suptitle(datecode)

    plt.savefig(os.path.join(path_to_fig, 'final_csd_thor', 'csd_{}.png'.format(datecode)))
    plt.savefig(os.path.join(path_to_fig, 'final_csd_thor', 'csd_{}.pdf'.format(datecode)))
    plt.close('all')










