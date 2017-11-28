""" 2017-0509 script to comapare the response between 1) preferred image, but noisy, 2) non-preferred image, but not noisy """

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import cPickle as pickle


import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data



""" load data """
try:
    dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
except:
    dir_tdt_tank = '/Volumes/Labfiles/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

list_date = [re.match('.*-(\d{6})-\d{6}', tankname).group(1) for tankname in list_name_tanks]

block_type = 'srv_mask'
# block_type = 'matchnot'
tf_cumsum = True

if block_type == 'matchnot':
    t_plot = [-0.200, 0.800]
else:
    t_plot = [-0.200, 0.800]
if tf_cumsum == True:
    t_plot = [0, 0.800]


def compute_decoding(data_neuro, data_df, label_name='stim_sname', electrode_type='GM32', cdtn=(1,70)):
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']
    if electrode_type == 'GM32':
        limit_ch = signal_info['channel_index'] <=32
    else:
        limit_ch = signal_info['channel_index'] > 32
    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    M = len(list_image_name)
    all_clf_score = np.zeros([M, M, len(ts)])*np.nan
    for indx_img1 in range(len(list_image_name)):
        for indx_img2 in range(indx_img1-1):
            fltr_tr = (data_df['stim_sname']==list_image_name[indx_img1]) | (data_df['stim_sname']==list_image_name[indx_img2])
            clf_score = pna.decode_over_time(data_neuro['data'], label=label, limit_tr=limit_tr & fltr_tr, limit_ch=limit_ch)
            all_clf_score[indx_img1, indx_img2, :] =  clf_score
    mean_clf_score = np.nanmean(all_clf_score, axis=(0,1))
    # plt.plot(ts, np.mean(np.vstack(all_clf_score), axis=0))
    return all_clf_score


def compute_decoding_of_day(tankname):

    date = re.match('.*-(\d{6})-\d{6}', tankname).group(1)

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    spike_bin_interval =0.050
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=1/spike_bin_interval,
                                                   chan_filter=range(1, 48 + 1))

    data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)

    data_neuro['ts'] = data_neuro['ts']+spike_bin_interval/2
    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']

    X = np.mean(data_neuro['data'][:, (ts >= 0.20) * (ts <= 0.40), :], axis=1)
    Y = np.array(data_df['stim_sname'])

    def decoding_over_noise(X=X, Y=Y, V4_IT='IT', nov_fam=0):
        if V4_IT == 'V4':
            tf_neuron = (data_neuro['signal_info']['channel_index'] <= 32)
        elif V4_IT == 'IT':
            tf_neuron = (data_neuro['signal_info']['channel_index'] > 32)
        X = X[:, tf_neuron]
        fltr_00 = data_neuro['cdtn_indx'][cdtn[nov_fam * 3]]
        fltr_50 = data_neuro['cdtn_indx'][cdtn[nov_fam * 3 + 1]]
        fltr_70 = data_neuro['cdtn_indx'][cdtn[nov_fam * 3 + 2]]
        X_00 = X[fltr_00, :]
        Y_00 = Y[fltr_00]
        X_50 = X[fltr_50, :]
        Y_50 = Y[fltr_50]
        X_70 = X[fltr_70, :]
        Y_70 = Y[fltr_70]
        clf = linear_model.LogisticRegression(solver='lbfgs', warm_start=False, multi_class='multinomial',
                                              fit_intercept=False)
        # print(model_selection.cross_val_score(clf, X_cur, Y_cur, cv=2))

        def get_score_p(clf, X, Y):
            dict_label2indx = dict(zip(clf.classes_, np.arange(len(clf.classes_))))
            Y_indx = np.array([dict_label2indx[y] for y in Y])
            proba_all = clf.predict_proba(X)
            proba_target = proba_all[np.arange(len(Y)), Y_indx]
            return np.mean(proba_target)

        n_splits = 20
        object_cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4)
        score_cv = []
        for indx_train, indx_test in object_cv.split(X_00, Y_00):
            clf.fit(X_00[indx_train, :], Y_00[indx_train])
            score_cv.append([get_score_p(clf, X_00[indx_test, :], Y_00[indx_test]), get_score_p(clf, X_50, Y_50),
                             get_score_p(clf, X_70, Y_70)])

        return np.mean(score_cv, axis=0)

    return decoding_over_noise(V4_IT='V4', nov_fam=0), decoding_over_noise(V4_IT='V4', nov_fam=1), \
           decoding_over_noise(V4_IT='IT', nov_fam=0), decoding_over_noise(V4_IT='IT', nov_fam=1)


"""" run code for all sesions, store decodign results (very slow) """
list_decoding_result = []
for tankname in list_name_tanks:
    try:
        print('decoding {}'.format(tankname))
        decoding_result = compute_decoding_of_day(tankname)
        list_decoding_result.append(decoding_result)
        plt.close('all')
    except:
        print('can not plot decoding results for {}'.format(tankname))

if tf_cumsum == True:
    signal_cum = '_cum'
else:
    signal_cum = ''
pickle.dump(list_decoding_result,
                    open('/shared/homes/sguan/Coding_Projects/support_data/Decoding_over_noise_{}'.format(block_type), "wb"))

plt.plot(np.mean(np.array(list_decoding_result), axis=0).transpose());  plt.legend([1,2,3,4])

"""" load data, plot """
list_cdtn = sorted(list_decoding_result[0]['img_GM32'].keys())
ts = np.array([-0.175, -0.125, -0.075, -0.025,  0.025,  0.075,  0.125,  0.175,
        0.225,  0.275,  0.325,  0.375,  0.425,  0.475,  0.525,  0.575,
        0.625,  0.675,  0.725,  0.775])
if tf_cumsum:
    ts =  np.array([0.025,  0.075,  0.125,  0.175,
            0.225,  0.275,  0.325,  0.375,  0.425,  0.475,  0.525,  0.575,
            0.625,  0.675,  0.725,  0.775])
list_decoding_xy =  ['img_GM32', 'noi_GM32', 'img_U16', 'noi_U16']
date_area = dict()
date_area['161015'] = 'IT'
date_area['161023'] = 'STS'
date_area['161026'] = 'STS'
date_area['161029'] = 'IT'
date_area['161118'] = 'STS'
date_area['161121'] = 'STS'
date_area['161125'] = 'STS'
date_area['161202'] = 'STS'
date_area['161206'] = 'IT'
date_area['161222'] = 'STS'
date_area['161228'] = 'IT'
date_area['170103'] = 'IT'
date_area['170106'] = 'STS'
date_area['170113'] = 'IT'
date_area['170117'] = 'IT'
date_area['170214'] = 'STS'
date_area['170221'] = 'STS'


decoding_ave ={}
decoding_ave['img_V4'] = {}
decoding_ave['noi_V4'] = {}
decoding_ave['img_IT'] = {}
decoding_ave['noi_IT'] = {}
decoding_ave['img_STS'] = {}
decoding_ave['noi_STS'] = {}
for i, cdtn in enumerate(list_cdtn):
    decoding_ave['img_V4'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_GM32'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result]), axis=0)

    decoding_ave['noi_V4'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_GM32'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result]), axis=0)

    decoding_ave['img_IT'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='IT']), axis=0)

    decoding_ave['noi_IT'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='IT']), axis=0)

    decoding_ave['img_STS'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='STS']), axis=0)

    decoding_ave['noi_STS'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='STS']), axis=0)

h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_V4'][cdtn])
    plt.plot(ts, decoding_ave['noi_V4'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_V4')
plt.savefig('./temp_figs/binary_decoding_ave_V4.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_V4.png')


h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_IT'][cdtn])
    plt.plot(ts, decoding_ave['noi_IT'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_IT')
plt.savefig('./temp_figs/binary_decoding_ave_IT.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_IT.png')


h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_STS'][cdtn])
    plt.plot(ts, decoding_ave['noi_STS'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_STS')
plt.savefig('./temp_figs/binary_decoding_ave_STS.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_STS.png')



""" test script """
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection



X = np.mean(data_neuro['data'][:, (ts>=0) * (ts<=0.40), :], axis=1)
Y = np.array(data_df['stim_sname'])

def decoding_over_noise(X=X, Y=Y, V4_IT= 'IT', nov_fam=0):
    if V4_IT == 'V4':
        tf_neuron = ( data_neuro['signal_info']['channel_index'] <= 32 )
    elif V4_IT == 'IT':
        tf_neuron = ( data_neuro['signal_info']['channel_index'] > 32 )
    X = X[:, tf_neuron]
    fltr_00 = data_neuro['cdtn_indx'][cdtn[nov_fam*3]]
    fltr_50 = data_neuro['cdtn_indx'][cdtn[nov_fam*3+1]]
    fltr_70 = data_neuro['cdtn_indx'][cdtn[nov_fam*3+2]]
    X_00 = X[fltr_00, :]
    Y_00 = Y[fltr_00]
    X_50 = X[fltr_50, :]
    Y_50 = Y[fltr_50]
    X_70 = X[fltr_70, :]
    Y_70 = Y[fltr_70]
    clf = linear_model.LogisticRegression(solver='lbfgs', warm_start=False, multi_class='multinomial', fit_intercept=False)
    # print(model_selection.cross_val_score(clf, X_cur, Y_cur, cv=2))

    def get_score_p(clf, X, Y):
        dict_label2indx = dict(zip(clf.classes_, np.arange(len(clf.classes_))))
        Y_indx = np.array([dict_label2indx[y] for y in Y])
        proba_all = clf.predict_proba(X)
        proba_target = proba_all[np.arange(len(Y)), Y_indx]
        return np.mean(proba_target)

    n_splits= 20
    object_cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4)
    score_cv= []
    for indx_train, indx_test in object_cv.split(X_00, Y_00):
        clf.fit(X_00[indx_train, :], Y_00[indx_train])
        score_cv.append([get_score_p(clf, X_00[indx_test,:], Y_00[indx_test]) , get_score_p(clf, X_50, Y_50), get_score_p(clf, X_70, Y_70)])

    return np.mean(score_cv, axis=0)


decoding_over_noise(V4_IT= 'IT', nov_fam=0)
print (decoding_over_noise(V4_IT= 'IT', nov_fam=0))
print (decoding_over_noise(V4_IT= 'IT', nov_fam=1))



