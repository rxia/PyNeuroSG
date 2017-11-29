""" script to test whether the information carried by IT and V4 is congruent or independent """


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


import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge
import sklearn
from sklearn import svm
from sklearn import cross_validation


""" load data """
# keyword_tank = '.*GM32.*U16.*161125.*'
keyword_tank = '.*GM32.*U16.*161222.*'
block_type = 'srv'
def gen_codeing_consistence_analysis(keyword_tank, brain_region='V4_IT'):
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), keyword_tank,
                                                               tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    t_plot = [0.050, 0.350]
    spike_bin_interval = 0.050
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=1 / spike_bin_interval,
                                               chan_filter=range(1, 48 + 1))

    grpby = ['stim_familiarized', 'mask_opacity_int']

    data_neuro = signal_align.neuro_sort(data_df, grpby, [], data_neuro)
    # data_neuro = signal_align.neuro_sort(data_df, ['', 'mask_opacity_int', ''], [], data_neuro)


    data_neuro['ts'] = data_neuro['ts'] + spike_bin_interval / 2
    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']
    N, T, M = data_neuro['data'].shape



    """ ==========  ==========  ==========  ========== """
    """ all pairs, all conditions """
    X_raw = np.mean(data_neuro['data'], axis=1)
    X = (X_raw - np.mean(X_raw, axis=0, keepdims=True)) \
        / np.std(X_raw, axis=0, keepdims=True)
    Y = np.array(data_df['stim_sname'])
    Y_uniq = np.unique(Y)


    """ all pairs """
    clf_V4 = sklearn.linear_model.LogisticRegression()
    clf_IT = sklearn.linear_model.LogisticRegression()
    dre_V4 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
    dre_IT = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
    if brain_region == 'V4_IT':
        indx_signal_V4 = (data_neuro['signal_info']['channel_index']<=32)
        indx_signal_IT = (data_neuro['signal_info']['channel_index']>32)
    elif brain_region == 'in_IT':
        indx_signal_V4 = (data_neuro['signal_info']['channel_index'] > 32) * (data_neuro['signal_info']['channel_index'] %2 ==0)
        indx_signal_IT = (data_neuro['signal_info']['channel_index'] > 32) * (data_neuro['signal_info']['channel_index'] %2 ==1)
    elif brain_region == 'in_V4':
        indx_signal_V4 = (data_neuro['signal_info']['channel_index'] <= 32) * (data_neuro['signal_info']['channel_index'] %2 ==0)
        indx_signal_IT = (data_neuro['signal_info']['channel_index'] <= 32) * (data_neuro['signal_info']['channel_index'] %2 ==1)
    list_color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    h_fig_lg, h_ax_lg = plt.subplots(nrows=2, ncols=3, figsize=(8,4), squeeze=False, sharex=True, sharey=True)
    h_fig_ld, h_ax_ld = plt.subplots(nrows=2, ncols=3, figsize=(8,4), squeeze=False, sharex=True, sharey=True)
    h_ax_lg = np.ravel(h_ax_lg)
    h_ax_ld = np.ravel(h_ax_ld)
    X_lg_cdtn = {}
    X_ld_cdtn = {}
    for i_c, c in enumerate(cdtn):
        X_lg = np.zeros([0,2])
        X_ld = np.zeros([0,2])
        indx_cdtn_c = data_neuro['cdtn_indx'][c]
        Y_uniq = np.unique(Y[indx_cdtn_c])
        for i in range(len(Y_uniq)):
            for j in range(len(Y_uniq)):
                if i==j:
                    continue
                else:
                    indx_subset = ((Y==Y_uniq[i]) | (Y==Y_uniq[j])) * pna.index_int2bool(indx_cdtn_c, N=N)
                    X_sub = X[indx_subset, :]
                    Y_sub = Y[indx_subset]
                    Y_sub_01 = (Y_sub == Y_uniq[j])

                    X_sub_V4 = X_sub[:, indx_signal_V4]
                    X_sub_IT = X_sub[:, indx_signal_IT]

                    X_lg_V4 = clf_V4.fit(X_sub_V4, Y_sub_01).predict_proba(X_sub_V4)[:, 1:]
                    X_lg_IT = clf_IT.fit(X_sub_IT, Y_sub_01).predict_proba(X_sub_IT)[:, 1:]
                    X_ld_V4 = dre_V4.fit(X_sub_V4, Y_sub_01).transform(X_sub_V4)
                    X_ld_IT = dre_IT.fit(X_sub_IT, Y_sub_01).transform(X_sub_IT)

                    # plt.plot(clf_V4.predict_proba(X_sub_V4)[:,1], clf_IT.predict_proba(X_sub_IT)[:,1], 'o')
                    plt.figure(h_fig_lg.number)
                    plt.axes(h_ax_lg[i_c])
                    plt.scatter(X_lg_V4[Y_sub == Y_uniq[i]], X_lg_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                    plt.scatter(X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)
                    plt.title(c)

                    plt.figure(h_fig_ld.number)
                    plt.axes(h_ax_ld[i_c])
                    plt.scatter(X_ld_V4[Y_sub == Y_uniq[i]], X_ld_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                    plt.scatter(X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)

                    X_lg = np.vstack( ( X_lg, np.hstack((X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]]) )))
                    X_ld = np.vstack( ( X_ld, np.hstack((X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]]) )))
                    plt.title(c)
        X_lg_cdtn[c] = X_lg
        X_ld_cdtn[c] = X_ld

    pickle.dump(X_lg, open('./temp_data/coding_consistency_{}_lg_{}.pkl'.format(brain_region, filename_common), 'wb'))
    pickle.dump(X_ld, open('./temp_data/coding_consistency_{}_ld_{}.pkl'.format(brain_region, filename_common), 'wb'))
    plt.figure(h_fig_lg.number)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.suptitle('trial_by_trial_decoding_consistency_{}_{}_logistic_regression'.format(brain_region, filename_common))
    plt.savefig('./temp_figs/trial_by_trial_decoding_consistency_{}_lg_{}.png'.format(brain_region, filename_common))
    plt.figure(h_fig_ld.number)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.suptitle('trial_by_trial_decoding_consistency_{}_{}_linear_discriminant'.format(brain_region, filename_common))
    plt.savefig('./temp_figs/trial_by_trial_decoding_consistency_{}_ld_{}.png'.format(brain_region, filename_common))



    """ correlation, test, the LDA results """
    h_fig_ld, h_ax_ld = plt.subplots(nrows=2, ncols=3, figsize=(8,4), squeeze=False, sharex=True, sharey=True)
    h_ax_ld = np.ravel(h_ax_ld)
    for i_c, c in enumerate(cdtn):
        X_ld = X_ld_cdtn[c]
        corr_empr = np.corrcoef(X_ld[:,0], X_ld[:,1] )[0,1]

        corr_permutation = [ np.corrcoef(X_ld[:,0], np.random.permutation(X_ld[:,1]) )[0,1] for temp in range(10000)]

        plt.axes(h_ax_ld[i_c])
        plt.plot(corr_empr, 0, 'ob', markersize=10)
        plt.hist(corr_permutation, bins=100, alpha=0.5)
        plt.title(c)
    plt.xlabel('corr between V4 & IT LDA projection')
    plt.legend(['empirical correlation', 'permutation test'])
    plt.suptitle('decoding_corr_{}_ld_by_{}_{}'.format(brain_region, grpby, filename_common))
    plt.savefig('./temp_figs/decoding_corr_{}_ld_{}.png'.format(brain_region, filename_common))




    """ correlation, test, the logistic regression results """
    h_fig_lg, h_ax_lg = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), squeeze=False, sharex=True, sharey=True)
    h_ax_lg = np.ravel(h_ax_lg)
    for i_c, c in enumerate(cdtn):
        X_lg = X_lg_cdtn[c]
        X_median = np.median(X_lg, axis=0)

        def cal_joint_proba_bin(X, X_cut):
            joint_count = np.zeros([2,2])
            joint_count[0, 0] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] <= X_cut[1]))
            joint_count[0, 1] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] >  X_cut[1]))
            joint_count[1, 0] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1] <= X_cut[1]))
            joint_count[1, 1] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1]  > X_cut[1]))
            joint_proba = 1.0*joint_count/X.shape[0]
            return joint_proba


        def cal_mutual_info(joint_proba):

            margn_proba0 = np.sum(joint_proba, axis=1, keepdims=True)
            margn_proba1 = np.sum(joint_proba, axis=0, keepdims=True)
            # return (joint_proba[0,0] + joint_proba[1,1]) - (margn_proba0[0]*margn_proba1[0] + margn_proba1[1]*margn_proba1[1])
            joint_proba_ind = margn_proba0 * margn_proba1
            return np.sum(joint_proba * np.log2(joint_proba/joint_proba_ind))

        def permute_X(X):
            X_permute = np.zeros(X.shape)
            X_permute[:, 0] = np.random.permutation(X[:, 0])
            X_permute[:, 1] = np.random.permutation(X[:, 1])
            return X_permute

        mutual_info_emp = cal_mutual_info(cal_joint_proba_bin(X_lg, X_median))
        mutual_info_permutation = [cal_mutual_info(cal_joint_proba_bin( permute_X(X_lg), X_median)) for temp in range(10000)]

        plt.axes(h_ax_lg[i_c])
        plt.plot(mutual_info_emp, 0, 'ob', markersize=10)
        plt.hist(mutual_info_permutation, bins=100, alpha=0.5)

        plt.title(c)
    plt.xlabel('mutual info between V4 & IT')
    plt.legend(['empirical correlation', 'permutation test'])
    plt.suptitle('decoding_mutural_info_{}_lg_by_{}_{}'.format(brain_region, grpby, filename_common))
    plt.savefig('./temp_figs/decoding_mutural_info_{}_lg_{}.png'.format(brain_region, filename_common))


""" run for everyday """
dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)


for name_tank in list_name_tanks:
    try:
        gen_codeing_consistence_analysis(name_tank, brain_region='in_V4')
    except:
        print('this day {} experience an error'.format(name_tank))
    plt.close('all')



""" load data """
temp_data_folder = './temp_data/'
pickle.dump(X_lg, open('./temp_data/coding_consistency_V4_IT_lg_{}.pkl'.format(filename_common), 'wb'))
pickle.dump(X_ld, open('./temp_data/coding_consistency_V4_IT_ld_{}.pkl'.format(filename_common), 'wb'))






""" ========== ---------- old code pool ---------- ========== """
""" decode """
X_raw = np.mean(data_neuro['data'], axis=1)
X = (X_raw - np.mean(X_raw, axis=0, keepdims=True)) \
    / np.std(X_raw, axis=0, keepdims=True)
Y = np.array(data_df['stim_sname'])
clf = sklearn.linear_model.LogisticRegression()
dre = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)

Y_uniq = np.unique(Y)

for mask_opacity in (0.0, 0.5, 0.7):
    i, j = np.random.permutation(len(Y_uniq))[0:2]
    indx_subset = ((Y==Y_uniq[i]) | (Y==Y_uniq[j])) * (data_df['mask_opacity']==mask_opacity)
    X_sub = X[indx_subset, :]
    Y_sub = Y[indx_subset]

    indx_signal_V4 = data_neuro['signal_info']['channel_index']<=32
    indx_signal_IT = data_neuro['signal_info']['channel_index']>32
    X_sub_V4 = X_sub[:, indx_signal_V4]
    X_sub_IT = X_sub[:, indx_signal_IT]

    clf_V4 = sklearn.linear_model.LogisticRegression()
    clf_IT = sklearn.linear_model.LogisticRegression()
    dre_V4 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
    dre_IT = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
    X_lg_V4 = clf_V4.fit(X_sub_V4, Y_sub).predict_proba(X_sub_V4)[:, 1]
    X_lg_IT = clf_IT.fit(X_sub_IT, Y_sub).predict_proba(X_sub_IT)[:, 1]
    X_ld_V4 = dre_V4.fit(X_sub_V4, Y_sub).transform(X_sub_V4)
    X_ld_IT = dre_IT.fit(X_sub_IT, Y_sub).transform(X_sub_IT)

    # plt.plot(clf_V4.predict_proba(X_sub_V4)[:,1], clf_IT.predict_proba(X_sub_IT)[:,1], 'o')
    h_fig, h_ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), squeeze=False)
    plt.axes(h_ax[0,0])
    for label in Y_uniq[i], Y_uniq[j]:
        plt.scatter(X_lg_V4[Y_sub==label], X_lg_IT[Y_sub==label], alpha=0.8, label=label)
    plt.legend()
    plt.title('logistic regression')
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.axes(h_ax[0,1])
    for label in Y_uniq[i], Y_uniq[j]:
        plt.scatter(X_ld_V4[Y_sub==label], X_ld_IT[Y_sub==label], alpha=0.8, label=label)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.title('linear discriminant analysis')





""" all pairs """
list_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
for mask_opacity in [0,0.5,0.7]:
    h_fig, h_ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), squeeze=False)
    X_lg_all = np.zeros([0,2])
    X_ld_all = np.zeros([0,2])
    for i in range(len(Y_uniq)):
        for j in range(len(Y_uniq)):
            if i==j:
                continue
            else:
                indx_subset = ((Y==Y_uniq[i]) | (Y==Y_uniq[j])) * (data_df['mask_opacity']==mask_opacity)
                X_sub = X[indx_subset, :]
                Y_sub = Y[indx_subset]
                Y_sub_01 = (Y_sub == Y_uniq[j])

                indx_signal_V4 = data_neuro['signal_info']['channel_index']<=32
                indx_signal_IT = data_neuro['signal_info']['channel_index']>32
                X_sub_V4 = X_sub[:, indx_signal_V4]
                X_sub_IT = X_sub[:, indx_signal_IT]

                clf_V4 = sklearn.linear_model.LogisticRegression()
                clf_IT = sklearn.linear_model.LogisticRegression()
                dre_V4 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
                dre_IT = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
                X_lg_V4 = clf_V4.fit(X_sub_V4, Y_sub_01).predict_proba(X_sub_V4)[:, 1:]
                X_lg_IT = clf_IT.fit(X_sub_IT, Y_sub_01).predict_proba(X_sub_IT)[:, 1:]
                X_ld_V4 = dre_V4.fit(X_sub_V4, Y_sub_01).transform(X_sub_V4)
                X_ld_IT = dre_IT.fit(X_sub_IT, Y_sub_01).transform(X_sub_IT)

                # plt.plot(clf_V4.predict_proba(X_sub_V4)[:,1], clf_IT.predict_proba(X_sub_IT)[:,1], 'o')
                plt.axes(h_ax[0, 0])
                plt.scatter(X_lg_V4[Y_sub == Y_uniq[i]], X_lg_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                plt.scatter(X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)


                plt.axes(h_ax[0,1])
                plt.scatter(X_ld_V4[Y_sub == Y_uniq[i]], X_ld_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                plt.scatter(X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)

                X_lg_all = np.vstack( ( X_lg_all, np.hstack((X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]]) )))
                X_ld_all = np.vstack( ( X_ld_all, np.hstack((X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]]) )))

    plt.axes(h_ax[0,0])
    plt.title('logistic regression')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.axes(h_ax[0,1])
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.title('linear discriminant analysis')
    plt.suptitle('trial_by_trial_decoding_consistency_V4_vs_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    plt.savefig('./temp_figs/trial_by_trial_decoding_consistency_V4_vs_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))



    """ correlation, test, the LDA results """
    corr_empr = np.corrcoef(X_ld_all[:,0], X_ld_all[:,1] )[0,1]

    corr_permutation = [ np.corrcoef(X_ld_all[:,0], np.random.permutation(X_ld_all[:,1]) )[0,1] for temp in range(10000)]

    plt.figure()
    plt.plot(corr_empr, 0, 'ob', markersize=10)
    plt.hist(corr_permutation, bins=100, alpha=0.5)

    # plt.plot(corr_empr, 0, '+k', markersize=10)
    plt.xlabel('corr between V4 & IT LDA projection')
    plt.xlabel('corr between V4 & IT, permutation test')
    plt.legend(['empirical correlation', 'permutation test'])
    plt.suptitle('decoding_corr_V4_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    # plt.savefig('./temp_figs/decoding_corr_V4_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))




    """ correlation, test, the logistic regression results """
    # plt.scatter(X_lg_all[:,0], X_lg_all[:,1], alpha=0.2, c=list_color[0], s=10)

    X_median = np.median(X_lg_all, axis=0)

    def cal_joint_proba_bin(X, X_cut):
        joint_count = np.zeros([2,2])
        joint_count[0, 0] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] <= X_cut[1]))
        joint_count[0, 1] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] >  X_cut[1]))
        joint_count[1, 0] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1] <= X_cut[1]))
        joint_count[1, 1] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1]  > X_cut[1]))
        joint_proba = 1.0*joint_count/X.shape[0]
        return joint_proba


    def cal_mutual_info(joint_proba):

        margn_proba0 = np.sum(joint_proba, axis=1, keepdims=True)
        margn_proba1 = np.sum(joint_proba, axis=0, keepdims=True)
        # return (joint_proba[0,0] + joint_proba[1,1]) - (margn_proba0[0]*margn_proba1[0] + margn_proba1[1]*margn_proba1[1])
        joint_proba_ind = margn_proba0 * margn_proba1
        return np.sum(joint_proba * np.log2(joint_proba/joint_proba_ind))

    def permute_X(X):
        X_permute = np.zeros(X.shape)
        X_permute[:, 0] = np.random.permutation(X[:, 0])
        X_permute[:, 1] = np.random.permutation(X[:, 1])
        return X_permute

    mutual_info_emp = cal_mutual_info(cal_joint_proba_bin(X_lg_all, X_median))
    mutual_info_permutation = [cal_mutual_info(cal_joint_proba_bin( permute_X(X_lg_all), X_median)) for temp in range(10000)]

    plt.figure()
    plt.plot(mutual_info_emp, 0, 'ob', markersize=10)
    plt.hist(mutual_info_permutation, bins=100, alpha=0.5)
    plt.xlabel('mutual info between V4 & IT LDA projection')
    plt.xlabel('mutual info between V4 & IT, permutation test')
    plt.legend(['empirical information', 'permutation test'])
    plt.suptitle('decoding_mutual_info_V4_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    # plt.savefig('./temp_figs/decoding_mutual_info_V4_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))






""" ==========  ==========  ==========  ========== """
""" all pairs, all conditions """

""" all pairs """
clf_V4 = sklearn.linear_model.LogisticRegression()
clf_IT = sklearn.linear_model.LogisticRegression()
dre_V4 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
dre_IT = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
indx_signal_V4 = (data_neuro['signal_info']['channel_index']<=32) * (np.random.rand(len(data_neuro['signal_info']))>0.5)
indx_signal_IT = (data_neuro['signal_info']['channel_index']<=32) - indx_signal_V4
list_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
for mask_opacity in [0,0.5,0.7]:
    h_fig, h_ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), squeeze=False)
    X_lg_all = np.zeros([0,2])
    X_ld_all = np.zeros([0,2])
    for i in range(len(Y_uniq)):
        for j in range(len(Y_uniq)):
            if i==j:
                continue
            else:
                i, j = np.random.permutation(len(Y_uniq))[0:2]
                indx_subset = ((Y==Y_uniq[i]) | (Y==Y_uniq[j])) * (data_df['mask_opacity']==mask_opacity)
                X_sub = X[indx_subset, :]
                Y_sub = Y[indx_subset]
                Y_sub_01 = (Y_sub == Y_uniq[j])

                X_sub_V4 = X_sub[:, indx_signal_V4]
                X_sub_IT = X_sub[:, indx_signal_IT]

                X_lg_V4 = clf_V4.fit(X_sub_V4, Y_sub_01).predict_proba(X_sub_V4)[:, 1:]
                X_lg_IT = clf_IT.fit(X_sub_IT, Y_sub_01).predict_proba(X_sub_IT)[:, 1:]
                X_ld_V4 = dre_V4.fit(X_sub_V4, Y_sub_01).transform(X_sub_V4)
                X_ld_IT = dre_IT.fit(X_sub_IT, Y_sub_01).transform(X_sub_IT)

                # plt.plot(clf_V4.predict_proba(X_sub_V4)[:,1], clf_IT.predict_proba(X_sub_IT)[:,1], 'o')
                plt.axes(h_ax[0, 0])
                plt.scatter(X_lg_V4[Y_sub == Y_uniq[i]], X_lg_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                plt.scatter(X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)


                plt.axes(h_ax[0,1])
                plt.scatter(X_ld_V4[Y_sub == Y_uniq[i]], X_ld_IT[Y_sub == Y_uniq[i]], alpha=0.2, c=list_color[0], s=10)
                plt.scatter(X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]], alpha=0.2, c=list_color[1], s=10)

                X_lg_all = np.vstack( ( X_lg_all, np.hstack((X_lg_V4[Y_sub == Y_uniq[j]], X_lg_IT[Y_sub == Y_uniq[j]]) )))
                X_ld_all = np.vstack( ( X_ld_all, np.hstack((X_ld_V4[Y_sub == Y_uniq[j]], X_ld_IT[Y_sub == Y_uniq[j]]) )))

    plt.axes(h_ax[0,0])
    plt.title('logistic regression')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.axes(h_ax[0,1])
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('V4')
    plt.ylabel('IT')
    plt.title('linear discriminant analysis')
    plt.suptitle('trial_by_trial_decoding_consistency_V4_vs_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    # plt.savefig('./temp_figs/trial_by_trial_decoding_consistency_V4_vs_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))



    """ correlation, test, the LDA results """
    corr_empr = np.corrcoef(X_ld_all[:,0], X_ld_all[:,1] )[0,1]

    corr_permutation = [ np.corrcoef(X_ld_all[:,0], np.random.permutation(X_ld_all[:,1]) )[0,1] for temp in range(10000)]

    plt.figure()
    plt.plot(corr_empr, 0, 'ob', markersize=10)
    plt.hist(corr_permutation, bins=100, alpha=0.5)

    # plt.plot(corr_empr, 0, '+k', markersize=10)
    plt.xlabel('corr between V4 & IT LDA projection')
    plt.xlabel('corr between V4 & IT, permutation test')
    plt.legend(['empirical correlation', 'permutation test'])
    plt.suptitle('decoding_corr_V4_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    # plt.savefig('./temp_figs/decoding_corr_V4_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))




    """ correlation, test, the logistic regression results """
    # plt.scatter(X_lg_all[:,0], X_lg_all[:,1], alpha=0.2, c=list_color[0], s=10)

    X_median = np.median(X_lg_all, axis=0)

    def cal_joint_proba_bin(X, X_cut):
        joint_count = np.zeros([2,2])
        joint_count[0, 0] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] <= X_cut[1]))
        joint_count[0, 1] = np.sum((X[:, 0] <= X_cut[0]) * (X[:, 1] >  X_cut[1]))
        joint_count[1, 0] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1] <= X_cut[1]))
        joint_count[1, 1] = np.sum((X[:, 0] >  X_cut[0]) * (X[:, 1]  > X_cut[1]))
        joint_proba = 1.0*joint_count/X.shape[0]
        return joint_proba


    def cal_mutual_info(joint_proba):

        margn_proba0 = np.sum(joint_proba, axis=1, keepdims=True)
        margn_proba1 = np.sum(joint_proba, axis=0, keepdims=True)
        # return (joint_proba[0,0] + joint_proba[1,1]) - (margn_proba0[0]*margn_proba1[0] + margn_proba1[1]*margn_proba1[1])
        joint_proba_ind = margn_proba0 * margn_proba1
        return np.sum(joint_proba * np.log2(joint_proba/joint_proba_ind))

    def permute_X(X):
        X_permute = np.zeros(X.shape)
        X_permute[:, 0] = np.random.permutation(X[:, 0])
        X_permute[:, 1] = np.random.permutation(X[:, 1])
        return X_permute

    mutual_info_emp = cal_mutual_info(cal_joint_proba_bin(X_lg_all, X_median))
    mutual_info_permutation = [cal_mutual_info(cal_joint_proba_bin( permute_X(X_lg_all), X_median)) for temp in range(10000)]

    plt.figure()
    plt.plot(mutual_info_emp, 0, 'ob', markersize=10)
    plt.hist(mutual_info_permutation, bins=100, alpha=0.5)
    plt.xlabel('mutual info between V4 & IT LDA projection')
    plt.xlabel('mutual info between V4 & IT, permutation test')
    plt.legend(['empirical information', 'permutation test'])
    plt.suptitle('decoding_mutual_info_V4_IT_{}_noise_{}'.format(filename_common, mask_opacity))
    # plt.savefig('./temp_figs/decoding_mutual_info_V4_IT_{}_noise_{}.png'.format(filename_common, mask_opacity))