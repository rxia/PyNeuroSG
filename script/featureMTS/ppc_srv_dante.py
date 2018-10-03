import store_hdf5
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp
import math
import time
import mne
from mne.connectivity import spectral_connectivity
import copy
import signal_align
import matplotlib as mpl
import matplotlib.pyplot as plt
import df_ana
import PyNeuroAna as pna
import PyNeuroPlot as pnp

task = 'srv'
if task=='srv':
    dates = ['161015','161029','161118','161121','161125','161202','161206','161228','170103','170106','170113','170117','170214','170221']
else:
    dates = ['161118','161121','161125','161202','161206','161228','170103','170106','170113','170117','170214','170221']

## The function for getting the depth profile of IT channels
# granular layer location is zero-indexed
path_to_U_probe_loc = './script/U16_location_Dante_Thor.csv'
loc_U16 = pd.read_csv(path_to_U_probe_loc)
loc_U16['date'] = loc_U16['date'].astype('str')
def signal_extend_loc_info(channel_index, day):
    signal_all = pd.DataFrame({'channel_index': channel_index, 'date': day})
    signal_extend = pd.merge(signal_all, loc_U16, 'left', on='date')

    signal_extend['channel_index_U16'] = signal_extend['channel_index'] - 32
    signal_extend['area'] = np.where((signal_extend['channel_index']<=32),['V4']*len(signal_extend), signal_extend['area'])

    signal_extend['depth'] = 0
    signal_extend['depth'] = (signal_extend['channel_index_U16']-1-signal_extend['granular']) * (signal_extend['area']=='TEd') \
                             +(16-signal_extend['channel_index_U16']-signal_extend['granular']) * (signal_extend['area']=='TEm')
    signal_extend['depth'] = np.where(signal_extend['area']=='V4',
                                     np.zeros(len(signal_extend))*np.nan, signal_extend['depth'])
    return signal_extend['depth']
##

ppc_V4_V4,ppc_V4_ITi,ppc_V4_ITs,ppc_ITi_V4,ppc_ITs_V4,ppc_ITi_ITi,ppc_ITi_ITs,ppc_ITs_ITi,ppc_ITs_ITs = [],[],[],[],[],[],[],[],[]
for day in dates:
    if task == 'srv':
        mat = sp.io.loadmat('../../analysis/dual_opto/results/d_{}/ppc_familiar_noise.mat'.format(day))  # load mat-file
    else:
        mat = sp.io.loadmat('../../analysis/dual_opto/results/d_{}/ppc_matchnot_familiar_noise.mat'.format(day))  # load mat-file
    for i in range(len(mat['ppc'])):
        mdata = mat['ppc'][i]  # variable in mat file
        if i==0:
            mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
            ppc = {n: mdata[n][0] for n in mdtype.names}
            ppc['ppc2'] = [ppc['ppc2']]
            ppc['time'], ppc['freq'] = ppc['time'][0], ppc['freq'][0]
            ppc['spk_channel'] = np.floor_divide(np.stack(ppc['labelcmb'][:,0])[:,0].astype('int'),10)
            ppc['lfp_channel'] = np.stack(ppc['labelcmb'][:,1])[:,0].astype('int')
            ppc['spk_depth'] = signal_extend_loc_info(ppc['spk_channel'],day)
            ppc['lfp_depth'] = signal_extend_loc_info(ppc['lfp_channel'],day)
        else:
            ppc['ppc2'].append(mdata['ppc2'][0])
    ppc_value = np.stack(ppc['ppc2'])
    for i in range(len(ppc['labelcmb'])):
        if ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]<=32:
            ppc_V4_V4.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0:
            ppc_V4_ITs.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0:
            ppc_V4_ITi.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]<=32 and ppc['spk_depth'][i]<0:
            ppc_ITs_V4.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]<=32 and ppc['spk_depth'][i]>0:
            ppc_ITi_V4.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0 and ppc['spk_depth'][i]<0:
            ppc_ITi_ITi.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0 and ppc['spk_depth'][i]<0:
            ppc_ITi_ITs.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0 and ppc['spk_depth'][i]>0:
            ppc_ITs_ITi.append(ppc_value[:,i,:,:])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0 and ppc['spk_depth'][i]>0:
            ppc_ITs_ITs.append(ppc_value[:,i,:,:])

ppc_V4_V4,ppc_V4_ITi,ppc_V4_ITs,ppc_ITi_V4,ppc_ITs_V4 = np.stack(ppc_V4_V4),np.stack(ppc_V4_ITi),np.stack(ppc_V4_ITs),np.stack(ppc_ITi_V4),np.stack(ppc_ITs_V4)
ppc_ITi_ITi,ppc_ITi_ITs,ppc_ITs_ITi,ppc_ITs_ITs = np.stack(ppc_ITi_ITi),np.stack(ppc_ITi_ITs),np.stack(ppc_ITs_ITi),np.stack(ppc_ITs_ITs)
freqs,times = ppc['freq'],ppc['time']

##
titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [ppc_V4_V4,ppc_V4_ITi,ppc_V4_ITs,ppc_ITi_V4,ppc_ITs_V4,ppc_ITi_ITi,ppc_ITi_ITs,ppc_ITs_ITi,ppc_ITs_ITs]
f_range = (freqs>5) & (freqs<40)
t_range = (times>-0.1) & (times<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('jet')
for i in range(9):
    for j in range(6):
        plt.subplot(6,9,j*9+i+1)
        plt.pcolormesh(times[t_range],freqs[f_range],np.nanmean(to_plot[i][:,j,f_range,:][:,:,t_range],axis=0))
        plt.clim(0,0.1)
        # plt.colorbar()
        plt.title(titles[i])
    # plt.savefig('./script/featureMTS/figs/d_{}_no_clim.pdf'.format(titles[i]))
    # plt.savefig('./script/featureMTS/figs/d_{}_no_clim.png'.format(titles[i]))

# plt.figure()
# plt.set_cmap('coolwarm')
# for i in range(4):
#     for j in range(3):
#         plt.subplot(4,3,i*3+j+1)
#         plt.pcolormesh(ppc['time'][t_range],ppc['freq'][f_range],(np.nanmean(to_plot[i][:,j+3,f_range,:][:,:,t_range],axis=0)-np.nanmean(to_plot[i][:,j,f_range,:][:,:,t_range],axis=0)))
#         plt.clim(-0.05,0.05)
#         plt.colorbar()

## Shuffle
ppc_V4_V4_shuffle,ppc_V4_ITi_shuffle,ppc_V4_ITs_shuffle,ppc_ITi_V4_shuffle,ppc_ITs_V4_shuffle,ppc_ITi_ITi_shuffle,ppc_ITi_ITs_shuffle,ppc_ITs_ITi_shuffle,ppc_ITs_ITs_shuffle = [],[],[],[],[],[],[],[],[]
for day in dates:
    mat = sp.io.loadmat('../../analysis/dual_opto/results/d_{}/ppc_shuffle_matchnot_familiar_noise_1_small.mat'.format(day))  # load mat-file
    mdata = mat['ppc_shuffle']  # variable in mat file
    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
    ppc = {n: mdata[n][0] for n in mdtype.names}
    ppc['ppc2'] = ppc['mean_2D'][0]
    ppc['time'], ppc['freq'] = ppc['time'][0].ravel(), ppc['freq'][0].ravel()
    ppc['labelcmb'] = ppc['labelcmb'][0]
    ppc['spk_channel'] = np.floor_divide(np.stack(ppc['labelcmb'][:,0])[:,0].astype('int'),10)
    ppc['lfp_channel'] = np.stack(ppc['labelcmb'][:,1])[:,0].astype('int')
    ppc['spk_depth'] = signal_extend_loc_info(ppc['spk_channel'],day)
    ppc['lfp_depth'] = signal_extend_loc_info(ppc['lfp_channel'],day)
    ppc_value = ppc['ppc2']
    for i in range(len(ppc['labelcmb'])):
        if ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]<=32:
            ppc_V4_V4_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0:
            ppc_V4_ITs_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]<=32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0:
            ppc_V4_ITi_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]<=32 and ppc['spk_depth'][i]<0:
            ppc_ITs_V4_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]<=32 and ppc['spk_depth'][i]>0:
            ppc_ITi_V4_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0 and ppc['spk_depth'][i]<0:
            ppc_ITi_ITi_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0 and ppc['spk_depth'][i]<0:
            ppc_ITi_ITs_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]<0 and ppc['spk_depth'][i]>0:
            ppc_ITs_ITi_shuffle.append(ppc_value[i])
        elif ppc['spk_channel'][i]>32 and ppc['lfp_channel'][i]>32 and ppc['lfp_depth'][i]>0 and ppc['spk_depth'][i]>0:
            ppc_ITs_ITs_shuffle.append(ppc_value[i])

ppc_V4_V4_shuffle,ppc_V4_ITi_shuffle,ppc_V4_ITs_shuffle,ppc_ITi_V4_shuffle,ppc_ITs_V4_shuffle = np.stack(ppc_V4_V4_shuffle),np.stack(ppc_V4_ITi_shuffle),np.stack(ppc_V4_ITs_shuffle),np.stack(ppc_ITi_V4_shuffle),np.stack(ppc_ITs_V4_shuffle)
ppc_ITi_ITi_shuffle,ppc_ITi_ITs_shuffle,ppc_ITs_ITi_shuffle,ppc_ITs_ITs_shuffle = np.stack(ppc_ITi_ITi_shuffle),np.stack(ppc_ITi_ITs_shuffle),np.stack(ppc_ITs_ITi_shuffle),np.stack(ppc_ITs_ITs_shuffle)

freqs_shuffle,times_shuffle = ppc['freq'],ppc['time']

##
titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [ppc_V4_V4_shuffle,ppc_V4_ITi_shuffle,ppc_V4_ITs_shuffle,ppc_ITi_V4_shuffle,ppc_ITs_V4_shuffle,ppc_ITi_ITi_shuffle,ppc_ITi_ITs_shuffle,ppc_ITs_ITi_shuffle,ppc_ITs_ITs_shuffle]
f_range = (freqs_shuffle>4) & (freqs_shuffle<40)
t_range = (times_shuffle>-0.1) & (times_shuffle<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('jet')
for i in range(9):
    for j in range(6):
        plt.subplot(6,9,j*9+i+1)
        plt.pcolormesh(times_shuffle[t_range],freqs_shuffle[f_range],np.nanmean(to_plot[i][:,:,:,j][:,f_range,:][:,:,t_range],axis=0))
        plt.clim(0,0.1)
        plt.colorbar()
        plt.title(titles[i])

##
titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [np.moveaxis(ppc_V4_V4[:,:,freqs<50,:],1,3) - ppc_V4_V4_shuffle,
           np.moveaxis(ppc_V4_ITi[:,:,freqs<50,:],1,3) - ppc_V4_ITi_shuffle,
           np.moveaxis(ppc_V4_ITs[:,:,freqs<50,:],1,3) - ppc_V4_ITs_shuffle,
           np.moveaxis(ppc_ITi_V4[:,:,freqs<50,:],1,3) - ppc_ITi_V4_shuffle,
           np.moveaxis(ppc_ITs_V4[:,:,freqs<50,:],1,3) - ppc_ITs_V4_shuffle,
           np.moveaxis(ppc_ITi_ITi[:,:,freqs<50,:],1,3) - ppc_ITi_ITi_shuffle,
           np.moveaxis(ppc_ITi_ITs[:,:,freqs<50,:],1,3) - ppc_ITi_ITs_shuffle,
           np.moveaxis(ppc_ITs_ITi[:,:,freqs<50,:],1,3) - ppc_ITs_ITi_shuffle,
           np.moveaxis(ppc_ITs_ITs[:,:,freqs<50,:],1,3) - ppc_ITs_ITs_shuffle]
f_range = (freqs_shuffle>4) & (freqs_shuffle<50)
t_range = (times_shuffle>-0.1) & (times_shuffle<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('coolwarm')
for i in range(9):
    max_image = []
    for j in range(6):
        plt.subplot(6,9,j*9+i+1)
        image = np.nanmean(to_plot[i][:,:,:,j][:,f_range,:][:,:,t_range],axis=0)
        max_image.append(np.max(image))
        plt.pcolormesh(times_shuffle[t_range],freqs_shuffle[f_range],image)
        plt.title(titles[i])
    for j in range(6):
        plt.subplot(6,9,j*9+i+1)
        plt.clim(-np.max(max_image),np.max(max_image))
        plt.colorbar()
##
titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [np.moveaxis(ppc_V4_V4[:,:,freqs<50,:],1,3) - ppc_V4_V4_shuffle,
           np.moveaxis(ppc_V4_ITi[:,:,freqs<50,:],1,3) - ppc_V4_ITi_shuffle,
           np.moveaxis(ppc_V4_ITs[:,:,freqs<50,:],1,3) - ppc_V4_ITs_shuffle,
           np.moveaxis(ppc_ITi_V4[:,:,freqs<50,:],1,3) - ppc_ITi_V4_shuffle,
           np.moveaxis(ppc_ITs_V4[:,:,freqs<50,:],1,3) - ppc_ITs_V4_shuffle,
           np.moveaxis(ppc_ITi_ITi[:,:,freqs<50,:],1,3) - ppc_ITi_ITi_shuffle,
           np.moveaxis(ppc_ITi_ITs[:,:,freqs<50,:],1,3) - ppc_ITi_ITs_shuffle,
           np.moveaxis(ppc_ITs_ITi[:,:,freqs<50,:],1,3) - ppc_ITs_ITi_shuffle,
           np.moveaxis(ppc_ITs_ITs[:,:,freqs<50,:],1,3) - ppc_ITs_ITs_shuffle]
f_range = (freqs_shuffle>4) & (freqs_shuffle<50)
t_range = (times_shuffle>-0.1) & (times_shuffle<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('coolwarm')
for i in range(9):
    for j in range(3):
        plt.subplot(3,9,j*9+i+1)
        plt.pcolormesh(times_shuffle[t_range],freqs_shuffle[f_range],np.nanmean(to_plot[i][:,:,:,j+3][:,f_range,:][:,:,t_range],axis=0)-np.nanmean(to_plot[i][:,:,:,j][:,f_range,:][:,:,t_range],axis=0))
        plt.clim(-0.005,0.005)
        plt.colorbar()
        plt.title(titles[i])

##
def remove_outlier(data,threshold=3):
    median = np.nanmedian(data, axis=0)
    if len(data.shape)==2:
        diff = np.sqrt(np.nansum((data - median) ** 2, axis=1))
    elif len(data.shape)==3:
        diff = np.sqrt(np.nansum((data - median) ** 2, axis=(1,2)))
    med_abs_deviation = np.nanmedian(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    data = data[modified_z_score < threshold, :]
    return data


titles = ['V4 spike - V4 LFP','V4 spike - ITi LFP','V4 spike - ITs LFP','ITi spike - V4 LFP','ITs spike - V4 LFP','ITi spike - ITi LFP','ITi spike - ITs LFP','ITs spike - ITi LFP','ITs spike - ITs LFP']
to_plot = [np.moveaxis(ppc_V4_V4[:,:,freqs<50,:],1,3) - ppc_V4_V4_shuffle,
           np.moveaxis(ppc_V4_ITi[:,:,freqs<50,:],1,3) - ppc_V4_ITi_shuffle,
           np.moveaxis(ppc_V4_ITs[:,:,freqs<50,:],1,3) - ppc_V4_ITs_shuffle,
           np.moveaxis(ppc_ITi_V4[:,:,freqs<50,:],1,3) - ppc_ITi_V4_shuffle,
           np.moveaxis(ppc_ITs_V4[:,:,freqs<50,:],1,3) - ppc_ITs_V4_shuffle,
           np.moveaxis(ppc_ITi_ITi[:,:,freqs<50,:],1,3) - ppc_ITi_ITi_shuffle,
           np.moveaxis(ppc_ITi_ITs[:,:,freqs<50,:],1,3) - ppc_ITi_ITs_shuffle,
           np.moveaxis(ppc_ITs_ITi[:,:,freqs<50,:],1,3) - ppc_ITs_ITi_shuffle,
           np.moveaxis(ppc_ITs_ITs[:,:,freqs<50,:],1,3) - ppc_ITs_ITs_shuffle]
f_range = (freqs_shuffle>4) & (freqs_shuffle<6)
f_range = (freqs_shuffle>6) & (freqs_shuffle<13)
t_range = (times_shuffle>-0.1) & (times_shuffle<0.4)
plt.figure(figsize=[12,8])
colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                    pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
cdtn_name = ['nov, 0%', 'nov, 50%', 'nov, 70%', 'fam, 0%', 'fam, 50%', 'fam, 70%']
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.title(titles[i])
    for j in range(6):
        data_to_show = np.nanmean(to_plot[i][:, :, :, j][:, f_range, :][:, :, t_range], axis=1)
        data_to_show = remove_outlier(data_to_show)
        mean = np.nanmean(data_to_show,axis=0)
        error = np.nanstd(data_to_show,axis=0)/np.sqrt(to_plot[i].shape[0])*2
        plt.plot(times_shuffle[t_range],mean,color=colors[j],linestyle=linestyles[j],label=cdtn_name[j])
        if j in [0,1,2]:
            plt.fill_between(times_shuffle[t_range], mean-error, mean+error,color=colors[j],alpha=0.2)
        else:
            plt.fill_between(times_shuffle[t_range], mean-error, mean+error,color=colors[j],alpha=0.4)
    if i==0:
        plt.legend()
    plt.ylim(-0.002,0.015)
    plt.ticklabel_format(style='sci',scilimits=(2,5),axis='y')

##
titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [ppc_V4_V4,ppc_V4_ITi,ppc_V4_ITs,ppc_ITi_V4,ppc_ITs_V4,ppc_ITi_ITi,ppc_ITi_ITs,ppc_ITs_ITi,ppc_ITs_ITs]
f_range = (freqs>4) & (freqs<50)
t_range = (times>-0.1) & (times<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('jet')
for i in range(9):
    plt.subplot(3,3,i+1)
    image = np.nanmean(remove_outlier(np.nanmean(to_plot[i][:,:,f_range,:][:,:,:,t_range],axis=0)),axis=0)
    plt.pcolormesh(times[t_range],freqs[f_range],image)
    plt.clim(0,0.08)
    plt.colorbar()
    # plt.title(titles[i])


titles = ['V4_V4','V4_ITi','V4_ITs','ITi_V4','ITs_V4','ITi_ITi','ITi_ITs','ITs_ITi','ITs_ITs']
to_plot = [np.moveaxis(ppc_V4_V4[:,:,freqs<50,:],1,3) - ppc_V4_V4_shuffle,
           np.moveaxis(ppc_V4_ITi[:,:,freqs<50,:],1,3) - ppc_V4_ITi_shuffle,
           np.moveaxis(ppc_V4_ITs[:,:,freqs<50,:],1,3) - ppc_V4_ITs_shuffle,
           np.moveaxis(ppc_ITi_V4[:,:,freqs<50,:],1,3) - ppc_ITi_V4_shuffle,
           np.moveaxis(ppc_ITs_V4[:,:,freqs<50,:],1,3) - ppc_ITs_V4_shuffle,
           np.moveaxis(ppc_ITi_ITi[:,:,freqs<50,:],1,3) - ppc_ITi_ITi_shuffle,
           np.moveaxis(ppc_ITi_ITs[:,:,freqs<50,:],1,3) - ppc_ITi_ITs_shuffle,
           np.moveaxis(ppc_ITs_ITi[:,:,freqs<50,:],1,3) - ppc_ITs_ITi_shuffle,
           np.moveaxis(ppc_ITs_ITs[:,:,freqs<50,:],1,3) - ppc_ITs_ITs_shuffle]
f_range = (freqs_shuffle>4) & (freqs_shuffle<50)
t_range = (times_shuffle>-0.1) & (times_shuffle<0.4)
plt.figure(figsize=[12,8])
plt.set_cmap('jet')
for i in range(9):
    plt.subplot(3,3,i+1)
    image = np.nanmean(remove_outlier(np.nanmean(to_plot[i][:,f_range,:][:,:,t_range],axis=-1)),axis=0)
    plt.pcolormesh(times_shuffle[t_range],freqs_shuffle[f_range],image)
    # plt.title(titles[i])
    plt.clim(0, 0.01)
    plt.colorbar()