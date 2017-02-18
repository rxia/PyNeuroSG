import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import spectral



def ComputeSpectrogram(data, data1=None, fs=1000.0, t_ini=0.0, t_bin=None, t_step=None, t_axis=1, batchsize=100, f_lim=None):
    """
    Compuate power spectrogram in sliding windows

    if a single data is give, returns power spectrum density Pxx over sliding windows;
    if two data are given, returns cross spectrum Pxy over sliding windows

    :param data:     LFP data, [ trials * timestamps * channels]
                            the dimension does not matter, as long as the time axis is provided in t_axis;
                            the resulting spcg will add another dimension (frequency) to the end
    :param fs:       sampling frequency
    :param t_ini:    the first timestamps
    :param t_bin:    duration of time bin for fft, will be used to find the nearest power of two, default to total_time/10
    :param t_step:   step size for moving window, default to t_bin / 8
    :param t_axis:   the axis index of the time in data
    :param batchsize: to prevent memory overloading problem (default to 100, make smaller if memory overload occurs)
    :param f_lim:    frequency limit to keep, [f_min, f_max], default to None
    :return:         [spcg, spcg_t, spcg_f]

           * spcg:     power spectogram, [ trials * frequencty * channels * timestamps] or [ trials * frequencty * timestamps]
           * spcg_t:   timestamps of spectrogram
           * spcg_f:   frequency ticks of spectrogram
    """

    if t_bin is None:     # default to total_time/10
        t_bin = 1.0*data.shape[t_axis]/fs /10

    nperseg = GetNearestPow2( fs * t_bin )                    # number of points per segment, power of 2
    if t_step is None:                                        # number of overlapping points of neighboring segments
        noverlap = int( np.round(nperseg * 0.875) )                # default 7/8 overlapping
    else:
        noverlap = nperseg - GetNearestPow2( fs * t_step )
    if noverlap > nperseg:
        noverlap = nperseg

    nfft = nperseg * 4                        # number of points for fft, determines the frequency resolution

    window = signal.hann(nperseg)             # time window for fft

    """ compute spectrogram, use batches to prevent memory overload """
    if batchsize is None:
        if data1 is None:
            [spcg_f, spcg_t, spcg] = signal.spectrogram(data, fs=fs, window=window, axis=t_axis,
                                                    nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        else:
            [spcg_f, spcg_t, spcg] = signal.spectral._spectral_helper(data, data1, fs=fs, window=window, axis=t_axis,
                                                        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                                                      scaling='density', mode='psd')
    else:
        N_trial = data.shape[0]
        N_batch = N_trial // batchsize
        if N_batch == 0:
            N_batch = 1
        list_indx_in_batch = np.array_split( range(N_trial), N_batch)
        spcg = []
        for indx_in_batch in list_indx_in_batch:
            if data1 is None:
                [spcg_f, spcg_t, spcg_cur] = signal.spectrogram(np.take(data,indx_in_batch,axis=0),
                                                            fs=fs, window=window, axis=t_axis,
                                                        nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            else:
                [spcg_f, spcg_t, spcg_cur] = signal.spectral._spectral_helper(np.take(data,indx_in_batch,axis=0),
                                                                          np.take(data1,indx_in_batch,axis=0),
                                                                          fs=fs, window=window, axis=t_axis,
                                                                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                                                              scaling='density', mode='psd')
            if len(spcg) == 0:
                spcg = spcg_cur
            else:
                spcg = np.concatenate([spcg, spcg_cur], axis=0)
            del spcg_cur    # release memory

    spcg_t = np.array(spcg_t) + t_ini
    spcg_f = np.array(spcg_f)

    if f_lim is not None:
        indx_f_keep = np.flatnonzero( (spcg_f>=f_lim[0]) * (spcg_f<f_lim[1]) )
        spcg = np.take(spcg, indx_f_keep, axis=t_axis)
        spcg_f = spcg_f[indx_f_keep]

    return [spcg, spcg_t, spcg_f]


def ComputeCoherogram(data0, data1, fs=1000.0, t_ini=0.0, t_bin=None, t_step=None, f_lim=None, batchsize=100,
                      tf_phase=False, tf_shuffle=False, tf_vs_shuffle = False,
                      data0_spcg=None, data1_spcg=None, data0_spcg_ave=None, data1_spcg_ave=None):
    """
    Compuate LFP-LFP coherence over sliding window, takes two [ trials * timestamps] arrays, or one [ trials * timestamps * 2] arrays

    :param data0:    LFP data, [ trials * timestamps]; if data1 is None, data0 contains both signals [ trials * timestamps * 2]
    :param data1:    LFP data, [ trials * timestamps]
    :param fs:       sampling frequency
    :param t_ini:    the first timestamps
    :param t_bin:    duration of time bin for fft, will be used to find the nearest power of two
    :param t_step:   step size for moving window, default to t_bin / 8
    :param tf_phase: true/false keep phase, if true, returning value cohg is complex, whose abs represents coherence, and whose angle represents phase (negative if data1 lags data0)
    :param t_axis:   the axis index of the time in data
    :param data0_spcg: the spcg_xx, if already calculated
    :param data1_spcg: the spcg_yy, if already calculated
    :return:         [cohg, spcg_t, spcg_f]

           * cohg:     power spectogram, [ frequencty * timestamps ]
           * spcg_t:   timestamps of spectrogram
           * spcg_t:   frequency ticks of spectrogram
    """

    if data1 is None:    # the input could be data0 contains both signals, and data1 is None
        if data0.shape[2] == 2:
            data1 = data0[:, :, 1]
            data0 = data0[:, :, 0]

    if tf_vs_shuffle:
        tf_shuffle = True
        tf_phase = False

    # Pxx
    if data0_spcg_ave is None:
        if data0_spcg is None:
            [spcg_xx, _, _] = ComputeSpectrogram(data0, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, batchsize=batchsize, f_lim=f_lim)
        else:
            spcg_xx = data0_spcg
        spcg_xx_ave = np.mean(spcg_xx, axis=0)
    else:
        spcg_xx = np.array([])
        spcg_xx_ave = data0_spcg_ave
    # Pyy
    if data1_spcg_ave is None:
        if data1_spcg is None:
            [spcg_yy, _, _] = ComputeSpectrogram(data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, batchsize=batchsize, f_lim=f_lim)
        else:
            spcg_yy = data1_spcg
        spcg_yy_ave = np.mean(spcg_yy, axis=0)
    else:
        spcg_yy = np.array([])
        spcg_yy_ave = data1_spcg_ave
    # Pxy
    [spcg_xy, spcg_t, spcg_f] = ComputeSpectrogram(data0, data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, batchsize=batchsize, f_lim=f_lim)
    spcg_xy_ave = np.mean(spcg_xy, axis=0)

    # cohreence
    cohg = np.abs(spcg_xy_ave)**2 / (spcg_xx_ave  *  spcg_yy_ave)

    if tf_shuffle:
        data0_shuffle = data0[np.random.permutation(data0.shape[0]), :]
        [spcg_xy_shuffle, spcg_t, spcg_f] = ComputeSpectrogram(data0_shuffle, data1, fs=fs, t_ini=t_ini, t_bin=t_bin,
                                                       t_step=t_step, batchsize=batchsize, f_lim=f_lim)
        spcg_xy_ave_shuffle = np.mean(spcg_xy_shuffle, axis=0)
        cohg_shuffle = np.abs(spcg_xy_ave_shuffle)**2 / (spcg_xx_ave  *  spcg_yy_ave)

        if tf_vs_shuffle:
            cohg = cohg-cohg_shuffle
        else:
            cohg = cohg_shuffle

    # if keeps phase, returns complex values, whose abs represents coherence, and whose angle represents phase (negative if data1 lags data0)
    if tf_phase:
        cohg = cohg * np.exp(np.angle(spcg_xy_ave) *1j )

    return [cohg, spcg_t, spcg_f]


def ComputeSpkTrnFieldCoupling(data_LFP, data_spk, fs=1000, measure='PLV', t_ini=0.0, t_bin=20, t_step=None,
                               batchsize=100, tf_phase=True, tf_shuffle=False, tf_vs_shuffle = False, f_lim=None):
    """
    Compuate spk-LFP coherence over sliding window, takes two [ trials * timestamps] arrays, or one [ trials * timestamps * 2] arrays

    based on paper: Vinck, M., Battaglia, F. P., Womelsdorf, T., & Pennartz, C. (2012). Improved measures of phase-coupling between spikes and the Local Field Potential

    :param data_LFP: LFP data, [ trials * timestamps]; if data_spk is None, dataLFP contains both signals [ trials * timestamps * 2]
    :param data_spk: spike data, [ trials * timestamps]
    :param fs:       sampling frequency
    :param measure:  what measures to use for spk-field coupling, 'PLV' (phase lock value, 1st order) or 'PPC' (pairwise phase consisntency, 2nd order)
    :param t_ini:    the first timestamps
    :param t_bin:    duration of time bin for fft, will be used to find the nearest power of two
    :param t_step:   step size for moving window, default to t_bin / 8
    :param t_axis:   the axis index of the time in data
    :param batchsize:process a a subset of trials at a time, to prevent memory overload
    :param tf_phase: true/false keep phase, if true, returning value coupling_value is complex, whose abs represents coupling value, and whose angle represents phase (positive if spk leads LFP)
    :param f_lim:    frequency limit
    :return:         [cohg, spcg_t, spcg_f]

           * cohg:     power spectogram, [ frequencty * timestamps ]
           * spcg_t:   timestamps of spectrogram
           * spcg_t:   frequency ticks of spectrogram
    """

    if data_spk is None:    # the input could be data0 contains both signals, and data1 is None
        if data_LFP.shape[2] == 2:
            data_spk = data_LFP[:, :, 1]
            data_LFP = data_LFP[:, :, 0]

    data_spk = (data_spk >0) * 1.0
    N_trial = data_spk.shape[0]

    if tf_vs_shuffle:
        tf_shuffle = True
        tf_phase = False

    def ComputeCouplingValue(data_LFP):
        # Pxy
        [spcg_xy, spcg_t, spcg_f] = ComputeSpectrogram(data_LFP, data_spk, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                                       batchsize=batchsize, f_lim=f_lim)

        # mask for blank windows (time window without spikes)
        [spcg_yy, _, _] = ComputeSpectrogram(data_spk, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                                       batchsize=batchsize, f_lim=f_lim)
        mask_spk_exist = np.all(spcg_yy>0, axis=1, keepdims=True)

        # get unit vectors X (complex values), (abs=0 if the window does not contain any spikes)
        phi = np.angle(spcg_xy)
        X = np.exp(1j*phi) * mask_spk_exist

        # comupute spiketrian-field phase lock value (PLV) (Figure 2 of paper)
        PLV = np.sum(X, axis=0)/N_trial

        if measure == 'PLV':      # ----- spiketrian-field phase lock value (PLV) (Figure 2 of paper) -----
            coupling_value = PLV
            if tf_phase is False:
                coupling_value = np.abs(PLV)
        elif measure == 'PPC':    # ----- spiketrian-field pairwise phase consistancy (PPC) (equation 5.6 of paper) -----
            PPC_sum = 0
            X_real = X.real
            X_imag = X.imag
            for i in range(N_trial):     # dot product for every pair of trials
                j = range(i+1, N_trial)
                PPC_sum = PPC_sum + np.sum( X_real[[i],:,:]*X_real[j,:,:] + X_imag[[i],:,:]*X_imag[j,:,:], axis=0 )
            PPC = PPC_sum/(N_trial*(N_trial-1)/2)

            if tf_phase:
                PPC = PPC*np.exp(1j*np.angle(PLV))   # add phase component from PLV
            coupling_value = PPC
        else:
            coupling_value = PLV

        return [coupling_value, spcg_t, spcg_f]

    if tf_shuffle is False:
        [coupling_value, spcg_t, spcg_f] = ComputeCouplingValue(data_LFP)
    else:
        [coupling_value_shuffle, spcg_t, spcg_f] = ComputeCouplingValue(data_LFP[np.random.permutation(data_LFP.shape[0]), :])
        if tf_vs_shuffle is True:
            [coupling_value, spcg_t, spcg_f] = ComputeCouplingValue(data_LFP)
            coupling_value = coupling_value - coupling_value_shuffle
        else:
            coupling_value = coupling_value_shuffle

    return [coupling_value, spcg_t, spcg_f]


def GetNearestPow2(n):
    """
    Get the nearest power of 2, for FFT

    :param n:  input number
    :return:   an int, power of 2 (e.g., 2,4,8,16,32...), nearest to n
    """
    return int(2**np.round(np.log2(n)))


def ComputeCrossCorlg(data0, data1, fs=1000.0, t_ini=0.0, t_bin=None, t_step=None, t_axis=1):
    """
    Compute Cross-correlogram

    :return:
    """

    if t_bin is None:     # default to total_time/10
        t_bin = 1.0*data0.shape[t_axis]/fs /10

    nperseg = GetNearestPow2( fs * t_bin )                    # number of points per segment, power of 2
    if t_step is None:                                        # number of overlapping points of neighboring segments
        noverlap = int( np.round(nperseg * 0.875) )                # default 7/8 overlapping
    else:
        noverlap = nperseg - GetNearestPow2( fs * t_step )
    if noverlap > nperseg:
        noverlap = nperseg

    data0_str = create_strided_array(data0, nperseg, noverlap)
    data1_str = create_strided_array(data1, nperseg, noverlap)

    t_grid = 1.0*np.arange(data0_str.shape[1])*(nperseg-noverlap)/fs + 0.5*nperseg/fs
    t_segm = 1.0*(np.arange(nperseg)-nperseg/2) /fs

    corss_corlg = data1_str*0
    for i in range(corss_corlg.shape[0]):
        for j in range(corss_corlg.shape[1]):
            corss_corlg[i,j,:] = sp.signal.correlate(data0_str[i,j,:], data1_str[i,j,:], mode='same')

    return (np.mean(corss_corlg, axis=0), t_grid, t_segm)

def create_strided_array(x, nperseg, noverlap):
    # Created strided array of data segments, from scipy.spectral._fft_helper
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)
    return result
