import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import spectral


def ComputeSpectrogram(data, data1=None, fs=1.0, t_ini=0.0, t_bin=20, t_step=None, t_axis=1, batchsize=100):
    """
    Compuate power spectrogram in sliding windows

        if a single data is give, returns power spectrum density Pxx over sliding windows;
        if two data are given, returns cross spectrum Pxy over sliding windows

    :param data:     LFP data, [ trials * timestamps * channels]
                            the dimension does not matter, as long as the time axis is provided in t_axis;
                            the resulting spcg will add another dimension (frequency) to the end
    :param fs:       sampling frequency
    :param t_ini:    the first timestamps
    :param t_bin:    during of time bin for fft, will be used to find the nearest power of two
    :param t_step:   step size for moving window, default to t_bin / 8
    :param t_axis:   the axis index of the time in data
    :param batchsize: to prevent memory overloading problem (default to 100, make smaller if memory overload occurs)
    :return:         [spcg, spcg_t, spcg_f]
           spcg:     power spectogram, [ trials * timestamps * channels * frequencty]
           spcg_t:   timestamps of spectrogram
           spcg_f:   frequency ticks of spectrogram
    """

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
                                                        nperseg=nperseg, noverlap=noverlap, nfft=nfft)
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

    return [spcg, spcg_t, spcg_f]



def ComputeCoherogram(data0, data1, fs=1.0, t_ini=0.0, t_bin=20, t_step=None, t_axis=1, batchsize=100, data0_spcg=None, data1_spcg=None, data0_spcg_ave=None, data1_spcg_ave=None):
    """
    Compuate cohrence over sliding window

    :param data0:    LFP data, [ trials * timestamps]
    :param data1:    LFP data, [ trials * timestamps]
    :param fs:       sampling frequency
    :param t_ini:    the first timestamps
    :param t_bin:    during of time bin for fft, will be used to find the nearest power of two
    :param t_step:   step size for moving window, default to t_bin / 8
    :param t_axis:   the axis index of the time in data
    :param data0_spcg: the spcg_xx, if already calculated
    :param data1_spcg: the spcg_yy, if already calculated
    :return:         [cohg, spcg_t, spcg_f]
           cohg:     power spectogram, [ timestamps * frequencty]
           spcg_t:   timestamps of spectrogram
           spcg_t:   frequency ticks of spectrogram
    """
    # Pxx
    if data0_spcg_ave is None:
        if data0_spcg is None:
            [spcg_xx, _, _] = ComputeSpectrogram(data0, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis, batchsize=batchsize)
        else:
            spcg_xx = data0_spcg
        spcg_xx_ave = np.mean(spcg_xx, axis=0)
    else:
        spcg_xx_ave = data0_spcg_ave
    # Pyy
    if data1_spcg_ave is None:
        if data1_spcg is None:
            [spcg_yy, _, _] = ComputeSpectrogram(data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis, batchsize=batchsize)
        else:
            spcg_yy = data1_spcg
        spcg_yy_ave = np.mean(spcg_yy, axis=0)
    else:
        spcg_yy_ave = data1_spcg_ave
    # Pxy
    [spcg_xy, spcg_t, spcg_f] = ComputeSpectrogram(data0, data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis, batchsize=batchsize)

    # cohreence
    cohg = np.abs( np.abs(np.mean(spcg_xy, axis=0))**2 / (spcg_xx_ave  *  spcg_yy_ave) )

    return [cohg, spcg_t, spcg_f]






def GetNearestPow2(n):
    """
    Get the nearest power of 2, for FFT

    :param n:  input number
    :return:   an int, power of 2 (e.g., 2,4,8,16,32...), nearest to n
    """
    return int(2**np.round(np.log2(n)))