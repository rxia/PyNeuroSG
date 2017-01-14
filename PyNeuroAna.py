import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import spectral


def ComputeSpectrogram(data, data1=None, fs=1.0, t_ini=0.0, t_bin=20, t_step=None, t_axis=1):
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
    :return:         [spcg, spcg_t, spcg_f]
           spcg:     power spectogram, [ trials * timestamps * channels * frequencty]
           spcg_t:   timestamps of spectrogram
           spcg_t:   frequency ticks of spectrogram
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

    # compute spectrogram
    if data1 is None:
        [spcg_f, spcg_t, spcg] = signal.spectrogram(data, fs=fs, window=window, axis=t_axis,
                                                nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    else:
        [spcg_f, spcg_t, spcg] = signal.spectral._spectral_helper(data, data1, fs=fs, window=window, axis=t_axis,
                                                    nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    spcg_t = np.array(spcg_t) + t_ini
    spcg_f = np.array(spcg_f)

    return [spcg, spcg_t, spcg_f]


def ComputeCoherogram(data0, data1, fs=1.0, t_ini=0.0, t_bin=20, t_step=None, t_axis=1):
    """
    Compuate cohrence over sliding window
    :param data0:    LFP data, [ trials * timestamps]
    :param data1:    LFP data, [ trials * timestamps]
    :param fs:       sampling frequency
    :param t_ini:    the first timestamps
    :param t_bin:    during of time bin for fft, will be used to find the nearest power of two
    :param t_step:   step size for moving window, default to t_bin / 8
    :param t_axis:   the axis index of the time in data
    :return:         [cohg, spcg_t, spcg_f]
           cohg:     power spectogram, [ timestamps * frequencty]
           spcg_t:   timestamps of spectrogram
           spcg_t:   frequency ticks of spectrogram
    """
    # Pxx
    [spcg_xx, _, _] = ComputeSpectrogram(data0, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis)
    # Pyy
    [spcg_yy, _, _] = ComputeSpectrogram(data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis)
    # Pxy
    [spcg_xy, spcg_t, spcg_f] = ComputeSpectrogram(data0, data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis)

    # cohreence
    cohg = np.abs(np.mean(spcg_xy, axis=0))**2 / ( np.mean(spcg_xx, axis=0) * np.mean(spcg_yy, axis=0) )

    return [cohg, spcg_t, spcg_f]




def GetNearestPow2(n):
    """
    Get the nearest power of 2, for FFT
    :param n:  input number
    :return:   an int, power of 2 (e.g., 2,4,8,16,32...), nearest to n
    """
    return int(2**np.round(np.log2(n)))