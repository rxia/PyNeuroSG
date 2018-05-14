import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import spectral
import scipy.ndimage.filters as spfltr
import sklearn
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.gaussian_process as gaussian_process
import sklearn.model_selection as model_selection
import scipy.optimize as optimize
from robust_csd import cal_robust_csd, cal_1dCSD, lfp_robust_smooth




""" ===== basic operation: smooth and average ===== """

def SmoothTrace(data, sk_std=None, fs=1.0, ts=None, axis=1):
    """
    smooth data using a gaussian kernel

    :param data:    a N dimensional array, default to [num_trials * num_timestamps * num_channels]
    :param sk_std:  smooth kernel (sk) standard deviation (std), default to None, do nothing
    :param fs:      sampling frequency, default to 1 Hz
    :param ts:      timestamps, an array, which can overwrite fs;   len(ts)==data.shape[axis] should hold,
    :param axis:    a axis of data along which data will be smoothed
    :return:        smoothed data of the same size
    """

    if ts is None:     # use fs to determine ts
        ts = np.arange(0, data.shape[axis])*(1.0/fs)
    else:              # use ts to determine fs
        fs = 1.0/np.mean(np.diff(ts))

    if sk_std is not None:  # condition for using smoothness kernel
        ts_interval = 1.0/fs  # get sampling interval
        kernel_std = sk_std / ts_interval  # std in frames
        kernel_len = int(np.ceil(kernel_std) * 3 * 2 + 1)  # num of frames, 3*std on each side, an odd number
        smooth_kernel = sp.signal.gaussian(kernel_len, kernel_std)
        smooth_kernel = smooth_kernel / smooth_kernel.sum()  # normalized smooth kernel

        # convolution using fftconvolve(), whihc is faster than convolve()
        data_smooth = sp.ndimage.convolve1d(data, smooth_kernel, mode='reflect', axis=axis)
    else:
        data_smooth = data

    return data_smooth



def AveOverTime(data, t_range=None, ts=None, t_axis=1, tf_count=False):
    """
    smooth data using a gaussian kernel

    :param data:    a N dimensional array, default to [num_trials * num_timestamps * num_channels]
    :param t_range: range of time to compute average on, e.g. [0.050, 0.400]
    :param ts:      timestamps, an array, which can overwrite fs;   len(ts)==data.shape[axis] should hold,
    :param t_axis:  axis of data along which data will be averaged
    :param tf_count:True/False output spike count instead of firing rate
    :return:        smoothed data of the same size
    """

    if ts is None:
        ts = np.arange(data.shape[t_axis])

    if t_range is None:
        t_range = ts[[0, -1]]

    # get sampling rate
    fs = 1.0/np.mean(np.diff(ts))

    indx_in_range = np.flatnonzero( (ts >= t_range[0]) * (ts < t_range[1]) )
    data_in_range = np.take(data, indx_in_range, axis=t_axis)
    data_ave = np.mean(data_in_range, axis=t_axis)

    if tf_count:
        data_ave = np.round(data_ave/fs*len(indx_in_range))

    return data_ave


def GroupWithoutAve(data_neuro, data=None):
    """
    for data_neuro object, get average response using the groupby information

    :param data_neuro: data neuro object after signal_align.neuro_sort() function, contains 'cdtn' and 'cdtn_indx' which directs the grouping rule
    :param data:       data, if not give, set to data_neuro['data].  It's zero-th dimension correspond to the index of data_neuro['cdtn_indx]
    :return:           data_groupave, e.g. array with the size of [num_cdtn * num_ts* num_chan]
    """

    if data is None:
        data = data_neuro['data']
    data_group = []
    for i, cdtn in enumerate(data_neuro['cdtn']):
        # ave = np.mean(np.take(data, data_neuro['cdtn_indx'][cdtn], axis=0), axis=0)
        data_group.append(data[data_neuro['cdtn_indx'][cdtn], :])
    return data_group


def GroupStat(data_neuro, data=None, axis=0, statfun='mean', **statfunargs):
    """
    for data_neuro object, get average response using the groupby information

    :param data_neuro: data neuro object after signal_align.neuro_sort() function, contains 'cdtn' and 'cdtn_indx' which directs the grouping rule
    :param data:       data, if not give, set to data_neuro['data].  It's zero-th dimension correspond to the index of data_neuro['cdtn_indx]
    :param statfun:    stat function to apply along axis, either strings in ['mean', 'std', 'median'] or function handels like np.min, note the function must contain axis argument
    :param statfunargs: any argument for statfun, e.g. q=20 for np.percentile
    :return:           data_groupstat, e.g. array with the size of [num_cdtn, :], all other dimentions are the same with data
    """

    assert 'cdtn' in data_neuro and 'cdtn_indx' in data_neuro
    if data is None:
        data = data_neuro['data']
    data_grpstat_shape = list(data.shape)
    data_grpstat_shape.pop(axis)
    data_grpstat_shape = [len(data_neuro['cdtn'])] + data_grpstat_shape
    data_groupstat = np.zeros(data_grpstat_shape)
    if statfun == 'mean':
        statfun = np.mean
    elif statfun == 'std':
        statfun = np.std
    elif statfun == 'median':
        statfun = np.median

    assert hasattr(statfun, '__call__')
    statfun_final = lambda x: statfun(x, axis=axis, **statfunargs)

    for i, cdtn in enumerate(data_neuro['cdtn']):
        data_groupstat[i, :] = statfun_final(data[data_neuro['cdtn_indx'][cdtn], :])

    return data_groupstat


def GroupAve(data_neuro, data=None, axis=0, return_std=False):
    """
    for data_neuro object, get average response using the groupby information

    :param data_neuro: data neuro object after signal_align.neuro_sort() function, contains 'cdtn' and 'cdtn_indx' which directs the grouping rule
    :param data:       data, if not give, set to data_neuro['data].  It's zero-th dimension correspond to the index of data_neuro['cdtn_indx]
    :return:           data_groupave, e.g. array with the size of [num_cdtn * num_ts* num_chan]
    """

    if data is None:
        data = data_neuro['data']
    data_grpave_shape = list(data.shape)
    data_grpave_shape[0] = len(data_neuro['cdtn'])
    data_groupave = np.zeros(data_grpave_shape)
    data_groupstd = np.zeros(data_grpave_shape)
    for i, cdtn in enumerate(data_neuro['cdtn']):
        data_groupave[i, :] = np.mean(data[data_neuro['cdtn_indx'][cdtn], :], axis=axis)
        data_groupstd[i, :] =  np.std(data[data_neuro['cdtn_indx'][cdtn], :], axis=axis)
    if return_std:
        return data_groupave, data_groupstd
    else:
        return data_groupave



def TuningCurve(data, label, type='', ts=None, t_window=None, limit=None, stat_t='mean', stat_trials='mean'):
    """
    Calculate the tuning curve of a neuron's response

    :param data:     2D array, [ num_trials * num_ts ]
    :param label:    1D array or list, [ num_trials ]
    :param type:     type of tuning curve, one of the following: 'rank'
    :param ts:       1D array of time stamps, [ num_ts ], default to 1 Hz sampling rate start from 0
    :param t_window: list, [ t_start, t_end ], if not give, use the full range
    :param limit:    1D array, boolean or index array, instructing whether to use a subset of trials
    :return:         [condition, activity], they both are 1D arrays, of the same length,
    """

    label = np.array(label)
    if limit is not None:
        data  = data[limit, :]
        label = label[limit]
    if ts is None:
        ts = np.arange(0, data.shape[1])*1.0
    if t_window is None:
        t_window = ts[[0,-1]]

    def InRange(x, x_range):
        return np.logical_and(x>=x_range[0], x<=x_range[1])

    # get the response in the t_window
    if stat_t == 'mean':
        response = np.mean(data[:, InRange(ts, t_window) ], axis=1)
    elif stat_t == 'median':
        response = np.median(data[:, InRange(ts, t_window)], axis=1)
    elif stat_t == 'sum':
        response = np.sum(data[:, InRange(ts, t_window)], axis=1)
    elif stat_t == 'std':
        response = np.std(data[:, InRange(ts, t_window)], axis=1)
    else:   # stat_t is a function handle
        response = stat_t(data[:, InRange(ts, t_window)], axis=1)


    # get the response over trials of the same condition
    if stat_trials == 'mean':
        stat_trials = np.mean
    elif stat_trials == 'median':
        stat_trials = np.median
    elif stat_trials == 'sum':
        stat_trials = np.sum
    elif stat_trials == 'std':
        stat_trials = np.std
    else:
        pass

    x = np.unique(label)
    y = np.zeros(len(x))
    for i, xx in enumerate(x):
        y[i] = stat_trials(response[label==xx])

    if type == 'rank':   # sort in descending order
        i_sort = np.argsort(y)[::-1]
        x = x[i_sort]
        y = y[i_sort]
    elif type == '':   # sort in descending order
        i_sort = np.argsort(x)
        x = x[i_sort]
        y = y[i_sort]

    return (x, y)



def cal_ISI(X, ts=None, bin=None):
    """
    Calculate inter-spike invervals
    :param X:    2D array of spike data (binary values), [ num_trials * num_ts ]
    :param ts:   timestamps of X (1D array) or sampling interval of ts (a scalar)
    :param bin:  bin centers for histogram
    :return:     (ISI, ISI_hist, bin). ISI: 1D array of all ISIs, ISI_hist: hist of ISIs, bin: bins of ISI_hist
    """

    X = X>0
    ISI = np.diff(np.flatnonzero(X))
    if ts is not None:
        if np.isscalar(ts):
            t_interval = ts
        else:
            t_interval= (ts[-1]-ts[0])/(len(ts)-1)
    else:
        t_interval= 1.0

    ISI = ISI* t_interval

    if bin is None:
        bin = np.arange(0,100)*t_interval

    (ISI_hist, _) = np.histogram(ISI, bins=center2edge(bin))

    return (ISI, ISI_hist, bin)



def cal_STA(X, Xt=None, ts=None, t_window=None, zero_point_zero = False):
    """
        Calculate spike triggered average
        :param X:    2D array of data (spike or LFP), [ num_trials * num_ts ]
        :param Xt:   2D array of spike data used as triggers (binary data), [ num_trials * num_ts ]
        :param ts:   timestamps for X
        :param t_window:  window for STA
        :return:     (sta, t_sta, st), sta: 1D array of STA; t_sta: timestamps; st: spike-triggered segments, 2D array [N_spikes, N_ts]
    """
    if Xt is None:
        Xt = X

    if ts is not None:
        if np.isscalar(ts):
            t_interval = ts
        else:
            t_interval= (ts[-1]-ts[0])/(len(ts)-1)
    else:
        t_interval= 1.0

    if t_window is None:
        indx_sta = np.arange(-100, 100+1)
    else:
        indx_sta = np.arange(int(t_window[0]/t_interval), int(t_window[1]/t_interval)+1)
    t_sta = indx_sta * t_interval

    X_flat = np.ravel(X)            # data in flat form
    Xt = Xt>0
    indx_trg = np.flatnonzero(Xt)   # indexes of triggers in flat form

    M = len(indx_trg)
    T = len(indx_sta)
    indx_trg_sta =  np.expand_dims(indx_sta, axis=0) + np.expand_dims(indx_trg, axis=1)

    # spike triggered segments, 2D array [N_spikes, N_ts]
    st = np.take(X_flat, indx_trg_sta, mode='wrap')
    # spike triggered average, 1D array [N_ts]
    sta = np.mean(st, axis=0)

    if zero_point_zero:
        sta[t_sta==0] = 0

    return (sta, t_sta, st)


""" ========== CSD estimation related ========== """
# moved to file robutst_csd.py


""" ===== decoding related ===== """


def decode_over_time(data, label, limit_tr=None, limit_ch=None, ts=None, ts_win_train=None):
    """ decoding realted, temporary """
    data = np.array(data)
    label = np.array(label)
    [N_tr, N_ts, N_ch] = data.shape

    """ normalized for every channels, so that every channel has mean=0, std=1 """
    mean_ch = np.mean(data, axis=(0, 1), keepdims=True)
    std_ch  = np.std (data, axis=(0, 1), keepdims=True) + 10**(-10)
    data_nmlz = (data - mean_ch) / std_ch

    """ select a subset of trials and channels """
    if limit_tr is None:
        limit_tr = range(N_tr)
    elif np.array(limit_tr).dtype == bool:
        limit_tr = np.flatnonzero(limit_tr)

    if limit_ch is None:
        limit_ch = range(N_ch)
    elif np.array(limit_ch).dtype == bool:
        limit_ch = np.flatnonzero(limit_ch)

    X = data_nmlz[limit_tr, :, :]
    X = X[:,:,limit_ch]
    y = label[limit_tr]

    """ classification model """
    clf = linear_model.LogisticRegression(solver='lbfgs', warm_start=True, multi_class='multinomial', fit_intercept=False)

    """ classification over every time point """
    clf_score = np.zeros(N_ts)
    clf_score_std = np.zeros(N_ts)
    if ts_win_train is None:
        for t in range(N_ts):
            cfl_scores = model_selection.cross_val_score(clf, X[:, t, :], y, cv=5)
            clf_score[t] = np.mean(cfl_scores)
            clf_score_std[t] = np.std(cfl_scores)
    else:
        pass
    return clf_score




""" ===== spike point process analysis related ===== """


def gen_gamma_knl(ts=np.arange(-0.1, 1.0, 0.001), k=1.0, theta=1.0, mu=None, sigma=None, normalize='max'):
    """
    tool function generate a gamma-distribution-like kernel (1D), used as a impulse response function

    could be parametrized using k and theta like Gamma distribution, or use mean and sigma like gaussian distribution.
    Note that mu=k*theta, sigma=sqrt(k*theta**2)

    e.g.  bump = gen_gamma_bump(ts, k=2, theta=0.20) is equivalent as bump = gen_gamma_bump(ts, mu=0.4, theta=0.08);

    :param ts:    timestamps, e.g. ts = np.arange(-0.1, 1.0, 0.001);
    :param k:     if parametrized as gamma: shape parameter, if 1, exp distribution, if large, becomes gaussian-like
    :param theta: if parametrized as gamma: scale parameter, mean = k*theta, var = k*theta**2
    :param mu:    alternative parametrization: mean is mu
    :param std:   alternative parametrization: var is std**2
    :return:      a gamma-like bump
    """
    if mu is not None and sigma is not None:
        theta = 1.0*sigma**2/mu
        k     = 1.0*mu**2/sigma**2
    ts_abs = np.abs(ts)
    bump = ts_abs**(k-1) * np.exp(-ts_abs/theta)
    bump = bump * (ts > 0)
    if normalize == 'max':
        bump = bump / np.max(bump)
    elif normalize == 'sum':
        bump = bump / np.sum(bump)
    return bump



def gen_knl_series(ts=np.arange(-0.1, 1.0, 0.001), scale=0.5, N=5, spacing_factor=np.sqrt(2), tf_symmetry=True):
    """
    tool function to generate a series of N gamma-like kernels distributed within the range defined by scale,
    used as the basis functions for the internal/external history term for neural point process
    test: plt.plot(pna.gen_bump_series().transpose())

    :param ts:    timestamps, e.g. ts = np.arange(-0.1, 1.0, 0.001);
    :param scale: scale of the bumps, which defines the mean of the broadest gamma bump
    :param N:     number of bumps
    :param spacing_factor: any number >=1, default to sqrt(2), if small, evenly spaces, if large, un-evenly spaced
    :param tf_symmetry: True or False, make ts symmetric about zero
    :return:      2D array of bump series, [N_bumps, N_ts]
    """

    if tf_symmetry:    # make sure that ts is symmetric about zero, convenient for convolution
        t_interval = ts[1]-ts[0]
        t_max = np.max(np.abs(ts))
        h_len_knl = int(t_max/t_interval)
        ts = np.arange(-h_len_knl, h_len_knl+1)*t_interval

    # k of gamma is exponent with base=spacing_factor
    list_k = spacing_factor ** (np.arange(N)+1)
    # theta of gamma makes the mean of the largest gamma equal to scale
    theta = scale/list_k[-1]
    # generate gamma series
    knl_series = np.array([gen_gamma_knl(ts, k=k, theta=theta) for k in list_k])
    return knl_series



def gen_delta_function_with_label(ts=np.arange(-0.2, 1.0, 0.01), t=np.zeros(1), y=np.ones(1),
                       tf_y_ctgr=False, tf_return_ctgr=False):
    """
    generate delta function at time t and with label y for every trial on time grid ts

    :param ts:   timestamps, grid of time, of length T
    :param t:    time of event onset of every trial, of length N, if the same across all trials, could be given as a scalar
    :param y:    labels of events, of length N: if tf_return_ctgr==True, could be any type; otherwise, must be numbers.
                    if the same across trials, could be given as a scalar
    :param tf_y_ctgr:      True/False of y being categorical
    :param tf_return_ctgr: True/False to return y cetegory lables
    :return:     delta_fun or (delta_fun, y_ctgy)

                    delta_fun if tf_return_ctgr==False, (delta_fun, y_ctgy) otherwise

                    detta_fun is [N*T] if tf_y_ctgr==False, [N*T*M] if tf
    """

    """  pre-process t and y, make sure that they are 1D array of lenth N  """
    t = np.array(t, ndmin=1)
    y = np.array(y, ndmin=1)
    if len(t) == 1:         # if t is scalar, repeat N times
        t = np.repeat(t, len(y))
    if len(y) == 1:         # if y is scalar, repeat N times
        y = np.repeat(y, len(t))
    if len(t) != len(y):    # if t and y are of different lengths
        raise('input t and y should be of the same length')


    """ produce Y: if y is continuous, make it [N*1]; if y is categorical (M categories), make it [N*M] binary values """
    if tf_y_ctgr:
        label_enc = preprocessing.LabelEncoder()
        enc = preprocessing.OneHotEncoder(sparse=False)
        y_code = label_enc.fit_transform(y)
        Y = np.array(enc.fit_transform(np.expand_dims( y_code , axis=1)))
        Y_label = label_enc.classes_
    else:
        Y = np.expand_dims(y, axis=1)
        Y_label = []
    T = len(ts)
    N, M = Y.shape


    """ generate gamma functions """
    delta_fun = np.zeros([N, T])
    i_t = [np.abs((ts-t_one)).argmin() for t_one in t]
    delta_fun[range(N), i_t]=1

    if tf_y_ctgr:
        delta_fun = np.dstack([delta_fun * Y[:, m:m+1] for m in range(M)])
    else:
        delta_fun *= Y

    if tf_return_ctgr:
        return delta_fun, Y_label
    else:
        return delta_fun


def fit_neural_point_process(Y, Xs, Xs_knls):
    """
    fit neural point process, based on Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005) A point process framework for relating neural spiking activity to spiking history, neural ensemble and extrinsic covariate effects. J Neurophysiology, 93, 1074-1089.

    :param Y:       Y to be predicted, binary, 2D of shape [N, T], N trials, T timestamps
    :param Xs:      Xs, used to predict Y, list of 2D arrays, where every array is of the same shape as Y
    :param Xs_knls: kernels for Xs, a list, where Xs_kernel[i] is a 2D array correspond to Xs[i],
                        Xs_kernel[i] is of shape [num_kernels, num_ts_for_kernel], every kernel works as the inpulse-response function, centered at zero
    :return:        regression object
    """

    """ make sure Y is binary; Xs, Xs_knls are lists """
    Y = np.array(Y>0)
    if type(Xs) is not (list or tuple):
        Xs = [Xs]
    if type(Xs_knls) is not (list or tuple):
        Xs_knls = [Xs_knls]

    """ X_base for fitting: [N,T,M], N trials, T time stamps, M total features  """
    N, T = Y.shape
    Ms= [len(history_term) for history_term in Xs_knls]
    M = np.sum(np.array(Ms))
    X_base = np.zeros([N, T, M])

    """ construct X_base, X_base[:,:,i] is the convolution of X and one kernel from the kernels for X """
    m = 0
    for i, X in enumerate(Xs):                 # for every X
        for j, knl in enumerate(Xs_knls[i]):   # for every kernel
            X_base[:,:,m] = spfltr.convolve1d(X, knl, axis=1)   # convolve along time axis
            m += 1

    """ GML regression, here logistic regression for binary Y """
    reg_X = np.reshape(X_base, [N * T, M])   # re-organized for regression, shape 2D: [N*T, M]
    reg_y = np.reshape(Y, [N * T])           # re-organized for regression, shape 1D: [N*T]

    reg = linear_model.LogisticRegression()  # regression model using sklearn
    reg.fit(reg_X, reg_y)                    # fit

    print(reg.score(reg_X, reg_y))

    return reg



""" ===== spectral analysis ===== """

def ComputeWelchSpectrum(data, data1=None, fs=1000.0, t_ini=0.0, t_window=None, t_bin=None, t_step=None, t_axis=1, batchsize=100, f_lim=None):

    ts = t_ini + 1.0/fs*np.arange(data.shape[t_axis])
    if t_window is not None:      # get the data of interest
        data = np.take(data, np.flatnonzero(np.logical_and(ts >= t_window[0], ts <= t_window[1])), axis=t_axis)
        if data1 is not None:
            data1 =np.take(data1, np.flatnonzero(np.logical_and(ts >= t_window[0], ts <= t_window[1])), axis=t_axis)
        t_ini = ts[ts>=t_window[0]][0]

    [spcg, spcg_t, spcg_f] = ComputeSpectrogram(data, data1=data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis,
                       batchsize=batchsize, f_lim=f_lim)

    spct = np.mean(spcg, axis=-1)
    return [spct, spcg_f]

def ComputeWelchCoherence(data, data1=None, fs=1000.0, t_ini=0.0, t_window=None, t_bin=None, t_step=None, t_axis=1, batchsize=100, f_lim=None):

    ts = t_ini + 1.0/fs*np.arange(data.shape[t_axis])
    if t_window is not None:      # get the data of interest
        data = np.take(data, np.flatnonzero(np.logical_and(ts >= t_window[0], ts <= t_window[1])), axis=t_axis)
        if data1 is not None:
            data1 =np.take(data1, np.flatnonzero(np.logical_and(ts >= t_window[0], ts <= t_window[1])), axis=t_axis)
        t_ini = ts[ts>=t_window[0]][0]

    [cohg, spcg_t, spcg_f] = ComputeCoherogram(data, data1=data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                       batchsize=batchsize, f_lim=f_lim)

    cohe = np.mean(cohg, axis=-1)
    return [cohe, spcg_f]


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
                      data0_spcg=None, data1_spcg=None, data0_spcg_ave=None, data1_spcg_ave=None, data01_spcg=None):
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

    # coherence
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




def ComputeSpcgMultiPair(data, ch_list0, ch_list1, fs=1000.0, t_ini=0.0, t_bin=None, t_step=None, f_lim=None, batchsize=100,
                      tf_shuffle=False, tf_vs_shuffle = False, tf_verbose = False):
    """
    Compute LFP spectrograms and cross-specgrograms over sliding window for all given pairs of channels,
    returns a dictionary containing all related info,
    This funciton calls the
    useful for computing coherence flexibly later using function ComputeCohgFromIntermediate()
    :param data:         LFP arrays of all channels, of shape [N_trials, N_ts]
    :param ch_list0:
    :param ch_list1:
    :param fs:
    :param t_ini:
    :param t_bin:
    :param t_step:
    :param f_lim:
    :param batchsize:
    :param tf_shuffle:
    :param tf_vs_shuffle:
    :param tf_verbose:
    :return:
    """

    spcg_xx = {}
    spcg_yy = {}
    spcg_xy = {}

    if tf_verbose:
        print('compute xx')
    for ch0 in ch_list0:
        if tf_verbose:
            print(ch0)
        data0 = data[:,:,ch0]
        [spcg_xx_cur, _, _] = ComputeSpectrogram(data0, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                                 batchsize=batchsize, f_lim=f_lim)
        spcg_xx[ch0] = spcg_xx_cur
    if tf_verbose:
        print('compute yy')
    for ch1 in ch_list1:
        if tf_verbose:
            print(ch1)
        data1 = data[:,:,ch1]
        [spcg_yy_cur, _, _] = ComputeSpectrogram(data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                                 batchsize=batchsize, f_lim=f_lim)
        spcg_yy[ch1] = spcg_yy_cur
    if tf_verbose:
        print('compute xy')
    for ch0 in ch_list0:
        data0 = data[:, :, ch0]
        for ch1 in ch_list1:
            if tf_verbose:
                print(ch0, ch1)
            data1 = data[:, :, ch1]
            [spcg_xy_cur, spcg_t, spcg_f] = ComputeSpectrogram(data0, data1, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                                     batchsize=batchsize, f_lim=f_lim)
            spcg_xy[(ch0, ch1)] = spcg_xy_cur
    spcg_multipair = {}
    spcg_multipair['ch_list0'] = ch_list0
    spcg_multipair['ch_list1'] = ch_list1
    spcg_multipair['spcg_xx'] = spcg_xx
    spcg_multipair['spcg_yy'] = spcg_yy
    spcg_multipair['spcg_xy'] = spcg_xy
    spcg_multipair['spcg_t'] = spcg_t
    spcg_multipair['spcg_f'] = spcg_f

    return spcg_multipair

def ComputeCohgFromIntermediate(spcg_multipair, limit_trial=None, tf_phase=False):

    if limit_trial is None:
        limit_trial = range(spcg_multipair['spcg_xx'][spcg_multipair['ch_list0'][0]].shape[0])

    spcg_multipair['cohg']={}

    for ch0 in spcg_multipair['ch_list0']:
        spcg_xx_ave = np.mean(spcg_multipair['spcg_xx'][ch0][limit_trial, :, :], axis=0)
        for ch1 in spcg_multipair['ch_list1']:
            spcg_yy_ave = np.mean(spcg_multipair['spcg_yy'][ch1][limit_trial, :, :], axis=0)
            spcg_xy_ave = np.mean(spcg_multipair['spcg_xy'][(ch0,ch1)][limit_trial,:,:], axis=0)
            # coherence
            cohg_cur = np.abs(spcg_xy_ave) ** 2 / (spcg_xx_ave * spcg_yy_ave)
            if tf_phase:
                cohg_cur = cohg_cur * np.exp(np.angle(spcg_xy_ave) * 1j)
            spcg_multipair['cohg'][(ch0,ch1)] = cohg_cur

    return spcg_multipair



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




"""  ===== simple statistics ===== """

def ErrIntvBinom(k=None, n=None, alpha=0.05, x=None, tf_err=True):
    """
    compute the confidence interval / error interval to estimate the probability p of binomial distribution, using grid search

    :param k:      int, number of samples==1
    :param n:      int, total number of samples
    :param alpha:  default to 0.05, confidence level
    :param x:      data array, used if k and n are not given:  k=sum(x==1), n=len(x)
    :param tf_err: True/False to return a) err interval (confidence_interval - p) or b) confidence interval
    :return:       (error_low, error_hight) or (p_low, p_high)
    """

    """ if n, k are not given, use x to get n and k """
    if (n is None) and (k is None):
        x = (np.array(x)>0.5).astype(float).ravel()
        n = len(x)
        k = np.sum(x)
    if n==0:   # if empty
        return np.array([np.nan, np.nan])
    """ best estimate of p """
    p = 1.0*k/n
    """ use grid search to compute confidence interval """
    p_grid = np.linspace(0,1,1001,endpoint=True)                # grid of p
    P_l = sp.stats.binom.cdf(k=k, n=n, p=p_grid)                # P(k or less success out of n trials)
    P_r = 1-sp.stats.binom.cdf(k=k-1, n=n, p=p_grid)            # P(k or more success out of n trials)
    intv_l = np.append(p_grid[P_r <= alpha*0.5], 0.0) .max()    # lower bound of p est: k or k+ success very rare
    intv_h = np.append(p_grid[P_l <= alpha*0.5], 1.0) .min()    # upper bound of p est: k or k- success very rare
    intv = np.array([intv_l, intv_h])                           # confidence interval of p estimate
    if tf_err:                                              # if returns error inverval
        return intv - p                                         # error interval of p estimate
    else:
        return intv                                             # confidence interval of p estimate


def cal_CohenD(data0, data1=None, axis=None, type_test='auto'):
    """
    calculate Cohen's d statistic for measuring effect size, as a complimentary test for t-test

    :param data0: a group of data
    :param data1: another group of data
    :param axis:  along which axis to compute d statistic
    :param type_test:  'auto', 'one-sample', 'independent', or 'paired'

         1) 'auto': determine based on the data input:
         use 'one-sample' if only data0 is give;
         use 'paired' if two data are of the same size; and
         use 'independent' if two data inputs are of different sizes

         2) 'one-sample': compute one-sample d stat using only data0

         3) 'independent': data0 and data1 are not directly paired, e.g., two groups of people under different treatment

         4) 'paired': data0 and data1 are directly paired, e.g., the same groups of people before and after treatment

    :return: d statistic
    """

    # determine test type automatically
    if type_test=='auto':
        if data1 is None:
            type_test = 'one-sample'
        elif data1.shape == data0.shape:
            type_test = 'paired'
        else:
            type_test = 'independent'

    if type_test == 'paired':
        data0 = data1-data0
    if type_test == 'paired' or type_test == 'one-sample':
        d = np.mean(data0, axis=axis) / np.std(data0, axis=axis)
    else:
        if axis is None:
            n0 = data0.size
            n1 = data1.size
        else:
            n0 = data0.shape[axis]
            n1 = data1.shape[axis]
        pooled_std = np.sqrt(( (n0-1)*np.std(data0, axis=axis)**2 + (n1-1)*np.std(data1, axis=axis)**2 )/(n0+n1-2))
        d = (np.mean(data1, axis=axis) - np.mean(data0, axis=axis))/pooled_std
    return d



""" ===== machine learning related ===== """

def LowDimEmbedding(data, type='PCA', para=None):
    """
    Low dimensional embedding of the data, using PCA or manifold leaning method

    :param data: N*M array, N data points of M dimensions
    :param type: embedding method
    :return:     N*2 array, 2D representation of all N data points
    """
    data = data/np.std(data)
    if type =='PCA':
        para = 2 if (para is None) else para
        model_embedding =  decomposition.PCA(n_components=para)
    elif type =='ICA':
        model_embedding =  decomposition.FastICA(n_components=2)
    elif type == 'Isomap':
        para = None if (para is None) else para
        model_embedding = manifold.Isomap(n_components=2, n_neighbors=para)
    elif type == 'MDS':
        model_embedding = manifold.MDS(n_components=2)
    elif type == 'TSNE':
        para = None if (para is None) else para
        model_embedding = manifold.TSNE(n_components=2, perplexity=para)
    elif type == 'SpectralEmbedding':
        model_embedding = manifold.SpectralEmbedding(n_components=2)
    else:
        model_embedding = decomposition.PCA(n_components=2)
    result_embedding = model_embedding.fit_transform(data)
    if result_embedding.shape[1]>=2:
        result_embedding = result_embedding[:, result_embedding.shape[1]-2:]
    return result_embedding


def DimRedLDA(X=None, Y=None, X_test=None, dim=2, lda=None, return_model=False):
    """
    supervised dimensionality reduction using LDA

    :param X:       data, [N*M], required if lda is not given
    :param Y:       labels, N,   required if lda is not given
    :param X_test:  testing data [N'*M], default to
    :param dim:     output dimension
    :param lda:     model object, if given. the function does not need X, Y to train the model
    :param return_model: if true returns the trained model, otherwise, returns X_test in low D, [N'*dim]
    :return:        the model (object) or the testing data in low D  [N'*dim]
    """
    if X_test is None:
        X_test = X
    if lda is None:
        lda = LinearDiscriminantAnalysis(n_components=dim)
        lda.fit(X, Y)
    X_2D = lda.transform(X_test)

    if return_model:
        return lda
    else:
        return X_2D






""" ========== Tool functons ========== """

def center2edge(centers):
    """
    tool function to get edges from centers for histogram. e.g. [0,1,2] returns [-0.5, 0.5, 1.5, 2.5]

    :param centers: centers (evenly spaced), 1D array of length N
    :return:        edges, 1D array of lenth N+1
    """

    centers = np.array(centers,dtype='float')
    edges = np.zeros(len(centers)+1)

    if len(centers) is 1:
        dx = 1.0
    else:
        dx = centers[-1]-centers[-2]
    edges[0:-1] = centers - dx/2
    edges[-1] = centers[-1]+dx/2
    return edges


def index_bool2int(index_bool):
    """
    tool function to transform bool index to int. e.g. turn [True, False, True] into [0, 2]

    :param index_bool: bool index, like [True, False, True]
    :return:           int index, like [0, 2]
    """
    return np.where(index_bool)


def index_int2bool(index_int, N=None):
    """
    tool function to transform int index to bool. e.g. turn into [0, 2] into [True, False, True]

    :param index_int: int index, like [0, 2]
    :param N:         length of boolean array
    :return:          bool index, like [True, False, True]
    """
    if N is None:
        N=np.max(index_int)+1
    index_bool = np.zeros(N)
    index_bool[index_int]=1
    return index_bool>0.5


def group_shuffle(indx_grps, n=None):
    """
    controlled shuffle within index groups
    :param indx_grps: list of indexes, eg. [[0,1,2,3], [4,5,6,7]], will shuffle within [0,1,2,3], and within [4,5,6,7], but not between them
    :param n:         number of elements to shufflt, defaul to None, s
    :return:          shuffled index, e.g. [0,3,1,2,5,4,7,6]
    """
    if n is None:
        n = np.max([np.max(grp) for grp in indx_grps]) + 1
    indx_shuffle = np.arange(n)
    for grp in indx_grps:
        indx_shuffle[grp] = np.random.permutation(grp)
    return indx_shuffle


def group_shuffle_data(data, indx_grps=None, axis=0):
    """
    controlled shuffle within index groups
    :param data:      data to be shuffled along an axis
    :param indx_grps: list of indexes, eg. [[0,1,2,3], [4,5,6,7]], will shuffle within [0,1,2,3], and within [4,5,6,7], but not between them
    :param axis:      along which to shuffle data
    :return:          shuffled data
    """

    n = data.shape[axis]
    if indx_grps is None:  # shuffle all if not given
        indx_grps = [np.arange(n)]

    indx_shuffle = group_shuffle(indx_grps, n)
    data_shuffle = np.take(data, indx_shuffle, axis=axis)
    return data_shuffle



""" ========== obsolete functions ========== """

def GP_ERP_smooth(lfp, ts=None, cs=None):
    """ obsolete function,  use gaussian process to smooth data, does not work well """
    kernel = 1.0 * gaussian_process.kernels.RBF(1.0)
    alpha = np.std(lfp)**2/10**6
    gp = gaussian_process.GaussianProcessRegressor(kernel= kernel, alpha=alpha)
    N,T = lfp.shape
    c_grid, t_grid = (np.arange(N), np.arange(T))
    x = np.expand_dims(c_grid, axis=1)
    c_grid_sm = c_grid
    x_sm = x
    lfp_smooth = np.zeros([len(c_grid_sm), len(t_grid)])
    for t in t_grid:
        y = lfp[:, t:t+1]
        gp.fit(x,y)
        lfp_smooth[:,t:t+1] = gp.predict(x_sm)
    return lfp_smooth


def quad_smooth_der(target, lambda_dev=1, lambda_der=0, add_edge=0, degree_der=3, return_CSD=False, x0=None):
    """
    --- OBSOLETE: use lfp_robust_smooth instead ---
    Quadratic smoothing using continuous (2nd-order) derivative assumption, used to calculate csd from lfp with noise

    :param target:     target signal to smooth, 1D array (recorded lfp with noise, e.g. slightly different gain between channels)
    :param lambda_dev: coefficient of the deviation  for the cost term, scalar or vector of the same shape as the target.  if target[i] is noisy, set lambda_dev[i] to be close to zero
    :param lambda_der: coefficient of the derivative for the cost term.  Larger values leads to more smoothed data
    :param add_edge:   number of channels to add outside both edges, to prevent boundary effect during computation (will be removed in result)
    :param return_CSD: true/false to return csd, affect the returned result
    :param x0:         initial value for optimization, default to None
    :return:           target_smoothed, or (target smoothed, csd)
    """
    lambda_dev = np.array(lambda_dev)
    def quad_cost(x, y):
        """ cost function where x is the variable and y is the target; penalize when 1) x devetates from y and 2) x is not smooth """

        """ dev (deviation) term: the smoothed data has to be similar with the original target data """
        if add_edge:
            dev = x[add_edge:0-add_edge]-y
        else:
            dev = x-y
        """ der (derivative) term: the smoothed data has to be smooth """
        if degree_der ==3:     # if assuming 3rd derivative is small, i.e., smooth 2nd derivative (csd from lfp)
            der = np.convolve(x, [-1, 3, -3, 1], mode='valid')   # third order derivative
        elif degree_der ==4:   # if assuming 4th derivative is small, i.e., smooth 3rd derivative
            der = np.convolve(x, [1, -4, 6, -4, 1], mode='valid')   # fourth order derivative
        """ tocal cost is the summation of dev and der """
        cost =  np.sum(lambda_dev* dev ** 2) + lambda_der * np.sum(der ** 2)
        return cost

    """ add edge to x0, initial values for optimization """
    if x0 is None:
        if add_edge:
            x0 = np.zeros(len(target)+2*add_edge)
        else:
            x0 = np.zeros(len(target))

    """ optimization step """
    res = optimize.minimize(lambda x: quad_cost(x, y=target), x0=x0, tol=np.std(target)/10**5)

    if add_edge:
        ret = res.x[add_edge:0-add_edge]
        if return_CSD:
            csd = np.convolve(res.x, [-1,2,-1], mode='same')
            csd = csd[add_edge:0-add_edge]
            ret = [ret, csd]
    else:
        ret = res.x
        if return_CSD:
            csd = csd = np.convolve(res.x, [-1,2,-1], mode='same')
            ret = [ret, csd]
    return ret


def lfp_cross_chan_smooth(lfp, method='der', lambda_dev=1, lambda_der=1, sigma_chan=0.5, sigma_t=0, tf_x0=True):
    """
    --- OBSOLETE: use lfp_robust_smooth instead ---
    Smooth the lfp across channels for csd estimation, to deal with varied gain across channels. either use 1) derivative-based method or 2) gaussian filter

    1) derivative-based: Quadratic smoothing using continuous (2nd-order) derivative assumption across channels, see function quad_smooth_der for details

    2) gaussian filter: use gaussian filter to smooth lfp across channels

    :param lfp:        lfp signal, [num_chan, num_timestamps]
    :param method:     smoothing method: 'der' for derivative-based method; 'gaussian' for gaussian filter
    :param lambda_dev: coefficient of the deviation  for the cost term, scalar or vector of the same shape as the target.  if target[i] is noisy, set lambda_dev[i] small or just zero, used for 'der' method
    :param lambda_der: coefficient of the derivative for the cost term.  Larger values leads to more smoothed result, used for 'der' method
    :param sigma_chan: std for gaussian smoothing across channels, used for 'gaussian' method
    :param sigma_t:    std for gaussian smoothing along time axis, set to 0 if do not smooth along time
    :param tf_x0:      true/false using the result of previous time point as the initial point for optimization, defult to True, which speeds up computation
    :return:           lfp of the same shape, smoothed
    """

    N, T = lfp.shape
    lfp_hat = lfp*0
    scale_lfp = np.percentile(np.abs(lfp), 98)    # used to normalize the data, convenient for optimization
    if method == 'der':
        x0= None
        for t in range(T):
            y = lfp[:,t]/scale_lfp
            x_smooth = quad_smooth_der(y, lambda_dev=lambda_dev, lambda_der=lambda_der, x0=x0, add_edge=0)
            lfp_hat[:, t] = x_smooth
            x0 = x_smooth
    elif method =='gaussian':
        lfp_hat = sp.ndimage.gaussian_filter1d(lfp/scale_lfp, sigma_chan, axis=0)
    else:
        raise Exception('method has to be either "der" or "gaussian"')

    if sigma_t>0:
        lfp_hat = sp.ndimage.gaussian_filter1d(lfp_hat, sigma_t, axis=1)
    return lfp_hat * scale_lfp