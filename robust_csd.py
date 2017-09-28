import numpy as np
import scipy as sp
import scipy.optimize as optimize

def cal_robust_csd(lfp, lambda_dev=1, lambda_der=1, sigma_t=0, tf_edge=True, spacing=1.0):
    """
    Robust estimation of CSD that deals with varied gain across channels. a wrapper funciton of cal_1dCSD() and lfp_robutst_smooth().

    FirstSmooth-out the varied gain of LFP signals across channels and deal with missing channels;
    Then, use 2nd order spatial derivative (the three point formula) to approximate 1D current source density (csd)

    :param lfp:        lfp signal, 2D array, with shape == [num_chan, num_timestamps]
    :param lambda_dev: coefficient of the deviation term   for the cost function, scalar or vector of length = number_channels
                            range between 0 and 1. if channel [i] is noisy, set lambda_dev[i] small or just zero
    :param lambda_der: coefficient of the derivative for the cost term, scalar.  Larger values leads to more smoothed result.
    :param sigma_t:    std for gaussian smoothing along time axis, set to 0 if do not smooth along time, the unit is number of timestampes
    :param tf_edge:    true/false to interpolate the two channels on the edge, affect the shape of the result
    :param spacing:    inter-channel distance, a scalar, affect the scale of the CSD
    :return:        csd, [N_channels-2, N_timestamps] or [N_channels, N_timestamps], dependign on tf_edge
    """

    # smooth lfp using function lfp_robust_smooth()
    lfp_smooth = lfp_robust_smooth(lfp, lambda_dev=lambda_dev, lambda_der=lambda_der, sigma_t=sigma_t)
    # compute CSD using function cal_1dCSD()
    csd = cal_1dCSD(lfp_smooth, axis_ch=0, tf_edge=tf_edge, spacing=spacing)
    return csd



def cal_1dCSD(lfp, axis_ch=0, tf_edge=False, spacing=1):
    """
    Use 2nd order spatial derivative (the three point formula) to approximate 1D current source density (csd)
    csd[i] = -( lfp[i-1]+lfp[i+1]-2*lfp[i] )/spacing**2

    :param lfp:     lfp signal , by default [N_channels, N_timestamps]
    :param axis_ch: axis of the channels, equals zero by default
    :param tf_edge: true/false to interpolate the two channels on the edge, affect the shape of the result
    :param spacing: inter-channel distance, a scalar, affect the scale of the CSD
    :return:        csd, [N_channels, N_timestamps]
    """
    N = lfp.shape[axis_ch]    # num of channels
    csd = -np.diff(np.diff(lfp, axis=axis_ch), axis=axis_ch)/spacing**2   # calculate csd using the three point formula
    if tf_edge:    # interpolate channels on the edge
        csd_edge_l = 2*np.take(csd,  0, axis=axis_ch) - np.take(csd,  1, axis=axis_ch)
        csd_edge_r = 2*np.take(csd, -1, axis=axis_ch) - np.take(csd, -2, axis=axis_ch)
    else:
        csd_dege_l = np.take(csd,  0, axis=axis_ch)*np.nan
        csd_edge_r = np.take(csd,  0, axis=axis_ch)*np.nan
    csd = np.concatenate([np.expand_dims(csd_edge_l, axis=axis_ch), csd, np.expand_dims(csd_edge_r, axis=axis_ch)], axis=axis_ch)
    return csd


def lfp_robust_smooth(lfp, lambda_dev=1, lambda_der=1, sigma_t=0, tf_x0_inherent=True, tf_grad=True):
    """
    Smooth the lfp across channels for robust CSD estimation, to deal with slightly varied gain across channels.
    The smoothing algorithm is to minimize a cost function that considers
      (1) the deviation from the empirical data, and (2) the smoothness of the 2nd order derivative (CSD)

    :param lfp:        lfp signal, 2D array, with shape == [num_chan, num_timestamps]
    :param lambda_dev: coefficient of the deviation term   for the cost function, scalar or vector of length = number_channels
                            range between 0 and 1. if channel [i] is noisy, set lambda_dev[i] small or just zero
    :param lambda_der: coefficient of the derivative for the cost term, scalar.  Larger values leads to more smoothed result.
    :param sigma_t:    std for gaussian smoothing along time axis, set to 0 if do not smooth along time, the unit is number of timestampes
    :param tf_x0_inherent:    true/false using the result of previous time point as the initial point for optimization, defult to True, which speeds up computation
    :param tf_grad:    true/false to use analytical form of gradient, which significantly speeds up the computation
    :return:           lfp of the same shape, smoothed
    """

    """ get the shape of data.  N: number of channels; T: number of timestamps """
    input_dimention = len(lfp.shape)
    if input_dimention == 1:      # if 1D, assumes it contains only 1 timestamps
        lfp=np.expand_dims(lfp, axis=1)
        N, T = lfp.shape
    elif input_dimention == 2:
        N, T = lfp.shape
    else:
        raise Exception('input lfp has to be 1D or 2D array')

    """ normalize data for the convenience of optimation """
    scale_lfp = np.nanstd(lfp)  # used to normalize the data, convenient for optimization
    lfp_nmlz = lfp/scale_lfp    # normalized lfp

    """ cost function """
    lambda_dev = np.array(lambda_dev)
    lambda_der = np.array(lambda_der)
    def quad_cost(x, y):
        """
        cost function where x is the variable and y is the target to be smoothed;
        penalize when 1) x devetates from y and 2) x is not smooth
        """

        """ dev (deviation) term: the smoothed data has to be similar with the original target data """
        dev = x-y
        """ der (derivative) term: the smoothed data has to be smooth across neighboring channels """
        # the 3rd spatial derivative has to be small if 2nd derivative (csd from lfp) is smooth
        der = np.convolve(x, [-1, 3, -3, 1], mode='valid')   # third order derivative
        """ tocal cost is the summation of quadratic term of dev and der """
        cost = np.sum(lambda_dev* dev ** 2) + lambda_der * np.sum(der ** 2)
        return cost

    """ analytical gradient of cost function """
    """ use if tf_grad=True; otherwise compute gradient numerically, which can be slow """
    if tf_grad:
        # construct ker_der_matrix, for calculating the gradient of the der term (enforcing smoothness)
        ker_der = np.array([-1,3,-3,1])
        ker_der_matrix = np.zeros([N,N])
        for i in range(N-3):
            ker_der_matrix[i, i:i+4] += ker_der*(-1)
        for i in range(1, N-2):
            ker_der_matrix[i, i-1:i+3] += ker_der * (+3)
        for i in range(2, N-1):
            ker_der_matrix[i, i-2:i+2] += ker_der * (-3)
        for i in range(3, N):
            ker_der_matrix[i, i-3:i+1] += ker_der * (+1)
        # finish constructing the ker_der_matrix

        def quad_cost_grad(x, y):
            """  analytical form of the gradient of the cost function """
            grad_dev = 2 * lambda_dev * (x - y)
            grad_der = 2 * lambda_der * np.matmul(ker_der_matrix, x)
            return grad_dev + grad_der

    """ options for the quadratic optimization process for smoothing LFP """
    x0 = np.zeros(N)        # initial value
    tol = 10**(-4)          # termination criterion: smaller value leads to more accurate results but slower
    lfp_hat = lfp * 0       # place holder of the smoothed data

    """ optimization process """
    for t in range(T):      # for every time point
        y = lfp_nmlz[:,t]                                                    # lfp over all channels at current time
        fun_cost = lambda x: quad_cost(x, y=y)                               # cost function for the current LFPs
        fun_grad = (lambda x: quad_cost_grad(x, y=y)) if tf_grad else None   # gradient function
        # optimization process
        res = optimize.minimize(fun_cost, x0=x0, jac=fun_grad, tol=tol)
        lfp_hat[:,t] = res.x
        if tf_x0_inherent:
            x0 = res.x

    """ smooth over time using gaussian kernel """
    if sigma_t>0:
        lfp_hat = sp.ndimage.gaussian_filter1d(lfp_hat, sigma_t, axis=1)

    """ put data back to its origianl range """
    lfp_smooth = lfp_hat * scale_lfp

    if input_dimention ==1:
        lfp_smooth = lfp_smooth.ravel()

    return lfp_smooth