"""
modeule for time series analysis, including: AR, VAR, Granger Causality, etc.

The core idea of time series analysis is to predict the future data using the previous data through a reg
Major difference from statsmodels.tsa: the input data type can be n array of [num_trials, num_timestamps, number_channels]
"""



import numpy as np
import scipy as sp
import sklearn
import sklearn.linear_model as linear_model
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')





""" ----------========== AR(p) process ==========---------- """

"""
An AR(p) process is auto-regressive model with p history terms:
x(t) = x(t-1)*phi(1) + x(t-2)*phi(2) + ... + x(t-p)*phi(p) + c + epsilon


"""

p = 2

phi = [0.7, -0.25]
x_ini = [0,0]

# realization of x
def AR_gen_rlzt(phi, x_ini, c=0, esp=0.1, T=20):
    p = len(phi)
    phi_flip = np.flip(phi, 0)
    x = np.zeros(T)
    x[:len(x_ini)] = x_ini
    for t in range(p,T):
        x[t] = c + np.dot(x[t-p:t], phi_flip) + esp*np.random.randn()
    return x

x_rlzt = AR_gen_rlzt(phi, x_ini, c=0.5, T=50, esp=0.3)

N = 20
T = 50
X_rlzt = np.stack([AR_gen_rlzt(phi, x_ini, c=0.5, T=T, esp=0.3) for i in range(N)], axis=0)

X_rlzt_mean = np.mean(X_rlzt, axis=0)
X_rlzt_std = np.std(X_rlzt, axis=0)
plt.plot(range(T), X_rlzt.transpose(), alpha=0.4)
plt.plot(range(T), X_rlzt_mean, color='k')
plt.fill_between(range(T), X_rlzt_mean-X_rlzt_std, X_rlzt_mean+X_rlzt_std, color='k', alpha=0.9)
plt.title('AR({}) process: {} realizations'.format(p, N))

def AR_fit(X_rlzt, p=1):
    N, T = X_rlzt.shape
    y_reg = np.concatenate([X_rlzt[:, t] for t in range(p, T)],  axis=0)
    x_reg = np.flip(np.concatenate([X_rlzt[:, t-p:t] for t in range(p, T)], axis=0), axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(x_reg, y_reg)
    return lr.coef_, lr.intercept_

AR_fit(X_rlzt, p=2)


""" ----------========== VAR(p) process  ==========---------- """

"""
x(t) = phi(1)*x(t-1) + x(2)*x(t-2) + ... + x(p)*x(t-p) + c + eps

-----
naming scheme:
X:   time series data, shape [M, T, M]:  N trials, T time stamps and M channels
x:   one trial of data, shape [T, M]
ts:  timestamps, array of length T
p:   number of history terms
phi: coefficients of history terms for regression, of shape [p, M, M]
c:   constant term for regression, length M
esp: error term, residue of regression, length M

-----
example parameters:
N = 200
T = 50
M = 2
p = 3
phi = np.array([[[0.5,0],[0,0.7]], [[0.25,0],[0,-0.25]], [[0.05,0],[0,0.05]]])   # [3,2,2]
c = np.array([0,0])
esp = np.array([0.1,0.1])
x_ini = np.array([0,0])
"""



def VAR_gen_rlzt(x_ini, phi, c=0.0, esp=0.1, T=20):
    """ generate one realization (sample) of VARp process using parameter (phi, c, and esp) over time T  """
    p, M, _ = phi.shape
    phi_flip = np.flip(phi, axis=0)
    x = np.zeros([T, M])
    x[:len(x_ini)] = x_ini
    for t in range(p,T):
        x[t, :] = c + np.sum([np.dot(phi_flip[pp], x[t-p+pp]) for pp in range(p)], axis=0) + esp*np.random.randn(M)
    return x   # shape [T, M]

def flatten_x_for_regress(x, p=1):
    """ preparing x for linear regression: input [T, M], output x_flat:[T-p, M*p], y_flat:[T-p, M]  """
    """ each row of x_flat is : [x_{t-1,1}, x_{t-1,2}, ... x_{t-1, M}, ...,  x_{t-p,1}, x_{t-p,2}, ... x_{t-p, M}] """
    x = np.array(x)
    T, M = np.shape(x)
    y_flat = x[p:]
    x_flat = np.concatenate([x[p-pp-1:T-1-pp] for pp in range(p)], axis=1)
    return x_flat, y_flat
# flatten_x_for_regress([[0,0],[1,1],[2,2],[3,3]], p=2)

def flatten_X_for_regress(X, p=1):
    """ preparing X for linear regression: input [N, T, M], output x_flat:[(T-p)*N, M*p], y_flat:[(T-p)*N, M]  """
    N, T, M = X.shape
    X_for_reg, Y_for_reg = zip(*[flatten_x_for_regress(x, p=p) for x in X])
    X_for_reg = np.concatenate(X_for_reg, axis=0)
    Y_for_reg = np.concatenate(Y_for_reg, axis=0)
    return X_for_reg, Y_for_reg

def VAR_fit(X, p=1, method_regression='LinearRegression'):
    """ fit a VARp process via linear regression """
    if method_regression == 'LinearRegression':
        lr = linear_model.LinearRegression()
    elif method_regression == 'RidgeCV':
        lr = linear_model.RidgeCV()
    else:
        warnings.warn('the given method_regression is not valid, use LinearRegression instead')
        lr = linear_model.LinearRegression()
    N, T, M = X.shape
    X_for_reg, Y_for_reg = flatten_X_for_regress(X, p=p)       # reformat data for linear regression

    lr.fit(X_for_reg, Y_for_reg)
    phi_hat = np.stack(np.split(lr.coef_, p, axis=1), axis=0)  # recover phi and c in the orignal format
    c_hat = lr.intercept_
    Y_hat_reg = lr.predict(X_for_reg)                          # get predicted values based on history term (VAR model)
    X_hat = X*np.nan
    X_hat[:,p:,:] = np.stack(np.split(Y_hat_reg, N, axis=0))   # re-format the predicted value in it's original structure as X
    return phi_hat, c_hat, X_hat

def plot_phi(phi, c=None):
    """ plot phi and c """
    [p, M, _] = phi.shape
    h_fig, h_ax = plt.subplots(M, M, squeeze=False, sharex='all', sharey='all')
    for i in range(M):
        for j in range(M):
            plt.axes(h_ax[i,j])
            plt.axhline(0, c='k', ls=':')
            plt.fill_between(np.arange(1, p + 1), phi[:, i, j], alpha=0.25)
            plt.stem(np.arange(1, p+1), phi[:,i,j])
            if c is not None:
                plt.bar(0, c[i])
            plt.title('{} to {}'.format(j, i))
    plt.suptitle('VAR: phi (stem plot) and c (bar at x=0)')
    plt.xlabel('p')
# plot_phi(phi)

def plot_X(X):
    """ plot samples """
    N, T, M = X.shape
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    h_fig, h_ax = plt.subplots(M, 1, squeeze=False, sharex='all', sharey='all')
    for m in range(M):
        plt.axes(h_ax[m, 0])
        plt.plot(range(T), X[:, :, m].transpose(), alpha=0.4)
        plt.plot(range(T), X_mean[:, m], color='k')
        plt.fill_between(range(T), X_mean[:, m] - X_std[:, m], X_mean[:, m] + X_std[:, m], color='k',
                         alpha=0.4)


class VARp(object):
    """ vector auto-regressive process object """

    def __init__(self, p=1, X=None, N=1, T=1, M=1, phi=None, c=None, esp=None):
        """ initialize VARp object either using data X, or (N,T,M); order p has to be given """
        self.X = X
        self.X_hat = None
        if X is not None:   # if X is given, determine dimension using X, otherwise use the given parameters
            [N, T, M] = X.shape
        self.N = N
        self.T = T
        self.M = M
        self.p = p       # get order p: how many history terms to consider
        self.phi = phi   # VAR parameter: coefficient matrix: how history predict present, shape [p, M, M]
        self.c = c       # VAR parameter: constant term of present, shape [M]
        self.esp = esp   # VAR parameter: error term, shape [M]

    def sample(self, X_ini = None):
        """ sample from the VAR process, phi and c has to be specified in the VAR object, returns array of shape [N, T, M] """
        if X_ini is None:
            X_ini = np.zeros([self.N, self.T, self.M])
        return np.stack([VAR_gen_rlzt(X_ini[i, :, :], phi=self.phi, c=self.c, esp=self.esp, T=self.T) for i in range(self.N)], axis=0)

    def fit(self, X=None, method_regression='LinearRegression'):
        """ sample from the VAR process, phi and c has to be specified in the VAR object """
        X = self.X if X is None else X
        phi_hat, c_hat, X_hat = VAR_fit(X, p=self.p, method_regression=method_regression)
        self.phi = phi_hat
        self.c   = c_hat
        self.X_hat = X_hat
        return phi_hat, c_hat, X_hat

    def plot_X(self):
        """ plot samples """
        plot_X(self.X)

    def plot_phi(self):
        """ plot phi and c """
        plot_phi(self.phi, self.c)

# def cal_pairwise_GC


""" script to test VAR object """
# ini object parameters
p = 2
phi = np.array([[[0.9, 0], [0.16, 0.8]], [[-0.5, 0], [-0.2, -0.5]]])
esp = np.array([1, 0.7**0.5])
c = np.array([0,0])
# var_proc = VARp(N=100, T=1000, M=2, p=3, phi=np.array([[[0.5,0.25],[0,0.7]], [[0.25,-0.1],[0,-0.25]], [[0.05,0],[0,0.05]]]), c=[0.1,0.2], esp=[1.0,1.0])
var_proc = VARp(N=500, T=100, M=2, p=3, phi=phi, c=c, esp=esp)
# sample
X_rlzt = var_proc.sample()
# initialize another VAR object using samples
var_proc = VARp(X=X_rlzt, p=3)
# plot data samples X
var_proc.plot_X()
# fit phi and c
var_proc.fit()
# plot phi and c
var_proc.plot_phi()


def cal_GC_pairwise(X, p=1):
    N, T, M = X.shape
    sigma_pair = np.zeros([M,M])
    for m in range(M):
        X_ind = X[:,:,m:m+1]
        var_ind = VARp(X=X_ind, p=p)
        var_ind.fit()
        residue = var_ind.X_hat - X_ind
        sigma_pair[m, m] = np.nanmean( residue ** 2)
    for i in range(M):
        for j in range(i+1, M):
            X_pair = X[:, :, [i,j]]
            var_pair = VARp(X=X_pair, p=p)
            var_pair.fit()
            residue = var_pair.X_hat - X_pair
            sigma_pair[i, j] = np.nanmean(residue[:, :, 0] ** 2)
            sigma_pair[j, i] = np.nanmean(residue[:, :, 1] ** 2)
    F_GC = np.zeros([M,M])*np.nan
    for m in range(M):
        F_GC[m, :] = np.log(sigma_pair[m, m]/sigma_pair[m, :])

    return sigma_pair, F_GC

temp= cal_GC_pairwise(X_rlzt, p=2)
print(temp)

temp = VAR_gen_rlzt(x_ini, phi)

X = VAR_gen_rlzt(phi, x_ini, T=10000, c=0.5, esp=0.3)

X_rlzt = np.stack([VAR_gen_rlzt(phi, x_ini, T=T, c=0.5, esp=0.3) for i in range(N)], axis=0)

X_rlzt_mean = np.mean(X_rlzt, axis=0)
X_rlzt_std = np.std(X_rlzt, axis=0)
plt.plot(range(T), X_rlzt[:,:,0].transpose(), alpha=0.4)
plt.plot(range(T), X_rlzt_mean[:,0], color='k')
plt.fill_between(range(T), X_rlzt_mean[:,0]-X_rlzt_std[:,0], X_rlzt_mean[:,0]+X_rlzt_std[:,0], color='k', alpha=0.2)
plt.plot(range(T), X_rlzt_mean[:,1], color='b')
plt.fill_between(range(T), X_rlzt_mean[:,1]-X_rlzt_std[:,1], X_rlzt_mean[:,1]+X_rlzt_std[:,1], color='b', alpha=0.2)


def flatten_x_for_regress(x, p=p):
    """ turn [p,M,M] to  """
    x = np.array(x)
    T, M = np.shape(x)
    x_flat = np.zeros([T-p, M])
    y_flat = x[p:]
    x_flat = np.concatenate([x[p-pp-1:T-1-pp] for pp in range(p)], axis=1)
    return x_flat, y_flat
# flatten_x_for_regress([[0,0],[1,1],[2,2],[3,3]], p=2)

def flatten_X_for_regress(X, p=p):
    N, T, M = X.shape
    X_for_reg, Y_for_reg = zip(*[flatten_x_for_regress(x, p=p) for x in X])
    X_for_reg = np.concatenate(X_for_reg, axis=0)
    Y_for_reg = np.concatenate(Y_for_reg, axis=0)
    return X_for_reg, Y_for_reg

def VAR_fit(X, p=p):
    N, T, M = X.shape
    X_for_reg, Y_for_reg = flatten_X_for_regress(X, p=p)   # reformat data for linear regression
    # lr = linear_model.LinearRegression()
    lr = linear_model.RidgeCV()
    lr.fit(X_for_reg, Y_for_reg)
    phi_hat = np.stack(np.split(lr.coef_, p, axis=1), axis=0)  # recover phi and c in the orignal format
    c_hat = lr.intercept_
    Y_hat_reg = lr.predict(X_for_reg)                          # get predicted values based on history term (VAR model)
    X_hat = X*np.nan
    X_hat[:,p:,:] = np.stack(np.split(Y_hat_reg, N, axis=0))   # re-format the predicted value in it's original structure as X
    return phi_hat, c_hat, X_hat

# phi_hat, c_hat, X_hat_full = VAR_fit(X_rlzt, p=p)


def VAR_esp(X, p=p, m_subset0=1):
    """ compute the noise residue that VAR fails to capture """
    X0 = X[:,:, :m_subset0]
    X1 = X[:,:, m_subset0:]
    phi_hat_full, c_hat_full, X_hat_full = VAR_fit(X, p=p)   # fit on full model
    phi_hat_0, c_hat_0, X_hat_0 = VAR_fit(X0, p=p)           # fit model only using ch subset 0
    phi_hat_1, c_hat_1, X_hat_1 = VAR_fit(X1, p=p)           # fit model only using ch subset 1
    eps_full = X_hat_full-X
    eps0 = X_hat_0 - X0
    eps1 = X_hat_1 - X1
    return eps_full, eps0, eps1

# eps_full, eps0, eps1 = VAR_esp(X_rlzt, p=p, m_subset0=1)

def VAR_GC(eps_full=None, eps0=None, eps1=None, X=None, p=p, m_subset0=1, n_tf=None):
    """ granger causality """
    if X is not None:
        eps_full, eps0, eps1 = VAR_esp(X, p=p, m_subset0=m_subset0)
    if n_tf is not None:
        eps_full = eps_full[n_tf, :, :]
        eps0 = eps0[n_tf, :, :]
        eps1 = eps1[n_tf, :, :]
    m_subset0 = eps0.shape[2]
    gc_1to0 = np.log( np.nanmean(eps0**2, axis=(0,2)) / np.nanmean(eps_full[:, :, :m_subset0]**2, axis=(0,2)))
    gc_0to1 = np.log( np.nanmean(eps1**2, axis=(0,2)) / np.nanmean(eps_full[:, :, m_subset0:]**2, axis=(0,2)))
    return gc_0to1, gc_1to0
VAR_GC(X=X_rlzt, p=p, m_subset0=1)
