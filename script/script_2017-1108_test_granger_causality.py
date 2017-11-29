""" test of granger causality """
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model as linear_model

""" ========== test AR process ========== """
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


""" ========== VAR process  ========== """
N = 200
T = 50
M = 2
p = 3
phi = np.array([[[0.5,0],[0,0.7]], [[0.25,0],[0,-0.25]], [[0.05,0],[0,0.05]]])
c = np.array([0,0])
esp = np.array([0.1,0.1])
x_ini = np.array([0,0])

# realization of x
def VAR_gen_rlzt(phi, x_ini, c=0.0, esp=0.1, T=20):
    p, M, _ = phi.shape
    phi_flip = np.flip(phi, axis=0)
    x = np.zeros([T, M])
    x[:len(x_ini)] = x_ini
    for t in range(p,T):
        x[t, :] = c + np.sum([np.dot(phi_flip[pp], x[t-p+pp]) for pp in range(p)], axis=0) + esp*np.random.randn(M)
    return x

temp = VAR_gen_rlzt(phi, x_ini)

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



""" use statsmodels to fit the simulated data """

import statsmodels.api as sm


# X_concat= X
X_concat = np.concatenate([ np.concatenate([XX, np.zeros([p, M])*np.nan], axis=0) for XX in X_rlzt], axis=0)
X_concat.shape

md = sm.tsa.VARMAX(X_concat, order=(p,0))

md_result = md.fit()

md_result.summary()