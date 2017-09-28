"""  script to test copulat model to capture the correlation """

import os
import sys
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')

import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import sklearn
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import datetime
import pickle

# ----- modules used to read neuro data -----
import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq

# ----- modules of the project PyNeuroSG -----
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32


dir_tdt_tank='/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/shaobo/data_dg'


keyword_block = 'd_.*srv.*'
keyword_tank = '.*GM32.*U16.*161125.*'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword=keyword_block, keyword_tank=keyword_tank,
                                                           tf_interactive=True, dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

""" waveform plot """
# pnp.SpkWfPlot(blk.segments[0])


data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.100, 0.600], type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1000)
data_neuro = signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro)

pnp.PsthPlot(data_neuro['data'], ts=data_neuro['ts'], sk_std=0.010, color_style='continuous')

t_focus = [0.030, 0.400]

data_neuro = signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48+1), sortcode_filter=(2,3))

spk_count = pna.AveOverTime(data_neuro['data'], ts=data_neuro['ts'], t_range=[0.050, 0.250], tf_count=True)


N_tr, N_ch = spk_count.shape
spk_hist2d_list = []
for i in range(N_ch):
    for j in range(N_ch):
        spk_hist2d, _, _ = np.histogram2d(spk_count[:,i], spk_count[:,j], bins=range(20))
        spk_hist2d_list.append(spk_hist2d)

reload(pnp)
pnp.DataFastSubplot(spk_hist2d_list, data_type='mesh')



""" test a distirubution """
y=np.arange(0,100,0.001)
mu= np.log(5)
sigma= np.log(5)
fy = lambda y: 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(np.log(y)-mu)**2/(2*sigma**2) -np.log(y))
plt.plot(y,fy(y))
plt.xlim(0, 40)
print( np.nansum(fy(y))*0.001 )




""" ===== test math ===== """
mu= 2
sigma = 0.5
mu= 0
sigma = 2
yy = np.arange(20)     # grid for plot
N = 1000               # number of samples to draw
X = sp.stats.norm(loc=mu, scale=sigma)     # rv
xs = X.rvs(size=N)                         # samples of X
ys = np.floor(np.exp(xs))                  # samples of Y

# empirical distribution
pmf_empirical, _ = np.histogram(ys, bins=pnp.center2edge(yy))
pmf_empirical = pmf_empirical * (1.0/N)
h_bar = plt.bar(yy, pmf_empirical, alpha=0.3)

# theoretical distribution
cdf_dscrlognorm = X.cdf(np.log(yy+1))
pmf_dscrlognorm =  np.insert(np.diff(cdf_dscrlognorm), 0, cdf_dscrlognorm[0])
h_line_true = plt.plot(yy, pmf_dscrlognorm, '-r', linewidth=3)
plt.title('log normal distribution')

# parameter fit
xs_hat = np.log(ys+0.5)
mu_hat = np.nanmean(xs_hat)
sigma_hat = np.nanstd(xs_hat)
X_hat = sp.stats.norm(loc=mu_hat, scale=sigma_hat)
cdf_dscrlognorm = X_hat.cdf(np.log(yy+1))
pmf_dscrlognorm =  np.insert(np.diff(cdf_dscrlognorm), 0, cdf_dscrlognorm[0])
h_line_est = plt.plot(yy, pmf_dscrlognorm, '+-b')
# plt.legend([h_bar, h_line_true[0], h_line_est[0]],['empirical', 'theoretical','estimate'])

# parameter fit: iterative sampling
N_rs = 10**6
N_y = 10**3
xs_hat = np.log(ys + 0.5)
mu_hat = np.nanmean(xs_hat)
sigma_hat = np.nanstd(xs_hat)
y_hist, _ = np.histogram(ys, bins= np.append(pnp.center2edge(np.arange(N_y)), np.inf) )
x_bins = np.log(np.arange(1,N_y+1))
N_interation = 10
for i in range(N_interation):
    print(mu_hat)
    X_rs = sp.stats.norm(loc=mu_hat, scale=sigma_hat)
    xs_rs = X_rs.rvs(size=N_rs)
    importance_ratio = 1.0*y_hist / np.histogram(np.floor(np.exp(xs_rs)), bins= np.append(pnp.center2edge(np.arange(N_y)), np.inf) )[0]
    xs_weight = importance_ratio[np.digitize(xs_rs, bins=x_bins)]
    mu_hat = np.average(xs_rs, weights=xs_weight)
    sigma_hat = np.sqrt(np.average((xs_rs-mu_hat)**2, weights=xs_weight))
    X_rs = sp.stats.norm(loc=mu_hat, scale=sigma_hat)
    cdf_dscrlognorm = X_rs.cdf(np.log(yy+1))
    pmf_dscrlognorm =  np.insert(np.diff(cdf_dscrlognorm), 0, cdf_dscrlognorm[0])
    h_line_true = plt.plot(yy, pmf_dscrlognorm, '--', c=np.ones(3)*1.0*i/N_interation)
# plt.legend()



# plt.plot(importance_ratio)
plt.plot(xs_rs, xs_weight, '.')



# test mvn cdf efficency:
from scipy.stats import mvn, multivariate_normal
for M in 2**np.arange(9):
    tic=time.time()
    temp = np.log2(mvn.mvnun(np.ones(M)*0,np.ones(M)*10,np.zeros(M),np.eye(M)))
    toc=time.time()
    h_ind = plt.loglog(M, toc-tic, 'ok')
for M in 2**np.arange(9):
    tic=time.time()
    temp = np.log2(mvn.mvnun(np.ones(M)*0,np.ones(M)*10,np.zeros(M),np.eye(M)*0.8+0.2))
    toc=time.time()
    h_cor = plt.loglog(M, toc-tic, '+r')
for M in 2**np.arange(9):
    N = 10000
    tic = time.time()
    dist = multivariate_normal(mean=np.zeros(M), cov=np.eye(M)*0.8+0.2)
    sps =  np.random.rand(N, M)*10
    logpdf_unnmlz = dist.logpdf(x=sps)
    temp = np.mean(dist.pdf(x=sps)*10**M)
    toc= time.time()
    h_pdf = plt.loglog(M, toc - tic, 'xg')
plt.title('mvn cdf computing time')
plt.xlabel('number of dimension')
plt.ylabel('time (sec)')
plt.legend([h_ind[0], h_cor[0], h_pdf[0]],['independent','correlated', 'pdf_mean'])
plt.savefig('./temp_figs/mvn_cdf.png')


# test: sample a normal distribution in a defined range
N_raw = 100000
N_flt = 10000
M = 100
dist = sp.stats.multivariate_normal(mean=np.zeros(M), cov=np.eye(M)*0.5+0.5)
x_low = np.ones(M)*0-1
x_high= np.ones(M)*0+1
x_raw = np.random.rand(N_raw, M) * (x_high-x_low) + x_low    # samples in regular grid
logpdf_unnmlz = dist.logpdf(x_raw)
logpdf_nmlz = logpdf_unnmlz - logpdf_unnmlz.max()
pdf_nmlz = np.exp(logpdf_nmlz) / sum(np.exp(logpdf_nmlz))
x_indx_flt = np.random.choice(len(x_raw), size=N_flt, p=pdf_nmlz)
x_flt = x_raw[x_indx_flt]
plt.hist2d(x_flt[:,0], x_flt[:,1],  bins=11)
plt.colorbar()





""" complete code for multi-variate non-negative correlated distribution """


""" generate samples and plot empirical distribution """

def pairplot_sample(X, bins=None):
    N,M = X.shape
    h_fig, h_ax = plt.subplots(M,M)
    for i in range(M):
        for j in range(M):
            plt.axes(h_ax[i,j])
            if i==j:
                if bins is None:
                    plt.hist(X[:,i])
                else:
                    plt.hist(X[:, i], bins=pnp.center2edge(bins))
            else:
                if bins is None:
                    plt.hist2d(X[:,i], X[:,j])
                else:
                    plt.hist2d(X[:, i], X[:, j], bins=pnp.center2edge(bins))
                h_ax[i, j].grid(False)


M=3    # number of samples
N=10000 # number of dim
# mu and sigma for mvn dist
mu = np.array([0,1,2])
sigma = np.eye(M)
sigma[0,1] = 0.5
sigma[1,0] = 0.5
sigma[0,2] = -0.5
sigma[2,0] = -0.5
sigma = sigma/2
# mvn dist object
rv_mvn = sp.stats.multivariate_normal(mean=mu, cov=sigma)
# samples as x
xs = rv_mvn.rvs(size=N)
# samples as y
ys = np.floor(np.exp(xs))
pairplot_sample(xs, bins=np.arange(-2,4,0.5))
plt.title('X distribution')
pairplot_sample(ys, bins=np.arange(0,20,1))
plt.title('Y distribution')

""" compute and plot pmf according to model """
# compute pdf:
def compute_dmvln_pmf(ys, mu, sigma, method='fast'):
    ys = np.array(ys, dtype=float)
    if method == 'fast':       # use the pdf of log normal distribution at y+0.5 to approximate pmf
        xs_hat = est_x(ys, method='fast')                          # estimate xs for give ys
        rv_mvn = sp.stats.multivariate_normal(mean=mu, cov=sigma)  # gaussian random variable for xs
        pdf_mvn = rv_mvn.pdf(xs_hat)                               # pdf of normal distribuion for x
        pdf_mvln= pdf_mvn/np.prod(ys+0.5, axis=1)                  # pdf of log normal ditribuion for y
        pmf = pdf_mvln *1.0                                        # ue one value (centroid of the interval) of pdf to approximate pmf
    elif method == 'accurate': # use mvn cdf of the right interval to accurately compute pmf
        ys[ys <= 0] = 0.01                                         # prevent log(0) error
        pmf = np.zeros(len(ys))
        for i, y in enumerate(ys):
            pmf_cur, _ = sp.stats.mvn.mvnun(np.log(y), np.log(y+1), mu, sigma)   # cdf of normal distribution on the corresponding interval
            pmf[i] = pmf_cur
    return pmf


def pmf_plot1d(mu, sigma, d=0, bins=np.arange(10), method='fast', plot_style='bar'):
    mu = mu[d]
    sigma = sigma[d][d]
    pmf = compute_dmvln_pmf( np.expand_dims(bins, axis=1), mu, sigma, method=method)
    if plot_style=='bar':
        plt.bar(bins, pmf, width=1.0)
    else:
        plt.plot(bins, pmf)

if False:
    values_mu = np.arange(0,4)
    values_sigma = np.arange(0.4,2,0.4)
    h_figh, h_ax = plt.subplots(2, 2, figsize=[8,6])
    plt.suptitle('pfm of DLN (discrete log normal distribution)')
    h_ax = np.ravel(h_ax)
    sigma = np.ones([1,1])
    for i, mu in enumerate(values_mu):
        plt.axes(h_ax[i])
        plt.title('mu={}'.format(mu))
        for sigma_scale in values_sigma:
            pmf_plot1d(np.expand_dims(mu, axis=1), sigma*sigma_scale, method='accurate', bins=np.arange(20), plot_style='line')
        plt.legend(values_sigma, title='sigma')
    plt.savefig('./temp_figs/pfm_DLN_distribution_by_parameters.png')

def pmf_plot2d(mu, sigma, d=[0,1], bins=np.arange(10), method='fast'):
    d=np.array(d)
    mu = mu[d]
    sigma = sigma[d][:,d]
    K = len(bins)
    pmf_grid = np.zeros([K, K])
    grid_d0, grid_d1 = np.meshgrid(bins, bins)
    pmf = np.zeros([K,K])
    grid_ravel = np.vstack([grid_d0.ravel(), grid_d1.ravel()]).transpose()
    pmf_ravel  = compute_dmvln_pmf(grid_ravel, mu, sigma, method=method)
    pmf2d = np.reshape(pmf_ravel, [K,K])
    plt.pcolormesh(pnp.center2edge(bins), pnp.center2edge(bins), pmf2d)


def pairplot_model(mu, sigma, bins=np.arange(10), method='fast'):
    M = len(mu)
    h_fig, h_ax = plt.subplots(M,M)
    for i in range(M):
        for j in range(M):
            plt.axes(h_ax[i,j])
            if i==j:
                pmf_plot1d(mu, sigma, d=i, bins=bins, method=method)
            else:
                pmf_plot2d(mu, sigma, d=[i,j], bins=bins, method=method)
                h_ax[i, j].grid(False)
pairplot_model(mu, sigma, bins=range(20), method='accurate')

if False:
    pmf_plot1d(mu, sigma, d=2, method='fast')
    pmf_plot1d(mu, sigma, d=2, method='accurate')

    pmf_plot1d(mu, sigma, d=0, method= 'fast')
    pmf_plot1d(mu, sigma, d=0, method= 'accurate')

    pmf_plot2d(mu, sigma, bins=np.arange(20), method='fast')

""" model fit """

# discrete multi variate log normal (dmvln) distribution
def est_x(ys, method='fast', N_MC=100, mu=None, sigma=None):
    N, M = ys.shape
    if method == 'fast':
        xs_hat = np.log(ys+0.5)
    elif method == 'MC':
        rv_mvn = sp.stats.multivariate_normal(mean=mu, cov=sigma)
        N_raw = N_MC*10
        xs_hat_list = []
        for y in ys:
            x_hat_unfm = np.log(np.random.rand(N_raw, M)+y)
            logpdf_mvn = rv_mvn.logpdf(x_hat_unfm)
            logpdf_mvln = logpdf_mvn - np.sum(x_hat_unfm, axis=1)
            logpdf_mvln_rltv = logpdf_mvln - logpdf_mvln.max()
            pdf_mvln_rltv = np.exp(logpdf_mvln_rltv)
            pdf_mvln_rltv = pdf_mvln_rltv/np.sum(pdf_mvln_rltv)
            x_hat_indx = np.random.choice(N_raw, size=N_MC, p=pdf_mvln_rltv)
            x_hat = x_hat_unfm[x_hat_indx]
            xs_hat_list.append(x_hat)
        xs_hat = np.vstack(xs_hat_list)
    return xs_hat

if False:   # test samples using est_x
    est_x(np.zeros([1,2]), method='fast')
    xs_sample = est_x(np.zeros([1,2]), method='MC', N_MC=1000, mu=np.zeros(2), sigma=np.eye(2)*0.1+0.9)
    plt.plot(xs_sample[:,0], xs_sample[:,1],'.')



N_raw = 100000
N_flt = 10000
M = 100
dist = sp.stats.multivariate_normal(mean=np.zeros(M), cov=np.eye(M)*0.5+0.5)
x_low = np.ones(M)*0-1
x_high= np.ones(M)*0+1
x_raw = np.random.rand(N_raw, M) * (x_high-x_low) + x_low    # samples in regular grid
logpdf_unnmlz = dist.logpdf(x_raw)
logpdf_nmlz = logpdf_unnmlz - logpdf_unnmlz.max()
pdf_nmlz = np.exp(logpdf_nmlz) / sum(np.exp(logpdf_nmlz))
x_indx_flt = np.random.choice(len(x_raw), size=N_flt, p=pdf_nmlz)
x_flt = x_raw[x_indx_flt]
plt.hist2d(x_flt[:,0], x_flt[:,1],  bins=11)
plt.colorbar()


def fit_cov(xs, method='empirical', model=None):
    if method=='empirical':
        cov = np.cov(xs_hat.transpose())
    elif method=='fa':
        model.fit(xs)
        cov = model.get_covariance()
    else:
        print('unsupported method')
    return cov


def fit_dmvln(ys, method_x='fast', method_cov='empirical', N_MC_itr=10, model=None):
    if method_x=='fast':
        xs_hat = est_x(ys, method='fast')
        mu_hat = np.mean(xs_hat, axis=0)
        if method_cov=='empirical':
            sigma_hat = fit_cov(xs_hat, method='empirical')
        elif method_cov=='fa':
            sigma_hat = fit_cov(xs_hat, method = 'fa', model=model)
    elif method_x=='MC':
        xs_hat = est_x(ys, method='fast')
        mu_hat = np.mean(xs_hat, axis=0)
        sigma_hat = np.cov(xs_hat.transpose())
        for i_MC in range(N_MC_itr):
            print(i_MC)
            xs_hat = est_x(ys, method='MC', mu=mu_hat, sigma=sigma_hat)
            mu_hat = np.mean(xs_hat, axis=0)
            if method_cov == 'empirical':
                sigma_hat = fit_cov(xs_hat, method = 'empirical')
            elif method_cov == 'fa':
                sigma_hat = fit_cov(xs_hat, method = 'fa', model=model)
    else:
        print('unsupported method')
    return mu_hat, sigma_hat, xs_hat, model



fit_dmvln(ys)
mu_hat, sigma_hat, xs_hat, model_fa = fit_dmvln(ys, method_x='MC')



""" test factor analysis """
model_fa =  sklearn.decomposition.FactorAnalysis(n_components=1)
model_fa.fit(xs)



""" new noise """
N=1000
M=5
Mh = 1
H = np.random.randn(N,Mh)
B = np.random.randn(N,M)
phi = np.random.rand(M,Mh)*2-1
b = np.random.rand(M)
X = np.dot(H, phi.transpose()) + b*B + np.random.rand(M)*2
Y = np.floor(np.exp(X))
xs = X
ys = Y
pairplot_sample(xs, bins=np.arange(-2,4,0.5))
# plt.savefig('./temp_figs/copula_show_xs.png')
pairplot_sample(ys, bins=np.arange(20))
# plt.savefig('./temp_figs/copula_show_ys_empirical.png')

# mu_hat, sigma_hat, xs_hat = fit_dmvln(ys, method_x='MC')
model_fa =  sklearn.decomposition.FactorAnalysis(n_components=3)
# model_fa.fit(xs_hat)

mu_hat, sigma_hat, xs_hat, model_fa = fit_dmvln(ys, method_x='fast', method_cov='fa', model=model_fa)
mu_hat, sigma_hat, xs_hat, model_fa = fit_dmvln(ys, method_x='MC', method_cov='fa', model=model_fa)
# print model_fa.components_
# print phi
# print model_fa.noise_variance_
# print b**2
pairplot_model(mu_hat, sigma_hat, bins=np.arange(20))
# plt.savefig('./temp_figs/copula_show_ys_fit.png')