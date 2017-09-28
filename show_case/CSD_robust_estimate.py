""" show case code for rubust CSD analysis """

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import robust_csd as csdr
import PyNeuroPlot as pnp

N = 16
T = 100
f = 1000.0
IED = 1.0
t = np.arange(T)/f
z = np.arange(N)*IED
sigma = 1.0


""" CSD profile: line source """
sigma = 1.0
x_lim=[-1.0, +1.0]
grid_size=0.001

# define line source
def fun_S(x):
    return 3.0*(x<=0.02)*(x>0) - 1.0*(x>=-0.06)*(x<0)
    # return (1.0-abs(x))*(abs(x)<=1.0)
    # return np.exp(-x**2/(0.2**2))

def fun_I(i):
    ss = np.arange(x_lim[0], x_lim[1], grid_size) + + grid_size/2
    return  np.sum(fun_S(ss)*np.sign(i-ss)/(i-ss)**2) / (4*np.pi*sigma) * grid_size

def fun_V(x, II=None, ii=None):
    if II is None:
        ii = np.arange(x_lim[0], x_lim[1], grid_size) + grid_size/2
        II = np.array([fun_I(ii_cur) for ii_cur in ii])
    return  np.sum(II[ii<=x]) / (-sigma) * grid_size

# plot
xx = np.arange(x_lim[0], x_lim[1], grid_size)
SS = fun_S(xx)
# II = np.array([fun_I(xx_cur) for xx_cur in xx])

II = np.array([fun_I(xx_cur) for xx_cur in xx])
ii = np.arange(x_lim[0], x_lim[1], grid_size) + grid_size/2
VV = np.array([fun_V(xx_cur, II=II, ii=ii) for xx_cur in xx])
# VV = np.array([fun_V(xx_cur) for xx_cur in xx])
h_fig, h_ax = plt.subplots(3,1, sharex=True)
plt.axes(h_ax[0])
plt.fill_between(xx, SS)
plt.axes(h_ax[1])
plt.fill_between(xx, II)
plt.axes(h_ax[2])
plt.fill_between(xx, VV)
plt.xlim([-1,1])




""" CSD profile: cylinder source """
sigma = 0.02
x_lim=[-5.0, +5.0]
grid_size=0.001
R = 0.4

# define line source
# type_source = 'rect'
type_source = 'rbf'

def fun_S(x):
    if type_source == 'rect':
        return 1.0*(x<=0.5)*(x>0) - 2.0*(x>=-0.25)*(x<0)
    elif type_source == 'rbf':
        return 1.0*np.exp(-(x-0.3)**2/(0.2**2)) - 2.0*np.exp(-(x+0.15)**2/(0.1**2))

def fun_I(i):
    ss = np.arange(x_lim[0], x_lim[1], grid_size) + + grid_size/2
    return  np.sum(fun_S(ss)*np.sign(i-ss)*( 1 - np.abs(i-ss)/np.sqrt( (i-ss)**2 + R**2 ) )) * grid_size / 2

def fun_V(x, II=None, ii=None):
    if II is None:
        ii = np.arange(x_lim[0], x_lim[1], grid_size) + grid_size/2
        II = np.array([fun_I(ii_cur) for ii_cur in ii])
    return  np.sum(II[ii<=x]) / (-sigma) * grid_size

# plot
xx = np.arange(x_lim[0], x_lim[1], grid_size)
SS = fun_S(xx)
# II = np.array([fun_I(xx_cur) for xx_cur in xx])

II = np.array([fun_I(xx_cur) for xx_cur in xx])
ii = np.arange(x_lim[0], x_lim[1], grid_size) + grid_size/2
VV = np.array([fun_V(xx_cur, II=II, ii=ii) for xx_cur in xx])
# VV = np.array([fun_V(xx_cur) for xx_cur in xx])
h_fig, h_ax = plt.subplots(3,1, sharex=True)
plt.axes(h_ax[0])
plt.fill_between(xx, SS, edgecolor='k')
plt.ylabel('source/sink')
plt.axes(h_ax[1])
plt.fill_between(xx, II, edgecolor='k')
plt.ylabel('current')
plt.axes(h_ax[2])
plt.fill_between(xx, VV, edgecolor='k')
plt.ylabel('potential')
plt.xlim([-3,3])
plt.suptitle('theoretical CSD profile, cylinder model, R={}'.format(R))
# plt.savefig('.temp_figs/CSD_theoretical_cylinder_R_{}.png'.format(R))


""" record using a linear probe and estimate """
spacing = 0.05
probe_loc = np.arange(-0.5, 0.5, spacing)
lfp_bar = np.array([fun_V(xx_cur, II=II, ii=ii) for xx_cur in probe_loc])
lfp_noi = lfp_bar * (np.random.randn(len(probe_loc))*(np.max(np.abs(lfp_bar)))/50+1)
csd_thr = csdr.cal_robust_csd(lfp_bar, lambda_der=0.0, spacing=spacing)*sigma
csd_nai = csdr.cal_robust_csd(lfp_noi, lambda_der=0.0, spacing=spacing)*sigma
csd_5pt = np.convolve(lfp_noi, [0.23,0.08,-0.62,0.08,0.23], mode='same')*(-sigma)/spacing**2
csd_smc = sp.ndimage.gaussian_filter1d(csd_nai, 1.0)
csd_sml = csdr.cal_robust_csd(sp.ndimage.gaussian_filter1d(lfp_noi, 1.0), lambda_der=0.0, spacing=spacing)*sigma
csd_rob = csdr.cal_robust_csd(lfp_noi, lambda_der=0.4, spacing=spacing)*sigma



h_fig, h_ax = plt.subplots(3,2, sharex=True, sharey=True)
h_ax = np.ravel(h_ax)
plt.axes(h_ax[0])
plt.fill_between(xx, VV, edgecolor='k', label='true')
plt.plot(probe_loc, lfp_noi, 'bo', label='measured')
plt.title('LFP')
plt.axes(h_ax[1])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.plot(probe_loc, csd_thr, 'k.-', label='theoretical')
plt.title('no noise ideal est')
plt.axes(h_ax[2])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.plot(probe_loc, csd_nai, 'b.--', label='naive est')
plt.title('naive 3-point est')
plt.axes(h_ax[3])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.plot(probe_loc, csd_5pt, 'c.--', label='five-point est')
plt.title('5-point est')
plt.axes(h_ax[4])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.plot(probe_loc, csd_smc, 'y.-', label='smooth_csd est')
plt.plot(probe_loc, csd_sml, 'b.--', label='smooth_lfp est')
plt.title('gaussian smooth est')
plt.axes(h_ax[5])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.plot(probe_loc, csd_rob, 'g.-',  label='robust est')
plt.title('robust est')
plt.xlim([-1.5,1.5])
plt.ylim([-np.max(np.abs(SS))*1.2, np.max(np.abs(SS))*1.2])
plt.savefig('./temp_figs/CSD_estimate.png')


""" different parameters for gaussian smooth and robust est """
h_fig, h_ax = plt.subplots(2,1, sharex=True)
num_smoothness = 11
list_lambda_der = np.linspace(0, 0.5, num=num_smoothness, endpoint=True)
list_ker_width = np.linspace(0, 1, num=num_smoothness, endpoint=True)
plt.axes(h_ax[0])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.title('CSD robust est')
plt.axes(h_ax[1])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.title('CSD gaussian smooth est')
for i_cur in np.arange(num_smoothness):
    plt.axes(h_ax[0])
    lambda_der_cur =  list_lambda_der[i_cur]
    csd_rob_cur = csdr.cal_robust_csd(lfp_noi, lambda_der=lambda_der_cur, spacing=spacing)*sigma
    plt.plot(probe_loc, csd_rob_cur, '.-', color=np.ones(3, dtype=float)*(1.0*i_cur/num_smoothness), label='robust est')
    plt.axes(h_ax[1])
    ker_width_cur = list_ker_width[i_cur]
    csd_sml_cur = csdr.cal_robust_csd(sp.ndimage.gaussian_filter1d(lfp_noi, ker_width_cur), lambda_der=0.0, spacing=spacing) * sigma
    plt.plot(probe_loc, csd_sml_cur, '.-', color=np.ones(3, dtype=float) * (1.0 * i_cur / num_smoothness),
             label='robust est')
plt.xlim([-0.8,0.8])
plt.ylim([-np.max(np.abs(SS))*1.2, np.max(np.abs(SS))*1.2])
plt.savefig('./temp_figs/CSD_estimate_gaussian_smooth_vs_robust_est.png')