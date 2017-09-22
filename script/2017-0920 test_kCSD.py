import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import robust_csd as csdr
import PyNeuroPlot as pnp
from pykCSD.pykCSD import KCSD   # use kCSD package, which only supports python2


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


""" record using a hypothetical electrode array """
spacing = 0.05
probe_loc = np.arange(-0.5, 0.5, spacing)
lfp_bar = np.array([fun_V(xx_cur, II=II, ii=ii) for xx_cur in probe_loc])
lfp_noi = lfp_bar * (np.random.randn(len(probe_loc))*(np.max(np.abs(lfp_bar)))/50+1)

def csd_est_using_kCSD(probe_loc, lfp_bar, lambd=0.0, sigma = sigma):
    elec_pos = np.expand_dims(probe_loc, axis=1)
    pots = np.expand_dims(lfp_bar, axis=1)
    params = {'gdX': np.diff(probe_loc)[0], 'gdY': np.diff(probe_loc)[0], sigma:sigma, lambd:lambd }

    k = KCSD(elec_pos, pots, params)
    k.solver.lambd = lambd
    k.estimate_pots()
    k.estimate_csd()
    return k.solver.space_X.ravel(), k.solver.estimated_csd.ravel()*sigma


# probe_loc_kcsd, csd_knl = csd_est_using_kCSD(probe_loc, lfp_noi, sigma = 1)
# plt.plot(probe_loc_kcsd, csd_knl)

""" smoothness """
num_smoothness = 11
list_lambda_der = np.linspace(0, 0.5, num=num_smoothness, endpoint=True)
list_ker_width = np.linspace(0, 1, num=num_smoothness, endpoint=True)
list_lambda_knl = np.linspace(0, 0.005, num=num_smoothness, endpoint=True)

h_fig, h_ax = plt.subplots(3,1, sharex=True, figsize=[4,6])
plt.axes(h_ax[0])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.title('CSD robust est')
plt.axes(h_ax[1])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.title('CSD gaussian smooth est')
plt.axes(h_ax[2])
plt.fill_between(xx, SS, edgecolor='k', label='actual')
plt.title('CSD kernel est')

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

    plt.axes(h_ax[2])
    lambda_knl_cur = list_lambda_knl[i_cur]
    probe_loc_kcsd, csd_knl = csd_est_using_kCSD(probe_loc, lfp_noi, lambd=lambda_knl_cur)
    plt.plot(probe_loc_kcsd, csd_knl, '.-', color=np.ones(3, dtype=float) * (1.0 * i_cur / num_smoothness),
             label='kernel est')

plt.xlim([-0.8,0.8])
plt.ylim([-np.max(np.abs(SS))*1.2, np.max(np.abs(SS))*1.2])
plt.savefig('./temp_figs/CSD_estimate_robust_vs_gaussian_vs_kernel.png')


