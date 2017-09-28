""" neural point process, GLM """
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import PyNeuroAna as pna
import PyNeuroPlot as pnp


def cal_P_I(s_type, ts):
    """
    Use as gamma function to define a inpulse response function for external stimuli

    P(x_{i,t}=1) = P_I_t + P_intrinsic{i, past(t)} + P_external{\i, past(t)}

    """
    k = s_type+1
    theta = 1.0/10
    I = pna.gen_gamma_knl(ts, k=k, theta=theta)
    if len(np.array(s_type).shape)>=2:
        I = I/np.max(I, axis=1, keepdims=True)
    else:
        I = I / np.max(I)
    return I



def gen_AR_realization(ts, P_I, AR_bump):
    """ generate AR process spikes """
    P = P_I
    X = np.zeros(P_I.shape)
    [N, T] = P_I.shape
    len_AR_bump = len(AR_bump)
    h_len_AR_bump = int((len_AR_bump-1)/2)
    for i, t in enumerate(ts):
        X[:, i] = gen_spk(P[:,i])
        P[:, np.arange(i - h_len_AR_bump, i - h_len_AR_bump + len_AR_bump) % T] \
            += X[:, np.array([i]) % T] * np.expand_dims(AR_bump, axis=0)
    return X



def gen_spk(P):
    """
    Generate spike using Bernoulli distribution on probability P
    """
    e_P = np.exp(P)
    return e_P/(1+e_P) > np.random.rand(*P.shape)


""" generate synthetic data """
M = 2   # number of neurons
N = 100 # number of trials
S = 1   # number of stimulus types
t_interval = 0.001
ts = np.arange(-0.2, 1.0, t_interval)   # time stamps

s = np.random.randint(S, size=N)
P_I = cal_P_I( np.expand_dims(s, 1), ts )/10

ts_AR_bump = np.arange(-50,50+1)*0.001
AR_bump = -pna.gen_gamma_knl(ts_AR_bump, k=1, theta=0.005, normalize='sum') \
          +pna.gen_gamma_knl(ts_AR_bump, k=2, theta=0.005, normalize='sum')


lambda_I = 30*P_I - 5
# X = gen_spk(P_I + 0.05)
X = gen_AR_realization(ts, lambda_I, 30*AR_bump)
(ISI, ISI_hist, ISI_hist_ts) = pna.cal_ISI(X, ts=ts)
(STA, t_STA, _) = pna.cal_STA(X, ts=ts, t_window=[-0.050, 0.050], zero_point_zero=True)


plt.figure()
plt.subplot2grid([2,2],[0,0])
pnp.PsthPlot(X, ts=ts, sk_std=0.01)
plt.title('PSTH')
plt.subplot2grid([2,2],[0,1])
plt.bar(ISI_hist_ts, ISI_hist, width=ts[1]-ts[0], align='center', color='k')
plt.title('ISI')
plt.subplot2grid([2,2],[1,1])
plt.plot(t_STA, STA, color='k')
plt.title('STA')

delta_stim_on = pna.gen_delta_function_with_label(ts=ts, t=np.zeros(N))
bases_stim_on = pna.gen_knl_series(ts=np.arange(0, 0.4, 0.001), scale=0.2, N=5, tf_symmetry=True)
bases_internal= pna.gen_knl_series(ts=np.arange(0, 0.1, 0.001), scale=0.2, N=5, tf_symmetry=True)
reg_X = [delta_stim_on, X]
reg_knls = [bases_stim_on, bases_internal]
reg_Y = X

reg = pna.fit_neural_point_process(Y=X,  Xs=reg_X, Xs_knls=reg_knls)

plt.subplot2grid([2,2],[1,0])
plt.plot( np.mean((bases_stim_on *np.expand_dims( np.squeeze(reg.coef_)[:5], axis=1)), axis=0))
plt.plot( np.mean((bases_internal*np.expand_dims( np.squeeze(reg.coef_)[5:], axis=1)), axis=0))

