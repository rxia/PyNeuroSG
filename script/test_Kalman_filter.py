import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt



dta = sm.datasets.webuse('lutkepohl2', 'http://www.stata-press.com/data/r12/')
dta.index = dta.qtr
endog = dta.ix['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]


exog = endog['dln_consump']
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2,0), trend='nc', exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
