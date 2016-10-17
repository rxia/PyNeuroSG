# exec(open("./temp_script.py").read())
# test matplitlib speed
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import PyNeuroPlot as pnp
import time

if 0:
    t=time.time()
    for i in range(100):
        temp=plt.subplot(10,10,i+1)
        temp=pnp.SignalPlot( np.arange(1000.0)/1000, np.random.rand(2000,1000,5)+i )
    print(time.time()-t);

if 0:
    t=time.time()
    h_f = plt.figure()
    for i in range(100):
        temp = h_f.add_subplot(10,10,i+1)
        temp = pnp.SignalPlot( np.arange(1000.0)/1000, np.random.rand(2000,1000,5)+i )
    print(time.time()-t)

if 1:
    t=time.time()
    [hf, hs] = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            plt.axes(hs[i,j])
            temp = pnp.SignalPlot( np.arange(1000.0)/1000, np.random.rand(2000,1000,5)+i )
    print(time.time()-t)