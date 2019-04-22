import numpy as np
import matplotlib.pyplot as plt
import ewtpy

T = 1000
t = np.arange(1,T+1)/T
f = np.cos(2*np.pi*0.8*t) + 2*np.cos(2*np.pi*10*t)+0.8*np.cos(2*np.pi*100*t)
ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 3)
plt.plot(f)
plt.plot(ewt)
plt.show()