import pywt
import matplotlib.pyplot as plt
import numpy as np
import h5py

# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
# coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
# plt.matshow(coef)
# plt.show()

#
# import pywt
# import numpy as np
# import matplotlib.pyplot as plt

filename_read = "data/breathing/camera/movements/K5.h5"

with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
widths = np.arange(1, 31)

#wavelet = pywt.Wavelet('db1')
#cwtmatr, freqs = pywt.cwt(s, widths, "shan")
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

#
# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)

plt.figure(3)
plt.plot(s)

plt.figure(2)
coef, freqs=pywt.cwt(s,widths,'gaus1')
plt.matshow(coef)
print(pywt.wavelist(kind='continuous'))

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(coef[0])
plt.subplot(2,1,2)
plt.plot(coef[len(coef)-1])
plt.show()
