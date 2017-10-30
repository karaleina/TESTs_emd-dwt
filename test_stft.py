from scipy import signal
import matplotlib.pyplot as plt
import h5py
import numpy as np

# READING DATA
filename_read = "data/breathing/camera/movements/K7.h5"
with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]


# CALCULATING
s = np.array(all.ravel())
fs = 33
f, t, Zxx = signal.stft(s, fs, nperseg=30)

# PRINTING
print(np.shape(Zxx))
print(t)
print(f)
N = int(len(f)/8)
plt.pcolormesh(t, f[0:], np.abs(Zxx[:,:]), vmin=0, vmax=np.max(f))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()