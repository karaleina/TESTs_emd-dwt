import pywt
import h5py
import numpy as np
import math
from matplotlib import pyplot as plt

filename_read = "data/breathing/camera/movements/K5.h5"
with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]
s = np.array(all.ravel())
print(len(s))


desired_level = 5
desired_samples = int(math.pow(2, desired_level))
for i in range(desired_samples):
    if len(s)%desired_samples == i:
        s = s[0:len(s)-i]

print(len(s))
coeffs = pywt.swt(s, "db6", level=desired_level)
print(len(coeffs))

plt.figure(1)
plt.plot(s)

for i in range(len(coeffs)):
    plt.figure(0)
    plt.subplot(len(coeffs), 2, i*2+1)
    plt.plot(coeffs[i][0])
    plt.subplot(len(coeffs), 2, i*2+2)
    plt.plot(coeffs[i][1])

plt.show()