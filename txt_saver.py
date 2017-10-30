import h5py
import numpy as np

for k in [5,6,7,8]:
    filename_read = "data/breathing/camera/movements_ref/oddech_w_ruchu_ref_K_" + str(k) + ".h5"
    with h5py.File(filename_read, 'r') as hf:
        s = hf['all'][:]
    start = 200

    f = h5py.File(filename_read, 'r')
    np.savetxt('txt_files/K_ref' + str(k) +'.txt', f['all'][start:len(s) - 500])
    f.close()
