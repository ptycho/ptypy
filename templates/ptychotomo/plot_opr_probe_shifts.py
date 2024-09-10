import numpy as np
import matplotlib.pyplot as plt
import sys

# Ground truth probe
#gtp = 'g_truth.npy'
gtp = 'diamond_probe.npy'

# Job ID
id = sys.argv[1]

# Copied from ptypy.utils.scripts
def mass_center(A):
    """
    Calculates mass center of n-dimensional array `A`.
    """
    axes = tuple(range(1, A.ndim + 1))
    return np.sum(A * np.indices(A.shape), axis=axes, dtype=float) / np.sum(A, dtype=float)

# Compute reference probe centre
pg = np.load(gtp)
ref_center = mass_center(np.abs(pg))

# Compute OPR probe shifts and save to file
ps = np.load('opr_probes_'+id+'.npy')
nps = ps.shape[0]
opr_shifts = np.zeros((nps,2))
for i in range(nps):
    opr_shifts[i] = mass_center(np.abs(ps[i])) - ref_center
np.savetxt('opr_probe_shifts_'+id+'.txt', opr_shifts, fmt='%+.2e')

# Load actual probe shifts
act_shifts = np.zeros((nps,2))
f = open('probe_shifts.txt','r')
i = 0
for line in f.readlines():
    if 'probe' in line:
        shx, shy = line.split('(')[1].split(')')[0].split(',')
        act_shifts[i,0] = float(shx)
        act_shifts[i,1] = float(shy)
        i += 1
f.close()

# Plot actual and OPR probe shifts
x = np.arange(nps)
plt.subplot(1,2,1)
plt.grid()
plt.scatter(x, act_shifts[:,0])
plt.scatter(x, opr_shifts[:,0])
plt.subplot(1,2,2)
plt.grid()
plt.scatter(x, act_shifts[:,1])
plt.scatter(x, opr_shifts[:,1])
plt.savefig('opr_probe_shifts_'+id+'.png')
