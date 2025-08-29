from ptypy.utils.metrics import fouriershellcorrelation, fourierringcorrelation, frc_plot
import numpy as np
import matplotlib.pyplot as plt

g_truth = np.load("/home/kpv14943/ptypy/rmap_242_glass.npy")[50:-20, 70:-70, 70:-70]
vol_joint = np.load("/home/kpv14943/ptypy/vol_200iters_subs_half.npy")[50:-20, 70:-70, 70:-70]
vol_conv = np.load("/home/kpv14943/ptypy/overall_vol_conv_REAL_400_subsampled_half.npy")[50:-20, 70:-70, 70:-70]

#FSC, T, fn = fourierringcorrelation(np.real(g_truth)[:, 100, :], np.real(vol_joint)[:, 100, :])
FSC_joint, _, fn = fouriershellcorrelation(np.real(g_truth), np.real(vol_joint))
# frc_plot(FSC, T, fn, 'z_joint_subs_half.png')

FSC_conv, _, _ = fouriershellcorrelation(np.real(g_truth), np.real(vol_conv))
# frc_plot(FSC, T, fn, 'z_conv_subs_half.png')


plt.figure()
plt.clf()
plt.plot(fn, FSC_joint.real, label="FRC joint")
plt.plot(fn, FSC_conv.real, label="FRC conv")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.xlabel("Spatial frequency/Nyquist [normalized units]")
plt.ylabel("Magnitude [normalized units]")
plt.show()
plt.savefig('FRC.png')
