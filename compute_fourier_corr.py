from ptypy.utils.metrics import fouriershellcorrelation, fourierringcorrelation, frc_plot
import numpy as np

g_truth = np.load("/home/kpv14943/ptypy/rmap_242_glass.npy")[30:-30, 60:-60, 60:-60]
vol_joint = np.load("/home/kpv14943/ptypy/vol_200iters_subs_half.npy")[30:-30, 60:-60, 60:-60]
vol_conv = np.load("/home/kpv14943/ptypy/overall_vol_conv_REAL_400_subsampled_half.npy")[30:-30, 60:-60, 60:-60]

#FSC, T, fn = fourierringcorrelation(np.real(g_truth)[:, 100, :], np.real(vol_joint)[:, 100, :])
FSC, T, fn = fouriershellcorrelation(np.real(g_truth), np.real(vol_joint))
frc_plot(FSC, T, fn, 'z_joint_subs_half.png')

FSC, T, fn = fouriershellcorrelation(np.real(g_truth), np.real(vol_conv))
frc_plot(FSC, T, fn, 'z_conv_subs_half.png')