# -*- coding: utf-8 -*-
import os
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.ndimage.interpolation import rotate
from utils.propagators import angular_spectrum
import tkFileDialog as filedialog
from Tkinter import Tk

from ptypy import io
from ptypy import utils as u
from ptypy.core import Base, Container

# Parameters
save = True
dialog = False
dpi = 200

pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir))
save_dir = pardir + "\\plots\\"

base_server = "Y:\AG_HHG\Data Analysis Tools\ptychography\Reconstruction\data\\"

base = pardir + "\\owncloud\\reconstruction\\src\\simulation\data_sstar_drift_alt\\dumps\\"
base = pardir + "\\owncloud\\reconstruction\\data\\2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\\dumps\\"

filepath = "512x512_freddy_prepro_pos_corr\\512x512_freddy_prepro_pos_corr_ML_0500.ptyr"
filepath = "no_cc_pos_corr_512x512_no_seed_prepro_freddy\\no_cc_pos_corr_512x512_no_seed_prepro_freddy_dm_1300.ptyr"

filepath = base + filepath

# 7um scan
# filepath = base_server + "\\2017_08_25\\sstar_center_6um_pinhole_7um_extent_bs\\dumps\\cc_no_pos_corr_512x512_high_thres_ps\\cc_no_pos_corr_512x512_high_thres_ps_DM_4950.ptyr"
# filepath = base_server + "\\2017_08_25\\sstar_center_6um_pinhole_7um_extent_bs\\dumps\\cc_no_pos_corr_512x512_high_thres\\cc_no_pos_corr_512x512_high_thres_DM_4950.ptyr"
# filepath = base_server + "\\2017_08_25\\sstar_center_6um_pinhole_7um_extent_bs\\dumps\\512x512_freddy_prepro\\512x512_freddy_prepro_DM_0300.ptyr"

# bs less pos 512x512
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_stiching_range\\512x512_cc_new_stiching_range_ML_0740.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_stiching_ps_bigger\\512x512_cc_new_stiching_ps_bigger_ML_0520.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_stiching_obj_smooth\\512x512_cc_new_stiching_obj_smooth_ML_0510.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_stiching_f_relax_fac\\512x512_cc_new_stiching_f_relax_fac_ML_0590.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_stiching_new_param\\512x512_cc_new_stiching_new_param_ML_0740.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_param_high_offset\\512x512_cc_new_param_high_offset_ML_0170.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_param_low_offset\\512x512_cc_new_param_low_offset_ML_0200.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\512x512_cc_new_param_medium_offset\\512x512_cc_new_param_medium_offset_ML_0200.ptyr"
# multi mode reconstructions:
filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\no_cc_pos_corr_512x512_no_seed\\no_cc_pos_corr_512x512_no_seed_ML_0650.ptyr"
filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\no_cc_pos_corr_512x512_no_seed_multi_mode_v1\\no_cc_pos_corr_512x512_no_seed_multi_mode_v1_dm_1300.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\no_cc_pos_corr_512x512_no_seed_multi_mode_v2\\no_cc_pos_corr_512x512_no_seed_multi_mode_v2_ML_0250.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\no_cc_pos_corr_512x512_no_seed_multi_mode_v3\\no_cc_pos_corr_512x512_no_seed_multi_mode_v3_DM_2950.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\no_cc_pos_corr_512x512_no_seed_multi_mode_v4\\no_cc_pos_corr_512x512_no_seed_multi_mode_v4_DM_2950.ptyr"


# 1024x1024
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_smooths_std_no_random\\1024x1024_smooths_std_no_random_ML_0740.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc\\1024x1024_cc_DM_1990.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_ps_big\\1024x1024_cc_ps_big_ML_0740.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_new_stiching\\1024x1024_cc_new_stiching_ML_0700.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_new_stiching_new_param\\1024x1024_cc_new_stiching_new_param_ML_0700.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_new_param_medium_offset\\1024x1024_cc_new_param_medium_offset_ML_0740.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_new_param_medium_offset_more_it\\1024x1024_cc_new_param_medium_offset_more_it_ML_1400.ptyr"
# filepath = base_server + "2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\dumps\\1024x1024_cc_new_param_medium_offset_no_smooth_more_it\\1024x1024_cc_new_param_medium_offset_no_smooth_more_it_ML_1300.ptyr"

# sstar with 3um pinhole
# filepath = base_server + "2017_09_28\sstar_3um_pinhole_no_HDR\dumps\\512x512_cc_new_param_medium_offset\\512x512_cc_new_param_medium_offset_ML_0700.ptyr"
# filepath = base_server + "2017_09_28\sstar_3um_pinhole_no_HDR\dumps\\350x350\\350x350_ML_0400.ptyr"

# grating with 3um pinhole
# filepath = base_server + "2017_09_28\\100nm_lines_50_pts_3um_pinhole_1um_step_HDR\\dumps\\512x512\\512x512_DM_0050.ptyr"


class ptry_reader():
    # Physical parameters
    distance = 28.e-3
    psize = 13.5e-6
    p_num = 1024
    lam = 18.1e-9

    # Plot parameters
    save = False
    font_size_label = 15
    font_size_ticks = 10

    def __init__(self, filepath, normalize_phase=True):

        if filepath is None:
            self.filepath = self.load_file()
        else:
            self.filepath = filepath

        # self.print_info()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mpl.rcParams["savefig.directory"] = os.path.dirname(save_dir)

        self.filename = os.path.split(filepath)[1]
        data = io.h5read(filepath)
        obj = data["content"]["obj"]["S00G00"]["data"]
        obj = obj[0, :, :]
        probe = data["content"]["probe"]["S00G00"]["data"]
        psize = data["content"]["probe"]["S00G00"]["_psize"]
        self.psize = psize

        # Decelerate object and probe instances
        self.obj = obj
        self.obj_int = (np.abs(obj))**2
        self.obj_phase = np.angle(obj)
        self.probe = probe
        self.probe_phase = np.angle(probe)
        self.probe_int = np.abs(probe)**2
        self.probe = probe
        self.p_num_probe = probe.shape

        # self.obj = np.copy(obj)
        self.p_num = self.obj_int.shape

        self.num_probes = probe.shape[0]
        self.num_probes = probe.shape[0]

        self.extent_y_obj = psize[0] * self.p_num[0]
        self.extent_x_obj = psize[1] * self.p_num[1]

        self.extent_y_probe = psize[0] * self.p_num_probe[1]
        self.extent_x_probe = psize[1] * self.p_num_probe[2]

        if normalize_phase:
            center_phase = self.obj_phase[self.p_num[0] / 2, self.p_num[1] / 2]
            self.obj_phase = np.angle(obj * np.exp(1j*(-1)*center_phase))

    def obj_cplot(self, fignum=0):
        P = Base()
        P.CType = np.complex128
        P.FType = np.float64

        P.obj = Container(P, "obj", data_type="complex")
        obj_storage = P.obj.new_storage(data=self.obj, psize=self.psize)
        u.plot_storage(obj_storage, fignum=fignum)

    def probe_cplot(self, fignum=0):
        P = Base()
        P.CType = np.complex128
        P.FType = np.float64

        P.obj = Container(P, "obj", data_type="complex")
        obj_storage = P.obj.new_storage(data=self.probe, psize=self.psize)
        u.plot_storage(obj_storage, fignum=fignum)

    def save_probe_mat(self, filename):
        print("Save probe at ", os.getcwd() + "\\" + filename)
        savemat(filename, {"probe": self.probe})

    def save_obj_mat(self,filename):
        print("Save obj at ", os.getcwd() + "\\" + filename)
        savemat(filename, {"obj": self.obj})

    def print_info(self):
        print(io.h5info(filepath))

    def plot_obj(self, imag=False, E_field=False, interpolation="none"):

        self.fig_lin_obj, self.ax_lin_obj = plt.subplots()
        self.ax_lin_obj.set_title("Object intensity lin scale")

        img = self.ax_lin_obj.imshow(self.obj_int, cmap="hot",
                                     interpolation=interpolation, vmin=0, vmax=1.)
        # img = self.ax_lin_obj.imshow(self.obj_int, extent=(-1/2.*self.extent_y_obj*1e6,1/2.* self.extent_y_obj*1e6,
        #                                                    -1/2.*self.extent_x_obj*1e6, 1/2.* self.extent_x_obj*1e6),
        #                              cmap="hot", interpolation=interpolation, vmin=0, vmax=1.0)
        cbar = self.fig_lin_obj.colorbar(img)
        cbar.set_label("Intensity", size=self.font_size_ticks)
        cbar.ax.tick_params(labelsize=self.font_size_ticks)
        if self.save:
            self.fig_lin_obj.savefig(save_dir + "obj_" + self.filename + ".png", dpi=dpi)

        plt.figure("Obj log scale")
        plt.imshow(np.log10(self.obj_int), interpolation="none", vmax=0.5, vmin=-3)
        plt.colorbar()
        if self.save:
            plt.savefig(save_dir + "obj_log_" + self.filename + ".png", dpi=dpi)

        plt.figure("Obj phase")
        plt.imshow(self.obj_phase, interpolation="none", extent=(-1/2.*self.extent_y_obj*1e6, 1/2.*self.extent_y_obj*1e6,
                                                                 -1/2.*self.extent_x_obj*1e6,
                                                                 1/2.*self.extent_x_obj*1e6))
        plt.colorbar()
        plt.xlabel(u"x-Position (µm)", fontsize=22)
        plt.ylabel(u"y-Position (µm)", fontsize=22)
        if self.save:
            plt.savefig(save_dir + "obj_phase_" + self.filename + ".png", dpi=dpi)

        if imag:
            plt.figure("Object imaginary part")
            plt.imshow(np.imag(self.obj), interpolation="none")
            plt.colorbar()

        if E_field:
            plt.figure("E_field")
            plt.imshow(np.real(self.obj), interpolation="none")
            plt.colorbar()

    def plot_probe(self, lineout=False, logplot=False):

        for i in range(self.num_probes):
            plt.figure("Probe " + str(i) +" lin scale intensity")
            plt.title("Probe lin. scale int.")
            plt.imshow(self.probe_int[i, :, :], interpolation="none",
                       extent=(-1 / 2. * self.extent_y_probe * 1e6, 1 / 2. * self.extent_y_probe * 1e6,
                               -1 / 2. * self.extent_x_probe * 1e6, 1 / 2. * self.extent_x_probe * 1e6)
                       , cmap="hot")
            plt.colorbar()
            if self.save:
                plt.savefig(save_dir + "probe_" + str(i) + "_" + self.filename + ".png", dpi=dpi)

            if logplot:
                plt.figure("Probe " + str(i) + " log scale intensity")
                plt.title("Probe log. scale int.")
                plt.imshow(np.log10(self.probe_int[i, :, :]), interpolation="none", vmin=0)
                plt.colorbar()
                if self.save:
                    plt.savefig(save_dir + "log_probe_" + str(i) + "_" + self.filename + ".png", dpi=dpi)

            plt.figure("Probe phase_" + str(i))
            plt.title("Probe phase")
            plt.imshow(self.probe_phase[i, :, :], interpolation="none",
                       extent=(-1 / 2. * self.extent_y_probe * 1e6, 1 / 2. * self.extent_y_probe * 1e6,
                               -1 / 2. * self.extent_x_probe * 1e6, 1 / 2. * self.extent_x_probe * 1e6)
                       )
            plt.colorbar()
            if self.save:
                plt.savefig(save_dir + "probe_phase_" + self.filename + ".png", dpi=dpi)

            if lineout is not False:
                shape = self.probe.shape
                lineout = np.abs(self.probe[i, shape[1]/2, :])**2
                plt.figure("lineout")
                plt.plot(lineout)

    @staticmethod
    def save_as_mat(filename, data_dict):
        savemat(filename, data_dict)

    def plot_vertical_line(self, x, ystart, yend):
        lineout = self.obj_int[ystart: yend, x]
        lineout = lineout - np.min(lineout)
        # lineout = lineout[::-1]
        xmin = 0
        xmax = lineout.shape[0] - 1
        x_scaled = np.linspace(xmin, xmax, lineout.shape[0])
        x_scaled *= self.psize[0]

        knife_edge = True
        if knife_edge:
            self.fit_error_func(x_scaled, lineout)
            self.interpolate(x_scaled, lineout)
            plt.figure("line")
            plt.plot(x_scaled, lineout, marker="x", linestyle="None", markersize=12)
        else:
            plt.figure("line")
            plt.plot(lineout, marker="x")

        self.ax_lin_obj.vlines(x, ystart, yend, color="green", lw=2)

    def plot_horizontal_line(self, y, xstart, xend):
        lineout = self.obj_int[y, xstart: xend]
        lineout = lineout - np.min(lineout)
        lineout = lineout[::-1]
        xmin = 0
        xmax = lineout.shape[0] - 1
        x = np.linspace(xmin, xmax, lineout.shape[0])
        x *= self.psize[0]

        knife_edge = True

        if knife_edge:
            self.fit_error_func(x, lineout)
            self.interpolate(x, lineout)
            plt.figure("line")
            plt.plot(x, lineout, marker="x", linestyle="None", markersize=12)
        else:
            plt.figure("line")
            plt.plot(x, lineout, marker="x")

        self.ax_lin_obj.hlines(y, xstart, xend, color="green", lw=2)

    def interpolate(self, x, line):
        from scipy.interpolate import interp1d
        f = interp1d(x, line, kind="cubic")
        x_fit = np.linspace(np.min(x), np.max(x), 1000)
        plt.figure("line")
        plt.plot(x_fit, f(x_fit), label="Interpolated")

    def fit_error_func(self, x, line):
        from scipy import special
        from scipy.optimize import curve_fit
        def error_func(x, a, b, c, d):
            return a * special.erf((1 / d) * (x - c)) + b

        xmin = np.min(x)
        xmax = np.max(x)
        xfit = np.linspace(xmin, xmax, 200)
        popt, pcov = curve_fit(error_func, x, line, p0=[1, 0, self.psize[0], self.psize[0]])
        print(np.sqrt(np.diag(pcov)))
        plt.figure("line")
        # plt.plot(xfit, error_func(xfit, *popt))
        ymax = popt[1] + popt[0]
        plt.hlines([ymax*0.9, ymax*0.1], 0, xmax, colors="b", linestyle="--")

    @staticmethod
    def load_file():
        Tk().withdraw()
        filepath = filedialog.askopenfilename(filetypes=(("Ptyr files", "*.ptyr"), ("All files", "*.*")))
        return filepath

    def rotate_img(self, angle):
        self.obj_int = rotate(self.obj_int, angle)
        # self.obj_phase = rotate(self.obj_phase, angle)
        # self.probe_int= rotate(self.probe_int, angle)
        # self.probe_phase = rotate(self.probe_phase, angle)
        self.extent_y_obj = self.psize[0] * self.obj_int.shape[0]
        self.extent_x_obj = self.psize[1] * self.obj_int.shape[1]

    def set_roi_obj(self, size=50, xmin=None, xmax=None, ymin=None, ymax=None):

        if xmin == None or xmax == None or ymin == None or ymax == None:
            y_shape = self.p_num[0]
            x_shape = self.p_num[1]
            self.obj_int = self.obj_int[y_shape/2 - size: y_shape/2 + size, x_shape/2 - size: x_shape/2 + size]
        else:
            self.obj_int = self.obj_int[ymin:ymax, xmin:xmax]
            self.obj_phase = self.obj_phase[ymin:ymax, xmin:xmax]
        # self.probe_int = self.probe_int[ymin:ymax, xmin:xmax]
        # self.probe_phase = self.probe_phase[ymin:ymax, xmin:xmax]

    def roi_plot(self, roisize=[-2.5, 2.5], interpolation="none"):

        plt.figure("Obj int roi")
        plt.imshow(self.obj_int, interpolation=interpolation,
                   extent=(-0.5*self.extent_y_obj*1e6, 0.5*self.extent_y_obj*1e6, -0.5*self.extent_x_obj*1e6,
                           0.5*self.extent_x_obj*1e6), vmin=0, vmax=0.3, cmap="hot")
        plt.xlim(roisize)
        plt.ylim(roisize)
        plt.colorbar()

        if self.save:
            plt.savefig(save_dir  + "Obj_int_roi_" + self.filename + ".png", dpi=dpi)

        # plt.figure("Obj int roi log")
        # plt.imshow(np.log10(self.obj_int), interpolation="none", extent=(-0.5*self.extent_y_obj*1e6, 0.5*self.extent_y_obj*1e6, -0.5*self.extent_x_obj*1e6, 0.5*self.extent_x_obj*1e6))
        # plt.xlim(roisize)
        # plt.ylim(roisize)

    def normalize_obj(self, radius=None, value=None):
        """
        Normalizes the obj by the largest number in the center
        :return:
        """
        if value is None:
            (y, x) = np.ogrid[-1*self.p_num[0]/2 : self.p_num[0]/2, -1*self.p_num[1]/2 : self.p_num[1]/2]
            mask = y**2 + x**2 < radius**2
            print("Normalize object with: " + str(np.max(self.obj_int*mask)))
            self.obj_int = self.obj_int/np.max(self.obj_int*mask)
        else:
            self.obj_int = self.obj_int/value

    def ring_lineout(self, center=(0, 0), radius=0, save_mat=False, interpolate=False):
        """
        Draws a circular lineout at a given center with a given radius. It will zoom in the center by a factor of 4.

        :param center: Tuple, in pixel.
        :param radius: in pixel
        """
        # ROI
        from scipy.ndimage.interpolation import zoom

        roi_size = 150
        roi = self.obj_int[center[0] - roi_size / 2.:center[0] + roi_size / 2.,
              center[1] - roi_size / 2.:center[1] + roi_size / 2.]

        plt.figure("ROI")
        plt.imshow(roi, interpolation="none", cmap="hot")

        zoomed = zoom(roi, 4)
        fig, ax = plt.subplots()
        ax.imshow(zoomed, interpolation="none", cmap="hot", vmax=0.3)

        center = (zoomed.shape[0]/2., zoomed.shape[1]/2.)

        x = np.arange(center[1] - radius, center[1] + radius, 0.01)
        first_lineout = np.array([])
        x_coord_array = np.array([])
        y_coord_array = np.array([])
        # mask = np.zeros_like(self.obj_int)
        mask = np.zeros_like(zoomed)
        old_xcoord = 0
        old_ycoord = 0

        for i in x:
            y = np.sqrt(radius ** 2 - (i - center[1]) ** 2) + float(center[0])
            y_coord = round(y)
            x_coord = round(i)
            if old_xcoord == x_coord and old_ycoord == y_coord:
                continue
            old_ycoord = y_coord
            old_xcoord = x_coord
            mask[y_coord, x_coord] = 1
            first_lineout = np.append(first_lineout, zoomed[y_coord, x_coord])
            x_coord_array = np.append(x_coord_array, x_coord)
            y_coord_array = np.append(y_coord_array, y_coord)

        second_lineout = np.array([])

        for i in x:
            y = -1 * np.sqrt(radius ** 2 - (i - center[1]) ** 2) + float(center[0])
            y_coord = round(y)
            x_coord = round(i)
            if old_xcoord == x_coord and old_ycoord == y_coord:
                continue
            old_ycoord = y_coord
            old_xcoord = x_coord
            mask[y_coord, x_coord] = 1
            x_coord_array = np.append(x_coord_array, x_coord)
            y_coord_array = np.append(y_coord_array, y_coord)
            second_lineout = np.append(second_lineout, zoomed[y_coord, x_coord])

        lineout = np.append(first_lineout, second_lineout[::-1])

        # x_lineout = np.arange(0, lineout.shape[0]) / float(lineout.shape[0]) * (
        # 2. * np.pi * self.psize[0] * radius * 1e9)

        x_lineout = np.arange(0, lineout.shape[0]) / float(lineout.shape[0]) * (
            2. * np.pi * self.psize[0] * radius * 1e9/4.)

        if save_mat:
            savemat("circ_lineout.mat", {"circ_lineout": lineout})

        plt.figure("lineout")
        if interpolate:
            from scipy.interpolate import interp1d
            f = interp1d(x_lineout, lineout, kind="cubic")
            x_new = np.linspace(np.min(x_lineout), np.max(x_lineout), 1000)
            plt.plot(x_new, f(x_new), color="red")
            plt.plot(x_lineout, lineout, "x", color="red")
        else:
            plt.plot(x_lineout, lineout, color="red")

        plt.ylabel("Intensity (a.u.)")
        plt.xlabel("Distance [nm]")

        # self.ax_lin_obj.plot(x_coord_array, y_coord_array, "o", ms=1, color="green")
        ax.plot(x_coord_array, y_coord_array, "o", ms=1, color="green")

    def propagate_probe(self, z, plot=True):

        probe = self.probe[0, :, :]

        print("start propagation for distance: " + str(z))
        probe_shape = probe.shape
        x = (np.arange(0, probe_shape[1], 1) - probe_shape[1] / 2) * self.psize[1]
        y = (np.arange(0, probe_shape[0], 1) - probe_shape[0] / 2) * self.psize[0]

        E_prop = angular_spectrum(probe, x, y, z, self.lam)
        if plot:
            plt.figure("Ampl for distance: " + str(z))
            plt.imshow(np.abs(E_prop)**2, interpolation="none", cmap="hot")
            plt.colorbar()

            plt.figure("Log. Ampl for distance: " + str(z))
            plt.imshow(np.log10(np.abs(E_prop) ** 2), interpolation="none", vmin=0)
            plt.colorbar()

            plt.figure("Phase for distance:  " + str(z))
            plt.imshow(np.angle(E_prop), interpolation="none")
            plt.colorbar()

        return E_prop

    def propagation_scan(self, z_start, z_stop, steps):
        plot = False
        shape = np.array([self.probe.shape[1], self.probe.shape[2]])

        z = np.linspace(z_start, z_stop, steps)

        propagation_horizontal = np.zeros([shape[0], steps,])
        propagation_vertical = np.zeros([shape[1], steps])

        from scipy.ndimage import center_of_mass

        for i, dis in enumerate(z):
            E_prop = self.propagate_probe(dis, plot=False)
            com = center_of_mass(np.abs(E_prop))
            x_center = com[1]
            y_center = com[0]
            print("center of mass: " + str(com))

            propagation_horizontal[:, i] = np.abs(E_prop[y_center, :])**2
            propagation_vertical[:, i] = np.abs(E_prop[:, x_center])**2

            if plot:
                plt.figure("propagated ampl.")
                plt.imshow(np.abs(E_prop), interpolation="none")

                plt.figure("propagated phase")
                plt.imshow(np.angle(E_prop), interpolation="none")

                plt.figure("vertical lineout")
                plt.plot(propagation_vertical[:, i])

                plt.figure("horizontal lineout")
                plt.plot(propagation_horizontal[:, i])

        y_extent_start = -1.*shape[0]/2.*self.psize[0]*1e6
        y_extent_stop = shape[0]/2.*self.psize[1]*1e6
        plt.figure("propagated horizontal ")
        plt.imshow(propagation_horizontal, interpolation="none", aspect="auto", extent=(z_start*1e6, z_stop*1e6,
                                                                                        y_extent_start, y_extent_stop),
                   cmap="hot", vmin=-1)
        plt.xlabel("Distance to the sample in um")
        plt.colorbar()

        y_extent_start = -1.*shape[1]/2.*self.psize[0]*1e6
        y_extent_stop = shape[1]/2.*self.psize[1]*1e6
        plt.figure("propagated vertical ")
        plt.imshow(propagation_vertical, interpolation="none", aspect="auto", extent=(z_start*1e6, z_stop*1e6,
                                                                                        y_extent_start, y_extent_stop),
                   cmap="hot", vmin=-1)
        plt.xlabel("Distance to the sample in um")
        plt.colorbar()

    def probe_lineout(self, vertical=True, horizontal=True):

        shape = self.probe[0, :, :].shape

        plt.figure("Amplitude Probe")
        plt.imshow(np.abs(self.probe[0, :, :]), interpolation="none", cmap="hot")
        plt.colorbar()

        plt.figure("Lineout")
        if vertical:
            v_lineout = np.abs(self.probe[0, :, shape[1]/2])

            y = np.arange(0, shape[1], dtype=np.float64)
            print(y)
            y *= self.psize[0]*1e6
            plt.plot(y, v_lineout, label="Vertical profile")

        if horizontal:
            h_lineout = np.abs(self.probe[0, shape[0]/2, :])

            x = np.arange(0, shape[0], dtype=np.float64)
            x *= self.psize[1]*1e6
            plt.plot(x, h_lineout, label="Horizontal profile")

        plt.legend()

if __name__ == "__main__":
    print("Open file: ", filepath)
    reader = ptry_reader(filepath)
    reader.normalize_obj(radius=50)

    # vertical line
    # reader.rotate_img(-3.7)

    # first horizontal line --> 44nm res
    # reader.rotate_img(1.3)

    # reader.set_roi_obj(size=200)
    # reader.probe_cplot()
    # reader.obj_cplot(1)
    reader.plot_obj(imag=False, E_field=False, interpolation="none")

    # vertical line
    # reader.plot_vertical_line(1082, 900, 920)

    # first horizontal line
    # xstart = 787
    # xend = 796
    # reader.plot_horizontal_line(636, xstart, xend)
    # reader.plot_horizontal_line(486, xstart, xend)

    # reader.plot_horizontal_line(1015, 860, 890)
    reader.roi_plot(interpolation="none")
    reader.plot_probe(lineout=False, logplot=True)
    # reader.probe_lineout()
    # reader.save_obj_mat("object.mat")
    # reader.save_probe_mat("probe.mat")

    plt.show()
