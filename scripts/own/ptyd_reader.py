import numpy as np
import matplotlib.pyplot as plt
import os
from ptypy import io
from scipy.io import savemat


pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir))

# experimental data
filepath = "\\owncloud\\reconstruction\\data\\2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\\scan_freddy.ptyd"
# filepath = "\data\\2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\\scan_medium_offset.ptyd"
# filepath = "\data\\2017_08_25\\sstar_center_6um_pinhole_15um_extent_bs_small\\scan.ptyd"
# filepath = "\data\\2017_09_28\\sstar_3um_pinhole_no_HDR\\scan.ptyd"
# filepath = "\data\\2017_09_28\\100nm_lines_50_pts_3um_pinhole_1um_step_HDR\\scan.ptyd"
# filepath = "\data\\2017_08_25\\sstar_center_6um_pinhole_7um_extent_bs\scan_freddy.ptyd"
# filepath = "\data\\2017_08_25\\sstar_center_6um_pinhole_7um_extent_bs\scan.ptyd"

# simulations
# filepath = "\\src\\simulation\data_sstar_simulation\\scan.ptyd"
# filepath = "\\src\\simulation\data_sstar_off_center_simulation\\scan.ptyd"
# filepath = "\\src\\simulation\data_sstar_off_center_drift\\scan.ptyd"

filepath = pardir + filepath

# filepath = "Y:\AG_HHG\Data Analysis Tools\ptychography\\results\sstar_5um_off_6um_pinhole_far_away\\results\\4p0_threshold_after_stiching\\scan_stiched.ptyd"


class PtydReader:
    font_size_label = 12
    font_size_ticks = 12
    num_instances = -1
    beamstop = None
    psize = 13.5e-6*2
    lam = 18.1e-9
    z = 27.6e-3

    def __init__(self, filepath, beamstop=False):
        PtydReader.num_instances += 2
        data = io.h5read(filepath)
        print(io.h5info(filepath))
        self.data = data["chunks"]["0"]
        self.beamstop = beamstop

    @staticmethod
    def get_num_instances():
        print(PtydReader.num_instances)
        return PtydReader.num_instances

    def plot_dp(self, i, show=True, mask_radius=False, scale_axis=False):
        print("position: ", self.data["positions"][i])
        dp = self.data["data"][i, :, :]
        # get rid of ugly white spaces
        dp += 0.01
        num = self.get_num_instances()

        # Mask for comparison with bs data
        if mask_radius is not False:
            x = np.arange(0, 1024, 1)
            y = np.arange(0, 1024, 1)
            y, x = np.meshgrid(x, y)
            mask = (x-512)**2 + (y-512)**2 > mask_radius**2
            dp = dp * mask

        plt.figure(num)
        plt.title("Diffraction pattern: " + str(i))
        plt.imshow(dp, interpolation="none", vmin=0, vmax=50)
                   # vmax=np.max(dp)/8)
        plt.colorbar()

        fig, ax = plt.subplots(tight_layout=True)

        if scale_axis == "camera_pos":
            shape = dp.shape[0]
            pos_start = 0
            pos_end = shape*2*13.5e-3
            temp = np.copy(dp)
            temp[temp < 0.1] = 0
            cax = ax.imshow(np.log10(temp), interpolation="none", extent=[pos_start, pos_end, pos_start, pos_end], vmin=1)
            cbar = fig.colorbar(cax)
            cbar.ax.tick_params(labelsize=self.font_size_ticks)
            cbar.set_label("Intensity (log)", size=self.font_size_label)
            plt.xlabel("Position on Camera (mm)", size=self.font_size_label)
            plt.ylabel("Position on Camera (mm)", size=self.font_size_label)

        elif scale_axis == "frequency":
            shape = dp.shape[0]
            theta = np.arctan(self.psize*shape/2/self.z)
            pos_start = np.sin(theta/2)*2/self.lam/1e6
            pos_end = pos_start*-1
            other = np.copy(dp)
            other[other <= 0.01] = 0
            cax = ax.imshow(np.log10(other), interpolation="none", extent=[pos_start, pos_end, pos_start, pos_end], vmin=-0.5)
            ax.set_title("Intensity log scale")
            cbar = fig.colorbar(cax)
            cbar.ax.tick_params(labelsize=self.font_size_ticks)
            cbar.set_label("Intensity (log)", size=self.font_size_label)
            plt.xlabel(r"$q_{x} \ [\mu m ^{-1}]$", size=self.font_size_label)
            plt.ylabel(r"$q_{y} \ [\mu m ^{-1}]$", size=self.font_size_label)

        else:
            plt.imshow(np.log10(dp), interpolation="none")
            plt.xlabel("x axis pixel", size=self.font_size_label)
            plt.ylabel("y axis pixel", size=self.font_size_label)
            plt.title("Diffraction pattern log scale", size=20)
            plt.colorbar()

        plt.tick_params(labelsize=self.font_size_ticks)
        if show:
            plt.show()

    def save_pos_as_mat(self, scan_no):
        dp = self.data["data"][scan_no, :, :]
        savemat("pos_"+str(scan_no).zfill(3) +".mat", {"diffraction_pattern": dp})

    def save_as_mat(self):
        for i in range(self.data["data"].shape[0]):
            self.save_pos_as_mat(i)

    def plot_all_pos(self, scale_axis=None, mask=False):
        """
        Plots all positions which are in the .ptyd file in log_10 and lin scale.

        :param scale_axis: (str) The way how the x-axis and the y-axis should be scaled.
        "cammera_pos": Diffraction pattern on the camera.
        """

        if self.beamstop:
            num_pos = self.data["data"].shape[0]/2              # number of given positions
            for i in range(num_pos):
                self.plot_dp(i, scale_axis=scale_axis)
                self.plot_dp(i + num_pos, scale_axis=scale_axis)
                if mask:
                    self.plot_mask(i)
                    self.plot_mask(i + num_pos)
                    plt.show()
        else:
            for i in range(self.data["data"].shape[0]):
                self.plot_dp(i, scale_axis=scale_axis)
                if mask:
                    self.plot_mask(i)
                    plt.show()

    def plot_mask(self, i):
        mask = self.data["weights"][i, :, :]
        plt.figure("Mask " + str(i))
        plt.imshow(mask, interpolation="none")

        # plt.figure("Mask + diffraction pattern " + str(i))
        # plt.imshow(mask * self.data["data"][i, :, :])
        # plt.colorbar()

    def printPositions(self):
        positions = self.data["positions"]
        print(positions)

    def crop_dp(self, size):
        shape = self.data["data"][0, :, :].shape
        print("shape")
        print(shape)
        self.data["data"] = self.data["data"][:, shape[0]/2 - size/2: shape[0]/2 + size/2, shape[1]/2 - size/2: shape[1]/2 + size/2]
        # self.data["weights"] =  self.data["weights"][:, shape[0]/2 - size/2: shape[0]/2 + size/2, shape[1]/2 - size/2: shape[1]/2 + size/2]

    def plot_sum_of_roi(self, x_min, x_max, y_min, y_max):
        pass

if __name__ == "__main__":
    print(filepath)
    scanFile = PtydReader(filepath, beamstop=False)
    # scanFile.save_pos_as_mat(0)
    # scanFile.crop_dp(1024)
    # scanFile.plot_dp(40)
    # scanFile.save_as_mat()
    scanFile.plot_all_pos(mask=False)

