from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import glob
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":

    pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

    # Office pc versions
    #base = "D:\Github\ptypy\\templates\positions_position_refinement\\"
    base = "./positions_position_refinement/"

    engine = "DM"

    filenames = "pos_*.txt"

    filepaths = glob.glob(base + filenames)
    norm_coord_idx = 0
    delimiter = ";"
    psize = 56e-8                 # 512x512 psize

    show_original = True
    simulation = True
    path_original = "./positions_theory.txt"

    # show trajectory of recovered positions
    for i, path in enumerate(filepaths):
        # iterate through all iterations of the reconstruction
        print("Load: " + path)
        data = np.loadtxt(path)
        if i == 0:
            # [number of iteration, positions, (y, x)]
            pos_trajec = np.zeros((len(filepaths), data.shape[0], 2))

        for j in range(data.shape[0]):
            # iterate through all positions
            pos_trajec[i, j, 0] = data[j, 0]
            pos_trajec[i, j, 1] = data[j, 1]

    print(pos_trajec.shape)
    # show reconstructed trajectories and cycle trough all positions
    for i in range(pos_trajec.shape[1]):
        if i == 0:
            plt.figure("pos trajectory")
            pos_trajec *= 1e6
            plt.plot(pos_trajec[0, i, 1], pos_trajec[0, i, 0], "x", ms=12, color="b", label="start")
        else:
            plt.plot(pos_trajec[0, i, 1], pos_trajec[0, i, 0], "x", ms=12, color="b")   # first iteration
        plt.plot(pos_trajec[:, i, 1], pos_trajec[:, i, 0], color="b")

    # show original positions which were used to calczlate the diffraction patterns
    if show_original:
        data_original = np.loadtxt(path_original) * 1e6
        offset = pos_trajec[-1, 0, :] - data_original[0, :]

        print(data_original[0, :])
        print(pos_trajec[-1, 0, :])

        data_original += offset
        for i in range(data_original.shape[0]):
            if i == 0:
                plt.plot(data_original[i, 1], data_original[i, 0], "x", ms=12, color="green", label="original")
            plt.plot(data_original[i, 1], data_original[i, 0], "x", ms=12, color="green")

    # label axes
    plt.xlabel("x Postion in um")
    plt.ylabel("y Positon in um")
    plt.legend()

    # Distance after position correction
    distance_after = np.zeros((pos_trajec.shape[1], 2))
    # Distance before positon correction
    distance_before = np.zeros((pos_trajec.shape[1], 2))

    if simulation:
        # caculate distance
        for i in range(pos_trajec.shape[1]):
            distance_after[i, 0] = pos_trajec[-1, i, 0] - data_original[i, 0]
            distance_after[i, 1] = pos_trajec[-1, i, 1] - data_original[i, 1]

            distance_before[i, 0] = pos_trajec[0, i, 0] - data_original[i, 0]
            distance_before[i, 1] = pos_trajec[0, i, 1] - data_original[i, 1]

        plt.figure("Residual Shift vertical")
        plt.plot(distance_before[:, 0], label="Before pos corr")
        plt.plot(distance_after[:, 0], label="After pos corr")
        plt.legend()

        plt.figure("Residual Shift horizontal")
        plt.plot(distance_before[:, 1], label="Before pos corr")
        plt.plot(distance_after[:, 1], label="After pos corr")

        norm_dis_before = np.linalg.norm(distance_before, axis=1)
        norm_dis_after = np.linalg.norm(distance_after, axis=1)

        plt.figure("Norm distance")
        plt.plot(old_div(norm_dis_before,psize*1e-6), label="Distance before pos corr")
        plt.plot(old_div(norm_dis_after,psize*1e-6), label="Distance after pos corr")
        plt.xlabel("Position")
        plt.ylabel("Distance in px")

        print("Mean distance to original position before pos corr: " + str(np.mean(old_div(norm_dis_before,psize*1e-6))))
        print("Mean distance to original position after pos corr: " + str(np.mean(old_div(norm_dis_after,psize*1e-6))))

        # calculate error by relativ positions
        # Contains the relative shift of the position to every other position for the positions that were used to calculate
        # the diffractio patterns
        relativ_shifts_original = np.zeros((data_original.shape[0], data_original.shape[0], 2))

        for i in range(relativ_shifts_original.shape[0]):
            for j in range(relativ_shifts_original.shape[0]):
                relativ_shifts_original[i, j, 0] = data_original[i, 0] - data_original[j, 0]
                relativ_shifts_original[i, j, 1] = data_original[i, 1] - data_original[j, 1]

        plt.figure("Relativ y-shifts original")
        plt.imshow(relativ_shifts_original[:, :, 0], interpolation="none")
        plt.colorbar()

        plt.figure("Relativ x-shifts original")
        plt.imshow(relativ_shifts_original[:, :, 1], interpolation="none")
        plt.colorbar()

        # calculate the relative shifts for the uncorrected data
        relativ_shifts_uncorrected = np.zeros((data_original.shape[0], data_original.shape[0], 2))

        for i in range(relativ_shifts_uncorrected.shape[0]):
            for j in range(relativ_shifts_uncorrected.shape[0]):
                relativ_shifts_uncorrected[i, j, 0] = pos_trajec[0, i, 0] - pos_trajec[0, j, 0]
                relativ_shifts_uncorrected[i, j, 1] = pos_trajec[0, i, 1] - pos_trajec[0, j, 1]

        plt.figure("Relative y-shifts uncorrected")
        plt.imshow(relativ_shifts_uncorrected[:, :, 0], interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        plt.figure("Relative x-shifts uncorrected")
        plt.imshow(relativ_shifts_uncorrected[:, :, 1], interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        plt.figure("Relative y-shifts uncorrected difference")
        plt.imshow(old_div(np.abs(relativ_shifts_uncorrected[:, :, 0] - relativ_shifts_original[:, :, 0]),psize*1e-6))
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        # calcuate the relativ shifts of the corrected positions
        relativ_shifts_corrected = np.zeros((data_original.shape[0], data_original.shape[0], 2))

        for i in range(relativ_shifts_corrected.shape[0]):
            for j in range(relativ_shifts_corrected.shape[0]):
                relativ_shifts_corrected[i, j, 0] = pos_trajec[-1, i, 0] - pos_trajec[-1, j, 0]
                relativ_shifts_corrected[i, j, 1] = pos_trajec[-1, i, 1] - pos_trajec[-1, j, 1]

        plt.figure("Relative y-shifts corrected")
        plt.imshow(relativ_shifts_corrected[:, :, 0], interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        plt.figure("Relative x-shifts corrected")
        plt.imshow(relativ_shifts_corrected[:, :, 1], interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        plt.figure("y relative Difference")
        plt.imshow(old_div(np.abs(relativ_shifts_corrected[:, :, 0] - relativ_shifts_original[:, :, 0]),psize*1e-6), interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        plt.figure("x relative Difference")
        plt.imshow(old_div(np.abs(relativ_shifts_corrected[:, :, 1] - relativ_shifts_original[:, :, 1]),psize*1e-6), interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        tot_dist = old_div(np.sqrt((relativ_shifts_corrected[:, :, 1] - relativ_shifts_original[:, :, 1])**2 + (relativ_shifts_corrected[:, :, 0] - relativ_shifts_original[:, :, 0])**2),psize*1e-6)
        plt.figure("Total distance in px")
        plt.imshow(tot_dist, interpolation="none")
        plt.xlabel("Position number")
        plt.ylabel("Position number")
        plt.colorbar()

        tot_dist_uncorr = old_div(np.sqrt((relativ_shifts_uncorrected[:, :, 1] - relativ_shifts_original[:, :, 1])**2 + (relativ_shifts_uncorrected[:, :, 0] - relativ_shifts_original[:, :, 0])**2),psize*1e-6)
        print("Mean distance error to all other positions corrected: " + str(np.mean(np.mean(tot_dist, axis=0))))
        print("Mean distance error to all other positions uncorrected: " + str(np.mean(np.mean(tot_dist_uncorr, axis=0))))
        plt.legend()

    plt.show()
