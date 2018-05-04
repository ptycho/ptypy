"""
Contains functions that are need for position correction.
"""

from ..core import View
from ..core.classes import DEFAULT_ACCESSRULE
from ..utils import parallel
from ..utils.verbose import log
import matplotlib.pyplot as plt
plt.switch_backend("QT5Agg")
import numpy as np
import os
import sys
import time


def save_pos(obj):
    """
    Saves the current positions in a .txt file.
    """

    if parallel.master:
        coords = []

        di_view_names = obj.di.views.keys()
        di_view_names.sort()

        for di_view_name in di_view_names:
            di_view = obj.di.views[di_view_name]
            coord = np.copy(di_view.pod.ob_view.coord)
            coords.append(coord)

        coords = np.asarray(coords)

        directory = "positions_" + sys.argv[0][:-3] + "\\"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory + "pos_" + str(obj.p.name) + "_" + str(obj.curiter).zfill(4) + ".txt", coords)


def single_pos_ref(obj, di_view_name):
    """
    Refines the positions by the following algorithm:

    A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
    An annealing algorithm to correct positioning errors in ptychography,
    Ultramicroscopy, Volume 120, 2012, Pages 64-72

    Describtion:
    Calculates random shifts around the original position and calculates the fourier error. If the fourier error
    decreased the randomly calculated postion will be used as new position.

    Parameters loaded by script:
    ----------------------------
    number_rand_shfits: Number of random shifts which are calculate for each position in every position correction
    iteration. Right now this should be always a mutliple of 4 since for every direction 1 shift is calculated.
    There is probably a better way to do this.
    amplitude: Distance around the originial position where the randomly calculated position lies.
    pos_ref_start: Iteration number the position correction should start.
    pos_ref_stop: Iteration number the position correction should stop.
    max_shift_allowed:  The distance that positions are allowed to drift away from the original position.

    :param obj: self of the calling engine.
    :param di_view_name: Name of the diffraction view for which a better position is searched for.

    :return: If a better coordinate (smaller fourier error) is found for a position, the new coordinate (meters)
    will be returned. Otherwise (0, 0) will be returned.
    """

    di_view = obj.di.views[di_view_name]
    number_rand_shifts = obj.p.number_rand_shifts           # should be a multiple of 4
    dcoords = np.zeros((number_rand_shifts + 1, 2)) - 1.
    pxl_size_obj = obj.ob.S.values()[0].psize[0]            # Pixel size in the object plane
    amplitude = obj.p.amplitude                             # still has to be replaced by parameter value
    end = obj.p.pos_ref_stop
    start = obj.p.pos_ref_start
    max_shift_allowed = obj.p.max_shift_allowed
    it = obj.curiter                                        # current iteration
    max_shift_dist = amplitude * (end - it) / (end - start) + pxl_size_obj/2.
    shape = obj.shape

    delta = np.zeros((number_rand_shifts, 2))               # coordinate shift
    errors = np.zeros(number_rand_shifts)                   # calculated error for the shifted position
    coord = np.copy(di_view.pod.ob_view.coord)

    obj.ar.coord = coord
    obj.ar.storageID = obj.temp_ob.storages["S00G00"].ID

    # Create temporal object view that can be shifted without reformatting
    ob_view_temp = View(obj.temp_ob, accessrule=obj.ar)
    dcoords[0, :] = ob_view_temp.dcoord

    # This can be optimized by saving existing iteration fourier error...
    error_inital = get_fourier_error(di_view, ob_view_temp.data)

    for i in range(number_rand_shifts):

        delta_y = np.random.uniform(0, max_shift_dist)
        delta_x = np.random.uniform(0, max_shift_dist)

        if i % 4 == 1:
            delta_y *= -1
            delta_x *= -1
        elif i % 4 == 2:
            delta_x *= -1
        elif i % 4 == 3:
            delta_y *= -1

        delta[i, 0] = delta_y
        delta[i, 1] = delta_x

        rand_coord = [coord[0] + delta_y, coord[1] + delta_x]
        norm = np.linalg.norm(rand_coord - obj.initial_pos[int(di_view_name[1:]), :])

        if norm > max_shift_allowed:
            # Positions drifted too far, skip this position
            log(4, "New position is too far away!!!", parallel=True)
            errors[i] = error_inital + 1.
            continue

        obj.ar.coord = rand_coord
        ob_view_temp = View(obj.temp_ob, accessrule=obj.ar)

        dcoord = ob_view_temp.dcoord                            # coordinate in pixel

        # check if new coordinate is on a different pixel since there is no subpixel shift, if there is no shift
        # skip the calculation of the fourier error
        if any((dcoord == x).all() for x in dcoords):
            errors[i] = error_inital + 1.
            continue

        dcoords[i + 1, :] = dcoord

        if ob_view_temp.dlow[0] < 0:
            ob_view_temp.dlow[0] = 0

        if ob_view_temp.dlow[1] < 0:
            ob_view_temp.dlow[1] = 0

        new_obj = np.zeros(shape, dtype=np.complex128)

        if ob_view_temp.data.shape[0] != shape[0] or ob_view_temp.data.shape[1] != shape[1]:
            # if the data of the view has the wrong shape, zero-pad the data
            # new data for calculating the fourier transform
            # calculate limits of the grid
            ymin = obj.ob.storages["S00G00"].grids()[0][0, 0, 0]
            ymax = obj.ob.storages["S00G00"].grids()[0][0, -1, -1]
            xmin = obj.ob.storages["S00G00"].grids()[1][0, 0, 0]
            xmax = obj.ob.storages["S00G00"].grids()[1][0, -1, -1]

            # check if the new array would be bigger
            new_xmin = rand_coord[1] - (pxl_size_obj * shape[1] / 2.)
            new_xmax = rand_coord[1] + pxl_size_obj * shape[1] / 2.
            new_ymin = rand_coord[0] - (pxl_size_obj * shape[0] / 2.)
            new_ymax = rand_coord[0] + pxl_size_obj * shape[0] / 2.

            idx_x_low = 0
            idx_x_high = shape[1]
            idx_y_low = 0
            idx_y_high = shape[0]

            if new_ymin < ymin:
                idx_y_low = shape[0] - ob_view_temp.data.shape[0]
            elif new_ymax > ymax:
                idx_y_high = ob_view_temp.data.shape[0]

            if new_xmin < xmin:
                idx_x_low = shape[1] - ob_view_temp.data.shape[1]
            elif new_xmax > xmax:
                idx_x_high = ob_view_temp.data.shape[1]

            new_obj[idx_y_low: idx_y_high, idx_x_low: idx_x_high] = ob_view_temp.data
        else:
            new_obj = ob_view_temp.data

        errors[i] = get_fourier_error(di_view, new_obj)
        del new_obj

    if np.min(errors) < error_inital:
        # if a better coordinate is found
        log(4, "New coordinate with smaller Fourier Error found!", parallel=True)
        arg = np.argmin(errors)
        new_coordinate = np.array([coord[0] + delta[arg, 0], coord[1] + delta[arg, 1]])
    else:
        new_coordinate = (0, 0)

    del ob_view_temp
    del di_view

    return new_coordinate


def pos_ref(obj):
    """
    Iterates trough all positions and refines the positions by a given algorithm. Right now
    the following algorithm is implemented:

    A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
    An annealing algorithm to correct positioning errors in ptychography,
    Ultramicroscopy, Volume 120, 2012, Pages 64-72

    :param obj: Self of the calling engine class.
    """
    log(4, "----------- START POS REF -------------")
    t_pos_s = time.time()

    di_view_names = obj.di.views.keys()
    # di_view_names.sort()

    # List of refined coordinates which will be used to reformat the object
    new_coords = np.zeros((len(di_view_names), 2))

    # Only used for calculating the shifted pos
    obj.temp_ob = obj.ob.copy()

    for i, di_view_name in enumerate(obj.di.views):
        di_view = obj.di.views[di_view_name]
        pos_num = int(di_view.ID[1:])

        if i == 0:
            # create accessrule
            obj.ar = DEFAULT_ACCESSRULE.copy()
            obj.ar.psize = obj.temp_ob.storages["S00G00"].psize
            obj.ar.shape = obj.shape

        if di_view.active:
            new_coords[pos_num, :] = single_pos_ref(obj, di_view_name)

    new_coords = parallel.allreduce(new_coords)

    for di_view_name in obj.di.views:
        di_view = obj.di.views[di_view_name]
        pos_num = int(di_view.ID[1:])
        if new_coords[pos_num, 0] != 0 and new_coords[pos_num, 1] != 0:
            log(4, "Position: " + str(pos_num))
            log(4, "Old coordinate: " + str(di_view.pod.ob_view.coord))
            log(4, "New coordinate: " + str(new_coords[pos_num, :]))
            di_view.pod.ob_view.coord = new_coords[pos_num, :]

    # Change the coordinates of the object
    obj.ob.reformat()

    # clean up
    del obj.ptycho.containers[obj.temp_ob.ID]
    del obj.temp_ob

    t_pos_f = time.time()
    log(4, "Pos ref time: " + str(t_pos_f - t_pos_s))


def get_fourier_error(di_view, object):
    """
    Calculates the fourier error for a given diffractin view and a numpy array which contains the corresponding object.
    (Stolen from the DM engine)

    :param di_view: View to the diffraction pattern of the given position.
    :param object: Numpy array which contains the needed object.
    :return: Fourier Error
    """

    af2 = np.zeros_like(di_view.data)
    # fmask = di_view.pod.mask

    for name, pod in di_view.pods.iteritems():
        af2 += np.abs(pod.fw(pod.probe*object))**2

    af = np.sqrt(af2)
    fmag = np.sqrt(np.abs(di_view.data))

    error = np.sum(di_view.pod.mask * (af - fmag)**2)

    del fmag
    del af2
    del af

    return error
