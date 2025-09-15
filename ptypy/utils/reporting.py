"""
Reporting functions

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
from .math_utils import abs2
import numpy as np
from scipy.interpolate import UnivariateSpline

def calculate_metrics(ptycho):
    metrics = {}
    metrics['probe_size'] = measure_probe_size(ptycho)
    metrics["average_step_size"] = calculate_average_step_size(ptycho)
    return metrics


def calculate_maps(ptycho):
    # Fluence map
    fluence = map(ptycho, ID='fluence')
    # Transmission map
    transmission = map(ptycho, ID='transmission')
    # Coverage map
    coverage = map(ptycho, ID='coverage')

    return {'fluence': fluence,
            'transmission': transmission,
            'coverage': coverage}


def map(ptycho, ID='fluence', mask=None):
    """
    Compute fluence or transmission map(s) for all object storages and modes

    Parameters
    ----------
    ptycho : The Ptycho object instance to draw information from.
    ID : one of 'fluence' (default), 'transmission' or 'coverage'
    """
    if ID not in ['fluence', 'transmission', 'coverage']:
        raise NotImplementedError(f'Unknown map {ID}')

    # Delete existing copy if it exists
    if f'C{ID}' in [c.ID for c in ptycho.obj.copies]:
        del ptycho.obj.owner._pool['C'][f'C{ID}']

    # Create copy of object container
    fmap = ptycho.obj.copy(ID=ID, fill=0., dtype='real')

    # Loop through storages and add the probe intensities
    for sname, sobj in fmap.S.items():
        for v in sobj.views:
            if not v.active:
                continue
            for pid, pod in v.pods.items():
                if ID == 'transmission':
                    sobj[v] += abs2(pod.probe*pod.object)
                elif ID == 'fluence':
                    sobj[v] += abs2(pod.probe)
                elif ID == 'coverage':
                    if mask is not None:
                        sobj[v] += mask
                    else:
                        sobj[v] += np.ones_like(pod.probe.real)

    return {sname: sobj.data[0] for sname, sobj in fmap.S.items()}

def measure_probe_size(ptycho):
    """
    main function for measuring the size of the reconstructed probes
    via different methods.
    """

    results = {}

    # for each storage do:
    for sname, sprobe in ptycho.probe.S.items():  
        probes = sprobe.data
        pixel_size = sprobe._psize
        results[sname] = {}

        probe_intensity = np.sum(abs2(probes), axis=0)

        # measure FWHM of projected intensities
        fwhm_x,fwhm_y = size_estimate_FWHM(probe_intensity)
        results[sname]['FWHM_px'] = (fwhm_y, fwhm_x)
        results[sname]['FWHM_m'] = (fwhm_y*pixel_size[0], fwhm_x*pixel_size[1])

        # measure probe size by 90% intensty criterion
        illumated_area = size_estimate_90pecent_intensity(probe_intensity)
        results[sname]['illumated_area'] = illumated_area

    return results

def size_estimate_FWHM(probe_intensity):
    """
    estimate the probe size via a simple FWHM measurment
    of the projected intensty profiles
    """
    projection_x = np.sum(probe_intensity, axis=0)
    projection_y = np.sum(probe_intensity, axis=1)
    Ny, Nx = np.shape(probe_intensity)

    # spline interpolate the profiles and find all the I_max/2 crossings
    edges_x = UnivariateSpline(np.linspace(0, Nx-1, Nx), projection_x-projection_x.max()/2).roots()
    edges_y = UnivariateSpline(np.linspace(0, Ny-1, Ny), projection_y-projection_y.max()/2).roots()
    
    # the distance between the first and the last I_max/2 crossing is the FWHM
    fwhm_x = abs(edges_x[0] - edges_x[-1]) # measured in pixels
    fwhm_y = abs(edges_y[0] - edges_y[-1]) # measured in pixels

    return fwhm_x,fwhm_y


def size_estimate_90pecent_intensity(probe_intensity):
    """
    estimate the probe size by masking the hottest pixels
    until 90% of the overall probe intensity is explained
    """

    threshold = find_threshold(probe_intensity, fraction=0.9)
    probe_mask = np.zeros_like(probe_intensity)
    probe_mask[probe_intensity>=threshold] = 1
    return probe_mask

def find_threshold(intensities, fraction=0.9):
    """
    Find the intensity threshold such that a specified fraction of the total intensity is below this threshold.

    Parameters:
    - intensities (numpy.ndarray): An array of pixel intensities.
    - fraction (float): The fraction of total intensity to use as a cutoff (default is 0.9 for 90%).

    Returns:
    - float: The intensity threshold.
    """
    import numpy as np

    # Flatten the array to 1D for processing and sort it
    sorted_vals = np.sort(intensities.ravel())
    # Compute the cumulative sum from the end to find the 90% cutoff
    cumulative = np.cumsum(sorted_vals[::-1])
    # Total intensity (sum of all pixel values)
    total = cumulative[-1]
    # Find the index where cumulative sum reaches fraction of total
    inverted_index = np.searchsorted(cumulative, fraction * total)
    # Reverse the index to get the correct position in the sorted array
    cutoff_index = len(sorted_vals) - inverted_index - 1 if inverted_index < len(sorted_vals) else 0

    return float(sorted_vals[cutoff_index])

def calculate_average_step_size(ptycho, number_of_neighbours=3):
    """
    Calculates the average step size
    """
    ass = {}
    for sname, sobj in ptycho.obj.S.items():
        allcoords = np.array([v.coord for v in sobj.views])
        distances = []
        for v in sobj.views:
            distxy = (allcoords - v.coord)
            distr = np.sqrt(distxy[:,0]**2 + distxy[:,1]**2)
            distances.append(np.sort(distr)[1:number_of_neighbours+1])
        ass[sname] = np.array(distances).mean()
    return ass
