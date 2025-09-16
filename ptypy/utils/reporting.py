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


    # ToDo: should actually go over all probe storages and all obj storages
    #       in two for loops

    # for each storage do:
    for sname, sprobe in ptycho.probe.S.items():  
        metrics[sname] = {}

        # probe size
        metrics[sname]['probe_size'] = measure_probe_size(ptycho, sname)

        # average step size
        metrics[sname]["average_step_size"] = calculate_average_step_size(ptycho, sname)

        # scanned vield of view
        metrics[sname]["field_of_view"] = measure_field_of_view(ptycho, sname, metrics[sname]['probe_size'])

        # area overlap
        metrics[sname]['area_overlap'] = calculate_area_overlap(ptycho, sname, metrics[sname]['probe_size'])

        # oversampling
        metrics[sname]['oversampling_area'] = calculate_geometric_oversampling(metrics[sname]['probe_size']['90perI_width_px'], sprobe.shape )
        metrics[sname]['oversampling_FWHM'] = calculate_geometric_oversampling(metrics[sname]['probe_size']['FWHM_px'], sprobe.shape )

        # some maps
        metrics[sname]['maps'] = {}
        for x in ['fluence', 'transmission', 'coverage']:
            metrics[sname]['maps'][x] = map(ptycho, ID=x)[sname]

    return metrics


def map(ptycho, ID='fluence', mask=None):
    """
    Compute fluence or transmission map(s) for all object storages and modes

    Parameters
    ----------
    ptycho : The Ptycho object instance to draw information from.
    ID : one of 'fluence' (default), 'transmission' or 'coverage'
    mask: a dictionary containing the thresholded mask of the probe for each probe storage
         (used only if ID=='coverage')
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
                probe_storage = pod.pr_view.storage.ID
                if ID == 'transmission':
                    sobj[v] += abs2(pod.probe*pod.object)
                elif ID == 'fluence':
                    sobj[v] += abs2(pod.probe)
                elif ID == 'coverage':
                    if mask is not None:
                        sobj[v] += mask[probe_storage]
                    else:
                        sobj[v] += np.ones_like(pod.probe.real)

    return {sname: sobj.data[0] for sname, sobj in fmap.S.items()}

def calculate_geometric_oversampling(probe_size, probe_shape):
    """
    Calulate the oversampling based on the evaluated probe size.
    Note that this likely underestimage the 

    Parameters:
    -----------
    probe_size: evaluated probe dimensions (Y, X)
    probe_shape: probe array shape (Y,X)

    returns:
    --------
    oversampling (Y, X)
    """
    return probe_shape[0]/probe_size[0], probe_shape[1]/probe_size[1]

def measure_probe_size(ptycho, sname):
    """
    main function for measuring the size of the reconstructed probes
    via different methods.
    """

    results = {}
    probes = ptycho.probe.S[sname].data
    pixel_size = ptycho.probe.S[sname]._psize

    probe_intensity = np.sum(abs2(probes), axis=0)

    # measure FWHM of projected intensities
    fwhm_x,fwhm_y = size_estimate_FWHM(probe_intensity)
    results['FWHM_px'] = (fwhm_y, fwhm_x)
    results['FWHM_m'] = (fwhm_y*pixel_size[0], fwhm_x*pixel_size[1])

    # measure probe area by 90% intensty criterion
    illuminated_area = size_estimate_90pecent_intensity(probe_intensity)
    results['90perI_image'] = illuminated_area

    # Total area
    results['90perI_area_px'] = np.sum(illuminated_area)
    results['90perI_area_sqm'] = np.sum(illuminated_area)*pixel_size[0]*pixel_size[1]

    # Extent in x and y
    sum_y = illuminated_area.sum(axis=0)
    sum_x = illuminated_area.sum(axis=1)
    y_nz = np.nonzero(sum_y)[0]
    x_nz = np.nonzero(sum_x)[0]
    wy, wx = (1 + y_nz[-1] - y_nz[0], 1 + x_nz[-1] - x_nz[0])
    results['90perI_width_px'] = (wy, wx)
    results['90perI_width_m'] = (wy*pixel_size[0], wx*pixel_size[1])

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


def measure_field_of_view(ptycho, sname, probe_size):
    """
    main function for measuring the scanned field of view
    via different methods.
    """

    results = {}
    pixel_size = ptycho.probe.S[sname]._psize

    # estimate FOV from coverage map
    FOV_from_coverage_px = FOV_estimate_coverage(ptycho, probe_size['90perI_image'], sname)
    results['coverage_px'] = FOV_from_coverage_px
    results['coverage_sqm'] = FOV_from_coverage_px * pixel_size[0] * pixel_size[1]

    # estimate from convex hull
    #ToDo

    return results


def FOV_estimate_coverage(ptycho, probe_mask, sname):
    """
    Estimating the imaged field of view from the probe intensity mask
    and coverage map funtion.
    """
    cmap = map(ptycho, ID='coverage', mask={sname: probe_mask})
    cm1 = cmap[sname] > 0.5
    return np.sum(cm1)

    #ToDo: add this to maps as well


def calculate_area_overlap(ptycho, sname, probe_size):
    """
    main function for measuring the scanned field of view
    via different methods.
    """

    results = {}

    # estimate via the coverage map
    area_overlap = estimate_area_overlap_via_coverage(ptycho, probe_size['90perI_image'], sname)
    results['from_coverage'] = area_overlap

    # estimate from convex hull
    #ToDo

    return results


def estimate_area_overlap_via_coverage(ptycho, probe_mask, sname):
    cmap = map(ptycho, ID='coverage', mask={sname: probe_mask})
    cm1 = cmap[sname] > 0.5
    return (cmap[sname]-cm1).sum() / cmap[sname].sum()




def calculate_average_step_size(ptycho, sname, number_of_neighbours=3):
    """
    Calculates the average step size is different ways:
    """
    ass = {}

    # from the n nearest neighbors.
    ass['from_nearest_neighbors'] = estimate_step_size_NN(ptycho, sname, number_of_neighbours)
    return ass


def estimate_step_size_NN(ptycho, sname, number_of_neighbours=3):
    """
    Estimating the average step size from nearest neighbors
    """
    allcoords = np.array([v.coord for v in ptycho.obj.S[sname].views])
    distances = []
    for v in ptycho.obj.S[sname].views:
        distxy = (allcoords - v.coord)
        distr = np.sqrt(distxy[:,0]**2 + distxy[:,1]**2)
        distances.append(np.sort(distr)[1:number_of_neighbours+1])
    ass = np.array(distances).mean()
    return ass


