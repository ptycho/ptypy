from math import floor


def max_fpb_from_scans(scans):
    """Find maximum 'frames_per_block' from different scan models.

    Parameters
    ----------
    scans : dict
        a dictionary contains different scan models, i.e. instances of
        ptypy.core.manager.ScanModel.

    Returns
    -------
    max_fpb : int
        the maximum number of 'frames_per_block' in these scan models,
        None if any one of them is GradFull as it is irrelevant.

    """
    max_fpb = 0
    for scan in scans.values():
        if scan.__class__.__name__ == "GradFull":
            # the 'frames_per_block' is irrelevant for the GradFull
            # model
            return None
        max_fpb = max(scan.max_frames_per_block, max_fpb)

    return max_fpb

def calculate_safe_fpb(mem_avail, mem_per_frame, nblk=3):
    """Return a safe value of 'frames_per_block' from memory information.

    Parameters
    ----------
    mem_avail : int
        the available GPU memory
    mem_per_frame : int
        the GPU memory required for a single frame
    nblk : int, optional
        the total number of blocks the data will be divided into.
        Default to 3.

    """
    if mem_per_frame < 0 or mem_avail < 0:
        msg = "Memory should be a positive number."
        raise ValueError(msg)

    return floor((mem_avail / nblk) / mem_per_frame)
