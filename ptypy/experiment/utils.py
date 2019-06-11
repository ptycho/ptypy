'''
Contains utilities for the file loading
'''

from .utils.verbose import log



class NexusCXI:

    def __init__(self, file_handle):
        '''
        passes in a file handle which is assumed to be safe.
        populates the data in ptypy friendly units
        '''

        self.energy = None  # needs to be keV
        self.extent = None  # m
        self.incident_beam_divergence = None  # mrad
        self.incident_beam_energy = None  # keV
        self.incident_energy_spread = None  # keV
        self.beam_center_x = None  # pixels
        self.beam_center_y = None  # pixels
        self.distance = None  # m
        self.detector_vector = None
        self.x_pixel_size = None  # m
        self.y_pixel_size = None  # m
        self.intensity_dataset = None
        self.fast_axis_dataset = None
        self.slow_axis_dataset = None
        self.monitor_dataset = None
        self.flatfield = None
        self.darkfield = None
        self.pixel_mask = None


        file_handle.visititems(self._get_NX_cxi_ptycho)  # populates the correct entry

        #now to get the rest of the information inside nxcxiptycho
        self.nxcxi_ptycho.visititems(self._populate_ptycho_information)


    def _get_NX_cxi_ptycho(self, name, obj):
        if "NX_class" in obj.attrs.keys():
            if str(obj.attrs["NX_class"]) in ["NXentry", "NXsubentry"]:
                if "definition" in obj.keys():
                    if str(obj["definition"][()]) == "NXcxi_ptycho":
                        self.nxcxi_ptycho = obj  # we can only have one of these per file.
                        return

    def _populate_ptycho_information(self, name, obj):
        if "NX_class" in obj.attrs.keys():
            nexus_class = str(obj.attrs["NX_class"])
            if (nexus_class == "NXdata") and (str(obj.attrs['signal']) == "data"):
                self.intensity_dataset = obj['data']
                required_axis = ['x', 'y']
                n_expected_axes = len(self.intensity_dataset.shape) - 2
                axes = obj.attrs['axes']
                if (len(axes)-2) != n_expected_axes:
                    raise IndexError('Axes:%s does not match the number of expected axes:%s' % (str(axes), n_expected_axes))
                if n_expected_axes == 2:
                    foo = obj[axes[-3]]
                    self.fast_axis_dataset = foo
                    self.slow_axis_dataset = obj[axes[-4]]
                if n_expected_axes == 1:
                    bar = obj[axes[-3:]]
                    self.fast_axis_dataset = bar
                    required_axis.remove(axes[-3])
                    self.slow_axis_dataset = obj[required_axis[0]]

            elif nexus_class == 'NXdetector':
                self.x_pixel_size = obj['x_pixel_size'][()]
                self.y_pixel_size = obj['y_pixel_size'][()]
                self.distance = obj['distance'][()]
                self.detector_vector = obj['transformations'].attrs['vector'][()]
                try:
                    self.darkfield = obj['darkfield'][...]
                except KeyError:
                    log(3, 'No darkfield supplied in nexus file.')
                try:
                    self.flatfield = obj['flatfield'][...]
                except KeyError:
                    log(3, 'No flatfield supplied in nexus file.')
                try:
                    self.pixel_mask = obj['pixel_mask'][...]
                except KeyError:
                    log(3, 'No mask supplied in nexus file.')


            elif nexus_class == 'NXbeam':
                self.incident_beam_divergence = obj['incident_beam_divergence'][()]
                self.incident_beam_energy = 6.63e-34*3e8/ (1.67e-19 * obj['energy'][()] )
                self.extent = obj['extent'][()]

            elif nexus_class == 'NXmonitor':
                self.monitor_dataset = obj['data'][()]

    def vectors_to_orientation(self, file_name):
        pass


if __name__=='__main__':
    import h5py as h5
    f = h5.File('/home/clb02321/PycharmProjects/fiddles/nexus_cxi/uptodate/example_nxptycho_cxi_compatible_raster.nxs','r')
    a = NexusCXI(f)

