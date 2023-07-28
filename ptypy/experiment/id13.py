"""  Implementation of PtyScan subclasses to hold nanomax scan data. The
     beamline is developing, as is the data format. """

from ..core.data import PtyScan
from .. import utils as u
from . import register
logger = u.verbose.logger

import numpy as np
try:
    import hdf5plugin
except ImportError:
    logger.warning('Couldnt find hdf5plugin - better hope your h5py has bitshuffle!')
import h5py
import os.path
from PIL import Image

@register()
class id13eh3_raw(PtyScan):
    """
    This class loads data written by BLISS at the ID13 nano focus endstation (EH3).

    Todo: 
    * add feature for normalization of the frames

    Defaults:

    [name]
    default = id13eh3_raw
    type = str
    help =

    [path]
    default = None
    type = str
    help = Path to where the data is at
    doc =

    [motor_auto_detect]
    default = False
    type = bool
    help = if True, will auto choose motor names (fly/mesh) from scan command in file
    doc =

    [flatfieldfile]
    default = None
    type = str
    help = Path to npy where the flatfield is ... already processed to be the factor for each pixel response
    doc =

    [scanNumber]
    default = None
    type = int
    help = Scan number or list of scan numbers
    doc =

    [sample_dir]
    default = None
    type = str
    help = sample subdirectory ... first in the RAW_DATA folder
    doc =

    [sample_sub_dir]
    default = None
    type = str
    help = sample subsubdirectory ... seconds in the RAW_DATA/sample/
    doc =

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the scan file
    doc =

    [cropOnLoad]
    default = True
    type = bool
    help = Only load the used bits of each detector frame
    doc =

    [cropOnLoad_y_lower]
    default = None
    type = int, list, tuple
    help = y-axis lower limit
    doc =

    [cropOnLoad_y_upper]
    default = None
    type = int, list, tuple
    help = y-axis upper limit
    doc =

    [cropOnLoad_x_lower]
    default = None
    type = int, list, tuple
    help = x-axis lower limit
    doc =

    [cropOnLoad_x_upper]
    default = None
    type = int, list, tuple
    help = x-axis upper limit
    doc =

    [tmp_center]
    default = None
    type = int, list, tuple
    help = x-axis upper limit
    doc =

    [xMotor]
    default = samx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samy
    type = str
    help = Which y motor to use
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [xMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor x axis relative to the lab x axis
    doc =

    [yMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor y axis relative to the lab y axis
    doc =

    [xyAxisSkewOffset]
    default = 0.0
    type = float
    help = Relative rotation angle beyond the expected 90 degrees between the motor x and y axes in degree
    doc = If for example the scanner is damaged and x and y end up not beeing perfectly under 90 degrees, this value can can be used to correct for that.

    [zDetectorAngle]
    default = 0.0
    type = float
    help = Relative rotation angle between the motor x and y axes and the detector pixel rows and columns in degree
    doc = If the Detector is mounted rotated around the beam axis relative to the scanning motors, use this angle to rotate the motor position into the detector frame of reference. The rotation angle is in mathematical positive sense from the motors to the detector pixel grid.

    [detector]
    default = 'mpxeh3he'
    type = str
    help = Which detector to use, can be pil100k or merlin

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Tiff file containing the mask or Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel, like alba2/1 for example
    doc =

    """

    def clean_mask(self, mask):
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        return mask

    def load_mask_h5(self):
        with h5py.File(self.info.maskfile, 'r') as hf:
            mask = np.array(hf.get('mask')) 
        return self.clean_mask(mask)

    def load_mask_tiff(self):
        with Image.open(self.info.maskfile) as im:
            mask = np.array(im) 
        return self.clean_mask(mask)

    def load_positions(self):

        scan_nmb_str = f'{self.info.scanNumber}.1'
        fpath_scan = f'{self.info.path}/{self.info.sample_dir}/{self.info.sample_sub_dir}/{self.info.sample_sub_dir}.h5' 

        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x/y axis is tilted with respect to the beam axis, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        logger.info("x and y motor angles result in multiplication by %.2f, %.2f" % (xCosFactor, yCosFactor))


        # check for fly/mesh automatially?
        if self.info.motor_auto_detect:
            with h5py.File(fpath_scan, 'r') as hf:
                scan_command = str(hf[f'{scan_nmb_str}/title'][()])
            if 'akmap' in scan_command:   # is a flyscan
                self.info.xMotor = 'nnp2_user_position'
                self.info.yMotor = 'nnp3_position'
                logger.info(f'auto detected a fly scan; using motors ({self.info.xMotor}, {self.info.yMotor})')        
            else:
                self.info.xMotor = 'nnp2'
                self.info.yMotor = 'nnp3'     
                logger.info(f'auto detected a step scan; using motors ({self.info.xMotor}, {self.info.yMotor})')        


        # read motor positions
        normdata, scanu, scanv, scanw, samr = [], [], [], [], []
        with h5py.File(fpath_scan, 'r') as hf:
            pos_x = list(hf[f'{scan_nmb_str}/measurement/{self.info.xMotor}'])
            pos_y = list(hf[f'{scan_nmb_str}/measurement/{self.info.yMotor}'])

        x = np.array(pos_x) * xFlipper * xCosFactor
        y = np.array(pos_y) * yFlipper * yCosFactor

        chi_rad_x = 0
        chi_rad_y = 0
        # if the detector and motor frame of reference are roated around the beam axis
        if self.info.zDetectorAngle != 0:
            chi_rad_x = self.info.zDetectorAngle / 180.0 * np.pi
            chi_rad_y = 1.*chi_rad_x
            logger.info("x and y motor positions were roated by %.4f degree to align with the detector pixel grid" % (self.info.zDetectorAngle))
        # if x and y are not under 90 degrees to each other
        if self.info.xyAxisSkewOffset != 0:
            chi_rad_x += -0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            chi_rad_y += +0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            logger.info("x and y motor positions were skewed by %.4f degree to each other" % (self.info.xyAxisSkewOffset))
        x, y = np.cos(chi_rad_x)*x-np.sin(chi_rad_y)*y, np.sin(chi_rad_x)*x+np.cos(chi_rad_y)*y
            
        # set minimum to zero so ptypy can work out the proper object size
        x -= np.min(x)
        y -= np.min(y)

        # put the two arrays together and express in [m]
        positions = -np.vstack((y, x)).T * 1e-6
        return positions

    def pad_to_size(self, frame, value):
        ny, nx = np.shape(frame)
        cy, cx = self.info.tmp_center
        dy, dx = self.info.shape
        ry, rx = dy//2 , dx//2
        pad_xl   = rx - cx
        pad_xu   = rx + cx - nx 
        pad_yl   = ry - cy
        pad_yu   = ry + cy - ny 
        return np.pad(frame, [[pad_yl,pad_yu],[pad_xl,pad_xu]], mode='constant', constant_values=[value])


    def load(self, indices):
        raw, weights, positions = {}, {}, {}

        scan_nmb_str = f'{self.info.scanNumber}.1'
        fpath_scan = f'{self.info.path}/{self.info.sample_dir}/{self.info.sample_sub_dir}/{self.info.sample_sub_dir}.h5' 

        # crop on load is requested, but the actual indices to crop are not yet defined
        if self.info.cropOnLoad and self.info.cropOnLoad_y_lower == None:
            
            # center of the diffraction patterns is not explicitly given
            if self.info.center==None:
                # requires to load the first frame and to find the center of mass there
                with h5py.File(fpath_scan, 'r') as fp:
                    frame = fp[f'{scan_nmb_str}/measurement/{self.info.detector}'][0]
                # and to mask the hot pixels ... sadly this will have double with self.load_weight
                mask = np.ones_like(frame)
                if self.info.detector == 'pilatus':
                    mask[np.where(frame < 0)] = 0
                if 'eiger' in self.info.detector:
                    mask[np.where(frame == 2**32-1)] = 0
                    mask[np.where(frame == 2**16-1)] = 0
                if self.info.maskfile:
                    if self.info.maskfile.endswith('.h5'):
                        mask2 = self.load_mask_h5()
                    else:
                        mask2 = self.load_mask_tiff()
                    mask *= mask2
                # now find the center of mass can be estimated using the ptypy internal function and make it integers
                self.info.center = u.scripts.mass_center(frame*mask)
                self.info.center = [int(x) for x in self.info.center]
                logger.info(f'Estimated the center of the (first) diffraction pattern to be {self.info.center}')

            # the center of the full frames is (now) known, and thus the indices for the cropping can be defined
            cy, cx  = self.info.center
            dy, dx  = self.info.shape
            logger.info(f'Found the center of the full frames at {self.info.center}')
            logger.info(f'Will crop all diffraction patterns on load to a size of {self.info.shape}')
            self.info.cropOnLoad_y_lower, self.info.cropOnLoad_x_lower = int(cy)-dy//2, int(cx)-dy//2
            self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_upper = self.info.cropOnLoad_y_lower+dy, self.info.cropOnLoad_x_lower+dx

            # the (temporary) center needs to be redefined for the cropped frames
            tmp_center_y, tmp_center_x = dy//2, dx//2

            # if the lower crop indices are negative, set them zero
            if self.info.cropOnLoad_y_lower<0:
                tmp_center_y += self.info.cropOnLoad_y_lower
                self.info.cropOnLoad_y_lower = 0
            if self.info.cropOnLoad_x_lower<0:
                tmp_center_x += self.info.cropOnLoad_x_lower
                self.info.cropOnLoad_x_lower = 0    
            # no need to have something similar for too large upper indices due to the way python slices arrays

            # now fix the new center
            self.info.tmp_center = (tmp_center_y, tmp_center_x)
            self.info.center = (dy//2, dx//2)
        
        # set the photon energy
        self.meta.energy = self.info.energy    #as not saved in the raw file from the beamline

        if self.info.flatfieldfile != None:
            f = np.load(self.info.flatfieldfile)
            if self.info.cropOnLoad:
                f = f[self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
        else:
            f = 1

        # actually loading the detector frames
        for ind in indices:
            with h5py.File(fpath_scan, 'r') as fp:
                # load only a cropped bit of the full frame
                if self.info.cropOnLoad:
                    frame = fp[f'{scan_nmb_str}/measurement/{self.info.detector}'][ind,self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                    #print('--', ind, np.shape(frame))
                    raw[ind] = self.pad_to_size(frame, -1) * f
                # load the full raw frame                
                else:	
                    raw[ind] = fp[f'{scan_nmb_str}/measurement/{self.info.detector}'][ind]
                # if there is I0 information, use it to normalize the just loaded frame                
                if self.info.I0!=None:
                    self.normdata = self.normdata.flatten()
                    #logger.info('normalizing frame %u by %f' % (ind, self.normdata[ind]))
                    #logger.info('hack! assuming mask = 2**32-1 when I0-normalizing')
                    msk = np.where(raw[ind] == 2**32-1)
                    raw[ind] = np.round(raw[ind]  * f / self.normdata[ind]).astype(raw[ind].dtype)
                    raw[ind][msk] = 2**32-1

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every single diffraction pattern ofthe whole scan.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]

        mask = np.ones_like(data)
        if self.info.detector == 'mpxeh3he':
            pass    # no automatic masking for the medipix yet
            
        logger.info("took account of the built-in mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        if self.info.maskfile:
            if self.info.maskfile.endswith('.h5'):
                mask2 = self.load_mask_h5()
            else:
                mask2 = self.load_mask_tiff()
                
            if self.info.cropOnLoad:
                mask2 = mask2[self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper, 
                                self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                mask2 = self.pad_to_size(mask2, 0)

            logger.info("loaded additional mask, %u x %u, sum %u, so %u masked pixels" %
                        (mask2.shape + (np.sum(mask2), np.prod(mask2.shape)-np.sum(mask2))))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        return mask


