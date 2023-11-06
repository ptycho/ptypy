"""
Current state: In progress, not checked to work yet.
                Connects to directly to streamer

After running the reconstruction script, variables of the LiveScan class can be found under
    P.model.scans['scan00'].ptyscan.__dict__

Old notes:
----------------------------------------------------------------------------
Things to fix / troubleshooting notes:
----------------------------------------------------------------------------
* Could you maybe add a weight to the iterations somehow so that the reconstructions from
    the earlier iterations (which are calculated with less frames) contributes with a
    lesser impact, or would this be impossible due to the iterative nature of the reconstruction?
* If num_iter have been reached before all patterns have been collected then
    Ptycho stops because it thinks it's finished!
    -> Define a number of iterations for recon that starts counting only after scan is over: yes!
* If 'p.scans.contrast.data.shape = 128' is included in the 'livescan.py' script then
    ptycho-reconstruction becomes super fast and stops before scan is over as above.
* Check: how to specify " kind = 'full_flat' " as input for save_run(..) that is called by Ptycho.
    Or rather, is there a way to save the pods to the .ptyr files as well?
* Iterations will not be performed while ptycho.model.new_data() has new data, meaning that if data is
    streamed seemingly continuous, Ptycho won't start with the iterations until all data has been acquired..
*Automatically check which iteration has the lowes error and chose that reconstruction as the final reconstruction instead of just the last one.

* Number of frames included in iteration  0    10    20    30  40  50  60  70  80  90  100
    min_frames = 10, DM.numiter = 10:
        check return:                     6*   8*9    0*
        latest_pos_index_received         27* 69*105 120*
        Repackaged data from frame        21* 61*96  120*
        .ptyr / error_local                -   21    96   120

    min_frames = 1, DM.numiter = 1:
        check return:                     0     0
        latest_pos_index_received         28    120
        Repackaged data from frame        28    120
        .ptyr / error_local               -     120
    min_frames = 1, DM.numiter = 10:
        check return:                     0     0
        latest_pos_index_received         29    120
        Repackaged data from frame        29    120
        .ptyr / error_local
 
SOLVED PROBLEMS:
✖✖✖✖✖✖✖✖✖ Ptycho keeps going in to LiveScan.check() after all frames have been acquired, which overwrites
            then self.end_of_scan..
            ✖ SOLUTION: Move 'self.end_of_scan = False' under LiveScan.init() instead of under LiveScan.check().
✖✖✖✖✖✖✖✖✖ There is still a small difference in the resulting object and exit wave compared to original script.
            Exit waves, masks, object etc get their values already during ptycho level 2 ( P.init_data() )!
            While the object is still just a uniform matrix at this point, the value it
            is filled with differs between livescan and 1222!
            - P.probe, Pobj, Pexit gets their first view and storage in line 981 of 'manager.py' (during P2) but are
                still just a zero matrix at this point.
            - P.probe is filled with values at line 1122 of 'manager.py'
            ✖ SOLUTION: Difference occurs in line359 of sample.py and comes from different precisions in variable k,
                which inherits the type from the energy! In 1222_...py energy gets read into a ndarray of
                size (1,) with type float64, whereas my energy is loaded directly as a float, i.e. float32!!
✖✖✖✖✖✖✖✖✖ Check if there is a way to see how many pods/frames that are included in each .ptyr file.
            ✖ SOLUTION: Using inspect.getouterframes(..) to retrieve variables from classes/methods/functions that
                calls on LiveScan. This is implemented in my fynction "self.BackTrace()"
"""

"""
To do:
* Check how the time for loading/Repacking data changes with the number of frames
-----------------------------------------------------------
Notes:
-----------------------------------------------------------
Subclasses of PtyScan can be made to override to tweak the methods of base class PtyScan.
Methods defined in PtyScan(object) are:
    ¨def __init__(self, pars=None, **kwargs):
    def initialize(self):
    ¨def _finalize(self):
    ^def load_weight(self):
    ^def load_positions(self):
    ^def load_common(self):
    def post_initialize(self):
    ¨def _mpi_check(self, chunksize, start=None):
    ¨def _mpi_indices(self, start, step):
    def get_data_chunk(self, chunksize, start=None):
    def auto(self, frames):
    ¨def _make_data_package(self, chunk):
    ¨def _mpi_pipeline_with_dictionaries(self, indices):
    ^def check(self, frames=None, start=None):
    ^def load(self, indices):
    ^def correct(self, raw, weights, common):
    ¨def _mpi_autocenter(self, data, weights):
    def report(self, what=None, shout=True):
    ¨def _mpi_save_chunk(self, kind='link', chunk=None):

¨: Method is protected (or private if prefix is __).
^: Description explicitly says **Override in subclass for custom implementation**.
"""

import numpy as np
import zmq
import time
import bitshuffle
import struct
import ptypy
from ptypy.core import Ptycho
from ptypy.core.data import PtyScan
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.experiment import register
from ptypy.utils.verbose import headerline
import inspect
from bitshuffle import decompress_lz4
import re
import h5py

logger = u.verbose.logger
def logger_info(*arg):
    """
    Just an alternative to commenting away logger messages.
    """
    return


##@defaults_tree.parse_doc('scandata.LiveScan')
@register()
class LiveScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = LiveScan
    help =

    [relay_host]
    default = 'tcp://127.0.0.1'
    type = str
    help = Name of the publishing host
    doc =

    [relay_port]
    default = 45678
    type = int
    help = Port number on the publishing host
    doc =

    [xMotor]
    default = sx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = sy
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

    [detector]
    default = 'diff'
    type = str
    help = Which detector from the contrast stream to use

    [block_wait_count]
    default = 0
    type = int
    help = Signals a WAIT to the model after this many blocks

    [start_frame]
    default = 1
    type = int
    help = Minimum number of frames loaded before starting iterations

    [frames_per_iter]
    default = None
    type = int
    help = Load a fixed number of frames in between each iteration

    [crop_at_RS]
    default = None
    type = int, tuple
    help = Cropping dimension of the diffraction, performed in the RelayServer.
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).

    [rebin_at_RS]
    default = None
    type = int
    help = Rebinning factor for the raw data frames used by the RelayServer.
      ``'None'`` or ``1`` both mean *no binning*

    [average_x_at_RS]
    default = None
    type = str
    help = Which x motor to use for averaging positions.
     Only used when there is 2 x- and y positions per frame.

    [average_y_at_RS]
    default = None
    type = str
    help = Which y motor to use for averaging positions.
     Only used when there is 2 x- and y positions per frame.

    [maskfile]
    default = None
    type = str
    help = Path to maskfile.h5

    [backgroundfile]
    default = None
    type = str
    help = Path to backgroundfile
    """


    def __init__(self, pars=None, **kwargs):

        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        p = self.DEFAULT.copy(depth=99)
        p.update(pars)
        p.update(kwargs)

        super(LiveScan, self).__init__(p, **kwargs) # To get the parent of LiveScan, e.g. PtyScan

        self.end_of_scan = False
        self.energy_replied = False
        self.interaction_started = False
        self.latest_frame_index_received = -1
        self.checknr = 0
        self.checknr_external = 0
        self.loadnr = 0
        self.checktottime = 0
        self.loadtottime = 0
        self.preprocess_RS = {}
        self.t = time.gmtime()
        self.t = f'{self.t[0]}-{self.t[1]:02d}-{self.t[2]:02d}__{self.t[3]:02d}-{self.t[4]:02d}'

        self.p = p


        try:
            self.BT_fname = re.sub(r'(.*/).*/.*', rf'\1backtrace_{time.strftime("%F_%H:%M:%S", time.localtime())}.txt', self.p.dfile)
        except:
            self.BT_fname = None
            print("Warning: Couldn't write a self.BackTrace-file.")
        self.BT_logfname = '/data/staff/nanomax/commissioning_2022-2/reblex/interaction_log.txt'##'/mxn/home/reblex/interaction_log.txt' # Will have to be updated on official release.

        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

    def initialize(self):
        # main socket: reporting images, positions and motor stuff from RelayServer

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("%s:%u" % (self.info.relay_host, self.info.relay_port))

        # !#
        if self.info.crop_at_RS is not None:
            self.preprocess_RS['shape'] = self.info.crop_at_RS
            self.preprocess_RS['center'] = self.p.center
        if self.info.rebin_at_RS is not None and self.info.rebin_at_RS != 1:
            self.preprocess_RS['rebin'] = self.info.rebin_at_RS
        if self.info.average_x_at_RS is not None:
            self.preprocess_RS['average_x_at_RS'] = self.info.average_x_at_RS
        if self.info.average_x_at_RS is not None:
            self.preprocess_RS['average_x_at_RS'] = self.info.average_x_at_RS
        # if self.info.maskfile is not None and (self.info.rebin_at_RS or self.info.crop_at_RS):
        #     self.preprocess_RS['maskfile'] = self.info.maskfile

        self.socket.send_json(['preprocess', self.preprocess_RS])
        self.socket.recv_json()
        # !#
        super(LiveScan, self).initialize()

        self.meta.energy = self.common['energy']  # common gets data into all the ranks

        if self.info.backgroundfile is not None:
            with h5py.File(self.info.backgroundfile, 'r') as fp:
                self.data_background = fp['/entry/instrument/zyla/data'] ### DEBUG: HARDCODED
                self.data_background = np.mean(self.data_background, axis=0)## check data type, change to float 32
            if self.info.crop_at_RS is not None:
                self.data_background = u.crop_pad_symmetric_2d(self.data_background, (self.info.crop_at_RS, self.info.crop_at_RS), center=self.p.center)[0]
            if self.info.rebin_at_RS is not None:
                self.data_background = u.rebin_2d(self.data_background, self.info.rebin_at_RS)[0]

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                self.mask_data = np.array(hf.get('mask'))
            if self.info.crop_at_RS is not None:
                self.mask_data = u.crop_pad_symmetric_2d(self.mask_data, (self.info.crop_at_RS, self.info.crop_at_RS), center=self.p.center)[0]
            if self.info.rebin_at_RS is not None:
                self.mask_data = u.rebin_2d(self.mask_data, self.info.rebin_at_RS)[0]
            logger_info('############## Loading mask! mask.shape = %s, np.sum(mask) = %s' % (str(self.mask_data.shape), str(np.sum(self.mask_data))))  ### DEBUG




    def load_common(self):
        """
        Load meta data such as energy
        :return:
        """

        logger.info('Waiting for energy-reply from RelayServer..')
        while not self.energy_replied:
            self.socket.send_json(['check_energy'])
            msg = self.socket.recv_json()
            if msg['energy'] != False:
                # self.meta.energy = np.float64([msg['energy']]) * 1e-3  ## Read energy from beamline snapshot
                common_dct = {'energy': np.float64([msg['energy']]) * 1e-3}
                logger.info('### Energy set to %f keV' % common_dct['energy'])
                self.energy_replied = True
                break
            time.sleep(1)
        return common_dct


    def check(self, frames=None, start=None):
        """
        Only called on the master node.

        Parameters
        ----------
        frames : int or None
            Number of frames requested.
        start : int or None
            Scanpoint index to start checking from.

        Returns
        -------
        frames_accessible : int
            Number of frames readable.

        end_of_scan : int or None
            is one of the following,
            - 0, end of the scan is not reached
            - 1, end of scan will be reached or is
            - None, can't say
        """
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        self.BackTrace()
        t0 = time.perf_counter()
        self.checknr += 1
        self.checknr_external += 1
        logger.info('check() has now been called %d times in total, and %d times externally.' % (self.checknr, self.checknr_external))  ##
        ###self.BackTrace(self.BT_fname, extra={'checknr': self.checknr, 'checknr_external': self.checknr_external, 'latest_frame_index_received': self.latest_frame_index_received, 'frames': frames, 'start': start})

        rank = parallel.rank
        logger.info("### I'm check-rank nr  = %s" % rank)
        if not self.interaction_started:
            self.BackTrace(plotlog=self.BT_logfname)

        self.socket.send_json(['check'])
        msg = self.socket.recv_json()
        logger.info('#### check message = %s' % msg)

        if self.checknr == 1:
            self.BackTrace(self.BT_fname,
                      extra={'frames': frames,
                             'p.frames_per_iter': self.p.frames_per_iter,
                             'min_frames':        self.min_frames})
        self.BackTrace(self.BT_fname,
                  extra={'checknr': self.checknr,
                         'return': [min(frames, msg[0]), msg[1]],
                         'frames_accessible': msg[0],
                         'start': start})
        ##### DEBUG:
        if min(frames, msg[0]) == 0:
            time.sleep(0.6)

        t1 = time.perf_counter()
        self.checktottime += t1 - t0
        logger.info('#### Time spent in check = %f, accumulated time = %f' % ((t1-t0), self.checktottime))
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().check() at time %s' % time.strftime("%H:%M:%S", time.localtime()), 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')
        return min(frames, msg[0]), msg[1]

        # while True:
        #     self.socket.send_json(['check'])
        #     msg = self.socket.recv_json()
        #     logger.info('#### check message = %s' % msg)
        #     self.latest_frame_index_received = msg[0] - 1 # could also set this to =msg[0] and delete the "+1" in the return..
        #     self.frames_accessible = msg[0]  # Total nr of frames accessible
        #     self.end_of_scan = msg[1]
        #
        #     self.BackTrace(self.BT_fname,
        #               extra={'checknr': self.checknr, 'checknr_external': self.checknr_external,
        #                      'return': [(self.latest_frame_index_received - start + 1), self.end_of_scan],
        #                      'latest_frame_index_received': self.latest_frame_index_received, 'p.num_frames': self.p.num_frames,
        #                      'frames': frames, 'start': start, 'frames_accessible': self.frames_accessible})
        #
        #     ## frames_per_iter = None, 1, 2, 3, 4, .. ## start_frame = 1, 2, 3, .., 55
        #     # Start iterations after all frames are gathered (start_frame= nr of frames):
        #     #     bwc=0, frames_per_iter=None, start_frame=55, min_frames=1
        #     # Check return whatever is available and iterate on that:
        #     #     bwc=1, frames_per_iter=None, start_frame=1, min_frames=1
        #     # Check return whatever is available, after at least start_frame has been recieved
        #     #     bwc=1, frames_per_iter=None, start_frame=14, min_frames=1
        #     # Check always return 2 frames:
        #     #     bwc=1, frames_per_iter=2, start_frame=2, min_frames=1?
        #     # Start iterations after least start_frame has been recieved, then Check always return 1 frame:
        #     #     bwc=1, frames_per_iter=1, start_frame=14, min_frames=1
        #     #replace start_frame with new ptycho param, use input param "frame"
        #
        #     if self.info.frames_per_iter == None and self.frames_accessible < self.info.start_frame and not self.end_of_scan:
        #         logger.info('---------------------------- nr 0, have %u frames in total, %u new ones, waiting...' % (self.frames_accessible, self.frames_accessible - start))
        #         time.sleep(1)
        #     elif self.info.frames_per_iter != None and self.frames_accessible - start < self.info.frames_per_iter and not self.end_of_scan:
        #         logger.info('---------------------------- nr 1, have %u frames in total, %u new ones, waiting...' % (self.frames_accessible, self.frames_accessible - start))
        #         time.sleep(1)
        #     elif self.info.frames_per_iter != None and self.frames_accessible - start >= self.info.frames_per_iter:
        #         logger.info('---------------------------- nr 2')
        #         if self.frames_accessible < self.info.start_frame:
        #             logger.info('---------------------------- nr 3, have %u frames in total, %u new ones, waiting...' % (self.frames_accessible, self.frames_accessible - start))
        #             time.sleep(1)
        #         elif self.checknr_external == 1 and self.frames_accessible >= self.info.start_frame:
        #             logger.info('---------------------------- nr 4, actually have %u frames, but will set it to %u' % (self.frames_accessible, start + self.info.start_frame))
        #             self.frames_accessible = start + self.info.start_frame
        #             break
        #         else:
        #             logger.info('---------------------------- nr 5, actually have %u frames, but will set it to %u' % (self.frames_accessible, start + self.info.frames_per_iter))
        #             self.frames_accessible = start + self.info.frames_per_iter
        #             break
        #     else:
        #         logger.info('---------------------------- nr 6')
        #         break
        #
        #     """# Start iterations after all frames are gathered (start_frame= nr of frames)
        #     if self.frames_accessible < self.info.start_frame:
        #         logger.info('have %u frames, waiting...' % self.frames_accessible)
        #         time.sleep(1)
        #     elif self.info.start_frame >= 2 and self.checknr_external == 1:
        #         break
        #     # Always return the same nr of frames
        #     elif self.info.frames_per_iter != None:  #and self.frames_accessible - start >= self.info.frames_per_iter:
        #         if self.frames_accessible - start >= self.info.frames_per_iter:
        #             logger.info('updating frames_accessible from %d to %d according to frames_per_iter' % (self.frames_accessible, self.info.frames_per_iter))
        #             self.frames_accessible = start + self.info.frames_per_iter
        #             break
        #         else:
        #             logger.info('not enough frames, have %u frames, waiting...' % self.frames_accessible)
        #             time.sleep(1)
        #         # self.checknr_external -= 1
        #         # self.check(frames=frames, start=start)
        #     # ### DEBUG: GPU memory error. Fixed nr of loaded frames each time.
        #     # elif self.latest_frame_index_received - start + 1 >= 1:
        #     #     self.latest_frame_index_received = start
        #     #     break
        #     else:
        #         break"""
        #
        # #
        # # if self.checknr_external >= 5 and (self.latest_frame_index_received - start + 1) <= 2:
        # #     return 0, self.end_of_scan
        #
        # logger.info('#### check return [self.frames_accessible - start), self.end_of_scan]  =  [(%d - %d), %s] = [%d, %s]' % (self.frames_accessible, start, self.end_of_scan, (self.frames_accessible-start), self.end_of_scan))
        # t1 = time.perf_counter()
        # self.checktottime += t1-t0
        # logger.info('#### Time spent in check = %f, accumulated time = %f' % ((t1-t0), self.checktottime))
        # logger.info(headerline('', 'c', '#'))
        # logger.info(headerline('Leaving LiveScan().check() at time %s' % time.strftime("%H:%M:%S", time.localtime()), 'c', '#'))
        # logger.info(headerline('', 'c', '#') + '\n')
        # #!#return (self.frames_accessible - start), self.end_of_scan
        # return (self.frames_accessible), self.end_of_scan


    def load(self, indices):
        """indices are generated by PtyScan's _mpi_indices method.
        It is a diffraction data index lists that determine
        which node contains which data."""
        ### ToDo: add feature for asking about IO data, and normalize with raw[i] = io[i] / np.mean(io[:i+1])
        ### ToDo: See if I can update the values of diffraction patterns after they have been loaded (to get a more accurate IO normalization)
        raw, weight, pos = {}, {}, {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().load() at time %s' % time.strftime("%H:%M:%S", time.localtime()), 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        t0 = time.perf_counter()
        self.loadnr += 1
        logger.info('load() has now been called %d times.' % self.loadnr)
        self.BackTrace()
        logger_info('### parallel.master = %s, parallel.size = %s' % (str(parallel.master), str(parallel.size)))  ### DEBUG

        logger.info('### indices = %s' % indices)  ### DEBUG
        self.socket.send_json(['load', {'frame': indices}])
        msgs = self.socket.recv_pyobj()
        buff = self.socket.recv(copy=True)
        imgs = decompress_lz4(np.frombuffer(buff, dtype=np.dtype('uint8')), msgs[0]['shape'], msgs[0]['dtype'])
        if self.loadnr == 1 and 'new_center' in msgs[0].keys():
            self.info.center = msgs[0]['new_center']
        if self.loadnr == 1 and 'RS_rebinned' in msgs[0].keys():
            print(f'self.info.psize = {self.info.psize}, self.info.rebin_at_RS = {self.info.rebin_at_RS}, self.meta.psize = {self.meta.psize}')  ###DEBUG
            if msgs[0]['RS_rebinned']:
                # Maybe not the best solution to let rebin_at_RS be of type bool,
                # since the only way of seeing the rebinning factor used in RS is
                # to compare the shapes in meta and info of the .ptyd file..
                self.info.shape = u.expect2(self.info.shape) // self.info.rebin_at_RS
                if self.info.psize is not None:
                    self.meta.psize = u.expect2(self.info.psize) * self.info.rebin_at_RS
                    self.info.psize = u.expect2(self.info.psize) * self.info.rebin_at_RS
            #     self.rebin = 1
            # else:
            #     # Setting this to False to get correct info when writing to .ptyd
            #     self.info.rebin_at_RS = False
        if self.loadnr == 1 and len(imgs) == 3:
            # Then imgs contain both diff, weights and mask.
            w = imgs[1]
            self.mask_data = imgs[2]
            imgs = imgs[0]
            print(f'w.shape = {w.shape}, imgs.shape = {imgs.shape}, mask.shape = {self.mask_data.shape}')  ### DEBUG
        elif self.info.rebin_at_RS:
            # Then imgs contain both diff and weights.
            print(f'len(imgs) = {len(imgs)}')### DEBUG
            w = imgs[1]
            imgs = imgs[0]
            print(f'w.shape = {w.shape}, imgs.shape = {imgs.shape}')### DEBUG
            print(f'w.shape = {w.shape}, imgs.shape = {imgs.shape}, mask.shape = {self.mask_data.shape}, bgdata.shape = {self.data_background.shape}')  ### DEBUG


        imgs = imgs.astype(np.float32)
        # repackage data and return
        for k, i in enumerate(indices):
            try:
                if self.info.backgroundfile is not None:
                    imgs[k] = imgs[k] - self.data_background
                raw[i] = imgs[k]
                raw[i][raw[i] <= 0] = 0  ##  Take care of overexposed pixels.

                logger_info('### i = %s, raw[i].shape = %s' % (str(i), str(raw[i].shape))) ### DEBUG

                xMotorKeys = self.info.xMotor.split('/')
                yMotorKeys = self.info.yMotor.split('/')
                x = y = msgs[k]
                for xkey in xMotorKeys:
                    x = x[xkey]
                for ykey in yMotorKeys:
                    y = y[ykey]

                if self.info.xMotorFlipped:
                    x *= -1
                if self.info.yMotorFlipped:
                    y *= -1

                pos[i] = np.array((y, x)) * 1e-9 #### CHECK IF THIS SHOULD BE MINUS!
                pos[i] = pos[i].reshape(len(pos[i]))
                if self.info.rebin_at_RS:
                    weight[i] = w[k]
                else:
                    weight[i] = np.ones_like(raw[i])
                    weight[i][np.where(raw[i] == 2 ** 32 - 1)] = 0
                    weight[i][np.where(raw[i] < 0)] = 0
                if self.info.maskfile:
                    weight[i] = weight[i] * self.mask_data
                logger_info('### weight[i].shape = %s' % str(weight[i].shape)) ### DEBUG
            except Exception as err:
                logger.info('### load exception')  ### DEBUG
                print('Error: ', err)
                break



        t1 = time.perf_counter()
        self.loadtottime += t1 - t0
        logger.info('#### Time spent in load = %f, accumulated time = %f' % ((t1 - t0), self.loadtottime))
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().load() at time %s' % time.strftime("%H:%M:%S", time.localtime()), 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

        return raw, pos, weight

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first
        frame.

        Will not be called on if I provide a weight in return load! => I can probably remove this
        """
        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask = np.array(hf.get('mask'))
            logger_info('############## Inside load_weight(), mask.shape = %s, np.sum(mask) = %s' % (str(mask.shape), str(np.sum(mask))))  ### DEBUG
            return mask

    def _finalize(self):
        """
        Close the socket to the RelayServer and tell it that it's OK to close the socket from its end as well.
        """
        super()._finalize()

        self.socket.send_json(['stop'])
        reply = self.socket.recv_json()
        logger.info('Closing the relay_socket at %s' % time.strftime("%H:%M:%S", time.localtime()))
        self.socket.close()
        self.context.term()
        time.sleep(1)

    def BackTrace(self, fname=None, writeBTsummary=True, plotlog=None, request_Ptycho=None, extra={}, verbose=False):
        """
        :param self:
        :param fname:   filename where self.BackTrace info will be written to
        :param plotlog: filename where ptypy interaction info will be written, used for jupyter liveplotting
        :param extra:   Extra parameters that should be written to the file 'fname'
        :param verbose: False,
                        1: Print traceback summary,
                        2: Print full traceback and names of local variables in each frame,
                        3: Print both verbose = 1 and 2
        :return:
        """
        logger.info(headerline('', 'c', '°'))
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        if verbose in [1, 3]:
            print(*[(frame.lineno, frame.filename, frame.function) for frame in calframe], "", sep="\n")
        if verbose in [2, 3]:
            print('calframe.__len__() = %d' % calframe.__len__())
            try:
                for fr in range(0, calframe.__len__()):
                    print(f'calframe[{fr}] =')
                    for calkey, calval in calframe[fr]._asdict().items():
                        print('\t', str(calkey + ':').ljust(14, ' '), calval)
                    print(f'calframe[{fr}][0].f_locals.keys() = \n\t', calframe[fr][0].f_locals.keys(), '\n')
            except:
                pass
        try:
            ptychoframe = [frame.filename for frame in calframe].index(ptypy.core.ptycho.__file__)
            ptycho_self = calframe[ptychoframe][0].f_locals['self']
            #### ptycho_self.print_stats()
            active_pods = sum(1 for pod in ptycho_self.pods.values() if pod.active)
            all_pods = len(ptycho_self.pods.values())
            print(f'---- Total Pods {all_pods} ({active_pods} active) ----')
        except:
            pass
        if calframe[ptychoframe].function == 'run':
            ptycho_engine = calframe[ptychoframe][0].f_locals['engine']
            print('ptycho_engine.curiter = ', ptycho_engine.curiter)
        if fname != None:
            with open(fname, 'a') as f:
                if extra != {}:
                    if 'min_frames' in extra:
                        extra_P = {'frames_per_block':     ptycho_self.frames_per_block,
                                   'min_frames_for_recon': ptycho_self.p.min_frames_for_recon,
                                   'numiter':              ptycho_self.p.engines.engine00.numiter,
                                   'numiter_contiguous':   ptycho_self.p.engines.engine00.numiter_contiguous}
                        f.write(f'{extra}, {extra_P} \n\n')
                    else:
                        f.write(f'\n{extra} \n')
                        f.write(f'Total Pods: {all_pods} ({active_pods} active), \n')
                        try:
                            f.write(f'Iteration:  {ptycho_engine.curiter}, \n\n\n')
                        except:
                            pass
        if writeBTsummary and fname != None and 'start' in extra.keys():
            fnamesummary = fname.replace('backtrace', 'backtrace-summary')
            try:
                with open(fnamesummary, 'r+') as f2:
                    text = f2.read()
            except:
                pass
            try:
                with open(fnamesummary, 'a+') as f2:
                    if f'iteration: {ptycho_engine.curiter}' not in text:
                        f2.write(f'start: {str(extra["start"]).ljust(4, " ")}, iteration: {str(ptycho_engine.curiter).ljust(5, " ")}, \n')
            except Exception as ex:
                pass
        if plotlog != None:
            if ptycho_self.interactor != None:
                self.interaction_started = True
                addr = ptycho_self.interactor.address
                port = ptycho_self.interactor.port
                print('++++++++++++++++ ptycho_self ++++++++++++++++')
                print(f'Connected to :::::: {addr}:{port}')
                print('+++++++++++++++++++++++++++++++++++++++++++++')
                print(f'writing to {plotlog}')
                with open(plotlog, 'w') as f3:
                    f3.write(f'address={addr}\nport={port}\ntime={time.strftime("%H:%M:%S", time.localtime())}')
        if request_Ptycho != None:
            print('++++++++++++++++ ptycho_self ++++++++++++++++')
            for key, val in ptycho_self.__dict__.items():
                print(key, ':', val)
            print()
            print(ptycho_self.interactor)
            print('+++++++++++++++++++++++++++++++++++++++++++++')

        logger.info(headerline('', 'c', '°') + '\n')


