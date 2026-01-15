# -*- coding: utf-8 -*-
"""\
Created on Nov 22 2013

@author: Bjeorn Enders
"""
import numpy as np
import time
import glob
from . import spec
from .. import utils as u
from ..core.data import PtyScan
from . import register

logger = u.verbose.logger

pp = u.Param()
pp.filename = './foo.ptyd'
pp.roi =None
pp.num_frames = 50
pp.save = 'extlink'


@register()
class FliSpecScanMultexp(PtyScan):
    """
    Defaults:

    [name]
    default = FliSpecScanMultexp
    type = str
    help =

    [base_path]
    default = '/data/CDI/opticslab_sxdm_2013/'
    type = str
    help =

    [scan_number]
    default = 74
    type = int
    help =

    [dark_number]
    default = 72
    type = int
    help =

    [exp_string]
    default = 'exp_time'
    type = str
    help =

    [hdr_thresholds]
    default = [500,50000]
    type = list
    help =

    [lam]
    default = 650e-9
    type = float
    help =

    [energy]
    default = None

    [z]
    default = 0.158
    type = float
    help =

    [psize_det]
    default = 24e-6
    type = float
    help =

    [center]
    default = 'auto'

    [orientation]
    default = (True,True,False)

    [base_path]
    default = '/data/CDI/opticslab_sxdm_2013/'
    type = str
    help =

    [scan_dir]
    default = 'ccdfli/S00000-00999/'
    type = str
    help =

    [log_file_pattern]
    default = '%(base_path)sspec/dat-files/spec_started_2013_11_21_1659.dat'
    type = str
    help =

    [data_dir_pattern]
    default = '%(base_path)s%(scan_dir)sS%(scan_number)05d/'
    type = str
    help =

    [dark_dir_pattern]
    default = '%(base_path)s%(scan_dir)sS%(dark_number)05d/'
    type = str
    help =

    """

    def __init__(self,pars=None,**kwargs):
        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        #self.p = pars
        super(FliSpecScanMultexp,self).__init__(p,**kwargs)
        logger.info(u.verbose.report(self.info))
        self.info.log_file = self.info.log_file_pattern % self.info
        self.info.dark_dir = self.info.dark_dir_pattern % self.info
        self.info.data_dir = self.info.data_dir_pattern % self.info
        self.nexp = len(glob.glob(self.info.dark_dir + '/ccd*_00000_??.raw'))

    def load_common(self):
        # Load 'spec' file
        self.specinfo = spec.SpecInfo(self.info.log_file)

        # Grab scan info
        try:
            self.scaninfo = self.specinfo.scans[self.info.scan_number]
        except:
            self.scaninfo = None

        common = u.Param()
        dark_imgs = []
        exposures =[]
        for j in range(self.nexp):
            darks,meta = u.image_read(self.info.dark_dir + '/ccd*_%02d.raw' % j)
            dark_imgs.append(np.array(darks,dtype=float).mean(0))
            exposures.append(meta[0][self.exp_string])

        # save in common dict/Param
        common.darks = np.asarray(dark_imgs)
        common.exposures = np.asarray(exposures)
        if self.scaninfo is not None:
            motor_mult = 1e-3
            x=self.scaninfo.data['samy'][::self.nexp]
            y=self.scaninfo.data['samx'][::self.nexp]
            common.positions_scan = motor_mult * np.array([x, y]).T
        else:
            common.positions_scan = None

        return common._to_dict()

    def check(self,frames_requested, start=0):

        npos = len(glob.glob(self.info.data_dir + '/ccd*_%02d.raw' % (self.nexp-1)))
        # essential!
        frames_accessible = min((frames_requested,npos-start))
        stop = self.frames_accessible + start
        # essential!
        #print start, npos, frames_requested
        #print frames_accessible
        return frames_accessible,(stop >= self.num_frames)


    def load(self,indices):
        raw = u.Param()
        raw = {}
        pos = {}
        weights = {}
        for j in indices:
            data,meta = u.image_read(self.info.data_dir + '/ccd*_%05d_??.raw' % j)
            raw[j] = np.asarray(data)

        return raw, pos, weights

    def correct(self, raw, weights, common):
        #chunk = u.Param()
        data = {}
        weights = {}
        expos = common['exposures']
        darks = common['darks']
        for j, rr in raw.items():
            data_hdr,lmask=u.hdr_image(rr, expos, thresholds=self.hdr_thresholds, dark_list=darks, avg_type='highest')
            data[j] = data_hdr
            weights[j] = lmask[-1]

        return data, weights

# if __name__ == '__main__':
#     u.verbose.set_level(3)
#     RS = RawScan(p,num_frames=50,roi=512 )
#     RS.initialize()
#     RS.report()
#     print('loading data')
#     msg = True
#     for i in range(200):
#         if msg is False:
#             break
#         time.sleep(1)
#         msg = RS.auto(10)
#         logger.info(u.verbose.report(msg), extra={'allprocesses': True})
#         u.parallel.barrier()

    #RS.report()
    #%RS.load_raw([0,1,2])
    #RS.prepare()




