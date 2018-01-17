'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
import numpy as np
from ptypy.gpu import data_utils as du
from ptypy.core import View, Container, Base, POD, geometry, xy
from ptypy import utils as u
from ptypy.resources import moon_pr, flower_obj


class DataUtilsTest(unittest.TestCase):
    '''
    tests the conversion between pods and numpy arrays
    '''

    def setUp(self):
        P = Base()
        P.CType = np.complex128
        P.FType = np.float64
        g = u.Param()
        g.energy = None  # u.keV2m(1.0)/6.32e-7
        g.lam = 5.32e-7
        g.distance = 15e-2
        g.psize = 24e-6
        g.shape = 256
        g.propagation = "farfield"
        G = geometry.Geo(owner=P, pars=g)
        fsize = G.shape * G.resolution
        P.probe = Container(P, 'Cprobe', data_type='complex')
        pr = -moon_pr(G.shape)
        pr = P.probe.new_storage(data=pr, psize=G.resolution)
        pos = u.Param()
        pos.model = "round"
        pos.spacing = fsize[0]/8
        pos.steps = None
        pos.extent = fsize*1.5
        positions = xy.from_pars(pos)
        P.obj = Container(P, 'Cobj', data_type='complex')
        oar = View.DEFAULT_ACCESSRULE.copy()
        oar.storageID = 'S00'
        oar.psize = G.resolution
        oar.layer = 0
        oar.shape = G.shape
        oar.active = True
        for pos in positions:
            # the rule
            r = oar.copy()
            r.coord = pos
            _V = View(P.obj, None, r)

        probe_ar = View.DEFAULT_ACCESSRULE.copy()
        probe_ar.psize = G.resolution
        probe_ar.shape = G.shape
        probe_ar.active = True
        probe_ar.storageID = pr.ID

        exit_ar = probe_ar.copy()
        exit_ar.layer = 0
        exit_ar.active = True

        diff_ar = probe_ar.copy()
        diff_ar.layer = 0
        diff_ar.active = True
        diff_ar.psize = G.psize
        mask_ar = diff_ar.copy()

        storage = P.obj.storages['S00']
        storage.fill(flower_obj(storage.shape[-2:]))
        P.exit = Container(P, 'Cexit', data_type='complex')
        P.diff = Container(P, 'Cdiff', data_type='real')
        P.mask = Container(P, 'Cmask', data_type='real')
        objviews = P.obj.views.values()
        pods = []

        for obview in objviews:
            # we keep the same probe access
            prview = View(P.probe, None, probe_ar)
            # For diffraction and exit wave we need to increase the
            # layer index as there is a new exit wave and diffraction
            # pattern for each
            # scan position
            exit_ar.layer += 1
            diff_ar.layer += 1
            exview = View(P.exit, None, exit_ar)
            maview = View(P.mask, None, mask_ar)
            diview = View(P.diff, None, diff_ar)
            views = {'probe': prview,
                     'obj': obview,
                     'exit': exview,
                     'diff': diview,
                     'mask': maview}
            pod = POD(P, ID=None, views=views, geometry=G)
            pods.append(pod)

        # We let the storage arrays adapt to the new Views.
        for C in [P.mask, P.exit, P.diff, P.probe]:
            C.reformat()

        # And the rest of the simulation fits in three lines of code!
        for pod in pods:
            pod.exit = pod.probe * pod.object
            pod.mask = np.ones_like(pod.diff)

        self.pods = pods

    def test_pod_to_numpy(self):


    def test_numpy_to_pod(self):
        pass

    def test_numpy_pod_consistency(self):
        pass

if __name__ == "__main__":
    unittest.main()
