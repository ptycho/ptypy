'''
A test for the Base
'''

import unittest
import ptypy.utils as u
import numpy as np
from ptypy.core import geometry, Container
from ptypy.resources import moon_pr
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

def get_P():
    P = Base()
    P.CType = np.complex128
    P.Ftype = np.float64
    return P

def set_up_geometry():
    P = get_P()
    g = u.Param()
    g.energy = None # u.keV2m(1.0)/6.32e-7
    g.lam = 5.32e-7
    g.distance = 15e-2
    g.psize = 24e-6
    g.shape = 256
    g.propagation = "farfield"
    G = geometry.Geo(owner=P, pars=g)
    return G

def set_up_probes(P,G):
    fsize = G.shape * G.resolution
    P.probe = Container(get_P(), 'Cprobe', data_type='complex')
    y, x = G.propagator.grids_sam
    apert = u.smooth_step(fsize[0]/5-np.sqrt(x**2+y**2), 1e-6)
    sh = (1,) + tuple(G.shape)
    #pr = P.probe.new_storage(shape=sh, psize=G.resolution)
    #pr.fill(-moon_pr(G.shape))
    pr2 = P.probe.new_storage(shape=sh, psize=G.resolution)
    pr2.fill(apert)
    pr3 = P.probe.new_storage(shape=sh, psize=G.resolution)
    y, x = pr3.grids()
    apert = u.smooth_step(fsize[0]/5-np.abs(x), 3e-5)*u.smooth_step(fsize[1]/5-np.abs(y), 3e-5)
    pr3.fill(apert)
    #return [pr, pr2, pr3]
    return [pr2, pr3]

def propagate_probes(G, probes):
    propagated_ill = []
    for pp in probes:
        pp.data *= np.sqrt(1e9/np.sum(pp.data*pp.data.conj()))
        propagated_ill.append(G.propagator.fw(pp.data[0]))
    return propagated_ill


class ProbeTest(unittest.TestCase):
    def test_probe(self):
        G = set_up_geometry()
        P = get_P()
        probes = set_up_probes(P,G)
        prop_probes = propagate_probes(G,probes)
        av = [np.mean(foo) for foo in prop_probes]
        #res = np.array([0.006+26.9j, 0.023+23.638j,-0.027+26.908j])
        res = np.array([0.023+23.638j,-0.027+26.908j])
        assert (np.round(np.array(av),3)==res).all(), "Probes are not equal"

if __name__ == '__main__':
    unittest.main()


