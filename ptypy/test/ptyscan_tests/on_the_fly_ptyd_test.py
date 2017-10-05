"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""
import ptypy
from ptypy import utils as u
import tempfile
import sys
import unittest

u.verbose.set_level(3)

TEMPDIR = tempfile.gettempdir()+'/.'+sys.argv[0].replace('.py', '')

# I should refactor this at some point -adp


class MakeSamplePtydTest(unittest.TestCase):
    DATA = u.Param(
        dfile=TEMPDIR + '/sample.ptyd',
        shape=128,
        num_frames=50,
        save=None,
        label=None,
        psize=1.0,
        energy=1.0,
        distance=1.0,
        center='fftshift',
        auto_center=None,
        rebin=None,
        orientation=None,
    )

    def setUp(self):
        # for verbose output
        u.verbose.set_level(3)

        data = self.DATA.copy()
        data.save = 'link'

        # This scan prepares the data and fills a '.ptyd' container
        if u.parallel.master:
            self.S1 = ptypy.core.data.MoonFlowerScan(data)
            self.S1.initialize()
        else:
            self.S1 = None

        # wait until master process complete
        u.parallel.barrier()

    def tearDown(self):
        import os
        for entry in os.listdir(TEMPDIR):
            os.remove(TEMPDIR + '/' + entry)
        os.rmdir(TEMPDIR)

    def _create_PtydScan(self, save='append', **kwargs):
        # the second process will aggregate the linked container to a
        # a new .ptyd file holding all data.
        # This may be done in parallel
        data = self.DATA.copy()
        dfile = str(data.dfile)

        # Base parameters
        from ptypy.core.data import PtyScan
        data = PtyScan.DEFAULT.copy()
        data.dfile = dfile.replace('.ptyd', '_aggregated.ptyd')
        data.save = save # maybe replace with merge in future
        data.update(**kwargs)

        return ptypy.core.data.PtydScan(data, source=dfile)

    def test_non_exisiting_chunk(self):
        try:
            S2 = self._create_PtydScan(save=None)
        except IOError:
            # expected behavior is an IOError if *.ptyd contains no data.
            pass

    def test_incomplete_scan(self):
        if u.parallel.master: msg = self.S1.auto(30)
        u.parallel.barrier()

        S2 = self._create_PtydScan(save=None)
        S2.initialize()
        msg2 = S2.auto(10)
        self.assertEqual(10, len(msg2['iterable']),
                         'There should be 10 frames available in Source ptyd')

        msg3 = S2.auto(30)
        self.assertEqual(20, len(msg3['iterable']),
                         'There should be 20 frames available in Source ptyd')

        msg4 = S2.auto(30)
        # print(u.verbose.report(msg4))
        from ptypy.core.data import WAIT
        self.assertEqual(msg4, WAIT,
                         "Last auto call should return 'wait' flag (data.WAIT) "
                         "but it returned %s." % str(msg4))

    def test_completed_scan(self):
        if u.parallel.master:
            msgs =[
                # 30 frames
                self.S1.auto(30, chunk_form='dp'),
                # should be 20 frames, but asked for hundred
                self.S1.auto(100, chunk_form='dp'),
                # source depleted
                self.S1.auto(100, chunk_form='dp')
            ]
        u.parallel.barrier()
        S2 = self._create_PtydScan(save='append')
        S2.initialize()
        msg = S2.auto(100)
        self.assertEqual(50, len(msg['iterable']),
                         'There should be 20 frames available in Source ptyd')

    def test_check(self):
        if u.parallel.master: msg = self.S1.auto(30)
        u.parallel.barrier()
        S2 = self._create_PtydScan(save='append')
        S2.initialize()
        print(S2.num_frames)
        print(S2.info.num_frames)
        print(S2.check(40))

if __name__ == '__main__':
    unittest.main()

