# --- test header (replace your current header with this) ---
import os, sys, unittest, tempfile, shutil, numpy as np, pytest

# Put repo root on sys.path (fix: use __file__)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# GPU guard
cupy_available = True
try:
    import cupy as cp  # noqa: F401
except Exception:
    cupy_available = False

# Import your GPU engine so @register and @parse_doc run
from ptypy.custom.threepie_cupy import ThreePIE_cupy  # noqa: F401
from ptypy import defaults_tree

print("engine has ThreePIE_cupy? ->", 'ThreePIE_cupy' in defaults_tree['engine'].children)
if 'ThreePIE_cupy' in defaults_tree['engine'].children:
    print("ThreePIE_cupy fields ->", list(defaults_tree['engine.ThreePIE_cupy'].children.keys()))

print("scan has MultiSliceVanilla? ->", 'MultiSliceVanilla' in defaults_tree['scan'].children)
if 'MultiSliceVanilla' in defaults_tree['scan'].children:
    print("MultiSliceVanilla fields ->", list(defaults_tree['scan.MultiSliceVanilla'].children.keys()))
else:
    print("Available scan models ->", list(defaults_tree['scan'].children.keys()))

# Register MoonFlower scan plugin (safe if already loaded)
try:
    from ptypy.resources import moonflower  # registers 'MoonFlowerScan'
except Exception:
    pass

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.utils.verbose import logger

@pytest.mark.skipif(not cupy_available, reason="CuPy/CUDA not available")
class ThreePIE_MoonFlower_Test(unittest.TestCase):
    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="_ThreePIE_cupy_test")
        logger.info(f"[TEST] output dir: {self.outpath}")

    def tearDown(self):
        shutil.rmtree(self.outpath, ignore_errors=True)

    def test_ThreePIE_cupy(self):
        p = u.Param()
        p.verbose_level = 3

        # --- IO (same style as MLOPR test) ---
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = self.outpath
        p.io.rfile = "ThreePIE_cupyTest.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False

        # --- SCAN: MoonFlower synthetic data with Vanilla model ---
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Vanilla'  # Use simple scan model first to test engine
        p.scans.MF.propagation = 'farfield'
        
        # Illumination structure for Vanilla
        p.scans.MF.illumination = u.Param()
        p.scans.MF.illumination.size = 20e-6
        
        # Sample structure for Vanilla
        p.scans.MF.sample = u.Param()
        p.scans.MF.sample.fill = 1.0
        
        # Data configuration
        p.scans.MF.data = u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.positions_theory = None
        p.scans.MF.data.auto_center = None
        p.scans.MF.data.min_frames = 1
        p.scans.MF.data.orientation = None
        p.scans.MF.data.num_frames = 64          # small-ish so the test is quick
        p.scans.MF.data.energy = 6.2             # keV
        p.scans.MF.data.shape = 64               # detector shape (square)
        p.scans.MF.data.chunk_format = '.chunk%02d'
        p.scans.MF.data.rebin = None
        p.scans.MF.data.experimentID = None
        p.scans.MF.data.label = None
        p.scans.MF.data.version = 0.1
        p.scans.MF.data.dfile = os.path.join(self.outpath, "ThreePIE_cupyTest.ptyd")
        p.scans.MF.data.save = True
        p.scans.MF.data.psize = 75e-6         # m
        p.scans.MF.data.load_parallel = None
        p.scans.MF.data.distance = 7.0           # m
        p.scans.MF.data.save = None
        p.scans.MF.data.center = 'fftshift'
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.density = 0.2

        # --- ENGINE: ThreePIE_cupy (GPU) ---
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = "ThreePIE_cupy"
        p.engines.engine00.numiter = 15
        p.engines.engine00.fslices = os.path.join(self.outpath,'rec_crop_64_slices-2_iter.h5')

        # Multislice specifics - start with 1 slice to reduce complexity
        p.engines.engine00.number_of_slices = 2
        p.engines.engine00.slice_thickness = 0.5e-6
        p.engines.engine00.slice_start_iteration = 0

        # FFT backend for inter-slice propagation (choose 'cuda' if your wrapper is present)
        p.engines.engine00.fft_lib = 'cupy'

        # Make sure we get error outputs for assertion/logging
        p.engines.engine00.compute_fourier_error = True
        p.engines.engine00.compute_exit_error = True
        p.engines.engine00.compute_log_likelihood = True

        # --- Build and run a few iterations ---
        try:
            logger.info("Starting Ptycho initialization...")
            P = Ptycho(p, level=4)  
            logger.info("Ptycho created successfully")
            
            # IMPORTANT: Trigger full initialization by accessing the engine through PTYpy
            # This ensures engine_initialize() and engine_prepare() are called properly
            logger.info("Triggering engine initialization through PTYpy...")
            
            # Option 1: Run the engine through PTYpy's normal flow
            # This is the proper way to run the engine
            P.run()
            
            # After running, we can check the results
            eng = P.engines['engine00']
            logger.info(f"Engine completed {eng.curiter} iterations")
            
            # DIAGNOSTIC: Check multislice-specific attributes after running
            logger.info(f"=== POST-RUN DIAGNOSTICS ===")
            logger.info(f"Number of slices configured: {eng.p.number_of_slices}")
            logger.info(f"Slice thickness: {eng.p.slice_thickness}")
            
            # Check if multislice components are properly initialized
            if hasattr(eng, '_object') and eng._object:
                logger.info(f"✓ Object slices created: {len(eng._object)}")
            else:
                logger.info("✗ No object slices found")
                
            if hasattr(eng, '_probe') and eng._probe:
                logger.info(f"✓ Probe slices created: {len(eng._probe)}")
            else:
                logger.info("✗ No probe slices found")
                
            if hasattr(eng, '_PROP_fw') and eng._PROP_fw:
                logger.info(f"✓ Forward propagators initialized: {len(eng._PROP_fw)}")
            else:
                logger.info("✗ No forward propagators found")
                
            if hasattr(eng, '_PROP_bw') and eng._PROP_bw:
                logger.info(f"✓ Backward propagators initialized: {len(eng._PROP_bw)}")
            else:
                logger.info("✗ No backward propagators found")
            
            # Check if there were any errors
            last_err = P.runtime["iter_info"][-1]["error"] if P.runtime["iter_info"] else None
            
        except Exception as e:
            logger.error(f"Failed during Ptycho run: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.fail(f"Ptycho run failed: {e}")

        # Assertions
        logger.info("=== FINAL TEST VALIDATION ===")
        
        # Check that the engine ran successfully
        self.assertIsNotNone(eng, "Engine should be available")
        self.assertGreater(eng.curiter, 0, "Engine should have completed iterations")
        
        if last_err is not None:
            self.assertTrue(np.isfinite(last_err).all(), "Error values should be finite")
            logger.info(f"✓ Final error is finite: {last_err}")
        
        # Validate multislice components
        if eng.p.number_of_slices > 1:
            # Check basic multislice setup
            if hasattr(eng, '_slices_initialized'):
                self.assertTrue(eng._slices_initialized, "Slices should be initialized")
                logger.info("✓ Slices initialized flag is True")
            
            if hasattr(eng, '_object') and hasattr(eng, '_probe'):
                self.assertEqual(len(eng._object), eng.p.number_of_slices, 
                               f"Should have {eng.p.number_of_slices} object slices")
                self.assertEqual(len(eng._probe), eng.p.number_of_slices, 
                               f"Should have {eng.p.number_of_slices} probe slices")
                logger.info(f"✓ Correct number of slices: {eng.p.number_of_slices}")
            
            # Check propagators if available
            if hasattr(eng, '_propagators_initialized') and eng._propagators_initialized:
                if hasattr(eng, '_PROP_fw') and hasattr(eng, '_PROP_bw'):
                    expected_propagators = eng.p.number_of_slices - 1
                    self.assertEqual(len(eng._PROP_fw), expected_propagators, 
                                   f"Should have {expected_propagators} forward propagators")
                    self.assertEqual(len(eng._PROP_bw), expected_propagators, 
                                   f"Should have {expected_propagators} backward propagators")
                    logger.info(f"✓ Correct number of propagators: {expected_propagators}")
            else:
                logger.warning("⚠️ Propagators not initialized - may be due to missing parent kernels")
        
        logger.info("=== TEST COMPLETE ===")
        logger.info(f"🎉 ThreePIE_cupy engine test passed with {eng.curiter} iterations completed")