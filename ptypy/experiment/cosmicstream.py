from __future__ import annotations

import threading
from typing import Any, Callable, Generator, List, Optional

import numpy as np
from cosmicstreams.PtychocamStream import PtychocamStream as PtychoStream
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.utils import parallel
from ptypy.utils.verbose import log

# Instructions:

####
# Install requirements...
####

# - Download ptypy
"""
git clone https://github.com/swelborn/ptypy.git
"""

# - Install ptypy requirements (_full.yml to be safe)
"""
mamba env create -f dependencies_full.yml
conda activate ptypy_full
"""

# - Copy this into a requirements.txt file.

"""
pydantic
matplotlib==3.7
"""

# - pip install it
"""
pip install -r requirements.txt
"""

# - install cosmicstreams
"""
git clone https://github.com/silvioachilles/cosmicstreams.git
pip install -e cosmicstreams/
"""

####
# Run...
####

"""
❯ python
Python 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from ptypy.experiment.cosmicstream import CosmicStreamLoader
>>> loader = CosmicStreamLoader()
>>> loader.initialize()
Listening on localhost:37013 for topic: b'start'
Listening on localhost:37013 for topic: b'frame'
Listening on localhost:37013 for topic: b'stop'
Listening on localhost:37013 for topic: b'abort'
Waiting for metadata...
"""


class Server(BaseModel):
    ip: str
    commandPort: int
    dataPort: int
    dataDir: str
    filePrefix: str


class ZonePlate(BaseModel):
    diameter: float
    outerZone: float
    A0: float
    A1: float
    type: str
    source: str


class Monitor(BaseModel):
    dwell: int


class Ptychography(BaseModel):
    position_jitter: float


class Zmq(BaseModel):
    connect: bool
    address: str


class Geometry(BaseModel):
    distance: float
    psize: float
    shape: int
    resolution: float
    rebin: int
    basis_vectors: List[List[float]]


class Region(BaseModel):
    xStart: float
    xStop: float
    xPoints: int
    yStart: float
    yStop: float
    yPoints: int
    xStep: float
    yStep: float
    xRange: float
    yRange: float
    xCenter: float
    yCenter: float
    zStart: float
    zStop: float
    zPoints: int
    zStep: int
    zRange: int
    zCenter: float


class ScanRegions(BaseModel):
    Region1: Region


class EnergyRegion(BaseModel):
    dwell: float
    start: float
    stop: float
    step: float
    nEnergies: int


class EnergyRegions(BaseModel):
    EnergyRegion1: EnergyRegion


class Image(BaseModel):
    type: str
    proposal: str
    experimenters: str
    sample: str
    x: str
    y: str
    defocus: bool
    mode: str
    scanRegions: ScanRegions
    energyRegions: EnergyRegions
    energy: str
    doubleExposure: bool


class PtychographyImage(BaseModel):
    type: str
    proposal: str
    experimenters: str
    sample: str
    x: str
    y: str
    defocus: bool
    mode: str
    scanRegions: ScanRegions
    energyRegions: EnergyRegions
    energy: str
    doubleExposure: bool


class Focus(BaseModel):
    type: str
    proposal: str
    experimenters: str
    sample: str
    x: str
    y: str
    defocus: bool
    mode: str
    scanRegions: ScanRegions
    energyRegions: EnergyRegions
    z: str
    doubleExposure: bool


class LineSpectrum(BaseModel):
    type: str
    proposal: str
    experimenters: str
    sample: str
    x: str
    y: str
    defocus: bool
    mode: str
    scanRegions: ScanRegions
    energyRegions: EnergyRegions
    energy: str
    doubleExposure: bool


class LastScan(BaseModel):
    Image: Image
    Ptychography_Image: PtychographyImage = Field(..., alias="Ptychography Image")
    Focus: Focus
    Line_Spectrum: LineSpectrum = Field(..., alias="Line Spectrum")


class CosmicMeta(BaseModel):
    header: str
    server: Server
    zonePlate: ZonePlate
    monitor: Monitor
    ptychography: Ptychography
    zmq: Zmq
    geometry: Geometry
    lastScan: LastScan
    repetition: int
    isDoubleExp: int
    pos_x: float
    pos_y: float
    step_size_x: float
    step_size_y: float
    num_pixels_x: int
    num_pixels_y: int
    background_pixels_x: int
    background_pixels_y: int
    dwell1: float
    dwell2: float
    energy: float
    energyIndex: int
    scanRegion: int
    dark_num_x: int
    dark_num_y: int
    exp_num_x: int
    exp_num_y: int
    exp_step_x: float
    exp_step_y: float
    double_exposure: bool
    exp_num_total: int
    translations: List[List[float]]
    output_frame_width: int
    detector_distance: float
    x_pixel_size: float
    y_pixel_size: float
    identifier: str
    illumination_real: List[List[float]]
    illumination_imag: List[List[float]]
    illumination_mask: List[List[bool]]
    dp_fraction_for_illumination_init: float
    dtype: Optional[str] = "float32"
    ptycho_shape: tuple[int, int, int] = (0, 0, 0)

    # For devtools pprint function.
    def __pretty__(
        self, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Custom pretty print to exclude 2D lists."""
        yield self.__repr_name__() + "("
        yield 1  # indentation level
        for name, value in self.__repr_args__():
            if isinstance(value, list) and all(isinstance(sub, list) for sub in value):
                if name is not None:
                    yield name + "="
                # Yield the shape of the 2D list (i.e., number of rows and columns)
                yield f"List[List], shape=({len(value)}, {len(value[0]) if len(value) > 0 else 0})"
                yield ","
                yield 0
                continue
            if name is not None:
                yield name + "="
            yield fmt(value)
            yield ","
            yield 0
        yield -1
        yield ")"

    @model_validator(mode="after")
    def set_ptycho_shape(self):
        total_acquisitions = self.exp_num_x * self.exp_num_y
        framewidth = self.output_frame_width
        self.ptycho_shape = (total_acquisitions, framewidth, framewidth)
        return self


class CosmicFrameMeta(BaseModel):
    shape_y: int
    shape_x: int
    dtype: str
    byteorder: str
    order: str
    identifier: str
    index: int
    posy: float
    posx: float


class CosmicAbortMeta(BaseModel):
    identifier: str


@register()
class CosmicStreamLoader(PtyScan):
    """
    Defaults:

    [name]
    default = 'CosmicStreamLoader'
    type = str
    help =

    [source]
    default = 'localhost'
    type = str
    help = Hostname for the publisher that this object will connect to

    [orientation]
    default = 3
    
    [load_parallel]
    default = None

    [psize]
    default = None

    [energy]
    default = None

    [distance]
    default = None

    [shape]
    default = None

    """

    def __init__(self, pars, **kwargs):
        super().__init__(pars, **kwargs)
        self.framecount: int = 0
        self.thread: Optional[threading.Thread] = None
        self._thread_end_of_scan: bool = False

    def initialize(self):
        self.metadata: Optional[CosmicMeta] = None
        if parallel.master:
            self.stream = PtychoStream(host_start=self.info.source)
            print("Waiting for metadata...")
            while True:
                if self.stream.has_scan_started():
                    meta_msg = self.stream.recv_start()
                    self.metadata = CosmicMeta(**meta_msg)
                    break
                else:
                    continue
            print("Metadata received.")

            md = self.metadata
            self.num_frames = md.exp_num_total // (md.double_exposure + 1)
            self.meta.energy = md.energy * 6.2425e15
            self.meta.psize = md.geometry.psize * 1e6
            self.info.psize = md.geometry.psize * 1e6 # need to set info.psize in case there is a rebining in PtyScan 
            self.meta.distance = md.geometry.distance * 1e3

            self._data = np.empty(
                shape=self.metadata.ptycho_shape, dtype=self.metadata.dtype
            )
            self._pos = np.empty(shape=(self.metadata.ptycho_shape[0], 2), dtype=float)
            print("Starting receiver thread...")
            self.thread = threading.Thread(target=self.receive_messages, daemon=True)
            self.thread.start()
            
        super().initialize()
        self.meta.energy = self.common["energy"]
        self.meta.psize = self.common["psize"]
        self.info.psize = self.common["psize"]
        self.meta.distance = self.common["distance"]
        
    def load_common(self):
        return self.meta

    def receive_messages(self):
        # TODO: Thread needs to terminate when scan has stopped.
        while True:
            if self.stream.has_frame_arrived():
                i, frame, idx, posy, posx, _ = self.stream.recv_frame()
                if idx < self._data.shape[0] and i == self.metadata.identifier:  # type: ignore
                    self._data[idx] = frame
                    self._pos[idx] = [posy, posx]
                    print(f"Received frame {idx}...")
                    self.framecount += 1
                else:
                    raise ValueError(
                        f"Received frame with wrong index {idx} or identifier {i}"
                    )
            elif self.stream.has_scan_aborted():
                print("Scan has been aborted")
                print("Receiving abort metadata...")
                self.stream.recv_abort()
                self._thread_end_of_scan = True

            elif self.stream.has_scan_stopped():
                print("Scan has stopped")
                print("Receiving stop metadata...")
                self.stream.recv_stop()
                self._thread_end_of_scan = True

    def check(self, frames=None, start=None):
        end_of_scan: int = 0  # could also be None if there was a condition that this streaming implementation wouldn't know
        frames_accessible: int = 0

        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        # Check how many frames are available
        new_frames = self.framecount - start
        # not reached expected nr. of frames
        if new_frames <= frames:
            # but its last chunk of scan so load it anyway
            if self.framecount == self.num_frames or self._thread_end_of_scan:
                frames_accessible = new_frames
                end_of_scan = 1
            # otherwise, do nothing
            else:
                frames_accessible = new_frames
                end_of_scan = 0
        # reached expected nr. of frames
        else:
            frames_accessible = frames
            end_of_scan = 0

        return frames_accessible, end_of_scan

    def load(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        log(4, "Loading...")
        log(4, f"indices = {indices}")
        for ind in indices:
            intensities[ind] = self._data[ind]
            positions[ind] = self._pos[ind]
            weights[ind] = np.ones_like(self._data[ind])

        return intensities, positions, weights

