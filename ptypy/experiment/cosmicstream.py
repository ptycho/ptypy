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


class PtyScanDefaults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "PtyScan"
    dfile: Optional[str] = None
    chunk_format: str = ".chunk%02d"
    save: Optional[str] = None
    auto_center: Optional[bool] = None
    load_parallel: str = "data"
    rebin: Optional[int] = None
    orientation: Optional[int | tuple | list] = None
    min_frames: int = 1
    positions_theory: Optional[NDArray] = None
    num_frames: Optional[int] = None
    label: Optional[str] = None
    experimentID: Optional[str] = None
    version: float = 0.1
    shape: int | tuple = 256
    center: list | str | tuple = "fftshift"
    psize: float | tuple = 0.000172
    distance: float = 7.19
    energy: float = 7.2
    add_poisson_noise: bool = False


class CosmicStreamLoaderParams(PtyScanDefaults):
    name: str = "CosmicStreamLoader"
    host_start: str = "localhost"


class CosmicStreamLoaderMeta(BaseModel):
    version: float
    num_frames: int
    label: Optional[str]
    shape: int | tuple
    psize: float | tuple
    energy: float
    center: list | str | tuple
    distance: float


@register()
class CosmicStreamLoader(PtyScan):
    """
    Defaults:

    [name]
    default = 'CosmicStreamLoader'
    type = str
    help =

    [host_start]
    default = 'localhost'
    type = str
    help = Hostname for the publisher that this object will connect to
    """

    def __init__(self, pars: Optional[CosmicStreamLoaderParams] = None, **kwargs):
        super().__init__(pars, **kwargs)
        self.p = CosmicStreamLoaderParams(**self.info)
        self._meta: Optional[CosmicStreamLoaderMeta] = None
        self.framecount: int = 0
        self.num_frames: int = 0
        self.thread: Optional[threading.Thread] = None

    def initialize(self):
        self.stream = PtychoStream(host_start=self.p.host_start)
        self.metadata: Optional[CosmicMeta] = None
        print("Waiting for metadata...")
        while True:
            if self.stream.has_scan_started():
                meta_msg = self.stream.recv_start()
                self.metadata = CosmicMeta(**meta_msg)
                break
            else:
                continue
        print("Metadata received.")

        def setup_params(meta: CosmicMeta):
            # TODO: change this to None, or dfile. for testing we can use uuid and saving
            self.p.dfile = None
            self.p.save = None
            self.p.auto_center = None
            self.p.load_parallel = "data"  # ?
            self.p.rebin = meta.geometry.rebin

            # TODO: do we need to invert/transpose this data?
            self.p.orientation = None
            self.p.min_frames = 1
            self.p.positions_theory = None
            self.p.num_frames = meta.exp_num_total // (meta.double_exposure + 1)
            self.p.experimentID = meta.identifier

            # TODO: this is the frame shape
            self.p.shape = meta.output_frame_width

            # TODO: i am not sure we have this, leave as default...
            # self.p.center = meta.geometry.center

            # TODO: this is 2.99e-11, not sure if it is right...
            self.p.psize = meta.geometry.psize

            # TODO: this is correct, but dunno units (0.000121, for ex.)
            self.p.distance = meta.geometry.distance

            # TODO: this is currently in J
            # self.p.energy = meta.energy / constants.e / 1000
            self.p.energy = meta.energy

            # TODO: not sure about this...
            self.p.add_poisson_noise = False

        def setup_meta(params: CosmicStreamLoaderParams):
            self._meta = CosmicStreamLoaderMeta(**params.model_dump())
            self.meta = u.Param(self._meta.model_dump())

        def setup_info(params: CosmicStreamLoaderParams):
            self.info.num_frames = params.num_frames
            self.info.experimentID = params.experimentID
            self.info.shape = params.shape
            self.info.distance = params.distance
            self.info.energy = params.energy
            self.info.dfile = params.dfile
            self.info.psize = params.psize

        setup_params(self.metadata)
        setup_meta(self.p)
        setup_info(self.p)

        self.orientation = self.p.orientation  # TODO: fix when we know orientation
        self.num_frames = self.p.num_frames if self.p.num_frames is not None else 0

        self._data = np.empty(
            shape=self.metadata.ptycho_shape, dtype=self.metadata.dtype
        )
        self._pos = np.empty(shape=(self.metadata.ptycho_shape[0], 2), dtype=float)

        print("Starting receiver thread...")
        self.thread = threading.Thread(target=self.receive_messages, daemon=True)
        self.thread.start()

    def receive_messages(self):
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
                # TODO: boolean for stop condition
            elif self.stream.has_scan_stopped():
                print("Scan has stopped")
                print("Receiving stop metadata...")
                self.stream.recv_stop()
                # TODO: boolean for stop condition

    def check(self, frames=None, start=None):
        end_of_scan: bool = False  # TODO: set stop condition from recv thread
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
            if self.framecount == self.num_frames:
                frames_accessible = new_frames
                end_of_scan = True
            # otherwise, do nothing
            else:
                frames_accessible = 0
                end_of_scan = False
        # reached expected nr. of frames
        else:
            frames_accessible = frames
            end_of_scan = False

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
            weights[ind] = np.ones(len(intensities[ind]))

        return intensities, positions, weights