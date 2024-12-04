import json

from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, Literal
from typing_extensions import Annotated
from annotated_types import Len

tuple_int_two = Annotated[tuple[int], Len(min_length=2, max_length=2)]
list_int_two = Annotated[list[int], Len(min_length=2, max_length=2)]
tuple_bool_three = Annotated[tuple[bool], Len(min_length=3, max_length=3)]
list_bool_three = Annotated[list[bool], Len(min_length=3, max_length=3)]
tuple_float_two = Annotated[tuple[float], Len(min_length=2, max_length=2)]
list_float_two = Annotated[list[float], Len(min_length=2, max_length=2)]


class VerboseLevel(str, Enum):
    """Verbosity level for information logging"""
    critical = "CRITICAL"
    error = "ERROR"
    warning = "WARNING"
    interactive = "INTERACTIVE"
    info = "INFO"
    inspect = "INSPECT"
    debug = "DEBUG"

class DataType(str, Enum):
    """Reconstruction floating number precision"""
    single = "single"
    double = "double"

class ReconstructionFileFormat(str, Enum):
    minimal = "minimal"
    dls = "dls"
    used_params = "used_params"
    
class InputOutputInteraction(BaseModel):
    """Options for the communications server"""
    active: bool = Field(True, 
        title="Interaction Activation Switch",
        description="If True the interaction starts, if False all interaction is turned off",
        json_schema_extra={"user_level": 0})

class InputOutputAutosave(BaseModel):
    """Options for automatic saving during reconstruction"""
    active: bool = Field(True, 
        title="Autosave Activation Switch",
        description="If ``True`` the current reconstruction will be saved at regular intervals.",
        json_schema_extra={"user_level": 0})

class InputOutputAutoplot(BaseModel):
    """Container for the plotting"""
    active: bool = Field(True, 
        title="AutoplotActivation Switch",
        description=" If ``True`` the current reconstruction will be plotted at regular intervals.",
        json_schema_extra={"user_level": 0})    

class InputOutput(BaseModel):
    """Global parameter container for I/O settings"""
    home: str = Field("./",
        title = "Base directory (home)",
        description="This is the root directory for all input/output operations. \
                     All other path parameters that are relative paths will be relative to this directory.",
        json_schema_extra={"user_level": 0})
    rfile: str = Field("recons/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr",
        title="Reconstruction File Name",
        description="Reconstruction file name or format string (constructed against runtime dictionary)",
        json_schema_extra={"user_level": 0})
    rformat: ReconstructionFileFormat = Field("minimal",
        title="Reconstruction File Format",
        description="Choose a reconstruction file format for after engine completion.",
        json_schema_extra={"user_level": 0})
    interaction: InputOutputInteraction
    autosave: InputOutputAutosave
    autoplot: InputOutputAutoplot

class LoadParallel(str, Enum):
    """Determines what will be loaded in parallel"""
    data = "data"
    common = "common"
    all = "all"
    none = "none"


class PtyScan(BaseModel):
    """
    PtyScan: A single ptychography scan, created on the fly or read from file.

    *BASECLASS*

    Objectives:
     - Stand alone functionality
     - Can produce .ptyd data formats
     - Child instances should be able to prepare from raw data
     - On-the-fly support in form of chunked data.
     - mpi capable, child classes should not worry about mpi
    """
    name: Literal["PtyScan"]
    dfile: str | None = Field(None,
        title = "Data File",
        description="File path where prepared data will be saved in the ``ptyd`` format.)",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    chunk_format: str = Field(".chunk%02d",
        title="Chunk Format",
        description="Appendix to saved files if save == 'link'",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    save: str | None = Field(None, 
        title="Saving Mode",
        description="Mode to use to save data to file. \
        <newline> \
        - ``None``: No saving \
        - ``'merge'``: attemts to merge data in single chunk **[not implemented]** \
        - ``'append'``: appends each chunk in master \\*.ptyd file \
        - ``'link'``: appends external links in master \\*.ptyd file and stores chunks separately \
        <newline> \
        in the path given by the link. Links file paths are relative to master file.",
        json_schema_extra={"user_level": 0},
        exclude_schema=False)
    auto_center: bool | None = Field(None,
        title="Auto Center",
        description="Determine if center in data is calculated automatically",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    load_parallel: LoadParallel = Field("data",
        title="Load Parallel",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    rebin: int | None = Field(None,
        ge=1, le=32,
        title="Rebin",
        description="Rebinning factor for the raw data frames. ``'None'`` or ``1`` both mean *no binning*",
        json_schema_extra={"user_level": 1},
        exclude_schema=True)
    orientation: int | tuple_bool_three | list_bool_three = Field(0,ge=0,le=7,
        title="Orientation",
        description="Choose \
       <newline> \
       - ``None`` or ``0``: correct orientation\
       - ``1``: invert columns (numpy.flip_lr)\
       - ``2``: invert rows  (numpy.flip_ud)\
       - ``3``: invert columns, invert rows\
       - ``4``: transpose (numpy.transpose)\
       - ``4+i``: tranpose + other operations from above\
       <newline>\
       Alternatively, a 3-tuple of booleans may be provided ``(do_transpose, \
       do_flipud, do_fliplr)``",
       json_schema_extra={"user_level": 1})
    min_frames: int = Field(1, ge=1,
        title="Min Frames",
        description="Minimum number of frames loaded by each node",
        json_schema_extra={"user_level": 2}, 
        exclude_schema=True)
    num_frames: int | None = Field(None,
        title="Num Frames",
        description="Maximum number of frames to be prepared",
        json_schema_extra={"user_level": 1})
    label: str | None = Field(None,
        title="Label",
        description="The scan label. Unique string identifying the scan",
        json_schema_extra={"user_level": 1},
        exclude_schema=True)
    shape: int | tuple_int_two = Field(256, 
        title="Shape",
        description="Shape of the region of interest cropped from the raw data.",
        json_schema_extra={"user_level": 1})
    center: list_int_two | tuple_int_two | str | None = Field("fftshift", 
        title="Center",
        description="Center (pixel) of the optical axes in raw data",
        json_schema_extra={"user_level": 1},
        exclude_schema=True)
    psize: float | tuple_float_two = Field(0.000172, ge=0,
        title="Pixelsize",
        description="Dimensions of the detector pixels (in meters)",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    distance: float = Field(7.19, ge=0,
        title="Distance",
        description="Sample to detector distance (in meters)",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    energy: float = Field(7.2, ge=0,
        title="Energy",
        description="Photon energy of the incident radia,tion in keV",
        json_schema_extra={"user_level": 0},
        exclude_schema=True)
    

class MoonflowerScan(PtyScan):
    name: Literal["MoonFlowerScan"] = Field("MoonFlowerScan")
    num_frames: int = Field(100,
        title="Num Frames",
        description="Maximum number of frames to be prepared",
        json_schema_extra={"user_level": 1})
    shape: int = Field(128, 
        title="Shape",
        description="Shape of the region of interest cropped from the raw data.",
        json_schema_extra={"user_level": 1})
    density: float = Field(0.2, 
        title="Density",
        descriminator="Position distance in fraction of illumination frame",
        json_schema_extra={"user_level": 0})
    model: str = Field("round",
        title="Model",
        description="The scan pattern",
        json_schema_extra={"user_level": 0})
    photons: float = Field(1e8,
        title="Photons",
        description="Total number of photons for Poisson noise",
        json_schema_extra={"user_level": 0})
    psf: float = Field(0,
        title="PSF",
        description="Point spread function of the detector",
        json_schema_extra={"user_level": 0})
    add_poisson_noise: bool = Field(True,
        title="Add Poisson Noise",
        description="Decides whether the scan should have poisson noise or not",
        json_schema_extra={"user_level": 0})
    

DataLoader = Annotated[Union[MoonflowerScan], Field(descriminator="name")]

class Propagation(str, Enum):
    farfield = "farfield"
    nearfield = "nearfield"

class FFTtype(str, Enum):
    numpy = "numpy"
    scipy = "scipy"
    fftw = "fftw"

class ApertureForm(str, Enum):
    circ = "circ"
    rect = "rect"
    none = None

class Aperture(BaseModel):
    rotate: float = Field(0,
        title="Rotate",
        description="Rotate aperture by this value")
    central_stop: float | None = Field(None,
        title="Central Stop",
        description="Size of central stop as a fraction of aperture.size")
    edge: float = Field(2.0, 
        title="Edge",
        description="Edge width of aperture (in pixels!)")
    form: ApertureForm = Field("circ",
        title="Form",
        description="One of None, 'rect' or 'circ'")
    offset: float | tuple_float_two | list_float_two = Field(0,
        title="Offset",
        description="Offset between center of aperture and optical axes")
    size: float | tuple_float_two | list_float_two | None = Field(None,
        title="Size",
        description="Aperture width or diameter")

class Diversity(BaseModel):
    noise: tuple_float_two | list_float_two = Field((0.5,1.0),
        title="Noise",
        description="Noise in each non-primary mode of the illumination.")
    power: float | tuple | list = Field(0.1,
        title="Power",
        description="Power of modes relative to main mode (zero-layer)")

class InitialModel(str, Enum):
    recon = "recon"
    stxm = "stxm"
    none = None

class IlluminationPropagation(BaseModel):
    antialiasing: int = Field(1, ge=1,
        title="Antialiasing",
        description="Antialiasing factor used when generating the probe. (numbers larger than 2 or 3 are memory hungry)")
    focussed: float | None = Field(None,
        title="Focussed",
        description="Propagation distance from aperture to focus")
    parallel: float | None = Field(None,
        title="Parallel",
        description="Parallel propagation distance")
    spot_size: float | None = Field(None,
        title="Spot Size",
        description="Focal spot diameter")

class InitialRecon(BaseModel):
    label: str | None = Field(None,
        title="Label",
        description="Scan label of diffraction that is to be used for probe estimate")
    rfile: str = Field("*.ptyr",
        title="Reconstruction File",
        description="Path to a ``.ptyr`` compatible file")

class Illumination(BaseModel):
    aperture: Aperture = Field(exclude_schema=True)
    diversity: Diversity = Field(exclude_schema=True)
    model: InitialModel | None = Field(None,
        title="Model",
        description="Type of illumination model",
        exclude_schema=True)
    photons: int | float | None = Field(None,
        title="Photons",
        description="Number of photons in the incident illumination",
        exclude_schema=False)
    propagation: IlluminationPropagation = Field(exclude_schema=True)
    recon: InitialRecon = Field(exclude_schema=True)
    
class Process(BaseModel):
    offset: tuple_int_two | list_int_two = Field((0,0),
        title="Offset",
        description="Offset between center of object array and scan pattern")
    zoom: float | tuple_float_two | list_float_two | None = Field(None,
        title="Zoom",
        description="Zoom value for object simulation.")
    formula: str | None = Field(None,
        title="Formula",
        description="A Formula compatible with a cxro database query,e.g. ``'Au'`` or ``'NaCl'`` or ``'H2O'``")
    density: float = Field(1,
        title="Density",
        description="Density in [g/ccm]. Only used if `formula` is not None")
    thickness: float = Field(1e-6,
        title="Thickness",
        description="Maximum thickness of sample. If ``None``, the absolute values of loaded source array will be used")
    ref_index: tuple_float_two | list_float_two = Field((0.5, 0.0),
        title="Refractive Index",
        description="Assigned refractive index, tuple of format (real, complex)")
    smoothing: int = Field(2,
        title="Smoothing",
        description="Smooth the projection with gaussian kernel of width given by `smoothing_mfs`")
    
class Sample(BaseModel):
    model: InitialModel | None = Field(None,
        title="Model",
        description="Type of initial object model",
        exclude_schema=True)
    fill: float | complex = Field(1,
        title="Fill",
        description="Default fill value")
    recon: InitialRecon = Field(exclude=True)
    process: Process = Field(exclude=True)
    diversity: Diversity = Field(exclude=True)

class BlockScanModel(BaseModel):
    propagation: Propagation = Field("farfield",
        title="Propagation",
        description="Either 'farfield' or 'nearfield'",
        json_schema_extra={"user_level": 1})
    ffttype: FFTtype = Field("scipy",
        title="FFT Type",
        description="FFT library",
        json_schema_extra={"user_level": 1})
    resample: int = Field(1, ge=1,
        title="Resample",
        description="Resampling fraction of the image frames w.r.t. diffraction frames.\
                    A resampling of 2 means that the image frame is to be sampled (in the detector plane) twice\
                    as densely as the raw diffraction data.",
        json_schema_extra={"user_level": 0})
    data: DataLoader
    Illumination: Illumination
    sample: Sample


class BlockVanilla(BlockScanModel):
    name: Literal["BlockVanilla"] = Field("BlockVanilla")

class BlockFull(BlockScanModel):
    name: Literal["BlockFull"]

Scan = Annotated[Union[BlockVanilla, BlockFull], Field(descriminator="name")]

class MFScan(BaseModel):
    MF: BlockVanilla

class BaseEngine(BaseModel):
    numiter: int = Field(20,
        title="Nr. of Iterations",
       description="Total number of iterations")
    numiter_contiguous: int = Field(1,
        title="Nr. of contiguous iterations",
        description="Number of iterations without interruption")
    probe_support: float = Field(0.7,
        title="Probe Support",
        description="Valid probe area as fraction of the probe frame")
    probe_fourier_support: float | None = Field(None,
        title="Probe Fourier Support",
        description="Valid probe area in frequency domain as fraction of the probe frame")
    record_local_error: bool = Field(False,
        title="Record Local Error Map",
        description="If True, save the local map of errors into the runtime dictionary.")
    

class DM(BaseEngine):
    """
    Difference Map Engine
    """
    name: Literal["DM"]

class ML(BaseEngine):
    """
    Maximum Likelihood Engine
    """
    name: Literal["ML"]


Engine = Annotated[Union[DM], Field(descriminator="name")]

class DMEngine(BaseModel):
  engine00 : DM

class BaseParamTree(BaseModel):
    verbose_level: VerboseLevel = Field("INFO", 
        title="Verbosity",
        json_schema_extra={"user_level": 0})
    data_type: DataType = Field("single",
        title="Data Type", 
        json_schema_extra={"user_level": 1})
    run: str | None = Field(None, 
        title="Run Label", 
        description="Reconstruction run identifier. If ``None``, \
                     the run name will be constructed at run time from other information",
        json_schema_extra={"user_level": 0},
        )
    frames_per_block: int = Field(100000,
        title="Frames Per Block",
        description="This parameter determines the size of buffer arrays for GPUs. \
                     Reduce this number if you run out of memory on the GPU.",
        json_schema_extra={"user_level":1},
        gt=1)
    io: InputOutput = Field(title="Input Output")


class MoonflowerParamTree(BaseParamTree):
    scans: MFScan
    engines: DMEngine


if __name__ == "__main__":
    print(json.dumps(MoonflowerParamTree.model_json_schema()))
