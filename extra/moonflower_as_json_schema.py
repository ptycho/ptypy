import json

from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, Literal, Optional
from typing_extensions import Annotated


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
    dfile: str = Field(None,
        title = "Data File",
        description="File path where prepared data will be saved in the ``ptyd`` format.)",
        json_schema_extra={"user_level": 0})
    chunk_format: str = Field(".chunk%02d",
        title="Chunk Format",
        description="Appendix to saved files if save == 'link'",
        json_schema_extra={"user_level": 0})
    save: str = Field(None, 
        title="Saving Mode",
        description="Mode to use to save data to file. \
        <newline> \
        - ``None``: No saving \
        - ``'merge'``: attemts to merge data in single chunk **[not implemented]** \
        - ``'append'``: appends each chunk in master \\*.ptyd file \
        - ``'link'``: appends external links in master \\*.ptyd file and stores chunks separately \
        <newline> \
        in the path given by the link. Links file paths are relative to master file.",
        json_schema_extra={"user_level":0})
    auto_center: bool = Field(None,
        title="Auto Center",
        description="Determine if center in data is calculated automatically",
        json_schema_extra={"user_level": 0})
    load_parallel: str 


class MoonflowerScan(PtyScan):
    name: Literal["MoonFlowerScan"]


DataLoader = Annotated[Union[MoonflowerScan], Field(descriminator="name")]


class BlockScanModel(BaseModel):
    data: DataLoader

class BlockVanilla(BlockScanModel):
    pass


Scan = Annotated[Union[BlockVanilla], Field(descriminator="name")]


class DM(BaseModel):
    """
    foo
    """
    name: Literal["DM"]


Engine = Annotated[Union[DM], Field(descriminator="name")]


class BaseParamTree(BaseModel):
    verbose_level: VerboseLevel = Field("INFO", 
        title="Verbosity",
        json_schema_extra={"user_level": 0})
    data_type: DataType = Field("single",
        title="Data Type", 
        json_schema_extra={"user_level": 1})
    run: str = Field(None, 
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
    scans: Scan
    engines: dict[str,Engine]


if __name__ == "__main__":
    print(json.dumps(MoonflowerParamTree.model_json_schema()))
