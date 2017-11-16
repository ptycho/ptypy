from ptypy.utils.descriptor import EvalDescriptor

root = EvalDescriptor('')

@root.parse_doc()
class FakePtycho(object):
    """
    Docs...

    Defaults:


    # A template tree under scan
    [scan]
    type = Param
    default = 
    help = 

    [scan.data]
    type = Param
    default =
    help = Data preparation parameters

    [scan.data.load_parallel]
    type = bool
    default = True
    help = Load 

    [scan.data.source]
    type = str
    default = 'ptyd'
    help = Data source

    [scan.model]
    type = @scanmodel.Vanilla, @scanmodel.Full
    default = @scanmodel.Vanilla
    help = Physical imaging model


    # The actual scans container
    [scans]
    type = Param
    default = None
    help = Container for scan instances

    [scans.*]
    type = @scan
    default = @scan
    help = Wildcard for scan instances

    """
    pass


@root.parse_doc('scanmodel.Vanilla')
class FakeVanillaScan(object):
    """
    Docs...

    Defaults:

    [name]
    type = str
    default = Vanilla
    help = Vanilla scan model

    [energy]
    type = float
    default = 9.3
    help = Photon energy
    """
    pass

@root.parse_doc('scanmodel.Full')
class FakeFullScan(FakeVanillaScan):
    """
    Docs...

    Defaults:

    [name]
    type = str
    default = Full
    help = Full scan model

    [probe_modes]
    type = int
    default = 1
    help = Number of mutually incoherent illumination modes
    """
    pass

