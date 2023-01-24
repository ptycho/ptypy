import numpy as np

# maps a numpy dtype to the corresponding C type
def map2ctype(dt):
    if dt == np.float32:
        return 'float'
    elif dt == np.float64:
        return 'double'
    elif dt == np.complex64:
        return 'complex<float>'
    elif dt == np.complex128:
        return 'complex<double>'
    elif dt == np.int32:
        return 'int'
    elif dt == np.int64:
        return 'long long'
    else:
        raise ValueError('No mapping for {}'.format(dt))
