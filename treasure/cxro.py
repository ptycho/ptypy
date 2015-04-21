
server = 'http://henke.lbl.gov'

POST_query = ('Material=Enter+Formula' +
             '&Formula=%(formula)s&Density=%(density)s&Scan=Energy' +
             '&Min=%(emin)s&Max=%(emax)s&Npts=%(npts)s&Output=Text+File')

def iofr(formula, energy,density=-1, npts=100):
    """\
    Query CXRO database for index of refraction values.

    Parameters:
    ----------
    formula: str
        String representation of the Formula to use.
    energy: float or (float,float)
        Either a single energy (in keV) or the minimum/maximum bounds
    npts: int [optional]
        Number of points between the min and max energies. 

    Returns:
        (energy, delta, beta), either scalars or vectors.
    """
    #from ptypy import utils as u
    import urllib
    import urllib2
    import numpy as np
    
    if np.isscalar(energy):
        emin = energy
        emax = energy
        npts = 1
    else:
        emin,emax = energy

    data = POST_query % {'formula':formula,
                     'emin':emin,
                     'emax':emax,
                     'npts':npts,
                     'density':density}

    url = server+'/cgi-bin/getdb.pl'
    print 'Querying CRXO database...'
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    t = response.read()
    datafile = t[t.find('/tmp/'):].split('"')[0]

    url = server + datafile
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    data = response.read()

    d = data.split('\n')
    dt = np.array([[float(x) for x in dd.split()] for dd in d[2:] if dd])

    print 'done, retrieved: ' +  d[0].strip()
    #print d[0].strip()
    if npts==1:
        return dt[-1,0], dt[-1,1], dt[-1,2]
    else:
        return dt[:,0], dt[:,1], dt[:,2]
