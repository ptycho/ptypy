
from ptypy import utils as u

prst = open('rst/parameters.rst','w')

Header=  '.. _parameters:\n\n'
Header+= '*************************\n'
Header+= 'Ptypy parameter structure\n'
Header+= '*************************\n\n'
prst.write(Header)

names = u.validator.parameter_descriptions
for name,desc in u.validator.parameter_descriptions.iteritems():
    if name=='':
        continue
    if hasattr(desc,'children') and desc.parent is u.validator.pdroot:
        prst.write('\n'+name+'\n')
        prst.write('='*len(name)+'\n\n')
    if hasattr(desc,'children') and desc.parent.parent is u.validator.pdroot:
        prst.write('\n'+name+'\n')
        prst.write('-'*len(name)+'\n\n')
    
    prst.write('.. py:data:: '+name)
    prst.write('('+', '.join([t for t in desc.type])+')')
    prst.write('\n\n')
    num = str(desc.ID) if hasattr(desc,'ID') else "None"
    prst.write('   *('+num+')* '+desc.shortdoc+'\n\n')
    prst.write('   '+desc.longdoc.replace('\n','\n   ')+'\n\n')
    prst.write('   *default* = ``'+str(desc.default))
    if desc.lowlim is not None and desc.uplim is not None:
        prst.write(' (>'+str(desc.lowlim)+', <'+str(desc.uplim)+')``\n')
    elif desc.lowlim is not None and desc.uplim is None:
        prst.write(' (>'+str(desc.lowlim)+')``\n')
    elif desc.lowlim is None and desc.uplim is not None:
        prst.write(' (<'+str(desc.uplim)+')``\n')
    else:
        prst.write('``\n')
        
    prst.write('\n')
prst.close()
