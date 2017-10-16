
from ptypy import utils as u

prst = open('rst/parameters.rst','w')

Header=  '.. _parameters:\n\n'
Header+= '************************\n'
Header+= 'Parameter tree structure\n'
Header+= '************************\n\n'
prst.write(Header)

names = u.descriptor.defaults_tree
for name, desc in names.descendants:
    if name == '':
        continue
    if desc.children and desc.parent is desc.root:
        prst.write('\n'+name+'\n')
        prst.write('='*len(name)+'\n\n')
    if desc.children and desc.parent.parent is desc.root:
        prst.write('\n'+name+'\n')
        prst.write('-'*len(name)+'\n\n')
    
    prst.write('.. py:data:: '+name)
    prst.write('('+', '.join([t for t in desc.options['type'].split(',')])+')')
    prst.write('\n\n')
    prst.write('   '+desc.help+'\n\n')
    prst.write('   '+desc.doc.replace('\n', '\n   ')+'\n\n')
    prst.write('   *default* = ``'+str(desc.options['default']))
    lowlim, uplim = desc.limits
    if lowlim is not None and uplim is not None:
        prst.write(' (>'+str(lowlim)+', <'+str(uplim)+')``\n')
    elif lowlim is not None and uplim is None:
        prst.write(' (>'+str(lowlim)+')``\n')
    elif lowlim is None and uplim is not None:
        prst.write(' (<'+str(uplim)+')``\n')
    else:
        prst.write('``\n')
        
    prst.write('\n')
prst.close()
