
from ptypy import defaults_tree

prst = open('rst/parameters.rst','w')

Header=  '.. _parameters:\n\n'
Header+= '************************\n'
Header+= 'Parameter tree structure\n'
Header+= '************************\n\n'
prst.write(Header)

for path, desc in defaults_tree.descendants:

    types = desc.type
    default = desc.default
    lowlim, uplim = desc.limits
    is_wildcard = (desc.name == '*')

    if is_wildcard:
        path = path.replace('*', desc.parent.name[:-1] + '_00')

    if path == '':
        continue
    if desc.children and desc.parent is desc.root:
        prst.write('\n'+path+'\n')
        prst.write('='*len(path)+'\n\n')
    if desc.children and desc.parent.parent is desc.root:
        prst.write('\n'+path+'\n')
        prst.write('-'*len(path)+'\n\n')

    prst.write('.. py:data:: '+path)

    if desc.is_symlink:
        tp = 'Param'
    else:
        tp = ', '.join([str(t) for t in types])
    prst.write(' ('+tp+')')
    prst.write('\n\n')

    if is_wildcard:
        prst.write('   *Wildcard*: multiple entries with arbitrary names are accepted.\n\n')

    prst.write('   '+desc.help+'\n\n')
    prst.write('   '+desc.doc.replace('<newline>','\n').replace('\n', '\n   ')+'\n\n')

    if desc.is_symlink:
        prst.write('   *default* = '+':py:data:`'+desc.path+'`\n')
    else:
        prst.write('   *default* = ``'+repr(default))
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
