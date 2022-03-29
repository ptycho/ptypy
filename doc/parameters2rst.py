
from ptypy import defaults_tree


def write_desc_recursive(prst, tree):
    for path, desc in tree.children.items():
        print(path)
        types = desc.type
        default = desc.default
        lowlim, uplim = desc.limits
        is_wildcard = (desc.name == '*')

        if is_wildcard:
            path = path.replace('*', desc.parent.name[:-1] + '_00')

        if path == '':
            continue

        if desc.children or desc.is_symlink:
            if desc.parent is desc.root:
                prst.write('\n' + path + '\n')
                prst.write('=' * len(path) + '\n\n')
            if desc.parent.parent is desc.root:
                prst.write('\n' + path + '\n')
                prst.write('-' * len(path) + '\n\n')

        prst.write('.. py:data:: ' + path)

        if desc.is_symlink:
            tp = 'Param'
        else:
            tp = ', '.join([str(t) for t in types])
        prst.write(' (' + tp + ')')
        prst.write('\n\n')

        if is_wildcard:
            prst.write('   *Wildcard*: multiple entries with arbitrary names are accepted.\n\n')

        # prst.write('   '+desc.help+'\n\n')
        prst.write('   ' + desc.help.replace('<newline>', '\n').replace('\n', '\n   ') + '\n\n')
        prst.write('   ' + desc.doc.replace('<newline>', '\n').replace('\n', '\n   ') + '\n\n')

        if desc.children:
            print('recursion ' + path)
            prst.write('\n')
            write_desc_recursive(prst, desc)
        elif desc.is_symlink:
            print('following symlink ' + path)
            prst.write('\n')
            write_desc_recursive(prst, desc.type[0])
        else:
            prst.write('   *default* = ``' + repr(default))
            if lowlim is not None and uplim is not None:
                prst.write(' (>' + str(lowlim) + ', <' + str(uplim) + ')``\n')
            elif lowlim is not None and uplim is None:
                prst.write(' (>' + str(lowlim) + ')``\n')
            elif lowlim is None and uplim is not None:
                prst.write(' (<' + str(uplim) + ')``\n')
            else:
                prst.write('``\n')

        prst.write('\n')


def write_desc_tree(prst, tree):
    for path, desc in tree.descendants:

        types = desc.type
        default = desc.default
        lowlim, uplim = desc.limits
        is_wildcard = (desc.name == '*')

        if is_wildcard:
            path = path.replace('*', desc.parent.name[:-1] + '_00')

        if path == '':
            continue
        if desc.children and desc.parent is desc.root:
            prst.write('\n' + path + '\n')
            prst.write('=' * len(path) + '\n\n')
        if desc.children and desc.parent.parent is desc.root:
            prst.write('\n' + path + '\n')
            prst.write('-' * len(path) + '\n\n')

        prst.write('.. py:data:: ' + path)

        if desc.is_symlink:
            tp = 'Param'
        else:
            tp = ', '.join([str(t) for t in types])
        prst.write(' (' + tp + ')')
        prst.write('\n\n')

        if is_wildcard:
            prst.write('   *Wildcard*: multiple entries with arbitrary names are accepted.\n\n')

        # prst.write('   '+desc.help+'\n\n')
        prst.write('   ' + desc.help.replace('<newline>', '\n').replace('\n', '\n   ') + '\n\n')
        prst.write('   ' + desc.doc.replace('<newline>', '\n').replace('\n', '\n   ') + '\n\n')

        if desc.is_symlink:
            prst.write('   *default* = ' + ':py:data:`' + desc.type[0].path + '`\n')
        else:
            prst.write('   *default* = ``' + repr(default))
            if lowlim is not None and uplim is not None:
                prst.write(' (>' + str(lowlim) + ', <' + str(uplim) + ')``\n')
            elif lowlim is not None and uplim is None:
                prst.write(' (>' + str(lowlim) + ')``\n')
            elif lowlim is None and uplim is not None:
                prst.write(' (<' + str(uplim) + ')``\n')
            else:
                prst.write('``\n')

        prst.write('\n')


prst = open('rst/parameters.rst','w')

Header = """\
.. _parameters:

************************
Parameter tree structure
************************
 
.. note::
   This section needs to be reworked to account for the
   recursive nature for |ptypy|_'s parameter tree. 
   Please use the examples in the templates folder to 
   craft your own scripts.
"""

prst.write(Header)
write_desc_tree(prst, defaults_tree)
# write_desc_recursive(prst, defaults_tree['ptycho'])
# write_desc_tree(prst, defaults_tree['io'])
# write_desc_tree(prst, defaults_tree['scan'])
# write_desc_tree(prst, defaults_tree['scandata'])
prst.close()
