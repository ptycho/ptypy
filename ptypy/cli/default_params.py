#!/usr/bin/env python3

from ptypy import utils as u
import textwrap
import argparse

def ptypy_csv2cp():
    opt = parse()
    convert_and_print(opt)

def parse():
    parser = argparse.ArgumentParser(description='Dump default parameter in ConfigParser format.')
    parser.add_argument('-p', '--path', dest='path', type=str, default=None,
                   help='A valid path in the parameter structure')
    parser.add_argument('-d', '--depth', dest='depth', type=int, default=1,
                   help='Recursion depth')
    pars = parser.parse_args()
    return pars

def convert_and_print(pars):
    root = u.validator.pdroot
    if pars.path:
        for node in pars.path.split('.'):
            root = root.children[node]
    print_param(root, depth=pars.depth)

def write(x):
    print(x)

def wrapdoc(x):
    return '\n  '.join(['\n  '.join(textwrap.wrap(line, 90, break_long_words=False, replace_whitespace=False)) for line in x.splitlines() if line.strip() != ''])

def print_param(entry, parent=None, depth=50):
    if parent is not None:
        if parent == '':
            entry_name = entry.name
        else:
            entry_name = '%s.%s' % (parent, entry.name)
        write('[%s]' % entry_name)
        write('default = %s' % entry.default)
        write('help = ' + wrapdoc(entry.shortdoc))
        write('doc = ' + wrapdoc(entry.longdoc))
        write('type = ' + ', '.join(entry.type))
        write('userlevel = %s' % entry.userlevel)
        if entry.choices: 
            write('choices = ' + ', '.join([str(x) for x in entry.choices]))
        if entry.lowlim: 
            write('lowlim = %s' % entry.lowlim)
        if entry.uplim: 
            write('uplim = %s' % entry.uplim)
        write('')
    else:
        entry_name = ''
    if entry.children and depth > 0:
        for childname, child in entry.children.items():
            print_param(child, entry_name, depth=depth-1)
    return

if __name__ == "__main__":
    ptypy_csv2cp()
