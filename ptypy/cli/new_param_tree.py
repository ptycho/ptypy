#!/usr/bin/env python3

# This script (ptypy.new) seems to be disfunctional
# and needs further checking

import sys
import argparse
from ptypy import defaults_tree

class MoreGentleParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

def ptypy_new():
    opt = parse()
    create_tree(opt)

def parse():
    parser = MoreGentleParser(description='Creates a new ptypy reconstruction script from defaults.')
    parser.add_argument('pyfile', type=str, 
                    help='target .py script file, existing files will be overrwritten')
    parser.add_argument('-u','--user-level', dest='ulevel', type=int, default=0,
                    help='use parameters up to this complexity level (0--2)')
    parser.add_argument('--short-doc', dest='sdoc', action='store_true',
                    help='include help info as inline comments')
    parser.add_argument('--long-doc', dest='ldoc', action='store_true',
                    help='include both help and doc info')
    pars = parser.parse_args()
    return pars

def create_tree(pars):
    if pars.sdoc:
        doc_level = 1  
    elif pars.ldoc:
        doc_level = 2  
    else:
        doc_level = 0
    defaults_tree.create_template(filename=pars.pyfile,
        user_level=pars.ulevel, doc_level=doc_level, start_at_root=True)

if __name__ == "__main__":
    ptypy_new()
