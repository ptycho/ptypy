#!/usr/bin/env python3

from ptypy import utils as u
from ptypy import io
import argparse

def ptypy_inspect():
    opt = parse()
    inspect(opt)

def parse():
    parser = argparse.ArgumentParser(description='Shows a summary of the content of h5 compatible file (.ptyr,.ptyd) in terminal')
    parser.add_argument('h5file', type=str, help='path to hdf5 compatible file')
    parser.add_argument('-p', '--path', dest='path', type=str, help='path within that hdf5 compatible file', default='/')
    parser.add_argument('--report', dest='report', action='store_true',
                        help='use ptypy.utils.verbose.report instead of ptypy.io.h5info (it will load everything to ram).')
    parser.add_argument('-d', '--max-depth', dest='depth', type=int,
                        help='maximum depth for inspection (not implemented yet)')
    args = parser.parse_args()
    return args

def inspect(args):
    if args.report:
        print(u.verbose.report(list(io.h5read(args.h5file, args.path, depth=args.depth).values())[0], noheader=True))
    else:
        io.h5info(args.h5file, args.path, depth=args.depth)

if __name__ == "__main__":
    ptypy_inspect()
