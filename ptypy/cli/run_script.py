#!/usr/bin/env python3

import argparse
import json
from ptypy.core import Ptycho
from ptypy.utils.verbose import log
from ptypy import utils as u


def ptypy_run():
    u.verbose.set_level(3)
    opt = parse()
    run(opt)

def parse():
    parser = argparse.ArgumentParser(description='Runs a ptychography from the command line.')

    parser.add_argument("json_file",
                        help="The path to the json configuration file.",
                        type=str)

    parser.add_argument('--output-folder', '-O',
                        dest="output_folder",
                        help="The path we want the outputs to exist in (will get created).",
                        type=str)

    parser.add_argument('--ptypy-level', '-L',
                        dest="ptypy_level",
                        help="The level we want to run to ptypy to.",
                        default=5,
                        type=str)

    parser.add_argument('--identifier', '-I',
                        dest="identifier",
                        help="This is the same as p.run.",
                        default=None,
                        type=str)

    parser.add_argument('--plot', '-P',
                        dest="plotting",
                        help="A switch for the plotting. 1: on, 0:off",
                        default=1,
                        type=int)

    args = parser.parse_args()
    return args

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, str):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.items()
        }
    # if it's anything else, return it in its original form
    return data

def get_parameters(args):
    in_dict = json.load(open(args.json_file), object_hook=_byteify)
    parameters_to_run = u.Param()
    if in_dict['base_file'] is not None:
        log(3, "Basing this scan off of the scan in {}".format(in_dict['base_file']))
        previous_scan = Ptycho.load_run(in_dict['base_file'], False)  # load in the run but without the data
        previous_parameters = previous_scan.p
        parameters_to_run.update(previous_parameters)
    if in_dict['parameter_tree'] is not None:
        parameters_to_run.update(in_dict['parameter_tree'], Convert=True)
    return parameters_to_run

def get_file_name(args):
    from datetime import datetime
    now = datetime.now()
    if args.identifier is not None:
        output_path = "{}scan_{}_{}".format(args.output_folder, args.identifier, now.strftime("%Y%m%d%H%M%S"))
    else:
        output_path = "{}scan_{}".format(args.output_folder, now.strftime("%Y%m%d%H%M%S"))
    log(3, "Output is going in: {}".format(output_path))
    return output_path

def run(args):
    parameters = get_parameters(args)
    parameters.run = args.identifier
    if args.plotting:
        log(3, "Turning the plotting on.")
        parameters.io.autoplot = u.Param(active=True)
    else:
        log(3, "Turning the plotting off.")
        parameters.io.autoplot = u.Param(active=False)
    # make sure we aren't randomly writing somewhere if this isn't set.
    if args.output_folder is not None:
        parameters.io.home = get_file_name(args)
        parameters.io.rfile = "%s.ptyr" % get_file_name(args)
        parameters.io.autosave = u.Param(active=True)
        log(3, "Autosave is on, with io going in {}, and the final reconstruction into {}".format(parameters.io.home,
                                                                                                parameters.io.rfile))
    else:
        parameters.io.rfile = None
        parameters.io.autosave = u.Param(active=False)
        log(3, "Autosave is off. No output will be saved.")

    Ptycho(parameters, level=args.ptypy_level)

if __name__ == "__main__":
    ptypy_run()
