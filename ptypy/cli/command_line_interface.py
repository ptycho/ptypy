# -*- coding: utf-8 -*-
"""
Wrapper to store nearly anything in an hdf5 file.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import argparse
import json
import re
from collections.abc import MutableMapping

from ptypy.core import Ptycho
from ptypy.utils.verbose import log
from ptypy import utils as u
from ptypy import load_ptyscan_module, load_gpu_engines

def ptypy_run():
    u.verbose.set_level("info")
    opt = parse()
    run(opt)

def parse():
    parser = argparse.ArgumentParser(description='A cloud-compatible command-line interface for ptypy')
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("-f", "--file", type=str,
                            help="Provide parameter configuration as a JSON or YAML file.")
    config_group.add_argument("-j", "--json", type=str,
                            help="Provide parameter configuration as a JSON string.")
    parser.add_argument('--output-folder', '-o', type=str,
                        help="The path we want the outputs to exist in (will get created).")
    parser.add_argument('--ptypy-level', '-l', default=5, type=str,
                        help="The level we want to run to ptypy to.")
    parser.add_argument('--identifier', '-i', type=str, default=None,
                        help="This is the same as p.run.")
    parser.add_argument('--plot', '-p', action="store_true", 
                        help="Turn on plotting")
    parser.add_argument('--ptyscan-modules', '-s', nargs="+", default=None,
                        help="A list of ptyscan modules to be loaded")
    parser.add_argument('--backends', '-b', nargs="+", default=None,
                        help="A list of CPU/GPU backends to be loaded")
    return parser.parse_args()

def run(args):

    # Load PtyScan modules
    if args.ptyscan_modules is not None:
        for ptyscan in args.ptyscan_modules:
            load_ptyscan_module(ptyscan)
        
    # Load CPU/GPU backends
    if args.backends is not None:
        for backend in args.backends:
            load_gpu_engines(backend)

    # Load parameter tree from file or JSON string
    if args.file:
        p = create_parameter_tree(load_config_as_dict_from_file(args.file))
    if args.json:
        p = create_parameter_tree(json.loads(args.json))
    p.run = args.identifier

    # TODO
    if args.plot:
        log("info", "Turning the plotting on.")
        #parameters.io.autoplot = u.Param(active=True)
        pass
    else:
        log("info", "Turning the plotting off.")
        p.io.autoplot = u.Param(active=False)

    # 
    if args.output_folder is not None:
        p.io.home = get_output_file_name(args)
        p.io.rfile = "%s.ptyr" % get_output_file_name(args)
        #parameters.io.autosave = u.Param(active=True)
        #log(3, "Autosave is on, with io going in {}, and the final reconstruction into {}".format(parameters.io.home,
        #                                                                                        parameters.io.rfile))
    else:
        p.io.rfile = None
        p.io.autosave = u.Param(active=False)
        log("info", "Autosave is off. No output will be saved.")

    # Substitute %(run) with in ptyscan 
    substitute_id_in_ptyscan(p)

    # Run PtyPy to given level
    P = Ptycho(p, level=args.ptypy_level)

    return P


def load_config_as_dict_from_file(config_file) -> dict:
    if config_file.endswith((".json", ".jsn")):
        param_dict = u.param_from_json(config_file)
    elif config_file.endswith((".yaml", "yml")):
        param_dict = u.param_from_yaml(config_file)
    else:
        raise FileExistsError(f"Cannot parse {config_file}, expecting a JSON or YAML config file")
    return param_dict

def create_parameter_tree(params) -> u.Param:
    parameters_to_run = u.Param()
    if params['base_file'] is not None:
        log(3, "Basing this scan off of the scan in {}".format(params['base_file']))
        previous_scan = Ptycho.load_run(params['base_file'], False)  # load in the run but without the data
        previous_parameters = previous_scan.p
        parameters_to_run.update(previous_parameters)
    if params['parameter_tree'] is not None:
        parameters_to_run.update(params['parameter_tree'], Convert=True)
    return parameters_to_run

def substitute_id_in_ptyscan(params):
    def _substitute(d, p):
        for k, v in d.items():
            if isinstance(v, MutableMapping):
                _substitute(v, p)
            elif isinstance(v, str) and re.search('%\(\w+\)s', v) is not None:
                # only perform substitution when the format %(...)s is present
                d[k] = v % p
    for scan_key, scan in params.scans.items():
        data_entry = scan.data
        _substitute(data_entry, params)

def get_output_file_name(args):
    from datetime import datetime
    now = datetime.now()
    if args.identifier is not None:
        output_path = "{}/scan_{}_{}".format(args.output_folder, args.identifier, now.strftime("%Y%m%d%H%M%S"))
    else:
        output_path = "{}/scan_{}".format(args.output_folder, now.strftime("%Y%m%d%H%M%S"))
    log("info", "Output is going in: {}".format(output_path))
    return output_path

if __name__ == "__main__":
    ptypy_run()
