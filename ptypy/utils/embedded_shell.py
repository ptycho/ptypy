# -*- coding: utf-8 -*-
"""
Tool to embed an ipython shell for debugging.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

__all__ = ['ipshell']

try:
    from IPython.config.loader import Config

    try:
        get_ipython
    except NameError:
        nested = 0
        cfg = Config()
        prompt_config = cfg.PromptManager
        prompt_config.in_template = 'In <\\#>: '
        prompt_config.in2_template = '   .\\D.: '
        prompt_config.out_template = 'Out<\\#>: '
    else:
        #print("Running nested copies of IPython.")
        #print("The prompts for the nested copy have been modified")
        cfg = Config()
        nested = 1

    # First import the embeddable shell class
    from IPython.frontend.terminal.embed import InteractiveShellEmbed

    # Now create an instance of the embeddable shell. The first argument is a
    # string with options exactly as you would type them if you were starting
    # IPython at the system command line. Any parameters you want to define for
    # configuration can thus be specified here.
    ipshell = InteractiveShellEmbed(config=cfg,
                                    banner1='Dropping into IPython. Hit ctrl-D to resume execution.',
                                    exit_msg='Leaving Interpreter, back to program.')
except:
    def ipshell():
        print('IPython shell embedding failed')
