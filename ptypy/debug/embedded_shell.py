# -*- coding: utf-8 -*-
"""
Tool to embed an ipython shell for debugging.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
from ..utils import log

__all__ = ['ipshell']

# Check if IPython is available
try:
    import IPython
# Assign warning to ipshell if not
except ImportError:
    log(3, 'IPython is not installed. Interactive shell is not available.')

    def ipshell():
        print('IPython is not installed. Interactive shell is not available.')
# Continue with setting up of embedded shell
else:
    # Get IPython version
    try:
        ip_version = IPython.version_info[0]
    # Assign random version if attribute does not exist
    except AttributeError:
        ip_version = 0

    # Load Config depending on IPython version
    if ip_version >= 4:
        from traitlets.config.loader import Config
    else:
        from IPython.config.loader import Config

    # Check whether embedded shell is nested in IPython or not
    try:
        get_ipython
    # Code embedded interpreter
    except NameError:
        banner = 'Dropping into IPython. Hit ctrl-D to resume execution.'
        exit_msg = 'Leaving Interpreter, back to program.'
        nested = 0
        if ip_version < 4:
            cfg = Config()
            prompt_config = cfg.PromptManager
            prompt_config.in_template = 'In <\\#>: '
            prompt_config.in2_template = '   .\\D.: '
            prompt_config.out_template = 'Out<\\#>: '
        else:
            cfg = Config()
            # Modify this for custom behaviour
            # cfg.TerminalInteractiveShell.prompts_class = CustomPrompt
    # Nested interpreter
    else:
        banner = '*** Nested interpreter ***'
        exit_msg = '*** Back in main IPython ***'
        nested = 1
        cfg = Config()

    # Embedded shell breaks tab completion in IPython version 4 and 0,
    # therefore deactivated
    if ip_version == 4 or ip_version == 0:
        def ipshell():
            print('Interactive shell deactivated in IPython version 4 and 0.\n'
                  'Please upgrade to a higher one to restore functionality.')
    else:
        # Import embeddable shell class
        from IPython.terminal.embed import InteractiveShellEmbed

        # Now create an instance of the embeddable shell. The first argument is
        # a string with options exactly as you would type them if you were
        # starting IPython at the system command line. Any parameters you want
        # to define for configuration can thus be specified here.
        ipshell = InteractiveShellEmbed(config=cfg,
                                        banner1=banner,
                                        exit_msg=exit_msg)
