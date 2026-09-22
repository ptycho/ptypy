import sys
import io
from importlib.resources import files
import contextlib
from glob import glob
import os

indent_keys = ['for', 'if', 'with', 'def', 'class']

@contextlib.contextmanager
def stdoutIO(stdout=None):
    sout = io.StringIO()
    old = sys.stdout
    if stdout is None:
        stdout = sout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old

def replace_rst_in_templates(template, outdir="./userguide/generated/legacy/rst/"):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    include_key = '!!include!!'
    with open(template, 'r') as ftmp:
        frst = open(outdir+"/"+os.path.split(template)[-1].replace('.tmp', '.rst'), 'w')
        for line in ftmp:
            if line.startswith(include_key):
                with open(line.split()[1], 'r') as f:
                    frst.writelines(f)
                continue
            else:
                frst.write(line)
        
def generate_legacy_tutorial_rst(script_name, outdir='userguide/generated/legacy'):

    _tutorial_dir = 'tutorial/'
    scr = "../../" + _tutorial_dir + script_name
    print("Generate legacy tutorial for script: ", scr)
    
    fpy = open(scr, 'r')
    name = scr.replace('.py', '').split(os.sep)[-1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    frst = open(outdir + os.sep + name + '.tut', 'w')
    fig_path = outdir + os.sep

    frst.write("""
.. note::
   This tutorial was generated from the python source
   :file:`[ptypy_root]/tutorial/%(fname)s` using :file:`ptypy/doc/%(this)s`.
   You are encouraged to modify the parameters and rerun the tutorial with::

     $ python [ptypy_root]/tutorial/%(fname)s

""" % {'fname': os.path.split(scr)[-1], 'this': script_name})

    was_comment = True

    shared = {}
    
    while True:
        line = fpy.readline()
        if not line:
            break
        print(line)
        if 'savefig' in line:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
            fig_name = name + '_%02d' % fig.number
            fname = fig_path + fig_name + '.png'
            #plt.tight_layout()
            fig.savefig(fname, dpi=300)
            frst.write('\nSee :numref:`%s` for the plotted image.\n\n' % fig_name)
            frst.write('.. figure:: '+'../'+os.sep+fig_name+'.png'+'\n')
            ratio = fig.get_figwidth()/fig.get_figheight()
            frst.write('   :width: %d ' % min(int(ratio * 60), 100)+'%\n')
            frst.write('   :figclass: highlights\n')
            frst.write('   :name: ' + fig_name+'\n\n')
            # jump the line to get the caption
            while True:
                line2 = fpy.readline()
                print(line2)
                if not line2.startswith('#'):
                    break
                frst.write('   '+line2[1:].strip()+'\n')
            frst.write('\n')
            fig.clf()
            continue

        if line.startswith('"""'):
            frst.write('.. parsed-literal::\n\n')
            while True:
                line2 = fpy.readline()
                if line2.startswith('"""'):
                    break
                frst.write('   ' + line2)
            continue

        decorator = False
        indent = False
        for key in indent_keys:
            if line.startswith(key):
                indent = True
                break

        if line.startswith('@'):
            indent = True
            decorator = True

        if indent:
            frst.write('\n::\n\n   >>> '+line)
            func = line
            pt = fpy.tell()  # Needed to rewind file when block is finished
            while True:
                line2 = fpy.readline()
                if line2.strip() and not line2.startswith('    '):
                    if decorator:
                        decorator = False
                    else:
                        frst.write('\n')
                        fpy.seek(pt)
                        break
                func += line2
                frst.write('   >>> '+line2)
                pt = fpy.tell()
            exec(func+'\n', shared)
            continue

        wline = line.strip()
        if not wline:
            frst.write('\n')
            continue

        with stdoutIO() as sout:         
            exec(wline, shared)
            out = sout.getvalue()
            sout.buf = ''


        if len(wline) > 0:
            if line.startswith('#'):  # This was '# ' before, but pycharm eats trailing whitespaces so the marker
                wline = line[2:]
                was_comment = True
                if not wline:
                    # just in case there is an empty line on purpose
                    frst.write('\n')
                else:
                    frst.write(wline)
            else:
                wline = '   >>> '+wline
                if was_comment:
                    wline = '\n::\n\n'+wline
                    was_comment = False

                frst.write(wline+'\n')

            #print out
            if out.strip():
                print(out)
                for l in out.split('\n'):
                    frst.write(' '*3+l+'\n')
                out = ''



