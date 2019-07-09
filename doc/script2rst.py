import sys
import io
import contextlib
import os

tutorial_dir = 'tutorial/'
scripts = ['minimal_script.py',
           'ptypyclasses.py',
           'simupod.py',
           'ownengine.py',
           'subclassptyscan.py']

if len(sys.argv) == 1:
    import pkg_resources
    import subprocess

    for script in scripts:
        scr = pkg_resources.resource_filename('ptypy', tutorial_dir+script)
        if not os.path.exists(scr):
            print('Using backup tutorial for %s' % script)
            scr = '../tutorial/'+script
        #subprocess.call(['python',sys.argv[0]+' '+scr]) # doesn't work
        os.system('python ' + sys.argv[0]+' '+scr)
    sys.exit()

indent_keys = ['for', 'if', 'with', 'def', 'class']

sout = io.StringIO()


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = sout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old
    

def exec2str(statement):
    with stdoutIO() as s:
        exec(statement)
    print(s.getvalue())

script_name = sys.argv[1]
fpy = open(script_name, 'r')
name = script_name.replace('.py', '').split(os.sep)[-1]
if not os.path.exists('_script2rst'):
    os.makedirs('_script2rst')
frst = open('_script2rst'+os.sep+name+'.tut', 'w')
fig_path = '_script2rst'+os.sep

frst.write("""
.. note::
   This tutorial was generated from the python source
   :file:`[ptypy_root]/tutorial/%(fname)s` using :file:`ptypy/doc/%(this)s`. 
   You are encouraged to modify the parameters and rerun the tutorial with::
   
     $ python [ptypy_root]/tutorial/%(fname)s

""" % {'fname': os.path.split(script_name)[-1], 'this': sys.argv[0]})
 
was_comment = True
def debug(x):
    print(x)

def check_for_fig(wline):
    if 'savefig' in wline:
        print('found fig')
        from matplotlib import pyplot as plt
        fig = plt.gcf()
        fig_name = name + '_%02d' % fig.number
        fname = fig_path + fig_name + '.png'
        plt.tight_layout()
        fig.savefig(fname, dpi=300)

        frst.write('.. figure:: '+'..'+os.sep+fname+'\n')
        frst.write('   :width: 70 %\n')
        frst.write('   :figclass: highlights\n')
        other = [s.strip() for s in wline.split(';')]
        print(other)
        if len(other) > 1:
            frst.write('   :name: '+other[1]+'\n\n')
        if len(other) > 2:
            frst.write('   '+other[2]+'\n\n')
        frst.write('\n')
        wline == ''
        return True
    else:
        return False

while True:
    line = fpy.readline()
    if line == '':
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
        frst.write('.. figure:: '+'..'+os.sep+fname+'\n')
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
        exec(func+'\n')
        continue
        
    wline = line.strip()
    if wline == '':
        frst.write('\n')
        continue
    
    with stdoutIO() as sout:
        exec(wline)
        out = sout.getvalue()
        sout.buf = ''
    if len(wline) > 0:
        if line.startswith('# '):
            wline = line[2:]
            #isfig = check_for_fig(wline)
            was_comment = True
            frst.write(wline)
        else:
            wline = '   >>> '+wline
            if was_comment:
                wline = '\n::\n\n'+wline
                was_comment = False
                
            frst.write(wline+'\n')
        
        #print out
        if out.strip() != '':
            #frst.write('\n')
            for l in out.split('\n'):
                frst.write(' '*3+l+'\n')
            out = ''
    """
    frst.write(wline+'\n')
    if out.strip()!='':
        frst.write('\n:Out:\n   ::\n\n')
        for l in out.split('\n'):
            frst.write(' '*6+l+'\n')
    """
    

