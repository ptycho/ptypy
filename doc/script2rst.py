import sys
import StringIO
import contextlib
import os

scripts = [
'../tutorial/minimal_script.py',
'../tutorial/ptypyclasses.py',
'../tutorial/simupod.py',
'../tutorial/ownengine.py',
'../tutorial/subclassptyscan.py',
]

if len(sys.argv)==1:
    for script in scripts:
        os.system('python '+sys.argv[0]+' '+script)
    sys.exit()

indent_keys = ['for','if','with','def','class']

sout = StringIO.StringIO()
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
    print s.getvalue()

script_name= sys.argv[1]
fpy=open(script_name,'r')
name = script_name.replace('.py','').split(os.sep)[-1]
frst=open('_script2rst'+os.sep+name+'.tut','w')
fig_path = '_script2rst'+os.sep

frst.write('.. note::\n   This tutorial was generated from the python source\
 :file:`ptypy/tutorial/%s` using :file:`ptypy/doc/%s`.\n\n' % (os.path.split(script_name)[-1],sys.argv[0]))
 
was_comment = True

def check_for_fig(wline):
    if 'savefig' in wline:
        print 'found fig' 
        from matplotlib import pyplot as plt
        fig = plt.gcf()
        fig_name = name + '_%02d'  % fig.number
        fname = fig_path + fig_name + '.png'
        fig.savefig(fname, dpi=300)

        frst.write('.. figure:: '+'..'+os.sep+fname+'\n')
        frst.write('   :width: 70 %\n')
        frst.write('   :figclass: highlights\n')
        other=[s.strip() for s in wline.split(';')]
        print other
        if len(other)>1:
            frst.write('   :name: '+other[1]+'\n\n')
        if len(other)>2:
            frst.write('   '+other[2]+'\n\n')
        frst.write('\n')
        wline==''
        return True
    else:
        return False

for line in fpy:
    print line
    if 'savefig' in line:
        from matplotlib import pyplot as plt
        fig = plt.gcf()
        fig_name = name + '_%02d'  % fig.number
        fname = fig_path + fig_name + '.png'
        fig.savefig(fname, dpi=300)
        frst.write('\nSee :numref:`%s` for the plotted image.\n\n' % fig_name)
        frst.write('.. figure:: '+'..'+os.sep+fname+'\n')
        frst.write('   :width: 70 %\n')
        frst.write('   :figclass: highlights\n')
        frst.write('   :name: ' + fig_name+'\n\n')
        # jump the line to get the caption
        for line2 in fpy:
            print line2
            if not line2.startswith('#'):
                break
            frst.write('   '+line2[1:].strip()+'\n')
        frst.write('\n')
        continue
        
    if line.startswith('"""'):
        frst.write('.. parsed-literal::\n\n')
        for line2 in fpy:
            if line2.startswith('"""'):
                break
            frst.write('   '+ line2)
        continue
    
    indent = False
    for key in indent_keys:
        if line.startswith(key): 
            indent = True
            break
            
    if indent:
        frst.write('\n::\n\n   >>> '+line)
        func = line
        for line2 in fpy:
            if not line2.startswith('    '):
                frst.write('\n')
                break
            func+=line2
            frst.write('   >>> '+line2)
        exec func+'\n'
        continue
        
    wline=line.strip()
    if wline=='':
        frst.write('\n')
        continue
    
    with stdoutIO() as sout:
        exec wline
        out = sout.getvalue()
        sout.buf =''
    if len(wline)>0:
        if line.startswith('# '):
            wline=line[2:]
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
        if out.strip()!='':
            #frst.write('\n')
            for l in out.split('\n'):
                frst.write(' '*3+l+'\n')
            out=''
    """
    frst.write(wline+'\n')
    if out.strip()!='':
        frst.write('\n:Out:\n   ::\n\n')
        for l in out.split('\n'):
            frst.write(' '*6+l+'\n')
    """
    

