import sys
import StringIO
import contextlib
import os
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def exec2str(statement):
    with stdoutIO() as s:
        exec(statement)
    print s.getvalue()

script_name= sys.argv[1]
fpy=open(script_name,'r')
name = script_name.replace('.py','').split(os.sep)[-1]
frst=open('rst'+os.sep+name+'.rst','w')
fig_stub = '_img'+os.sep+name

was_comment = True

def check_for_fig(wline):
    if wline.startswith('make fig'):
        print 'found fig' 
        from matplotlib import pyplot as plt
        fig = plt.gcf()
        fname = fig_stub +'_%03d.png' % fig.number
        print fname
        fig.savefig(fname)
        frst.write('.. figure:: '+'..'+os.sep+fname+'\n')
        frst.write('   :width: 70 %\n')
        frst.write('   :figclass: highlights\n')
        other=[s.strip() for s in wline.split(';')]
        print other
        if len(other)>1:
            frst.write('   :alt: '+other[1]+'\n\n')
        if len(other)>2:
            frst.write('   '+other[2]+'\n\n')
        frst.write('\n')
        wline==''
        return True
    else:
        return False

for line in fpy:
    if line.startswith('"""'):
        frst.write('.. parsed-literal::\n\n')
        for line2 in fpy:
            if line2.startswith('"""'):
                break
            frst.write('   '+ line2)
        continue
    
    wline=line.strip()
    if wline=='':
        frst.write('\n')
        continue
        
    with stdoutIO() as s:
        exec(wline)
    out = s.getvalue()
    if len(wline)>0:
        if line.startswith('#'):
            wline=line[1:].strip()
            isfig = check_for_fig(wline)
            was_comment = True
            if not isfig:
                frst.write(wline+'\n')
        else:
            wline = '   >>> '+wline
            if was_comment:
                wline = '\n'+wline
                was_comment = False
                
            frst.write(wline+'\n')
            
        if out.strip()!='':
            #frst.write('\n')
            for l in out.split('\n'):
                frst.write(' '*3+l+'\n')
    """
    frst.write(wline+'\n')
    if out.strip()!='':
        frst.write('\n:Out:\n   ::\n\n')
        for l in out.split('\n'):
            frst.write(' '*6+l+'\n')
    """
    

