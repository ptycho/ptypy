import sys
import StringIO
import contextlib

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

name= sys.argv[1]
fpy=open(name,'r')
frst=open('rst/'+name.replace('.py','.rst').split('/')[-1],'w')

was_comment = True

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
            was_comment = True
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
    


