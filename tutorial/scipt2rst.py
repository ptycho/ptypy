import sys
import io
import contextlib
"""
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = io.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

with stdoutIO() as s:
    execfile(sys.argv[1])

print "out:", s.getvalue()
"""
name= sys.argv[1]
fpy=open(name,'r')
frst=open(name.replace('.py','.rst'),'w')

for line in fpy:
    if line.startswith('#'):
        wline=line[1:].strip()
    else:
        wline = '>>> '+line
    frst.write(wline)



