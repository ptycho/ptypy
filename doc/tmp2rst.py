"""
This convenience script works on all scripts that are marked .tmp in
the rst/ folder. It actually literally includes all references that
are marked with the .. include:: statement and spits out a patched up 
version.
"""

from glob import glob
import os

include_key = '!!include!!'
alltmp = glob('rst_templates/*.tmp')
#print alltemp

for tmp in alltmp:
    with open(tmp, 'r') as ftmp:
        frst = open('rst/'+os.path.split(tmp)[-1].replace('.tmp', '.rst'), 'w')
        for line in ftmp:
            if line.startswith(include_key):
                with open(line.split()[1], 'r') as f:
                    frst.writelines(f)
                continue
            else:
                frst.write(line)
