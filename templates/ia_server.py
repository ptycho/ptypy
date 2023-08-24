from ptypy.io import interaction
from ptypy.utils import verbose
import sys

if len(sys.argv) >1:
    ip = sys.argv[1]
else:
    ip = 0.0.0.0

if len(sys.argv) >2:
    port = int(sys.argv[2])
else:
    port =  5860


import time
verbose.set_level(5)
S = interaction.Server(
    address="tcp://"+ip,
    port=port,
)
S.activate()
keys=set()
i=0
while True:
    i+=1
    S.process_requests()
    nkeys = set(S.objects.keys())
    diff = nkeys - keys
    if diff:
        for keys in diff:
            print(keys)
        keys=nkeys
    time.sleep(0.01)
    # if i%100==0:
    #     print(S.objects)
