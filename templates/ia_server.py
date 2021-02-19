from ptypy.io import interaction
from ptypy.utils import verbose

import time
verbose.set_level(5)
S = interaction.Server(
    address="tcp://127.0.0.1",
    port=55060,
)
S.activate()
keys=set()
while True:
    S.process_requests()
    nkeys = set(S.objects.keys())
    diff = nkeys - keys
    if diff:
        for keys in diff:
            print(keys)
        keys=nkeys
    time.sleep(0.01)