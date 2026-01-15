import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

frame = 10
if len(sys.argv)>2:
    frame = int(sys.argv[2])
    
if len(sys.argv)>1:
    pngs = [sys.argv[1]]
else:
    import glob
    pngs = glob.glob('_script2rst/*.png')
    
for png in pngs:
    im=Image.open(png)
    ar = np.array(im)
    bg = ar[0,0].copy()
    ar-=bg
    #print bg,ar,ar.shape
    t =ar[:,:,:2].sum(-1)
    th = np.nonzero(t.sum(0))[0]
    tv = np.nonzero(t.sum(1))[0]
    
    im2=im.crop( (th.min()-frame,tv.min()-frame,th.max()+frame, tv.max()+frame))
    im2.save(png)
    #plt.imshow(im2)
    #plt.show()

    
    
