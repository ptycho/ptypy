"""
Created on 2013-04-02

@author: Bjoern Enders
"""

from scipy import ndimage as ndi
import numpy as np
from misc import *
import parallel

__all__ = ['hdr_image','mirror','pad_lr','crop_pad_axis','crop_pad',
            'xradia_star','png2mpg','mass_center','phase_from_dpc',
            'radial_distribution','str2range','stxm_analysis','stxm_analysis']

def str2range(s):
    """
    generates an index list
    range_from_string('1:4:2') == range(1,4,2)
    BUT
    range_from_string('1') == range(1,2)
     
    Author: Bjoern Enders
    """
    start = 0
    stop = 1
    step = 1
    l=s.split(':')
    
    il = [int(ll) for ll in l]
    
    if len(il)==0:
        pass    
    elif len(il)==1:
        start=il[0]; stop=start+1
    elif len(il)==2:
        start, stop= il
    elif len(il)==3:
        start, stop, step = il
        
    return range(start,stop,step)


def hdr_image(img_list, exp_list, thresholds=[3000,50000], dark_list=[],avg_type='highest',mask_list=[],ClipLongestExposure=False,ClipShortestExposure=False):
    """
    generate high dynamic range image from a list of images "img_list" and 
    exposure information "exp_list".
    
    Usage:
    >> dark_list,meta=io.image_read('/path/to/dark/images/ccd*.raw')
    >> img_list,meta=io.image_read('/path/to/images/ccd*.raw')
    >> exp_list=[meta[j]['exposure_time__key'] for j in range(len(meta))]
    >> hdr,masks = hdr_image(img_list, exp_list, dark_list=dark_list)
        
    PARAMETERS:
    
    img_list : sequence of images (as 2d np.array)
    exp_list : associated exposures to each element of above sequence
    thresholds: sequence of upper limit (overexposure) and lower limit (noise floor) in the images
    dark_list : single or sequence of dark images (as 2d np.array)
    avg_type : 'highest' -> the next longest exposure is used to replace overexposed pixels.
               'other_string' -> each overexposed pixel is raplaced by the pixel average of 
                                 all other images with valid pixel values for that pixel.
    mask_list : provide additional masking (dead pixels, hot pixels),
                single or sequence of 2d np.array
    ClipLongestExposure (False) : if True, also mask the noise floor in the longest exposure
    ClipShortestExposure (False) : if True, also mask the overexposed pixels in the shortest exposure
    
    """
    min_exp=min(exp_list)
    #print min_exp
    max_exp=max(exp_list)
    #print max_exp
    if len(mask_list)==0:
        mask_list=[np.ones(img.shape) for img in img_list]
    elif len(mask_list)==1:
        mask_list = mask_list*len(img_list)
        
    if len(dark_list)==0:
        dark_list=[np.zeros(img.shape) for img in img_list]
    elif len(dark_list)==1:
        dark_list = dark_list*len(img_list)
    # convert to floats except for mask_list 
    img_list=[img.astype(np.float) for img in img_list]
    dark_list=[dark.astype(np.float) for dark in dark_list]   
    exp_list=[np.float(exp) for exp in exp_list]
    mask_list=[mask.astype(np.int) for mask in mask_list]
         
    for img,dark,exp,mask in zip(img_list, dark_list,exp_list,mask_list):
        img[:]=abs(img-dark)
        #img[img<0]=0
        #figure();imshow(img);colorbar()
        maskhigh = ndi.binary_erosion(img < thresholds[1])
        masklow = ndi.binary_dilation(img > thresholds[0])
        if abs(exp-min_exp)/exp < 0.01 and not ClipShortestExposure:
            mask *= masklow
        elif abs(exp-max_exp)/exp < 0.01 and not ClipLongestExposure:
            mask *= maskhigh
        else:
            mask *= masklow*maskhigh    

    if avg_type=='highest':
        ix=list(np.argsort(exp_list))
        ix.reverse()
        mask_sum= mask_list[ix[0]]
        #print ix,img_list
        img_hdr = np.array(img_list[ix[0]]) * max_exp/exp_list[ix[0]] 
        for j in range(1,len(ix)):
            themask = (1-mask_sum)*mask_list[ix[j]]
            #figure();imshow(mask_sum);
            #figure();imshow(themask);
            mask_sum[themask.astype(bool)] = 1
            img_hdr[themask.astype(bool)] = img_list[ix[j]][themask.astype(bool)] * max_exp/exp_list[ix[j]]     
    else:
        mask_sum=np.zeros_like(mask_list[0]).astype(np.int)
        img_hdr = np.zeros_like(img_list[0])
        for img,exp,mask in zip(img_list,exp_list,mask_list):
            img = img * max_exp/exp
            img_hdr += img * mask
            mask_sum += mask
        mask_sum[mask_sum==0]=1
        img_hdr = img_hdr / (mask_sum*1.)
    
    return img_hdr,mask_list

def png2mpg(listoffiles,framefile='frames.txt',fps=5,bitrate=2000,codec='wmv2',Encode=True,RemoveImages=False):
    """
    makes movie (*.mpg) from png or jpeg 
    Usage:
    
    - png2mpg(['/path/to/image_000.png'])
        1) search for files similar to image_*.png in '/path/to/'
        2) found files get listed in a file '/path/to/frames.txt'
        3) calls mencoder to use that file to encode a movie with the default args.
        4) movie is in the same folder as 'frames.txt' 
    
    - png2mpg(['/path1/to/imageA_040.png','/path2/to/imageB_001.png'],framefile='./banana.txt')
        generates list file 'banana_text' in current folder
        list file contains in order every path compatible with wildcard
            '/path1/to/imageA_*.png'
            '/path2/to/imageB_*.png'
    
    - str=png2mpg(...,Encode=False)
        returns encoder string. Use os.system(encoderstring) for later encoding
        
    PARAMETER:
    fps : desired fps-rate
    bitrate : encoding detail, determines video quality
    conde : defines the used codec
    """
    import os
    import glob
    import re
    framelist=[]
    for frame_or_list in listoffiles:
        if not os.path.isfile(frame_or_list):
            raise ValueError('File %s not found' % frame_or_list)
        else:
            head,tail=os.path.split(frame_or_list)
            if os.path.splitext(tail)[1] in ['.txt','.dat']:
                print('Found text file - try to interpret it as a list of image files.')
                temp=open(frame_or_list)
                line='uiui'
                while 'EOF' not in line and line!='':
                    line=temp.readline()
                    content=line.strip()
                    #print content
                    # look if the referenced file exists and then add the file to the list
                    if os.path.isfile(content):
                        framelist.append(content)
                    elif os.path.isfile(head+os.sep+content):
                        framelist.append(head+os.sep+content)
                    else:
                        print('File reference %s not found, continueing..' % content)
                        continue
                temp.close()
            else:
                #trying to find similar images
                body,imagetype=os.path.splitext(tail)
                #replace possible numbers by a wildcard
                newbody=re.sub('\d+','*',body)
                wcard=head+os.sep+newbody+imagetype
                #print wcard
                imagfiles=glob.glob(wcard)
                imagfiles.sort()
                framelist+=imagfiles
                
    if os.path.split(framefile)[0]=='':                        
        ff=head+os.sep+framefile
    else:
        ff=framefile
        
    newframefile=open(ff,'w')
    #newframefile.writelines(framelist)
    for frame in framelist:
        newframefile.write('/'+os.path.relpath(frame,'/')+'\n')
    newframefile.close()

    last=os.path.relpath(framelist[-1])
    savelast=os.path.split(last)[0]+'/last'+os.path.splitext(last)[1]
    #print last
    #print savelast
    os.system('cp %s %s' % (last,savelast))

    frametype=os.path.splitext(frame)[1].split('.')[-1]    
    body=os.path.splitext(ff)[0] 
    
    mencode_dict=dict(
        listf=os.path.relpath(ff),
        outputf=os.path.relpath(body),
        fps=fps,
        frametype=frametype,
        bitrate=bitrate,
        codec=codec
    )
    encodepattern='mencoder ' \
    +'mf://@%(listf)s ' \
    +'-o %(outputf)s.mpg ' \
    +'-mf type=%(frametype)s:fps=%(fps)d ' \
    +'-ovc lavc ' \
    +'-lavcopts vbitrate=%(bitrate)d:vcodec=%(codec)s ' \
    +'-oac copy'  
    
    encoderstring=encodepattern % mencode_dict  
    
    if Encode:
        try:
            os.system(encoderstring)
        except OSError:
            print('Encoding failed - exiting')
        if RemoveImages:
            nff=open(ff,'r')
            line='haha'
            while line!='':
                line=nff.readline().strip()
                print('Removing %s' % line)
                #os.remove(line)
                try:
                    os.remove(line)
                except OSError:
                    print OSError
                    print('Removing %s failed .. continuing' % line)
                    continue
            nff.close()
            if line=='':
                try:
                    print('Removing %s' % ff)
                    os.remove(ff)
                except OSError:
                    print('Removing %s failed' % ff)                
    else:
        return encoderstring        
     
def mirror(A,axis):
    """\
    mirrors array A along one axis 
    """
    return np.flipud(A.swapaxes(axis,0)).swapaxes(0,axis)
    
def pad_lr(A,axis,l,r,fillpar=0.0, filltype='scalar'):
    """\
    Pads ndarray 'A' orthogonal to 'axis' with 'l' layers (pixels,lines,planes,...)
    on low side an 'r' layers on high side. 
    if filltype=
        'scalar' : uniformly pad with fillpar
        'mirror' : mirror A
        'periodic' : well, periodic fill
        'custom' : pad according arrays found in fillpar
         
    """ 
    fsh=np.array(A.shape)
    if l>fsh[axis]: #rare case
        l-=fsh[axis]
        A=pad_lr(A,axis,fsh[axis],0,fillpar, filltype)
        return pad_lr(A,axis,l,r,fillpar, filltype)
    elif r>fsh[axis]: 
        r-=fsh[axis]
        A=pad_lr(A,axis,0,fsh[axis],fillpar, filltype)
        return pad_lr(A,axis,l,r,fillpar, filltype)
    elif filltype=='mirror':        
        left=mirror(np.split(A,[l],axis)[0],axis)
        right=mirror(np.split(A,[A.shape[axis]-r],axis)[1],axis)
    elif filltype=='periodic':
        right=np.split(A,[r],axis)[0]
        left=np.split(A,[A.shape[axis]-l],axis)[1]
    elif filltype=='project':
        fsh[axis]=l
        left=np.ones(fsh,A.dtype)*np.split(A,[1],axis)[0]
        fsh[axis]=r
        right=np.ones(fsh,A.dtype)*np.split(A,[A.shape[axis]-1],axis)[1] 
    if filltype=='scalar' or l==0:
        fsh[axis]=l
        left=np.ones(fsh,A.dtype)*fillpar
    if filltype=='scalar' or r==0:
        fsh[axis]=r
        right=np.ones(fsh,A.dtype)*fillpar 
    if filltype=='custom':
        left=fillpar[0].astype(A.dtype)
        rigth=fillpar[1].astype(A.dtype)   
    return np.concatenate((left,A,right),axis=axis)


def _roll_from_pixcenter(sh,center):
    """\
    returns array of ints as input for np.roll
    use np.roll(A,-roll_from_pixcenter(sh,cen)[ax],ax) to put 'cen' in geometric center of array A
    """
    sh=np.array(sh)
    if center != None:
        if center=='fftshift':
            cen=sh//2.0
        elif center=='geometric':
            cen=sh/2.0-0.5
        elif center=='fft':
            cen=sh*0.0
        elif center is not None:
            cen=sh*np.asarray(center) % sh - 0.5
            
        roll=np.ceil(cen - sh/2.0) % sh
    else:
        roll=np.zeros_like(sh)
    return roll.astype(int)
    


def _translate_to_pix(sh,center):
    """\
    takes arbitrary input and translates it to a pixelpositions with respect to sh.
    """
    sh=np.array(sh)
    if center=='fftshift':
        cen=sh//2.0
    elif center=='geometric':
        cen=sh/2.0-0.5
    elif center=='fft':
        cen=sh*0.0
    elif center is not None:
        cen=sh*np.asarray(center) % sh - 0.5

    return cen
    
    

def crop_pad_axis(A,hplanes,axis,roll=0,fillpar=0.0, filltype='scalar'):
    """\
    crops or pads a volume array 'A' at beginning and end of axis 'axis' 
    with a number of hyperplanes specified by 'hplanes'

    Paramters:
    -------------
    A : nd-numpy array
    
    hplanes: tuple or scalar int
    axis: int, axis to be used for cropping / padding
    roll: int, roll array backwards by this number prior to padding / cropping. the roll is reversed afterwards
   
    if 'hplanes' is,
    -scalar and negativ : 
        crops symmetrically, low-index end of axis is preferred if hplane is odd,
    -scalar and positiv : 
        pads symmetrically with a fill specified with 'fillpar' and 'filltype'
        look at function pad_lr() for detail.
    -is tupel : function pads /crops asymmetrically according to the tupel.
    
    Usage:
    -------------
    A=np.ones((8,9))
    B=crop_pad_axis(A,2,0)
    -> a total of 2 rows, one at top, one at bottom (same as crop_pad_axis(A,(1,1),0))
    B=crop_pad_axis(A,(-3,2),1)
    -> crop 3 columns on left side and pad 2 columns on right
    V=np.random.rand(3,5,5)
    B=crop_pad_axis(V,-2,0)
    -> crop one plane on low-side and high-side (total of 2) of Volume V
    B=crop_pad_axis(V,(3,-2),1,filltype='mirror')
    -> mirror volume 3 planes on low side of row axis, crop 2 planes on high side
    
    Author: Bjoern Enders
    """
    if np.isscalar(hplanes):
        hplanes=int(hplanes)
        r=np.abs(hplanes) / 2 * np.sign(hplanes)
        l=hplanes - r
    elif len(hplanes)==2:
        l=int(hplanes[0])
        r=int(hplanes[1])
    else:
        raise RuntimeError('unsupoorted input for \'hplanes\'')
        
    if roll!=0:
        A=np.roll(A,-roll,axis=axis)
        
    if l<=0 and r<=0:
        A=np.split(A,[-l,A.shape[axis]+r],axis)[1]
    elif l>0 and r>0:
        A=pad_lr(A,axis,l,r,fillpar,filltype)
    elif l>0 and r<=0:
        A=pad_lr(A,axis,l,0,fillpar,filltype)
        A=np.split(A,[0,A.shape[axis]+r],axis)[1]
    elif l<=0 and r>0:
        A=pad_lr(A,axis,0,r,fillpar,filltype)
        A=np.split(A,[-l,A.shape[axis]],axis)[1]
        
        
    if roll!=0:
        return np.roll(A,roll+r,axis=axis)
    else:
        return A

 
def crop_pad(A,hplane_list,axes=None,cen=None,fillpar=0.0,filltype='scalar'):
    """\
    crops or pads a volume array 'A' with a number of hyperplanes according to parameters in 'hplanes'
    wrapper for crop_pad_axis
    
    Parameters
    ----------------------
    hplane_list : 
     -list of scalars or tupels counting the number of hyperplanes to crop / pad 
     -see crop_pad_axis() for detail
     -if N=len(hplane_list) has less entries than dimensions of A, the last N axes are used 

    axes: list of axes to be used for cropping / padding, has to be same length as hplanes
    
    cen: center of array, padding/cropping occurs at cen + A.shape / 2
    
    Usage:
    ----------------------
    V=np.random.rand(3,5,5)
    B=crop_pad(V,[3,4])
    ->  pads 4 planes of zeros on the last axis (2 on low side and 2 on high side),
        and pads 3 planes of zeros on the second last axis (2 on low side and 1 on high side)
        equivalent: B=crop_pad(V,[(2,1),(2,2)])
                    B=crop_pad(V,[(2,1),(2,2)], axes=[-2,-1],fillpar=0.0,filltype='scalar')
    
    C=pyE17.utils.fgrid_2d((4,5))
    cropped_fgrid=crop_pad(V,[-2,4],cen='fft')
    -> note that cropping/ padding now occurs at the start and end of fourier coordinates
    -> useful for cropping /padding high frequencies in fourier space.
    
    Author: Bjoern Enders
    """
    if axes is None:
        axes=np.arange(len(hplane_list))-len(hplane_list)
    elif not(len(axes)==len(hplane_list)):
        raise RuntimeError('if axes is specified, hplane_list has to be same length as axes')
    
    sh=np.array(A.shape)
    roll = _roll_from_pixcenter(sh,cen)
        
    for ax,cut in zip(axes,hplane_list):
        A=crop_pad_axis(A,cut,ax,roll[ax],fillpar,filltype)
    return A
        

def xradia_star(sh,spokes=48,std=0.5,minfeature=5,ringfact=2,rings=4,contrast=1.,Fast=False):
    """\
    creates an Xradia-like star pattern on on array of shape sh
    std: "resolution" of xradia star, i.e. standard deviation of the 
         errorfunction used for smoothing the step (in pixel)
    spokes : number of spokes
    minfeature : smallest spoke width (in pixel)
    ringfact : factorial increase in featuresize from ring to ring.
    rings : number of rings
    contrast : minimum contrast, set to 0 for gradual color change from zero to 1 
               set to 1 for no gradient in the spokes
                  
    Fast : if set to False, the error function is evaluated at the edges
            -> preferred when using fft, as its features are less prone to antiaaliasing
           if set to True, simple boolean comparison will be used instead and the 
           result is later blurred with a gaussian filter.
            -> roughly a factor 2 faster
    """
    from scipy.ndimage import gaussian_filter as gf
    from scipy.special import erf
    
    def step(x,a,std=0.5):
        if not Fast:
            return 0.5*erf((x-a)/(std*2))+0.5
        else:
            return (x>a).astype(float)
            
    def rect(x,a):
        return step(x,-a/2.,std) * step(-x,-a/2.,std)
    
    def rectint(x,a,b):
        return step(x,a,std) * step(-x,-b,std)
    
    ind=np.indices(sh)
    cen=(np.array(sh)-1)/2.0
    ind=ind-cen.reshape(cen.shape+len(cen)*(1,))
    z=ind[1]+1j*ind[0]
    spokeint,spokestep=np.linspace(0.0*np.pi,1.0*np.pi,spokes/2,False,True)   
    spokeint+=spokestep/2

    r=np.abs(z)
    r0=(minfeature/2.0)/np.sin(spokestep/4.)
    rlist=[]
    rin=r0
    for ii in range(rings):
        if rin > max(sh)/np.sqrt(2.):
            break
        rin*=ringfact
        rlist.append((rin*(1-2*np.sin(spokestep/4.)),rin))
        
    spokes=np.zeros(sh)
    contrast= np.min((np.abs(contrast),1))
    
    mn=min(spokeint)
    mx=max(spokeint)
    for a in spokeint:
        color = 0.5-np.abs((a-mn)/(mx-mn)-0.5)
        spoke=step(np.real(z*np.exp(-1j*(a+spokestep/4))),0)-step(np.real(z*np.exp(-1j*(a-spokestep/4))),0)
        spokes+= (spoke*color+0.5*np.abs(spoke))*(1-contrast) + contrast*np.abs(spoke)
    
    spokes*=step(r,r0)
    spokes*=step(rlist[-1][0],r)
    for ii in range(len(rlist)-1):
        a,b=rlist[ii]
        spokes*=(1.0-rectint(r,a,b))

    if Fast:
        return gf(spokes,std)
    else:
        return spokes
        
def mass_center(A, axes=None):
    """\
    returns mass center of n-dimensional array 'A' 
    along tuple of axis 'axes'
    """
    A=np.asarray(A)
    
    if axes is None:
        axes=tuple(range(1,A.ndim+1))
    else:
        axes=tuple(np.array(axes)+1)
        
    return np.sum(A*np.indices(A.shape),axis=axes)/np.sum(A)
    
def radial_distribution(A,radii=None):
    """\
    return radial mass distribution
    radii: sequence of radii to calculate enclosed mass
    """
    if radii is None:
        radii=range(1,np.max(A.shape))
       
    coords=np.indices(A.shape)-np.reshape(mass_center(A),(A.ndim,) +A.ndim*(1,))
    masses=[np.sum(A*(np.sqrt(np.sum(coords**2,0)) < r)) for r in radii]

    return radii,masses


def stxm_analysis(storage,probe=None):
    """
    Performs a stxm analysis on a storage using the pods.
    This function is MPI compatible.
    
    Parameters:
    ----------
    
    storage : A ptypy.core.Storage instance
    
    probe   : None, scalar or array
           
            if None, picks a probe from the first view's pod
            if scalar, uses a Gaussian with probe as standard deviation
            else: attempts to use passed value directly as 2d-probe
           
    Returns:
    --------
    trans, dpc_row, dpc_col : Nd-array of shape storage.shape
            
            trans 
                is transmission 
            dpc_row 
                is differential phase contrast along row-coordinates,
                i.e. vertical direction (y-direction)
            dpc_col
                is differential phase contrast along column-coordinates,
                i.e. horizontal direction (x-direction)
            
    """
    s=storage
    
    # prepare buffers
    trans = np.zeros_like(s.data)
    dpc_row = np.zeros_like(s.data)
    dpc_col = np.zeros_like(s.data)
    nrm = np.zeros_like(s.data)+1e-10
    
    t2=0.
    # pick a single probe view for preparation purpose:
    v = s.views[0]
    pp = v.pods.values()[0].pr_view
    if probe is None:
        pr = np.abs(pp.data).sum(0)
    elif np.isscalar(probe):
        x,y = grids(pp.shape[-2:])
        pr = np.exp(-(x**2+y**2)/probe**2)
    else:
        pr = np.asarray(probe)
        assert pr.shape == pp.shape[-2:],'stxm probe has not the same shape as a view to this storage'
        
    for v in s.views:
        pod = v.pods.values()[0]
        if not pod.active: continue
        t = pod.diff.sum()
        if t > t2: t2=t
        ss = (v.layer,slice(v.roi[0,0],v.roi[1,0]),slice(v.roi[0,1],v.roi[1,1]))
        #bufview=buf[ss]
        m = mass_center(pod.diff) #+ 1.
        q = pod.di_view.storage._to_phys(m)
        dpc_row[ss]+=q[0]*v.psize[0]*pr *2*np.pi/pod.geometry.lz
        dpc_col[ss]+=q[1]*v.psize[1]*pr *2*np.pi/pod.geometry.lz
        trans[ss]+=np.sqrt(t)*pr
        nrm[ss]+=pr
        
    parallel.allreduce(trans)
    parallel.allreduce(dpc_row)
    parallel.allreduce(dpc_col)
    parallel.allreduce(nrm)
    dpc_row/=nrm
    dpc_col/=nrm
    trans/=nrm * np.sqrt(t2)
    
    return trans,dpc_row,dpc_col

def stxm_init(storage,probe=None):
    trans,dpc_row,dpc_col = stxm_analysis(storage,probe)
    s.data = trans*np.exp(-1j*phase_from_dpc(dpc_row,dpc_col))
    

def phase_from_dpc(dpc_row,dpc_col):
    """
    Implements fourier integration method for two diffential quantities.
    
    Assumes 2 arrays of N-dimensions who contain differential quantities 
    in X and Y, i.e. the LAST two dimensions (-2 & -1) in this case.
    
    Parameters:
    -----------
    
    dpc_row : Nd-array
            Differential information along 2nd last dimension
    dpc_col : Nd-array
            Differential information along last dimension
    
    """
    py=-dpc_row
    px=-dpc_col
    sh = px.shape
    sh = np.asarray(sh)
    fac = np.ones_like(sh)
    fac[-2:]=2
    f = np.zeros(sh*fac,dtype=np.complex) 
    c = px+1j*py
    f[...,:sh[-2],:sh[-1]]=c
    f[...,:sh[-2],sh[-1]:]=c[...,:,::-1]
    f[...,sh[-2]:,:sh[-1]]=c[...,::-1,:]
    f[...,sh[-2]:,sh[-1]:]=c[...,::-1,::-1]
    # fft conform grids in the boundaries of [-pi:pi]
    g = grids(f.shape,psize=np.pi/np.asarray(f.shape),center='fft')
    qx = g[-2]
    qy = g[-1]
    inv_qc = 1./(qx+1j*qy)
    inv_qc[...,0,0]=0
    nf=np.fft.ifft2(np.fft.fft2(f)*inv_qc)
    
    return np.real(nf[...,:sh[-2],:sh[-1]])

