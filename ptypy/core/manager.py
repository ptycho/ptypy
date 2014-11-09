# -*- coding: utf-8 -*-
"""
Scan management

This main task of this module is to prepare the data structure for 
reconstruction, taking a data feed and connecting individual diffraction
measurements to the other containers. The way this connection is done
is defined by the user through a model definition. The connections are
described by the POD objects. This module also takes care of initializing
containers according to user-defined rules.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import weakref

from .. import utils as u
from ..utils.verbose import logger
from classes import *
from classes import DEFAULT_ACCESSRULE
from classes import MODEL_PREFIX

from ..utils import parallel
import geometry
from ..utils import prop
import illumination
import sample
import geometry
import model
import xy
import data

# please set these globally later
FType = np.float64
CType = np.complex128

__all__ = ['ModelManager']

DESCRIPTION = u.Param()

coherence_DEFAULT = u.Param(
        Nprobe_modes = 1,   # number of mutually spatially incoherent probes per diffraction pattern
        Nobject_modes = 1,  # number of mutually spatially incoherent objects per diffraction pattern
        energies = [1.0]    # list of energies / wavelength relative to mean energy / wavelength
    )

scan_DEFAULT = u.Param(
        illumination = {},      # All information about the probe
        sample = {},            # All information about the object
        geometry = {},          # Geometry of experiment - most of it provided by data
        xy = {},                # Information on scanning paramaters to yield position arrays - If positions are provided by the DataScan object, set xy.scan_type to None
        coherence = coherence_DEFAULT,
        if_conflict_use_meta = False, # take geometric and position information from incoming meta_data if possible parameters are specified both in script and in meta data
        source = None,
        sharing = None,
        tags = "", # For now only used to declare an empty scan
    )

model_DEFAULT = u.Param(scan_DEFAULT,
        sharing = {},              # rules for linking
        scans = {},             # Sub-structure that can contain scan-specific parameters
    )

class ModelManager(object):
    """
    ModelManager : Manage object creation and update according to the physical
    model of the ptychography experiment
    """
    DEFAULT = model_DEFAULT
    _PREFIX = MODEL_PREFIX
    
    def __init__(self, ptycho, pars=None, scans=None, **kwargs):
        """
        ModelManager : Manage object creation and update.
        
        The main task of ModelManager is to follow the rules for a given
        reconstruction model and create:
         1. the probe, object, exit, diff and mask containers
         2. the views
         3. the PODs 
         
        A ptychographic problem is defined by the combination of one or
        multiple scans. ModelManager uses encapsulate
        scan-specific elements in .scans und .scans_pars
        
        Parameters
        ----------
        ptycho: Ptycho
                The parent Ptycho object
        pars : dict or Param
               Input parameters (see ModelManager.DEFAULT)
               If None uses defaults
        scans : dict or Param
                scan-specific parameters, Values should be dict Param that
                follow the structure of `pars` (see ModelManager.DEFAULT)
                If None, tries in ptycho.p.scans else becomes empty dict  
        """

        # Initialize the input parameters
        p = u.Param(self.DEFAULT)
        if pars is not None: p.update(pars)
        self.p = p

        self.ptycho = ptycho
        # abort if ptycho is None:
        if self.ptycho is None:
            return
        # Prepare the list of scan_labels (important because they are sorted)
        # BE I don't think this is the way to go. This is only needed for sharing 
        # For the user it might be better to mark the sharing behavior directly
        self.scan_labels = []
        
        # store scan specifics
        self.scans_pars = scans if scans is not None else self.ptycho.p.get('scans',u.Param())
        # Scan dictionary 
        # This will store everything scan specific and will hold
        # references to the storages of the respective scan
        self.scans = u.Param()
        
        # update self.scans from information already available
        for label in self.scans_pars.keys():
            self.prepare_scan(label)
            
        # sharing dictionary that stores sharing behavior
        self.sharing = {'probe_ids':{},'object_ids':{}}
        # Initialize sharing rules for POD creations
        self.sharing_rules = model.parse_model(p.sharing,self.sharing)
        
        self.label_idx = len(self.scans) # this start is a little arbitrary
        
    def _to_dict(self):
        # delete the model class. We do not really need to store it
        del self.sharing_rules
        return self.__dict__.copy()
        
    @classmethod
    def _from_dict(cls,dct):
        # create instance
        inst = cls(None,None)
        # overwrite internal dictionary
        inst.__dict__= dct
        return inst
        
    def prepare_scan(self,label=None):
        """
        Prepare scan specific parameters and create a label if necessary
        """
        if label is None:
            label = 'Scan%05d' % self.label_idx
            self.label_idx +=1
            
        try:
            # return the scan if already prepared
            return self.scans[label]
              
        except KeyError:
            # get standarad parameters
            # Create a dictionary specific for the scan.
            scan = u.Param() 
            self.scans[label] = scan
            scan.label = label
            # make a copy of model ditionary
            scan.pars = self.p.copy(depth=3)  
            # Look for a scan-specific entry in the input parameters
            scan_specific_parameters = self.scans_pars.get(label, None)
            scan.pars.update(scan_specific_parameters,Replace=False)
            
            # prepare the tags
            t = scan.pars.tags
            if str(t)==t:
                scan.pars.tags = [tag.strip().lower() for tag in t.split(',')] 
            # also create positions
            scan.pos_theory = xy.from_pars(scan.pars.xy)
            return scan
            
    def _update_stats(self, scan,mask_views=None,diff_views=None):
        """
        (Re)compute the statistics for the data stored in a scan.
        These statistics are:
         * Itotal: The integrated power per frame
         * DPCX, DPCY: The center of mass in X and Y
         * max/min/mean_frame: pixel-by-pixel maximum, minimum and 
           average among all frames.
        """
        if mask_views is None:
            mask_views = scan.mask_views
        if diff_views is None:
            diff_views = scan.diff_views
        # Reinitialize containers
        Itotal = []
        #DPCX = []
        #DPCY = []
        max_frame = np.zeros(scan.diff_views[0].shape)
        min_frame = np.zeros_like(max_frame)
        mean_frame = np.zeros_like(max_frame)
        norm = np.zeros_like(max_frame)
        # Useful quantities
        #sh0,sh1 = scan.geo.N
        #x = np.arange(s1, dtype=float)
        #y = np.arange(s0, dtype=float)
        
        for maview,diview in zip(mask_views,diff_views):
            if not diview.active:
                continue
            dv = diview.data
            m = maview.data
            v = m*dv
            #mv = dv.pod.ma_view.data # pods may not yet exist, since mask & data are not linked yet
            #S0 = np.sum(mv*dv, axis=0)
            #S1 = np.sum(mv*dv, axis=1)
            #I0 = np.sum(S0)
            Itotal.append(np.sum(v))
            #DPCX.append(np.sum(S0*x)/I0 - sh1/2.)
            #DPCY.append(np.sum(S1*y)/I0 - sh0/2.)
            max_frame[max_frame < v] = v[max_frame < v]
            min_frame[min_frame > v] = v[min_frame > v]
            mean_frame += v
            norm += m
        
        parallel.allreduce(mean_frame)
        parallel.allreduce(norm)
        parallel.allreduce(max_frame,parallel.MPI.MAX)
        parallel.allreduce(max_frame,parallel.MPI.MIN)
        mean_frame /= (norm + (norm == 0))
        
        scan.diff.max_power = parallel.MPImax(Itotal)
        scan.diff.tot_power = parallel.MPIsum(Itotal)
        scan.diff.mean = mean_frame
        scan.diff.max = max_frame
        scan.diff.min = min_frame
        
        info={'label':scan.label,'max':scan.diff.max_power,'tot':scan.diff.tot_power,'mean':mean_frame.sum()}
        logger.info('--- Scan %(label)s photon report ---\nTotal photons   : %(tot).2e \nAverage photons : %(mean).2e\nMaximum photons : %(max).2e\n' %info +'-'*34)

    def make_datasource(self,data_pars=None):
        """
        This function will make a static datasource from
        parameters in the self.scans dict.
        For any additional file in data.filelist it will create a new entry in 
        self.scans with generic parameters given by the current model.
        """
        if data_pars is not None:
            filelist = data_pars.get('filelist')
        if filelist is not None:
            for f in filelist:
                scan = self.prepare_scan()
                scan.source = f
        # now there should be little suprises. every scan is listed in 
        # self.scans
        labels=[]
        sources =[]
        pars =[]
        for label,scan in self.scans.items():
            if scan.pars.source is None: 
                sources.append(self.ptycho.paths.get_data_file(label=label)) 
            else:
                sources.append(scan.pars.source)
            labels.append(label)
            pars.append(None)
    
        return data.StaticDataSource(sources,pars,labels)
        
    def new_data(self):
        """
        Get all new diffraction patterns and create all views and pods
        accordingly.
        """
        parallel.barrier()
        # Nothing to do if there are no new data.
        if not self.ptycho.datasource.data_available: return 'ok'
        
        logger.info('Processing new data.')
        used_scans = []
        not_initialized = []
        
        for dp in self.ptycho.datasource.feed_data(): 
            """
            A dp (data package) contains the following:
            
            common : dict or Param
                    Meta information common to all datapoints in the 
                    data package. Variable names need to be consistent with those 
                    in the rest of ptypy package.
                    (TODO further description)
                    
                    Important info:
                    ------------------------
                    shape : (tuple or array) 
                           expected frame shape
                    label : (string)
                            a string signifying to which scan this package belongs
                   
            
            iterable : An iterable structure that yields for each iteration
                       a dict with the following fields:
            
                        data     : (np.2darray, float) 
                                    diffraction data 
                                    In MPI case, data can be None if distributed to other nodes
                        mask     : (np.2darray, bool) 
                                    masked out areas in diffraction data array
                        index    : (int)
                                    diffraction datapoint index in scan
                        position : (tuple or array)
                                    scan position 
            """
            meta = dp['common']
            label = meta['label']
            
            # we expect a string for the label.
            assert label == str(label)

            used_scans.append(label)
            logger.info('Importing data from %s as scan %s.' % (meta['label_original'],label))
            
            # prepare scan dictionary or dig up the already prepared one
            scan = self.prepare_scan(label)
            scan.meta = meta
            
            # empty buffer
            scan.iterable = []
            
            # prepare the scan geometry if not already done.
            # FIXME: It should be allowed to let "N" be figured out from the data
            # but for now it needs to be set in the initial parameters.
            if scan.get('geometries') is None:              
                # ok now that we have meta we can check if the geometry fits
                scan.geometries=[]
                geo = scan.pars.geometry
                for key in ['lam','energy','psize_det','z','prop_type']:
                    if geo[key] is None or scan.pars.if_conflict_use_meta:
                        mk = scan.meta.get(key)
                        if mk is not None: # None existing key or None values in meta dict are treated alike
                            geo[key] = mk
                                 

                for ii,fac in enumerate(self.p.coherence.energies):
                    geoID = geometry.Geo._PREFIX + '%02d' %ii + label
                    g = geometry.Geo(self.ptycho,geoID,pars=geo)
                    # now we fix the sample pixel size, This will make the frame size adapt
                    g.p.psize_sam_is_fix = True
                    g.energy *= fac
                    scan.geometries.append(g)
                    
                # create a buffer
                scan.iterable = []
                
                scan.diff_views = []
                scan.mask_views = []
                
                # Remember the order in which these scans were fed the manager
                self.scan_labels.append(label)
                
                # Remember also that these new scans are probably not initialized yet
                not_initialized.append(label)
                               
            # Buffer incoming data and evaluate if we got Nones in data
            for dct in dp['iterable']:
                dct['active'] = dct['data'] is not None
                scan.iterable.append(dct)

        # Ok data transmission is over for now.
        # Lets see what scans have received data and create the views for those
        for label in used_scans:
            
            # get scan Param
            scan = self.scans[label]

            # pick one of the geometries for calculating the frame shape
            geo = scan.geometries[0]
            sh = np.array(scan.meta.get('shape',geo.N))
            
            # Storage generation if not already existing
            if scan.get('diff') is None:
                # this scan is brand new so we create storages for it
                scan.diff = self.ptycho.diff.new_storage(shape=(1,sh[-2],sh[-1]), psize=geo.psize_det, padonly=True, layermap=None)
                old_diff_views = []
                old_diff_layers= []
            else:
                # ok storage exists already. Views most likely also. Let's do some analysis and deactivate the old views
                old_diff_views = self.ptycho.diff.views_in_storage(scan.diff,active=False)
                old_diff_layers =[]
                for v in old_diff_views:
                    old_diff_layers.append(v.layer) 
                    #v.active = False
                    
            # same for mask
            if scan.get('mask') is None:
                scan.mask = self.ptycho.mask.new_storage(shape=(1,sh[-2],sh[-1]), psize=geo.psize_det, padonly=True, layermap=None)
                old_mask_views = []
                old_mask_layers= []
            else:
                old_mask_views = self.ptycho.mask.views_in_storage(scan.mask,active=False)
                old_mask_layers =[]
                for v in old_mask_views:
                    old_mask_layers.append(v.layer) 
                    #v.active = False

            # Prepare for View genereation
            AR_diff_base = DEFAULT_ACCESSRULE.copy()
            AR_diff_base.shape = geo.N #None
            AR_diff_base.coord = 0.0
            AR_diff_base.psize = geo.psize_det
            AR_mask_base = AR_diff_base.copy()
            AR_diff_base.storageID = scan.diff.ID
            AR_mask_base.storageID = scan.mask.ID
            
            diff_views = []
            mask_views = []
            positions = []
            positions_theory = xy.from_pars(scan.pars.xy)
            
            for dct in scan.iterable:
                index = dct['index']
                active = dct['active']
                #tpos = positions_theory[index]
                if not scan.pars.if_conflict_use_meta and positions_theory is not None:
                    pos = positions_theory[index]
                else:
                    pos = dct.get('position') #,positions_theory[index])
                
                AR_diff = AR_diff_base #.copy()
                AR_mask = AR_mask_base #.copy()
                AR_diff.layer = index
                AR_mask.layer = index
                
                # check here: is there already a view to this layer? Is it active?
                try: 
                    old_view = old_diff_views[old_diff_layers.index(index)]
                    old_active = old_view.active
                    old_view.active = active
                    # also set this for the attached pods' exit views
                    for pod in old_view.pods.itervalues():
                        pod.ex_view.active = active
                        
                    logger.debug('Diff view with layer/index %s of scan %s exists. \nSetting view active state from %s to %s' % (index,label,old_active,active)  )
                except ValueError:
                    v =  View(self.ptycho.diff, accessrule=AR_diff, active=active)
                    diff_views.append(v)
                    logger.debug('Diff view with layer/index %s of scan %s does not exist. \nCreating view with ID %s and set active state to %s' % (index,label,v.ID,active))
                    # append position also
                    positions.append(pos)
                
                try: 
                    old_view = old_mask_views[old_mask_layers.index(index)]
                    old_view.active = active
                except ValueError: 
                    v = View(self.ptycho.mask, accessrule=AR_mask, active=active)  
                    mask_views.append(v)
            # so now we should have the right views to this storages. Let them reformat()
            # that will create the right sizes and the datalist access
            scan.diff.reformat()
            scan.mask.reformat()
            #parallel.barrier()
            #print 'this'
            
            for dct in scan.iterable:
                parallel.barrier()
                if not dct['active']:
                    continue
                data = dct['data']
                idx = dct['index']
                scan.diff.datalist[idx][:]=data #.astype(scan.diff.dtype)
                scan.mask.datalist[idx][:]=dct.get('mask',np.ones_like(data)) #.astype(scan.mask.dtype)
                #del dct['data']

            #print 'hallo'
            scan.diff.nlayers = parallel.MPImax(scan.diff.layermap)+1
            scan.mask.nlayers = parallel.MPImax(scan.mask.layermap)+1
            # empty iterable buffer
            #scan.iterable = []
            scan.new_positions = positions
            scan.new_diff_views = diff_views
            scan.new_mask_views = mask_views
            scan.diff_views += diff_views
            scan.mask_views += mask_views
        
            self._update_stats(scan)
        # Create PODs
        new_pods, new_probe_ids, new_object_ids = self._create_pods(used_scans)
        logger.info('Process %d created %d new PODs, %d new probes and %d new objects.' % (
                            parallel.rank,len(new_pods), len(new_probe_ids), len(new_object_ids)),extra={'allprocesses':True})
        
        # Adjust storages      
        self.ptycho.probe.reformat(True)
        self.ptycho.obj.reformat(True)
        self.ptycho.exit.reformat()
        
        self._initialize_probe(new_probe_ids)
        self._initialize_object(new_object_ids)
        self._initialize_exit(new_pods)
        
    def _initialize_probe(self,probe_ids):
        """
        initializes the probe storages referred to by the probe_ids
        """
        for pid,labels in probe_ids.items():
            
            # pick scanmanagers from scan_label for illumination parameters
            # for now, the scanmanager of the first label is chosen
            scan = self.scans[labels[0]]
            illu_pars = scan.pars.illumination
            
            # pick storage from container
            s = self.ptycho.probe.S.get(pid)
            if s is None:
                continue
            else: 
                logger.info('Initializing probe storage %s using scan %s' %(pid,scan.label))
            
            # find out energy or wavelength. Maybe store that information 
            # in the storages in future too avoid this weird call.
            lam = s.views[0].pod.geometry.lam
            
            # make this a single call in future
            pr = illumination.from_pars(s.shape[-2:],s.psize,lam,illu_pars)
            pr = illumination.create_modes(s.shape[-3],pr)
            s.fill(pr.probe)
            s.reformat() # maybe not needed
            s.model_initialized = True
            
    def _initialize_object(self,object_ids):
        """
        initializes the probe storages referred to by the object_ids
        """
        for oid,labels in object_ids.items():
            
            # pick scanmanagers from scan_label for illumination parameters
            # for now, the scanmanager of the first label is chosen
            scan = self.scans[labels[0]]
            sample_pars = scan.pars.sample
            
            # pick storage from container
            s = self.ptycho.obj.S.get(oid)
            if s is None or s.model_initialized:
                continue
            else: 
                logger.info('Initializing object storage %s using scan %s' %(oid,scan.label))
                
            if sample_pars.source == 'diffraction':
                logger.info('STXM initialization using diffraction data')
                trans,dpc_row,dpc_col = u.stxm_analysis(s)
                s.fill(trans*np.exp(1j*u.phase_from_dpc(dpc_row,dpc_col)))
            else:
                # find out energy or wavelength. Maybe store that information in the storages too in future
                lam = s.views[0].pod.geometry.lam
                
                # make this a single call in future
                obj = sample.from_pars(s.shape[-2:],lam,sample_pars)
                obj = sample.create_modes(s.shape[-3],obj)
                s.fill(obj.obj)
            
            s.reformat() # maybe not needed
            s.model_initialized = True
            
    def _initialize_exit(self,pods):
        """
        initializes exit waves ousing the pods
        """
        for pod in pods:
            if not pod.active:
                continue 
            pod.exit = pod.probe * pod.object
            
    def _create_pods(self, new_scans):
        """
        Create all pods associated with the scan labels in 'scans'.
        
        Return the list of new pods, probe and object ids (to allow for
        initialization).
        """
        
        new_pods = []
        new_probe_ids = {}
        new_object_ids = {}

        # Get a list of probe and object that already exist
        existing_probes = self.ptycho.probe.S.keys()#self.sharing_rules.probe_ids.keys()
        existing_objects = self.ptycho.obj.S.keys()#self.sharing_rules.object_ids.keys()
        logger.info('Found these probes:\n' + '\n'.join(existing_probes)) 
        logger.info('Found these objects:\n' + '\n'.join(existing_objects))
        #exit_index = 0
        # Loop through scans
        for label in new_scans:
            scan = self.scans[label]
            
            positions = scan.new_positions
            di_views = scan.new_diff_views
            ma_views = scan.new_mask_views
            # Loop through diffraction patterns             
            for i in range(len(di_views)):
                dv,mv = di_views.pop(0), ma_views.pop(0)
                #print dv
                index = dv.layer
                
                # object and probe position
                pos_pr = u.expect2(0.0)
                pos_obj = positions[i] if 'empty' not in scan.pars.tags else 0.0
                
                # Compute sharing rules
                alt_obj = scan.pars.sharing.object_shared_with
                alt_pr = scan.pars.sharing.probe_shared_with
                obj_label = label if alt_obj is None else alt_obj
                pr_label = label if alt_pr is None else alt_pr
                
                t, object_id = self.sharing_rules(obj_label, index)
                probe_id, t = self.sharing_rules(pr_label, index)

                # For multiwavelength reconstructions: loop here over
                # geometries, and modify probe_id and object_id.
                for ii,geometry in enumerate(scan.geometries):
                    suffix='G%02d' % ii
                    
                    # make new IDs and keep them in record
                    # sharing_rules is not aware of IDs with suffix
                    probe_id_suf = probe_id + suffix
                    if probe_id_suf not in new_probe_ids.keys() and probe_id_suf not in existing_probes:
                        new_probe_ids[probe_id_suf]= self.sharing_rules.probe_ids[probe_id]
                        
                    object_id_suf = object_id +suffix
                    if object_id_suf not in new_object_ids.keys() and object_id_suf not in existing_objects:
                        new_object_ids[object_id_suf]=self.sharing_rules.object_ids[object_id]
                        
                    # Loop through modes
                    for pm in range(scan.pars.coherence.Nprobe_modes):
                        for om in range(scan.pars.coherence.Nobject_modes):
                            # Make a unique layer index for exit view
                            # The actual number does not matter due to the layermap access
                            exit_index = index * 10000 + pm * 100 + om
                            # Create views
                            # please Note that mostly references are passed,
                            # i.e. the views do mostly not own the accessrule contents
                            pv = View(container=self.ptycho.probe,
                                      accessrule={'shape':geometry.N,
                                                    'psize':geometry.psize_sam,
                                                    'coord':pos_pr,
                                                    'storageID': probe_id_suf,
                                                    'layer': pm},
                                       active=True)
                            ov = View(container=self.ptycho.obj,
                                      accessrule={'shape':geometry.N,
                                                    'psize':geometry.psize_sam,
                                                    'coord':pos_obj,
                                                    'storageID': object_id_suf,
                                                    'layer': om},
                                       active=True)
                            """
                            ev = View(container=self.ptycho.exit,
                                      accessrule={'shape':geometry.N,
                                                    'psize':geometry.psize_sam,
                                                    'coord':pos_pr,
                                                    'storageID': probe_id+object_id_suf, #BE :combined ID -- needed for sharing
                                                    'layer': exit_index},
                                       active=dv.active) #BE: exit waves take a lot of space -> inactivate now if there is no active diff view. 
                            """
                            views={'probe':pv,'obj':ov,'diff':dv,'mask':mv}
                            pod = POD(ptycho=self.ptycho, ID=None, views=views, geometry=geometry, meta=None)
                            new_pods.append(pod)
                            pod.probe_weight = scan.pars.sharing.probe_share_power
                            pod.object_weight = scan.pars.sharing.object_share_power
                            pod.is_empty = True if 'empty' in scan.pars.tags else False
                            exit_index += 1

            # delete buffer & meta (meta may be filled with a lot of stuff)
            scan.iterable =[]
            #scan.meta={}
            
        return new_pods, new_probe_ids, new_object_ids
        

    def collect_diff_mask_meta(self,label=None,filename =None,save=False,dtype=None,**kwargs):
        """
        attempt to save diffraction data
        
        PARAMETERS
        -----------
        
        label : str
                ptypy label of the scan to save 
                if None, tries to save ALL diffraction data 
                
        filename : str
                override the file path to write to
                will change `data_filename` in `scan_info` dict
        
        all other kwargs are added to 'scan_info' key in the '.h5' file
        """
        
        if label is None:
            scans={}
            for l in self.scans.keys():
                scans[l]=self.collect_diff_mask_meta(l,filename,save,dtype,**kwargs)
            return scans
        else:
            dct ={}
            # get the scan
            scan = self.scans[label]
            for kind in ['mask','diff']:
                storage=scan[kind]
                # fresh copy
                new = [data.copy() if data is not None else None for data in storage.datalist]
                Nframes = len(new)
                if parallel.MPIenabled:
                    logger.info('Using MPI to gather arrays for storing %s' % kind)
                    for i in range(Nframes):
                        if parallel.master:
        
                            # Root receives the data if it doesn't have it yet
                            if new[i] is None:
                                new[i] = parallel.receive()
                                logger.info('%s :: Frame %d/%d received at process %d' %(kind.upper(),i,Nframes,parallel.rank) , extra={'allprocesses':True})

                            parallel.barrier()
        
                        else:
                            if new[i] is not None:
                                # Send data to root.
                                parallel.send(new[i])
                                #logger.info('Process %d - Send frame %d of %s' % (parallel.rank,i,kind), extra={'allprocesses':True})
                                sender=parallel.rank
                                logger.info('%s :: Frame %d/%d send from process %d' %(kind.upper(),i,Nframes,parallel.rank) , extra={'allprocesses':True})

                            parallel.barrier()

                    parallel.barrier()
                    
                # storing as arrays
                if parallel.master:
                    key = 'data' if kind=='diff' else kind
                    dct[key] = np.asarray(new)
                    
            # save if you are master
            if parallel.master: 
               
                # get meta data
                meta = self.scans[label]['meta']
                # update with geometric info
                meta.update(scan.pars.geometry.copy())
                
                # translate to scan_info and ditch variables not in data.DEFAULT_scan_info
                from data import MT as LeTraducteur
                dct['scan_info'] = LeTraducteur.as_scan_info(self.scans[label]['meta'])
                
                # overwrite filename
                if filename is not None:
                    dct['scan_info']['data_filename']= filename
                    
                filename = dct['scan_info'].get('data_filename')
                
                # add other kwargs to scan_info
                dct['scan_info'].update(kwargs)
                dct['scan_info']['shape']=dct['data'].shape
                
                # switch data type for data if wanted (saves space)
                if dtype is not None:
                    dct['data']=dct['data'].astype(dtype)
                    
                if save:
                # cropping
                    from .. import io
                    filename = u.clean_path(filename)
                    logger.info('Saving to ' +filename)
                    io.h5write(filename,dct)
                    logger.info('Saved')
                    return filename
            
            return dct

