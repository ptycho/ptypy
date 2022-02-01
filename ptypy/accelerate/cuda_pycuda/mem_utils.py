import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.tools import DeviceMemoryPool
from collections import deque

def make_pagelocked_paired_arrays(ar, flags=0):
    mem = cuda.pagelocked_empty(ar.shape, ar.dtype, order="C", mem_flags=flags)
    mem[:] = ar
    return gpuarray.to_gpu(mem), mem


class GpuData:
    """
    Manages one block of GPU data with corresponding CPU data.
    Keeps track of which cpu array is currently on GPU by its id,
    and transfers if it's not already there.

    To be used for the exit wave, ma, and mag arrays.
    Note: Allocator should be pooled for best performance
    """

    def __init__(self, nbytes, syncback=False):
        """
        New instance of GpuData. Allocates the GPU-side array.

        :param nbytes: Number of bytes held by this instance.
        :param syncback: Should the data be synced back to CPU any time it's swapped out
        """

        self.gpu = None
        self.gpuraw = cuda.mem_alloc(nbytes)
        self.nbytes = nbytes
        self.nbytes_buffer = nbytes
        self.gpuId = None
        self.cpu = None
        self.syncback = syncback
        self.ev_done = None

    def _allocator(self, nbytes):
        if nbytes > self.nbytes:
            raise Exception('requested more bytes than maximum given before: {} vs {}'.format(nbytes, self.nbytes))
        return self.gpuraw

    def record_done(self, stream):
        self.ev_done = cuda.Event()
        self.ev_done.record(stream)

    def to_gpu(self, cpu, id, stream):
        """
        Transfer cpu array to GPU on stream (async), keeping track of its id
        """
        if self.gpuId != id:
            if self.syncback:
                self.from_gpu(stream)
            self.gpuId = id
            self.cpu = cpu
            if self.ev_done is not None:
                self.ev_done.synchronize()
            self.gpu = gpuarray.to_gpu_async(cpu, allocator=self._allocator, stream=stream)
        return self.gpu

    def from_gpu(self, stream):
        """
        Transfer data back to CPU, into same data handle it was copied from
        before.
        """
        if self.cpu is not None and self.gpuId is not None and self.gpu is not None:
            if self.ev_done is not None:
                stream.wait_for_event(self.ev_done)
            self.gpu.get_async(stream, self.cpu)
            self.ev_done = cuda.Event()
            self.ev_done.record(stream)

    def resize(self, nbytes):
        """
        Resize the size of the underlying buffer, to allow re-use in different contexts.
        Note that memory will only be freed/reallocated if the new number of bytes are
        either larger than before, or if they are less than 90% of the original size -
        otherwise it reuses the existing buffer
        """
        if nbytes > self.nbytes_buffer or nbytes < self.nbytes_buffer * .9:
            self.nbytes_buffer = nbytes
            self.gpuraw.free()
            self.gpuraw = cuda.mem_alloc(nbytes)
        self.nbytes = nbytes
        self.reset()

    def reset(self):
        """
        Resets handles of cpu references and ids, so that all data will be transfered
        again even if IDs match.
        """
        self.gpuId = None
        self.cpu = None
        self.ev_done = None

    def free(self):
        """
        Free the underlying buffer on GPU - this object should not be used afterwards
        """
        self.gpuraw.free()
        self.gpuraw = None


class GpuData2(GpuData):
    """
    Manages one block of GPU data with corresponding CPU data.
    Keeps track of which cpu array is currently on GPU by its id,
    and transfers if it's not already there.

    To be used for the exit wave, ma, and mag arrays.
    Note: Allocator should be pooled for best performance
    """

    def __init__(self, nbytes, syncback=False):
        """
        New instance of GpuData. Allocates the GPU-side array.

        :param nbytes: Number of bytes held by this instance.
        :param syncback: Should the data be synced back to CPU any time it's swapped out
        """
        self.done_what = None
        super().__init__(nbytes, syncback)

    def record_done(self, stream, what):
        assert what in ['dtoh','htod','compute']
        self.ev_done = cuda.Event()
        self.ev_done.record(stream)
        self.done_what = what

    def to_gpu(self, cpu, ident, stream):
        """
        Transfer cpu array to GPU on stream (async), keeping track of its id
        """
        ident = id(cpu) if ident is None else ident
        if self.gpuId != ident:
            if self.ev_done is not None:
                stream.wait_for_event(self.ev_done)
            # Safety measure. This is asynchronous, but it should still work
            # Essentially we want to copy the data held in gpu array back to its CPU
            # handle before the buffer can be reused.
            if self.done_what != 'dtoh' and self.syncback:
                # uploads on the download stream, easy to spot in nsight-sys
                self.from_gpu(stream)
            self.gpuId = ident
            self.cpu = cpu
            self.gpu = gpuarray.to_gpu_async(cpu, allocator=self._allocator, stream=stream)
            self.record_done(stream, 'htod')
        return self.ev_done, self.gpu

    def from_gpu(self, stream):
        """
        Transfer data back to CPU, into same data handle it was copied from
        before.
        """
        if self.cpu is not None and self.gpuId is not None and self.gpu is not None:
            # Wait for any action recorded with this array
            if self.ev_done is not None:
                stream.wait_for_event(self.ev_done)
            self.gpu.get_async(stream, self.cpu)
            self.record_done(stream, 'dtoh')
            # Mark for reuse
            self.gpuId = None
            return self.ev_done
        else:
            return None

class GpuDataManager:
    """
    Manages a set of GpuData instances, to keep several blocks on device.

    Currently all blocks must be the same size.

    Note that the syncback property is used so that during fourier updates,
    the exit wave array is synced bck to cpu (it is updated),
    while during probe update, it's not.
    """

    def __init__(self, nbytes, num, max=None, syncback=False):
        """
        Create an instance of GpuDataManager.
        Parameters are the same as for GpuData, and num is the number of
        GpuData instances to create (blocks on device).
        """
        self._syncback = syncback
        self._nbytes = nbytes
        self.data = []
        self.max = max
        for i in range(num):
            self.add_data_block()

    def add_data_block(self, nbytes=None):
        """
        Add a GpuData block.

        Parameters
        ----------
        nbytes - Size of block

        Returns
        -------
        """
        if self.max is None or len(self)<self.max:
            nbytes=nbytes if nbytes is not None else self._nbytes
            self.data.append(GpuData2(nbytes, self._syncback))

    @property
    def syncback(self):
        """
        Get if syncback of data to CPU on swapout is enabled.
        """
        return self._syncback

    @syncback.setter
    def syncback(self, whether):
        """
        Adjust the syncback setting
        """
        self._syncback = whether
        for d in self.data:
            d.syncback = whether

    @property
    def nbytes(self):
        """
        Get the number of bytes in each block
        """
        return self.data[0].nbytes

    @property
    def memory(self):
        """
        Get all memory occupied by all blocks
        """
        m = 0
        for d in self.data:
            m += d.nbytes_buffer
        return m

    def __len__(self):
        return len(self.data)

    def reset(self, nbytes, num):
        """
        Reset this object as if these parameters were given to the constructor.
        The syncback property is untouched.
        """
        sync = self.syncback
        # remove if too many, explictly freeing memory
        for i in range(num, len(self.data)):
            self.data[i].free()
        # cut short if too many
        self.data = self.data[:num]
        # reset existing
        for d in self.data:
            d.resize(nbytes)
        # append new ones
        for i in range(len(self.data), num):
            self.data.append(GpuData2(nbytes, sync))

    def free(self):
        """
        Explicitly clear all data blocks - same as resetting to 0 blocks
        """
        self.reset(0, 0)


    def to_gpu(self, cpu, id, stream, pop_id="none"):
        """
        Transfer a block to the GPU, given its ID and CPU data array
        """
        idx = 0
        for x in self.data:
            if x.gpuId == id or x.gpuId == pop_id:
                break
            idx += 1
        if idx == len(self.data):
            idx = 0
        else:
            pass
        m = self.data.pop(idx)
        self.data.append(m)
        #print("Swap %s for %s and move from %d to %d" % (m.gpuId,id,idx,len(self.data)))
        ev, gpu = m.to_gpu(cpu, id, stream)
        # return the wait event, the gpu array and the function to register a finished computation
        return ev, gpu, m

    def sync_to_cpu(self, stream):
        """
        Sync back all data to CPU
        """
        for x in self.data:
            x.from_gpu(stream)

## looks useful, but probably unused

class ManagedPool:

    def __init__(self, nbytes=None):

        self.dmp = DeviceMemoryPool()
        self.nbytes_allocated = 0
        self.nbytes = nbytes if nbytes is not None else cuda.mem_get_info()[0]
        # this one keeps the refs alive
        self.dev_data = {}
        self.upstream = None
        self.downstream = None
        self.ev_computed = {}
        self.set_io_streams()

    def set_io_streams(self, downstream=None, upstream=None):
        self.upstream = cuda.Stream() if upstream is not None else upstream
        self.downstream = cuda.Stream() if downstream is not None else downstream

    def computed(self, ary, ev):
        self.ev_computed[id(ary)]=ev

    def _allocator(self, nbytes):
        # this one gets called if
        return self.dmp.allocate(nbytes)

    def get_array(self, ary, stream=None):
        pass

    def set_array(self, ary, synchback=None, stream=None):
        """
        Schedule an (asynchronous) array transfer to gpu or return array if the data is already there.
        """
        if stream is None:
            stream = self.downstream
        n = id(ary)
        if synchback is not None:
            # get the last event
            if n in self.ev_computed:
                self.upstream.wait_for_event(self.ev_computed[n])
            self.dev_data[n].get_async(self.upstream, synchback)
            ev = cuda.Event()
            ev.record(self.upstream)
        gpu = gpuarray.to_gpu_async(ary, allocator=self._allocater, stream=stream)
        self.dev_data[id] = gpu


## unused

class MemoryManager:

    def __init__(self, fraction=0.7):
        self.fraction = fraction
        self.dmp = DeviceMemoryPool()
        self.queue_in = cuda.Stream()
        self.queue_out = cuda.Stream()
        self.mem_avail = None
        self.mem_total = None
        self.get_free_memory()
        self.on_device = {}
        self.on_device_inv = {}
        self.out_events = deque()
        self.bytes = 0

    def get_free_memory(self):
        self.mem_avail, self.mem_total = cuda.mem_get_info()

    def device_is_full(self, nbytes = 0):
        return (nbytes + self.bytes) > self.mem_avail

    def to_gpu(self, ar, ev=None):
        """
        Issues asynchronous copy to device. Waits for optional event ev
        Emits event for other streams to synchronize with
        """
        stream = self.queue_in
        id_cpu = id(ar)
        gpu_ar = self.on_device.get(id_cpu)

        if gpu_ar is None:
            if ev is not None:
                stream.wait_for_event(ev)
            if self.device_is_full(ar.nbytes):
                self.wait_for_freeing_events(ar.nbytes)

            # TOD0: try /except with garbage collection to make sure there is space
            gpu_ar = gpuarray.to_gpu_async(ar, allocator=self.dmp.allocate, stream=stream)

            # keeps gpuarray alive
            self.on_device[id_cpu] = gpu_ar

            # for deleting later
            self.on_device_inv[id(gpu_ar)] = ar

            self.bytes += gpu_ar.mem_size * gpu_ar.dtype.itemsize


        ev = cuda.Event()
        ev.record(stream)
        return ev, gpu_ar


    def wait_for_freeing_events(self, nbytes):
        """
        Wait until at least nbytes have been copied back to the host. Or marked for deletion
        """
        freed = 0
        if not self.out_events:
            #print('Waiting for memory to be released on device failed as no release event was scheduled')
            self.queue_out.synchronize()
        while self.out_events and freed < nbytes:
            ev, id_cpu, id_gpu = self.out_events.popleft()
            gpu_ar = self.on_device.pop(id_cpu)
            cpu_ar = self.on_device_inv.pop(id_gpu)
            ev.synchronize()
            freed += cpu_ar.nbytes
            self.bytes -= gpu_ar.mem_size * gpu_ar.dtype.itemsize

    def mark_release_from_gpu(self, gpu_ar, to_cpu=False, ev=None):
        stream = self.queue_out
        if ev is not None:
            stream.wait_for_event(ev)
        if to_cpu:
            cpu_ar = self.on_device_inv[id(gpu_ar)]
            gpu_ar.get_asynch(stream, cpu_ar)

        ev_out = cuda.Event()
        ev_out.record(stream)
        self.out_events.append((ev_out, id(cpu_ar), id(gpu_ar)))
        return ev_out