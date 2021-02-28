import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.tools import DeviceMemoryPool

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


class GpuDataManager:
    """
    Manages a set of GpuData instances, to keep several blocks on device.

    Note that the syncback property is used so that during fourier updates,
    the exit wave array is synced bck to cpu (it is updated),
    while during probe update, it's not.
    """

    def __init__(self, nbytes, num, syncback=False):
        """
        Create an instance of GpuDataManager.
        Parameters are the same as for GpuData, and num is the number of
        GpuData instances to create (blocks on device).
        """
        self.data = [GpuData(nbytes, syncback) for _ in range(num)]

    @property
    def syncback(self):
        """
        Get if syncback of data to CPU on swapout is enabled.
        """
        return self.data[0].syncback

    @syncback.setter
    def syncback(self, whether):
        """
        Adjust the syncback setting
        """
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
            self.data.append(GpuData(nbytes, sync))

    def free(self):
        """
        Explicitly clear all data blocks - same as resetting to 0 blocks
        """
        self.reset(0, 0)


    def to_gpu(self, cpu, id, stream):
        """
        Transfer a block to the GPU, given its ID and CPU data array
        """
        idx = 0
        for x in self.data:
            if x.gpuId == id:
                break
            idx += 1
        if idx == len(self.data):
            idx = 0
        else:
            pass
        m = self.data.pop(idx)
        self.data.append(m)
        return m.to_gpu(cpu, id, stream)

    def record_done(self, id, stream):
        for x in self.data:
            if x.gpuId == id:
                x.record_done(stream)
                return
        raise Exception('recording done for id not in pool')


    def sync_to_cpu(self, stream):
        """
        Sync back all data to CPU
        """
        for x in self.data:
            x.from_gpu(stream)


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
        super().__init__(nbytes, syncback)

    def to_gpu(self, cpu, stream, upstream=None, ident=None):
        """
        Transfer cpu array to GPU on stream (async), keeping track of its id
        """
        ident = id(cpu) if ident is None else ident
        upstream = stream if upstream is None else upstream
        if self.gpuId != ident:
            # wait for any action recorded with that array
            if self.ev_done is None:
                stream.wait_for_event(self.ev_done)
                upstream.wait_for_event(self.ev_done)
            if self.syncback:
                ev = self.from_gpu(upstream)
                if ev is not None:
                    stream.wait_for_event(ev)
            self.gpuId = ident
            self.cpu = cpu
            self.gpu = gpuarray.to_gpu_async(cpu, allocator=self._allocator, stream=stream)
            self.record_done(stream)
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
            ev = cuda.Event()
            ev.record(stream)
            return ev
        else:
            return None

class GpuDataManager2:
    """
    Manages a set of GpuData instances, to keep several blocks on device.

    Note that the syncback property is used so that during fourier updates,
    the exit wave array is synced bck to cpu (it is updated),
    while during probe update, it's not.
    """

    def __init__(self, nbytes, num, syncback=False):
        """
        Create an instance of GpuDataManager.
        Parameters are the same as for GpuData, and num is the number of
        GpuData instances to create (blocks on device).
        """
        self._syncback = syncback
        self.data = []
        for i in range(num):
            self.add_data_block(nbytes)

    def add_data_block(self, nbytes):
        """
        Add a GpuData block.

        Parameters
        ----------
        nbytes - Size of block

        Returns
        -------
        """
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


    def to_gpu(self, cpu, id, stream, upstream=None, pop_id=None):
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
        return m.to_gpu(cpu, id, stream, upstream)

    def sync_to_cpu(self, stream):
        """
        Sync back all data to CPU
        """
        for x in self.data:
            x.from_gpu(stream)

class EvData:

    def __init__(self):
        self.ev_download = None
        self.ev_upload = None
        self.ev_cycle = None
        self.ev_compute = None

    def record_download(self, stream):
        ev = cuda.Event()
        ev.record(stream)
        self.ev_download = ev
        return ev

    def record_upload(self, stream):
        ev = cuda.Event()
        ev.record(stream)
        self.ev_upload = ev
        return ev

    def record_compute(self, stream):
        ev = cuda.Event()
        ev.record(stream)
        self.ev_cycle = ev
        return ev

    def record_cycle(self, stream):
        ev = cuda.Event()
        ev.record(stream)
        self.ev_compute = ev
        return ev

    @property
    def is_on_dev(self):
        ev_d = self.ev_download
        ev_u = self.ev_upload
        if ev_d is not None and ev_d.query():
            if ev_u is None:
                return True
            else:
                if ev_u.query():
                    # upload event has happened
                    if ev_d.time_since(ev_u) > 0:
                        return True
        return False

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