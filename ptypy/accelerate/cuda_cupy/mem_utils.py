import numpy as np
import cupy as cp
import cupyx
from collections import deque


def make_pagelocked_paired_arrays(ar):
    mem = cupyx.empty_pinned(ar.shape, ar.dtype, order="C")
    mem[:] = ar
    return cp.asarray(mem), mem


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
        self.gpuraw = cp.cuda.alloc(nbytes)
        self.nbytes = nbytes
        self.nbytes_buffer = nbytes
        self.gpuId = None
        self.cpu = None
        self.syncback = syncback
        self.ev_done = None

    def _allocator(self, nbytes):
        if nbytes > self.nbytes:
            raise Exception('requested more bytes than maximum given before: {} vs {}'.format(
                nbytes, self.nbytes))
        return self.gpuraw

    def record_done(self, stream):
        self.ev_done = cp.cuda.Event()
        with stream:
            self.ev_done.record()

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
            alloc = cp.cuda.get_allocator()
            try:
                cp.cuda.set_allocator(self._allocator)
                with stream:
                    self.gpu = cp.asarray(cpu)
            finally:
                cp.cuda.set_allocator(alloc)
        return self.gpu

    def from_gpu(self, stream):
        """
        Transfer data back to CPU, into same data handle it was copied from
        before.
        """
        if self.cpu is not None and self.gpuId is not None and self.gpu is not None:
            if self.ev_done is not None:
                stream.wait_event(self.ev_done)
            cp.cuda.runtime.memcpyAsync(dst=self.cpu.ctypes.data,
                                        src=self.gpu.data.ptr,
                                        size=self.gpu.nbytes,
                                        kind=2,  # d2h
                                        stream=stream.ptr)
            self.ev_done = cp.cuda.Event()
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
            self.gpuraw.mem.free()
            self.gpuraw = cp.cuda.alloc(nbytes)

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
        self.gpuraw.mem.free()
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
        assert what in ['dtoh', 'htod', 'compute']
        self.ev_done = cp.cuda.Event()
        with stream:
            self.ev_done.record()
        self.done_what = what

    def to_gpu(self, cpu, ident, stream):
        """
        Transfer cpu array to GPU on stream (async), keeping track of its id
        """
        ident = id(cpu) if ident is None else ident
        if self.gpuId != ident:
            if self.ev_done is not None:
                stream.wait_event(self.ev_done)
            # Safety measure. This is asynchronous, but it should still work
            # Essentially we want to copy the data held in gpu array back to its CPU
            # handle before the buffer can be reused.
            if self.done_what != 'dtoh' and self.syncback:
                # uploads on the download stream, easy to spot in nsight-sys
                self.from_gpu(stream)
            self.gpuId = ident
            self.cpu = cpu
            alloc = cp.cuda.get_allocator()
            try:
                cp.cuda.set_allocator(self._allocator)
                with stream:
                    self.gpu = cp.asarray(cpu)
            finally:
                cp.cuda.set_allocator(alloc)
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
                stream.wait_event(self.ev_done)
            cp.cuda.runtime.memcpyAsync(dst=self.cpu.ctypes.data,
                                        src=self.gpu.data.ptr,
                                        size=self.gpu.nbytes,
                                        kind=2,  # d2h
                                        stream=stream.ptr)
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
        if self.max is None or len(self) < self.max:
            nbytes = nbytes if nbytes is not None else self._nbytes
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

