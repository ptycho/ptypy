## Author: Benedikt J. Daurer

import numpy as np
import scipy.ndimage as ndimage
from ptypy.utils.utils import rmask

def sample_volume(N):
    vol = np.zeros((N,N,N))
    xx,yy,zz = np.meshgrid(np.arange(N)-N/2, np.arange(N)-N/2,np.arange(N)-N/2)
    m = lambda r,dx,dy,dz: np.sqrt((xx+dx)**2 + (yy+dy)**2 + (zz+dz)**2) < r
    vol[m(N//6,0,0,0)] = 1
    vol[m(N//8,N//7,N//7,N//7)] = 2
    vol[m(N//12,N//12,0,0)] = 5
    vol[m(N//12,0,N//6,0)] = 5
    return ndimage.gaussian_filter(vol,1)

def refractive_index_map(Nx):
    beta = np.log(sample_volume(Nx)+20)-np.log(20)
    delta = 0.1 * sample_volume(Nx)
    return delta + 1j * beta

class Ptycho2DSimulation:
    def __init__(self, scan_points_xy = 16, scan_step_size = 4, detector_shape = (32, 32), probe_diameter = 16):
        self.scan_points = scan_points_xy
        self.scan_stepsize = scan_step_size
        self.detector_shape = detector_shape
        self.probe_diameter = probe_diameter

        # Create initial positions, probe, object
        self.create_positions()
        self.create_circular_probe()
        self.create_object()

    def create_positions(self):
        forward = True
        position_number = 0
        n, step = self.scan_points, self.scan_stepsize
        self.pos = np.zeros([2, n*n], dtype=np.int32)

        for j in range(n):
            for i in range(n):
                if forward:
                    self.pos[1, position_number] = i * step
                    self.pos[0, position_number] = j * step
                else:
                    self.pos[1, position_number] = (n-i-1) * step
                    self.pos[0, position_number] = j * step
                position_number += 1
            forward = not forward

        # Calculate global shape
        n_pos = self.pos.shape[-1]
        n_pos_x = np.ceil(np.sqrt(n_pos))
        n_pos_y = np.ceil(np.sqrt(n_pos))
        self.global_shape = np.array(self.detector_shape)
        self.global_shape[0] += (n_pos_y-1)*self.scan_stepsize
        self.global_shape[1] += (n_pos_x-1)*self.scan_stepsize

    def add_jitter_to_positions(self, amplitude=1):
        self.pos += np.random.randint(amplitude+1, size=self.pos.shape)

    def set_global_shape(self):
        shy = self.pos[0].max() + self.detector_shape[0] + 1
        shx = self.pos[1].max() + self.detector_shape[1] + 1
        self.global_shape = (shy, shx)

    def create_circular_probe(self, signal=1e8, random=False):
        self.probe = np.ones(self.detector_shape, dtype=np.complex64)
        radius= np.array(self.probe_diameter/2.)
        self.probe[~rmask(self.detector_shape, radius)] = 0.
        self.probe = self.probe * np.exp(1j*self.probe)
        if random:
            rng = np.random.default_rng(2022)
            randomized_amp = (rng.random(self.probe.shape) * 0.1 + 1) * np.abs(self.probe)
            randomized_phs = np.angle(self.probe) * rng.random(self.probe.shape)
            self.probe = randomized_amp * np.exp(1j * randomized_phs)
        self.probe *= (np.sqrt(signal) / np.abs(self.probe).sum())

    def create_focussed_probe(self, signal=1e8, phase_factor=0.01, ap_scale=0.5, aspect=1, shifty=0, rotation=0):
        ap_size = ap_scale*self.probe_diameter
        ap_amp = np.ones(self.detector_shape, dtype=np.float32)
        ap_amp[~rmask(self.detector_shape, ap_size/2., cent=(self.detector_shape[0]//2-shifty, self.detector_shape[1]//2))] = 0.
        ap_amp = ndimage.gaussian_filter(ap_amp,1)

        cent = np.divide(self.detector_shape,2)
        x = np.arange(0,self.detector_shape[0]) - cent[0] + shifty
        y = np.arange(0,self.detector_shape[1]) - cent[1]
        yy, xx = np.meshgrid(y, x)
        xxx = xx * np.cos(rotation/180 * np.pi) + yy * np.sin(rotation/180 * np.pi)
        yyy = -xx * np.sin(rotation/180 * np.pi) + yy * np.cos(rotation/180 * np.pi)
        ap_phase = np.exp(-1j * np.pi * (phase_factor * xxx**2 + phase_factor * aspect * yyy**2))
        self.probe = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ap_amp * ap_phase)))
        self.probe *= (np.sqrt(signal) / np.abs(self.probe).sum())
        return

    def create_object(self):
        im_mod = np.ones(self.global_shape, dtype=np.float32)
        im_phase = np.zeros(self.global_shape, dtype=np.float32)
        self.obj = np.sqrt(im_mod) * np.exp(1j * im_phase)

    def cut(self, ar, pos, shape):
        centre = np.divide(shape, 2)
        centre = np.floor(centre)
        shape = np.int32(np.round(shape))
        ar_cut = np.copy(ar[pos[0]:(pos[0] + shape[0]), pos[1]:(pos[1] + shape[1])])
        return ar_cut

    def create_data(self, total_dose=1e3, dtype=np.uint32, poisson=True):
        global_shape = np.int32(self.obj.shape)
        local_shape = np.int32(self.probe.shape)
        self.data = np.zeros([self.pos.shape[1], local_shape[0], local_shape[1]], dtype=np.float64)
        self.exit_wave = np.zeros([self.pos.shape[1], local_shape[0], local_shape[1]], dtype=np.complex64)
        self.local_obj = np.zeros([self.pos.shape[1], local_shape[0], local_shape[1]], dtype=np.complex64)

        for frame_number in range(self.pos.shape[1]):
            self.local_obj[frame_number, :, :] = self.cut(self.obj, self.pos[:,frame_number], local_shape)
            self.exit_wave[frame_number, :, :] = self.local_obj[frame_number, :, :] * self.probe
            data_temp = np.square(np.abs(np.fft.fftshift(np.fft.fft2(self.exit_wave[frame_number, :, :]))))
            self.data[frame_number, :, :] = data_temp

        #self.data *= total_dose / self.data.sum()
        if poisson:
            self.data = np.random.poisson(self.data).astype(dtype)
        else:
            self.data = self.data.astype(dtype)
        self.nphotons = self.data.sum()