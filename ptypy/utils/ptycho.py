# We don't need this actually, but keeping it here for reference.

import numpy as np
from ptypy.utils.utils import scan_pos_pix, scan_view_limits, free_nf

FFT = lambda x: np.fft.ifftshift(np.fft.fftn(x, norm='ortho'))
IFFT = lambda x: np.fft.ifftn(np.fft.fftshift(x), norm='ortho')

class Ptycho2DReconstruction():

    def __init__(self, data, positions, probe, alpha=1e-4, beta = 1):

        self.data = data
        self.pos = positions
        self.probe = np.copy(probe)

        self.alpha = alpha
        self.beta = beta

        # Convert scan positions to pixel units
        posx_pix = self.pos[1,:]
        posy_pix = self.pos[0,:]

        # Dimensions
        self.num_positions = len(posx_pix)
        self.num_projections = len(data)

        # Calculate lower and higher limits for each scan view
        xlow, xhigh, ylow, yhigh = scan_view_limits(posx_pix, posy_pix, self.probe.shape)
        self.xlow = xlow
        self.ylow = ylow
        self.xhigh = xhigh
        self.yhigh = yhigh

        # Calculate mean power in diffraction data
        mean_power = np.mean(np.abs(self.data))

        # Create initial object array
        obj_shape = (yhigh.max()+1, xhigh.max()+1)
        self.obj = np.ones( (self.num_projections,) + obj_shape, dtype=complex)

        # Create probe array
        self.probe *= np.sqrt(mean_power / np.mean(np.abs(self.probe)**2))

        # Create more variables
        self.lerror = np.zeros((self.num_projections,self.num_positions))
        self.gerror = np.zeros(self.num_projections)
        self.fft_scaling = 1./np.sqrt(np.prod(self.probe.shape))
        self.positions = np.arange(self.num_positions)
        self.iter = 0

    def update_all(self, iterations=1):
        for k in range(iterations):
            for j in range(self.num_projections):
                self.update_single(j)
            self.iter += 1

    def object_to_stack(self):
        stack = np.ones((self.num_projections, self.num_positions, self.data.shape[-1], self.data.shape[-2]), dtype=complex)
        for j in np.arange(self.num_projections):
            for i in np.arange(self.num_positions):
                stack[j,i] = self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]]
        return stack

class Ptycho2DReconstructionEPIE(Ptycho2DReconstruction):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def update_single(self,j):

        # Randomize the order of the scan positions
        np.random.shuffle(self.positions)

        # Going through all the scan positions
        for i in self.positions:

                # The object view
                ov  = self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]]

                # The exit wave view
                ev = self.probe*ov

                # The model in Fourier space
                fmodel = FFT(ev)

                # Keep track of the error
                self.lerror[j,i] = np.sum((np.abs(fmodel)**2 - self.data[j,i])**2) / np.prod(fmodel.shape)

                # Apply the data constraint
                fmodel = np.sqrt(self.data[j,i]) * np.exp(1j * np.angle(fmodel))

                # New exit wave
                emodel = IFFT(fmodel)

                # Update the object
                ov += self.beta * np.conj(self.probe) / np.max(np.abs(self.probe)**2) * (emodel - ev)

                # Update the probe
                self.probe += self.beta * np.conj(ov) / np.max(np.abs(self.obj[j])**2) * (emodel - ev)

                # Power correction for probe
                #probe *= np.sqrt(mean_power / (np.abs(probe)**2).mean())

        # Save average error
        self.gerror[j] = np.mean(self.lerror[j])
        print("ptychographic update, iteration = {:04d}, projection index = {:04d}, error = {:1.2e}".format(self.iter, j, self.gerror.mean()), end="\r")

class Ptycho2DReconstructionRAAR(Ptycho2DReconstruction):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.beta = 0.9
        self.num_inner = 5


        # Create exit wave array and initialise it
        self.exit = np.tile(np.ones_like(self.probe), (self.num_projections,self.num_positions,1,1))
        for j in range(self.num_projections):
            for i in self.positions:
                self.exit[j,i] = self.probe * self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]]

        # Create arrays
        self.probe = np.tile(self.probe, (self.num_projections,1,1))
        self.obj_norm = np.zeros_like(self.obj)
        self.probe_norm = np.zeros_like(self.probe)

    def update_single(self,j):

        # Fourier update
        for i in self.positions:

            # The object view
            ov  = self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]]

            # Keep track of the error
            fmodel = FFT(self.probe[j] * ov) # this is only necessary for the calculating the error (not for the update)
            self.lerror[j,i] = np.sum((np.abs(fmodel)**2 - self.data[j,i])**2) / np.prod(fmodel.shape)

            # Update the exit array
            f =  FFT(2 * self.probe[j] * ov - self.exit[j,i])
            fm = np.sqrt(self.data[j,i]) / (np.abs(f) + 1e-10)
            self.exit[j,i] = self.beta * IFFT(fm * f) + (1 - 2*self.beta) * self.probe[j] * ov + self.beta * self.exit[j,i]

        # The inner loop (overlap)
        for k in range(self.num_inner):

            # Object update
            self.obj[j,:] = 0.
            self.obj_norm[j,:] = 0.
            for i in self.positions:
                self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]] += self.probe[j].conj() * self.exit[j,i]
                self.obj_norm[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]] += np.abs(self.probe[j])**2
            self.obj[j] /= (self.obj_norm[j] + 1e-10)

            # Probe update
            self.probe[j,:] = 0.
            self.probe_norm[j,:] = 0.
            for i in self.positions:
                self.probe[j] += self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]].conj() * self.exit[j,i]
                self.probe_norm[j] += np.abs(self.obj[j,self.ylow[i]:self.yhigh[i],self.xlow[i]:self.xhigh[i]])**2
            self.probe[j] /= (self.probe_norm[j] + 1e-10)

        # Save average error
        self.gerror[j] = np.mean(self.lerror[j])
        print("ptychographic update, iteration = {:04d}, projection index = {:04d}, error = {:1.2e}".format(self.iter, j, self.gerror.mean()), end="\r")



class Ptycho3DReconstruction():

    def __init__(self, data, probe, alpha=1e-4, beta = 1):

        self.data = data
        self.probe = np.copy(probe)

        self.alpha = alpha
        self.beta = beta

        # Dimensions
        self.num_positions = len(data[0])
        self.num_projections = len(data)

        # Calculate mean power in diffraction data
        mean_power = np.mean(np.abs(self.data))

        # Create initial object array
        self.obj = np.ones(self.data.shape, dtype=complex)

        # Create probe array
        self.probe *= np.sqrt(mean_power / np.mean(np.abs(self.probe)**2))

        # Create more variables
        self.lerror = np.zeros((self.num_projections,self.num_positions))
        self.gerror = np.zeros(self.num_projections)
        self.fft_scaling = 1./np.sqrt(np.prod(self.probe.shape))
        self.iter = 0

    def update_fourier(self):
        pass

    def update_object(self):
        pass

    def update_probe(self):
        pass

    def update_fourier_all(self):
        for j in range(self.num_projections):
            self.update_fourier(j)
        self.iter += 1

    def update_probe_all(self):
        for j in range(self.num_projections):
            self.update_probe(j)

    def update_object_all(self):
        for j in range(self.num_projections):
            self.update_object(j)

    def update_object_epie_all(self):
        for j in range(self.num_projections):
            self.update_object_epie(j)

    def update_object_sdr_all(self):
        for j in range(self.num_projections):
            self.update_object_sdr(j)


class Ptycho3DReconstructionRAAR(Ptycho3DReconstruction):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.beta = 0.9
        self.beta_object = 1

        # Create exit wave array and initialise it
        self.exit = np.tile(np.ones_like(self.probe), (self.num_projections,self.num_positions,1,1))
        for j in range(self.num_projections):
            self.exit[j] = self.probe * self.obj[j]

        # Create arrays
        self.probe = np.tile(self.probe, (self.num_projections,1,1))
        self.probe_norm = np.zeros_like(self.probe)

    def update_fourier(self,j):

        for i in range(self.num_positions):

            # The object view
            ov  = self.obj[j,i]

            # Keep track of the error
            fmodel = FFT(self.probe[j] * ov) # this is only necessary for the calculating the error (not for the update)
            self.lerror[j,i] = np.sum((np.abs(fmodel)**2 - self.data[j,i])**2) / np.prod(fmodel.shape)

            # Update the exit array
            f =  FFT(2 * self.probe[j] * ov - self.exit[j,i])
            fm = np.sqrt(self.data[j,i]) / (np.abs(f) + 1e-10)
            self.exit[j,i] = self.beta * IFFT(fm * f) + (1 - 2*self.beta) * self.probe[j] * ov + self.beta * self.exit[j,i]

        # Save average error
        self.gerror[j] = np.mean(self.lerror[j])
        print("ptychographic update, iteration = {:04d}, projection index = {:04d}, error = {:1.2e}".format(self.iter, j, self.gerror.mean()), end="\r")

    def update_object(self,j):
        for i in range(self.num_positions):
            self.obj[j,i] = self.probe[j].conj() * self.exit[j,i] / (np.abs(self.probe[j])**2 + 1e-10)

    def update_probe(self,j):
        self.probe[j,:] = 0.
        self.probe_norm[j,:] = 0.
        for i in range(self.num_positions):
            self.probe[j] += self.obj[j,i].conj() * self.exit[j,i]
            self.probe_norm[j] += np.abs(self.obj[j,i])**2
        self.probe[j] /= (self.probe_norm[j] + 1e-10)

    def update_object_epie(self, j):
        for i in range(self.num_positions):
            self.obj[j,i] = self.obj[j,i] - self.beta_object * self.probe[j].conj() * (self.obj[j,i] * self.probe[j] - self.exit[j,i]) / np.max(np.abs(self.probe[j])**2)

    def update_object_sdr(self, j):
        for i in range(self.num_positions):
            max_norm = np.max(np.abs(self.probe[j])**2)
            self.obj[j,i] = ((1 - self.beta_object) * max_norm * self.obj[j,i] + self.beta_object * self.probe[j].conj() * self.exit[j,i]) / ((1 - self.beta_object) * max_norm + self.beta_object * np.abs(self.probe[j]**2))

