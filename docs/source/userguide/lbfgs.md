# L-BFGS Engine

**Limited-memory BFGS (L-BFGS) is a gradient-based reconstruction engine that serves as an alternative to the ML engine. It still uses the maximum likelihood approach, but with a limited-memory BFGS optimizer in place of the nonlinear CG optimizer used in the ML engine.**

### Maximum Likelihood

The maximum-likelihood approach allows one to explicitly define a noise model for the ptychographic reconstruction. For ptychography, the negative log-likelihood can be written as

$$
\mathcal{L} = - \log \prod_k \prod_q p(I_{kq}|PO)
$$

where $p(I_{kq}|PO)$ is the probability of measuring a photon $I_{kq}$ at scan view $k$ and detector pixel $q$ given probe $P$ and object $O$. For a Gaussian distribution, the negative log-likelihood reduces to a least squares sum

$$
\mathcal{L} = \sum_k \sum_q \frac{(|\Psi_{kq}|^2 - I_{kq})^2}{2\sigma^2_{kq}}.
$$

For more details, see the original ML paper by [Thibault P. and Guizar-Sicairos M. (2012)](http://dx.doi.org/10.1088/1367-2630/14/6/063004).

### Limited-memory BFGS

The limited memory BFGS optimization algorithm is a [quasi-Newton method](https://en.wikipedia.org/wiki/Quasi-Newton_method#Search_for_extrema:_optimization) that approximates the famous [Broyden–Fletcher–Goldfarb–Shanno (BFGS)](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm) optimization algorithm using a limited amount of memory.

For ptychography, the negative log-likelihood $\mathcal{L}$ can be minimised using limited-memory BFGS with respect to $P$ and $O$ by defining the gradient

$$
g = \left(\frac{\partial \mathcal{L}}{\partial P}, \frac{\partial\mathcal{L}}{\partial O}\right)
$$

which in turn can be used to define the search direction for the $n$-th iteration step as

$$
\Delta^{(n)} = -{H^{(n)}} g^{(n)}
$$

where $H^{(n)}$ is the limited-memory BFGS approximation to the inverse Hessian

$$
H^{(n)} \approx
\left(
\begin{aligned}
    \frac{\partial^2\mathcal{L}}{\partial P^2}, \, \frac{\partial^2\mathcal{L}}{\partial O \partial P} \\
    \frac{\partial^2\mathcal{L}}{\partial P \partial O}, \, \frac{\partial^2\mathcal{L}}{\partial O^2}
\end{aligned}
\right)^{-1}.
$$

For more details, see the original limited-memory BFGS paper by [Liu D. C. and Nocedal J. (1989)](http://dx.doi.org/10.1007/BF01589116).

### Example Script

Note that all the usual ML engine parameters, including preconditioners and regularisers, also apply to the L-BFGS engine with the notable exception of the smoothing preconditioner (as this would require the ability to invert the Gaussian filter i.e. exactly de-blur).

```python
import ptypy
import ptypy.utils as u

from ptypy.custom import LBFGS

p = u.Param()
p.verbose_level = "interactive"
p.io = u.Param()
p.io.rfile = None
p.io.autosave = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# Live-plotting
p.io.autoplot = u.Param()
p.io.autoplot.active=True
p.io.autoplot.threaded = False
p.io.autoplot.layout = "jupyter"
p.io.autoplot.interval = 10

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = "Full"
p.scans.MF.data= u.Param()
p.scans.MF.data.name = "MoonFlowerScan"
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None
p.scans.MF.data.density = 0.2
p.scans.MF.data.photons = 1e8
p.scans.MF.data.psf = 0.

# Define reconstruction engines
p.engines = u.Param()

# L-BFGS (LBFGS) engine
p.engines.engine00 = u.Param()
p.engines.engine00.name = "LBFGS"
p.engines.engine00.ML_type = "Gaussian"
p.engines.engine00.numiter = 300
p.engines.engine00.numiter_contiguous = 10
p.engines.engine00.reg_del2 = True
p.engines.engine00.reg_del2_amplitude = 1.
p.engines.engine00.scale_precond = True
p.engines.engine00.scale_probe_object = 1.

P = ptypy.core.Ptycho(p,level=5)
```

