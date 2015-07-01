****
Math
****

.. math::
   \phi_{j,\gamma}^\mathrm{it}(\mathbf x) &= 2\, p_{j,\gamma}^\mathrm{it}(\mathbf x)\cdot o_{j,\gamma}^\mathrm{it}(\mathbf x -\mathbf x_j)-\psi_{j,\gamma}^\mathrm{it-1}(\mathbf x) 

.. math::
   \Upsilon_j(\mathbf{s}) = 1-M_{j}(\mathbf{s}) +\frac{M_{j}(\mathbf{s})\sqrt{I_{j}(\mathbf{s})}}{\sqrt{\sum_{\tilde\gamma} \left|\mathcal{P}_{j,\tilde\gamma}\left \{\phi_{j,\tilde \gamma}^\mathrm{it}(\mathbf x)\right\} \right|^2}}

.. math::
   \phi^\mathrm{it+1}_{j,\gamma}(\mathbf x) &= \hat{\mathcal{P}}_{j,\gamma}\left\{\mathcal{P}_{j,\gamma} \left\{\phi_{j,\gamma}^\mathrm{it}(\mathbf{x})\right\}(\mathbf{s}) \cdot \Upsilon_j(\mathbf{s}) \right\} \\
   \psi^\mathrm{it+1}_{j,\gamma}(\mathbf x)  &= \psi^\mathrm{it}_{j,\gamma}(\mathbf x)  +  \phi_{j,\gamma}^\mathrm{it+1}(\mathbf x)  - p_{j,\gamma}^{\mathrm{it}}(\mathbf x) \cdot o_{j,\gamma}^{\mathrm{it}}(\mathbf x-\mathbf x_j)
