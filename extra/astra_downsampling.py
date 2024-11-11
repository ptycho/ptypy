import ptypy.utils.tomo as tu
import astra
import numpy as np
import matplotlib.pyplot as plt

# Parameters
size = 64
nangles = 180
angles = np.linspace(0, np.pi, nangles, endpoint=True)
downsample = 1

# Create simulated volume
rmap = tu.refractive_index_map(size)

# Create geometry
vectors = np.zeros((nangles,12))
for i in range(len(vectors)):

    # ray direction
    vectors[i,0] = np.sin(angles[i])
    vectors[i,1] = -np.cos(angles[i])
    vectors[i,2] = 0

    # center of detector
    vectors[i,3] = 0
    vectors[i,4] = 0
    vectors[i,5] = 0
    
    # vector from detector pixel (0,0) to (0,1)
    vectors[i,6] = np.cos(angles[i])
    vectors[i,7] = np.sin(angles[i])
    vectors[i,8] = 0
    
    # vector from detector pixel (0,0) to (1,0)
    vectors[i,9] = 0
    vectors[i,10] = 0
    vectors[i,11] = 1

# Projection geometry
proj_geom = astra.create_proj_geom('parallel3d_vec',  size, size, vectors)

# Volume geometry
vol_geom = astra.create_vol_geom(size, size, size)

# Create projections
proj_id, proj_data = astra.create_sino3d_gpu(rmap.real, proj_geom, vol_geom)

# Create downsampled geometry
vectors_downsampled = np.zeros((nangles,12))
for i in range(len(vectors)):

    # ray direction
    vectors_downsampled[i,0] = np.sin(angles[i])
    vectors_downsampled[i,1] = -np.cos(angles[i])
    vectors_downsampled[i,2] = 0

    # center of detector
    vectors_downsampled[i,3] = 0
    vectors_downsampled[i,4] = 0
    vectors_downsampled[i,5] = 0
    
    # vector from detector pixel (0,0) to (0,1)
    vectors_downsampled[i,6] = np.cos(angles[i]) / downsample
    vectors_downsampled[i,7] = np.sin(angles[i]) / downsample
    vectors_downsampled[i,8] = 0
    
    # vector from detector pixel (0,0) to (1,0)
    vectors_downsampled[i,9] = 0
    vectors_downsampled[i,10] = 0
    vectors_downsampled[i,11] = 1 / downsample 

# Downsampled projection geometry
proj_geom_downsampled = astra.create_proj_geom('parallel3d_vec',  size, size, vectors_downsampled)

# Downsampled volume geometry
size_d = int(size/downsample)
vol_geom_downsampled = astra.create_vol_geom(size_d, size_d, size_d)

# Create projections associated with downsampled geometry
proj_id_downsampled = astra.data3d.create("-sino", proj_geom_downsampled, proj_data)

# Reconstruction
rec_id = astra.data3d.create('-vol', vol_geom_downsampled)
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id_downsampled
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
recons = astra.data3d.get(rec_id)

# Shapes
print(rmap.shape, recons.shape)

# Plotting
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6,4), dpi=100)
for i in range(3):
    for j in range(2):
        ax = axes[j,i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    axes[0,i].set_title("sum(axis=%d)" %i)
    axes[0,i].imshow(rmap.real.sum(axis=i))
    axes[1,i].imshow(recons.sum(axis=i))
axes[0,0].set_ylabel("Original")
axes[1,0].set_ylabel("Recons")
plt.show()