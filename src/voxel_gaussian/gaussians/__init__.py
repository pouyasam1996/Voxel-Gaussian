"""Gaussian model and rendering. May import: common only (never slam or tsdf).

camera.py     render camera (world -> camera), rotations, blob shapes, projection
renderer.py   tiled renderer: image, depth and alpha, differentiable
model.py      Gaussians3D: the blob parameters
init.py       one flat blob per surface voxel of the voxel map
densify.py    split blobs whose colour is still wrong, remove faded ones (Step 5)
renderer_gsplat.py               optional, NVIDIA GPUs only
"""
