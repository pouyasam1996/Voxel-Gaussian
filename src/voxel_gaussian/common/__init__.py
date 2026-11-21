"""Shared by every package; imports nothing else from the project.

env.py     machine paths from VG_* environment variables / .env      (Stage 0)
config.py  base.yaml + experiment + --set overrides -> typed Config  (Stage 0)
run.py     one output folder per run, with the merged config         (Stage 0)
device.py  cuda, then mps, then cpu (VG_DEVICE overrides)             (Stage 0)
camera.py, rotation.py, io.py                                         (Stage 1)
"""
