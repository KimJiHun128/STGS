# Environment-specific pip requirements

STGS uses three isolated environments because its upstream components require different Python and PyTorch generations. Do not install all three requirement files into one environment.

The files in this directory contain the direct runtime dependencies used by this project. The `conda_envs/*.yml` files remain the authoritative, full environment snapshots, including transitive and system-level packages.

## SurgTPGS / 4DGS

```bash
conda create -n SurgTPGS python=3.7 -y
conda activate SurgTPGS
pip install -r requirements/surgtpgs.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

This environment runs feature extraction, the autoencoder, Gaussian training, rendering, and evaluation.

## SAM3 segmentation

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install -r requirements/sam3.txt
```

The final editable entry installs the checked-out `submodules/sam3` source, so initialize the Git submodule first.

## SamGeo tracking

```bash
conda create -n samgeo_track python=3.10 -y
conda activate samgeo_track
pip install -r requirements/samgeo-track.txt
```

This environment runs `sam3_samgeo_tracking.py` and `tracking_merge.py`.

GPU-enabled PyTorch wheels are platform- and CUDA-dependent. The pinned versions mirror the current exported environments, but on a different CUDA/driver stack it is safer to install the appropriate PyTorch build first and then install the remaining packages.
