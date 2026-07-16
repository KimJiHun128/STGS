# STGS

STGS is a surgical-video understanding pipeline that builds temporally consistent instances and semantic features before learning them in a 4D Gaussian representation. Its main contributions are the SAM3-based initialization, bidirectional SamGeo3 tracking, track association and postprocessing, and track-aware semantic feature construction.

The project integrates three technical layers that require separate Python environments:

- **STGS segmentation and semantics**: initialization, tracking, bidirectional merging, semantic feature extraction/aggregation, and feature compression implemented for this project.
- **SAM3 and SamGeo3**: external models/libraries used by the STGS segmentation and tracking implementation.
- **Gaussian Splatting backend**: the final training, rendering, and evaluation stage is based on the SurgTPGS codebase.

## Pipeline overview

```text
Video frames
    │
    ▼
1. SAM3 instance initialization                [sam3]
   sam3_seg_mask.py
    │
    ▼
2. Forward and backward video tracking         [samgeo_track]
   sam3_samgeo_tracking.py
    │
    ▼
3. Bidirectional track merging                 [samgeo_track]
   tracking_merge.py
    │
    ▼
4. Track-aware CLIP feature extraction         [STGS]
   sam3_tracking_feature_save.py
    │
    ▼
5. Semantic feature compression to 3D          [STGS]
   autoencoder/train.py + autoencoder/test.py
    │
    ▼
6. Semantic 4D Gaussian Splatting training     [GS backend]
   train.py
    │
    ▼
7. RGB/depth/semantic rendering                [GS backend]
   render.py
    │
    ▼
8. Text-prompted semantic evaluation           [GS backend]
   eval_fine.py
```

The main orchestration script is [`sam3_tracking.sh`](./sam3_tracking.sh). It documents all eight stages and switches environments with `conda run` at each environment boundary.

> **Current script state:** stages 1 and 2 are enabled in `sam3_tracking.sh`. Stages 3–8 are present but commented out for experiment control. To run the complete pipeline, enable them in order. Stage 4 expects the output of stage 3 at `samgeo_track/merged/id_maps`, so the merge stage must not be skipped.

## Environments

The environments are intentionally separate because the Gaussian backend and the newer SAM3/SamGeo libraries require incompatible Python and PyTorch versions. The `SurgTPGS` environment name is retained for compatibility with the existing local environment and scripts; it does not mean that the STGS tracking or semantic-feature pipeline comes from SurgTPGS.

| Environment | Python in exported YAML | Main responsibility |
|---|---:|---|
| `sam3` | 3.12 | SAM3 image segmentation |
| `samgeo_track` | 3.10 | SamGeo3 video tracking and track merging |
| `SurgTPGS` | 3.7 | STGS semantic-feature processing plus the GS backend |

Create the environments from the provided exports:

```bash
conda env create -f conda_envs/sam3.yml
conda env create -f conda_envs/samgeo_track.yml
conda env create -f conda_envs/SurgTPGS.yml
```

The YAML files are full snapshots. Smaller pip-oriented dependency lists are also provided for each component:

```text
requirements/surgtpgs.txt
requirements/sam3.txt
requirements/samgeo-track.txt
```

See [`requirements/README.md`](./requirements/README.md) for installation commands and CUDA/PyTorch notes. The environments must remain separate; do not install all three files into one environment.

Install the CUDA extensions used by the Gaussian backend inside the legacy-named `SurgTPGS` environment:

```bash
conda activate SurgTPGS
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

This environment uses PyTorch 1.13.1 with CUDA 11.7. A matching CUDA toolkit is therefore recommended when building the extensions:

```bash
export PATH=/usr/local/cuda-11.7/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-11.7
```

## Repository setup

Clone the repository together with the SAM3 Git submodule:

```bash
git clone --recurse-submodules https://github.com/KimJiHun128/STGS.git STGS
cd STGS
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Only `submodules/sam3` is registered as a Git submodule. The following Gaussian-backend dependencies are vendored source directories in this repository:

```text
submodules/depth-diff-gaussian-rasterization/
submodules/segment-anything-langsplat/
submodules/simple-knn/
```

## Data and checkpoints

Expected top-level layout:

```text
STGS/
├── data/
│   ├── cholecseg_sub/
│   │   ├── video01_00080/
│   │   ├── video01_00240/
│   │   ├── video01_15019/
│   │   ├── video12_15750/
│   │   └── video17_01803/
│   └── endovis_2018/
│       ├── seq_5_sub/
│       └── seq_9_sub/
├── ckpts/
│   ├── model_final_cholecseg.pth
│   ├── model_final_endovis.pth
│   └── sam_vit_h_4b8939.pth
└── submodules/
    └── sam3/
```

A sequence is expected to contain the camera/depth metadata required by the Gaussian backend in addition to these relevant folders:

```text
<sequence>/
├── images/       # ordered video frames
├── masks/        # optional FOV masks
└── test_seg/     # ground-truth semantic masks for evaluation
```

The original processed CholecSeg/EndoVis datasets and pretrained checkpoints are described in the upstream [SurgTPGS repository](https://github.com/lastbasket/SurgTPGS).

## Running the integrated pipeline

Select datasets near the bottom of `sam3_tracking.sh`, then run:

```bash
bash sam3_tracking.sh
```

The script resolves paths from its own location, so it can be launched from a different working directory.

### Stage 1 — SAM3 instance initialization

```bash
conda run -n sam3 python sam3_seg_mask.py \
  --dataset_path data/cholecseg_sub/video12_15750
```

SAM3 segments the first and last frames. The implementation performs score/area filtering, rough-mask removal, mask IoU NMS, optional non-overlapping instance decomposition (SNID), optional residual-region mask decomposition (RRMD), and FOV filtering.

Output:

```text
<sequence>/image_sam3_seg/
├── <first-frame>/
│   ├── masks_stack.npy
│   ├── mask_*.png
│   └── vis_final_masks_inside_fov.png
└── <last-frame>/
    └── ...
```

SNID/RRMD can be controlled explicitly:

```bash
python sam3_seg_mask.py --dataset_path <sequence> --enable_snid --enable_rrmd
python sam3_seg_mask.py --dataset_path <sequence> --disable_snid --disable_rrmd
```

### Stage 2 — forward/backward SamGeo3 tracking

```bash
conda run -n samgeo_track python sam3_samgeo_tracking.py \
  --dataset_path data/cholecseg_sub/video12_15750
```

The first-frame masks initialize forward tracking and the last-frame masks initialize backward tracking. By default, both directions run in one command.

Output:

```text
<sequence>/samgeo_track/
├── forward/
│   ├── id_maps/
│   ├── instance_masks/
│   ├── id_maps_color/
│   └── tracked_postprocessed.mp4
└── backward/
    └── ...
```

For a single direction:

```bash
python sam3_samgeo_tracking.py \
  --dataset_path <sequence> \
  --single_direction \
  --prompt_mode first
```

### Stage 3 — bidirectional track merging

```bash
conda run -n samgeo_track python tracking_merge.py \
  --dataset_path data/cholecseg_sub/video12_15750
```

The merge step matches forward/backward tracks using accumulated overlap, unions matched masks, absorbs contained or very small tracks, and applies FOV/gap/hole postprocessing.

Output:

```text
<sequence>/samgeo_track/merged/
├── id_maps/
├── instance_masks/
├── id_maps_color/
└── tracked_postprocessed.mp4
```

### Stage 4 — track-aware CLIP features

```bash
conda run -n SurgTPGS python sam3_tracking_feature_save.py \
  --dataset_path data/cholecseg_sub/video12_15750 \
  --image_folder images \
  --track_out_name samgeo_track/merged \
  --feature_agg area_weighted \
  --save_name language_features_fine_area_weighted
```

Aggregation modes:

- `none`: compute features independently per frame.
- `mean`: average features for the same tracked instance over time.
- `area_weighted`: average over time using instance area as the weight.

Each output folder contains paired files:

```text
frame_XXXXXX_endo_s.npy   # pixel-to-feature/instance map
frame_XXXXXX_endo_f.npy   # CLIP feature table
```

### Stage 5 — autoencoder compression

```bash
conda run -n SurgTPGS python autoencoder/train.py \
  --dataset_name cholecseg_sub/video12_15750 \
  --dataset_path data/cholecseg_sub/video12_15750 \
  --input_dir_name language_features_fine_area_weighted \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --lr 0.0007 \
  --vlm clip_fine

conda run -n SurgTPGS python autoencoder/test.py \
  --dataset_name cholecseg_sub/video12_15750 \
  --dataset_path data/cholecseg_sub/video12_15750 \
  --input_dir_name language_features_fine_area_weighted \
  --output_dir_name language_features_fine_area_weighted_dim3 \
  --vlm clip_fine
```

The autoencoder reduces CLIP features to three dimensions for Gaussian training. Checkpoints are stored below `autoencoder/ckpt/`, namespaced by the feature folder and dataset.

### Stage 6 — semantic 4DGS training

```bash
conda run -n SurgTPGS python train.py \
  -s data/cholecseg_sub/video12_15750 \
  --expname language_features_fine_area_weighted_dim3/cholecseg_sub/video12_15750_0 \
  --configs arguments/endonerf/default.py \
  --feature_level 0 \
  --language_features_name language_features_fine_area_weighted_dim3 \
  --vlm clip_fine
```

Models and training artifacts are written below `output/`.

### Stage 7 — rendering

```bash
conda run -n SurgTPGS python render.py \
  --model_path output/language_features_fine_area_weighted_dim3/cholecseg_sub/video12_15750_0 \
  --configs arguments/endonerf/default.py
```

This renders train/test/video views, including RGB, depth, and semantic features.

### Stage 8 — text-prompted evaluation

```bash
conda run -n SurgTPGS python eval_fine.py \
  --dataset_name language_features_fine_area_weighted_dim3/cholecseg_sub/video12_15750_0 \
  --output_path output \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --gt_path data/cholecseg_sub/video12_15750/test_seg \
  --ckpt_path autoencoder/ckpt/language_features_fine_area_weighted/cholecseg_sub/video12_15750/best_ckpt.pth \
  --clip_ckpt_path ckpts/model_final_cholecseg.pth \
  --level 0 \
  --thresholds 0.3,0.4,0.5,0.6 \
  --vlm fine
```

The evaluator decodes rendered semantic features, compares them with CLIP text embeddings for the dataset classes, generates semantic masks, and evaluates them against `test_seg`.

## Baseline and standalone scripts

The following standalone scripts are retained for baseline reproduction and stage-by-stage execution:

- `pre_data.sh`: baseline frame-wise SAM/CLIP preprocessing through `preprocess_fine.py`.
- `pre_VL_features.sh`: standalone autoencoder workflow.
- `train.sh`, `render.sh`, `eval_fine.sh`: standalone Gaussian training/evaluation loops.

For STGS, use `sam3_tracking.sh` and the stage commands documented above. The baseline scripts are kept only for comparison and independent execution.

## Ablation workflow

[`ablation_video12_15750.sh`](./ablation_video12_15750.sh) runs the configured Video12 ablations across the integrated pipeline. It currently compares:

- all optional components disabled;
- SNID and RRMD enabled, with single-direction tracking and no temporal feature aggregation.

Summarize completed runs with:

```bash
python summarize_ablation_video12.py
```

## Main source files

| Path | Role |
|---|---|
| `sam3_seg_mask.py` | STGS initial-instance generation using SAM3 |
| `sam3_samgeo_tracking.py` | STGS forward/backward tracking using SamGeo3 |
| `tracking_merge.py` | STGS bidirectional track association and postprocessing |
| `sam3_tracking_feature_save.py` | STGS tracked-instance semantic feature construction |
| `autoencoder/` | STGS semantic feature compression and reconstruction |
| `train.py` | semantic 4D Gaussian training using the GS backend |
| `render.py` | RGB/depth/semantic rendering |
| `eval_fine.py` | text-prompted semantic segmentation evaluation |
| `scene/`, `gaussian_renderer/`, `utils/` | SurgTPGS-derived 4DGS backend and rendering implementation |
| `conda_envs/` | reproducible exports for the three environments |

## Gaussian backend attribution

Only the final Gaussian Splatting backbone, rendering, and associated evaluation code are derived from SurgTPGS. The STGS segmentation, tracking, track merging, and track-aware semantic feature pipeline described above are project-specific implementations.

> Yiming Huang, Long Bai, Beilei Cui, Kun Yuan, Guankun Wang, Mobarak I. Hoque, Nicolas Padoy, Nassir Navab, and Hongliang Ren. *SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting*. MICCAI 2025.

- [Paper](https://arxiv.org/abs/2506.23309)
- [Project page](https://lastbasket.github.io/MICCAI-2025-SurgTPGS/)
- [Original repository](https://github.com/lastbasket/SurgTPGS)

```bibtex
@misc{huang2025surgtpgssemantic3dsurgical,
  title={SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting},
  author={Yiming Huang and Long Bai and Beilei Cui and Kun Yuan and Guankun Wang and Mobarak I. Hoque and Nicolas Padoy and Nassir Navab and Hongliang Ren},
  year={2025},
  eprint={2506.23309},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2506.23309}
}
```

Please also cite the relevant SAM3 and SamGeo/segment-geospatial projects when publishing results obtained with their components.
