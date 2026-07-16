#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

DATASET_PATH="${PROJECT_ROOT}/data/cholecseg_sub/video12_15750"
DATASET_NAME="cholecseg_sub/video12_15750"
LEVEL="0"
EVAL_THRESHOLD="0.4"

run_one() {
  local exp_tag="$1"          # ex: all_off / snid_rrmd_on_only
  local snid_flag="$2"        # --enable_snid or --disable_snid
  local rrmd_flag="$3"        # --enable_rrmd or --disable_rrmd

  local clip_save_name="language_features_ablation_${exp_tag}_none"
  local dim3_save_name="${clip_save_name}_dim3"
  local exp_name="${dim3_save_name}/${DATASET_NAME}_${LEVEL}"
  local model_path="output/${exp_name}"

  echo ""
  echo "========================================"
  echo "[Ablation] ${exp_tag}"
  echo "========================================"

  # 1) Segmentation (SNID / RRMD toggle)
  conda run -n sam3 python sam3_seg_mask.py \
    --dataset_path "${DATASET_PATH}" \
    ${snid_flag} \
    ${rrmd_flag}

  # 2) Tracking (forward only)
  conda run -n samgeo_track python sam3_samgeo_tracking.py \
    --dataset_path "${DATASET_PATH}" \
    --single_direction \
    --prompt_mode first

  # 4) CLIP feature extraction (ISTC off => feature_agg none)
  conda run -n SurgTPGS python sam3_tracking_feature_save.py \
    --dataset_path "${DATASET_PATH}" \
    --image_folder images \
    --track_out_name samgeo_track/forward \
    --feature_agg none \
    --save_name "${clip_save_name}"

  # 5) pre_VL_features
  conda run -n SurgTPGS python autoencoder/train.py \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --input_dir_name "${clip_save_name}" \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --lr 0.0007 \
    --vlm clip_fine

  conda run -n SurgTPGS python autoencoder/test.py \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --input_dir_name "${clip_save_name}" \
    --output_dir_name "${dim3_save_name}" \
    --vlm clip_fine

  # 6) 4DGS train
  conda run -n SurgTPGS python train.py \
    -s "${DATASET_PATH}" \
    --expname "${exp_name}" \
    --configs arguments/endonerf/default.py \
    --feature_level "${LEVEL}" \
    --language_features_name "${dim3_save_name}" \
    --vlm clip_fine

  # 7) render
  conda run -n SurgTPGS python render.py \
    --model_path "${model_path}" \
    --configs arguments/endonerf/default.py

  # 8) evaluation
  conda run -n SurgTPGS python eval_fine.py \
    --dataset_name "${exp_name}" \
    --output_path "output" \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --gt_path "${DATASET_PATH}/test_seg" \
    --ckpt_path "autoencoder/ckpt/${clip_save_name}/${DATASET_NAME}/best_ckpt.pth" \
    --clip_ckpt_path "ckpts/model_final_cholecseg.pth" \
    --level "${LEVEL}" \
    --thresholds "${EVAL_THRESHOLD}" \
    --vlm fine
}

# A) all OFF: SNID off, RRMD off, bi-directional off, ISTC off(none)
run_one "all_off" "--disable_snid" "--disable_rrmd"

# B) only SNID+RRMD ON: bi-directional off, ISTC off(none)
run_one "snid_rrmd_on_only" "--enable_snid" "--enable_rrmd"

echo ""
echo "[Done] Ablation runs finished."
echo "Now run: python summarize_ablation_video12.py"
