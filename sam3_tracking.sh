#!/usr/bin/env bash
set -euo pipefail

cd /home/jihun/PycharmProjects/SurgTPGS
# bash sam3_tracking.sh

# -----------------------------
# Step 4/5 folder settings
# -----------------------------
# Step4(CLIP feature) output folder name under each dataset_path
CLIP_SAVE_NAME="language_features_fine_area_weighted"
# Step5(autoencoder test) output folder name (auto: ${CLIP_SAVE_NAME}_dim3)
DIM3_SAVE_NAME="${CLIP_SAVE_NAME}_dim3"
# Step6(4DGS train) feature level list
TRAIN_LEVELS=("0")
# Step8(evaluation) relevancy threshold sweep
# - single: "0.4"
# - sweep : "0.3,0.4,0.5,0.6"
EVAL_THRESHOLDS="0.3,0.4,0.5,0.6"

run_pipeline() {
  local dataset_path="$1"
  local dataset_name="$2"
  echo ""
  echo "========================================"
  echo "[Dataset] ${dataset_path}"
  echo "========================================"

  # 먼저 png2jpg_move / png2jpg_move_mask 전처리 필요시 수행

#  # 1) segmentation: first/last frame -> image_sam3_seg
#  # 입력:   ${dataset_path}/images, (옵션) ${dataset_path}/masks(FOV)
#  # 출력:   ${dataset_path}/image_sam3_seg
#  conda run -n sam3 python sam3_seg_mask.py \
#    --dataset_path "${dataset_path}"
#
#  # 2) tracking: forward + backward in one run -> samgeo_track/{forward,backward}
#  # 입력:   ${dataset_path}/images, ${dataset_path}/image_sam3_seg
#  # 출력:   ${dataset_path}/samgeo_track/forward,backward
#  conda run -n samgeo_track python sam3_samgeo_tracking.py \
#    --dataset_path "${dataset_path}"
#
#  # 3) merge: bi-directional merge -> samgeo_track/merged
#  # 입력:   ${dataset_path}/samgeo_track/forward,backward
#  # 출력:   ${dataset_path}/samgeo_track/merged
#  conda run -n samgeo_track python tracking_merge.py \
#    --dataset_path "${dataset_path}"

  # 4) CLIP feature extraction
  # 입력:   ${dataset_path}/images, ${dataset_path}/samgeo_track/merged/id_maps
  # 출력:   ${dataset_path}/${CLIP_SAVE_NAME}
  conda run -n SurgTPGS python sam3_tracking_feature_save.py \
    --dataset_path "${dataset_path}" \
    --save_name "${CLIP_SAVE_NAME}"
#
#  # 5) pre_VL_features (autoencoder train/test)
#  # 입력(train/test): ${dataset_path}/${CLIP_SAVE_NAME}
#  # 출력(train):      autoencoder/ckpt/${CLIP_SAVE_NAME}/${dataset_name}
#  # 출력(test):       ${dataset_path}/${DIM3_SAVE_NAME}
#  conda run -n SurgTPGS python autoencoder/train.py \
#    --dataset_name "${dataset_name}" \
#    --dataset_path "${dataset_path}" \
#    --input_dir_name "${CLIP_SAVE_NAME}" \
#    --encoder_dims 256 128 64 32 3 \
#    --decoder_dims 16 32 64 128 256 256 512 \
#    --lr 0.0007 \
#    --vlm clip_fine
#
#  conda run -n SurgTPGS python autoencoder/test.py \
#    --dataset_name "${dataset_name}" \
#    --dataset_path "${dataset_path}" \
#    --input_dir_name "${CLIP_SAVE_NAME}" \
#    --output_dir_name "${DIM3_SAVE_NAME}" \
#    --vlm clip_fine
#
#  # 6) 4DGS train (train.sh 단계 통합)
#  # 입력:   ${dataset_path}/images, ${dataset_path}/${DIM3_SAVE_NAME}
#  # 출력:   output/${DIM3_SAVE_NAME}/${dataset_name}_${level}
#  for level in "${TRAIN_LEVELS[@]}"
#  do
#    conda run -n SurgTPGS python train.py \
#      -s "${dataset_path}" \
#      --expname "${DIM3_SAVE_NAME}/${dataset_name}_${level}" \
#      --configs arguments/endonerf/default.py \
#      --feature_level "${level}" \
#      --language_features_name "${DIM3_SAVE_NAME}" \
#      --vlm clip_fine
#  done
#
#  # 7) render (render.sh 단계 통합)
#  # 입력:   output/${DIM3_SAVE_NAME}/${dataset_name}_${level}
#  # 출력:   ${model_path}/train|test|video/ours_*/...
#  for level in "${TRAIN_LEVELS[@]}"
#  do
#    local model_path="output/${DIM3_SAVE_NAME}/${dataset_name}_${level}"
#    conda run -n SurgTPGS python render.py \
#      --model_path "${model_path}" \
#      --configs arguments/endonerf/default.py
#  done
#
#  # 8) evaluation (eval_fine.sh 단계 통합)
#  # 입력:
#  #   - 렌더 결과: output/${DIM3_SAVE_NAME}/${dataset_name}_${level}/test/ours_3000
#  #   - GT 세그:   ${dataset_path}/test_seg
#  #   - AE ckpt:   autoencoder/ckpt/${CLIP_SAVE_NAME}/${dataset_name}/best_ckpt.pth
#  # 출력:
#  #   - result_thr_*.json, seg_separated_thr_*
#  for level in "${TRAIN_LEVELS[@]}"
#  do
#    local eval_dataset_name="${DIM3_SAVE_NAME}/${dataset_name}_${level}"
#    local gt_folder="${dataset_path}/test_seg"
#    local ae_ckpt_path="autoencoder/ckpt/${CLIP_SAVE_NAME}/${dataset_name}/best_ckpt.pth"
#    local clip_ckpt_path="ckpts/model_final_cholecseg.pth"
#    if [[ "${dataset_name}" == endovis_2018/* ]]; then
#      clip_ckpt_path="ckpts/model_final_endovis.pth"
#    fi
#
#    conda run -n SurgTPGS python eval_fine.py \
#      --dataset_name "${eval_dataset_name}" \
#      --output_path "output" \
#      --encoder_dims 256 128 64 32 3 \
#      --decoder_dims 16 32 64 128 256 256 512 \
#      --gt_path "${gt_folder}" \
#      --ckpt_path "${ae_ckpt_path}" \
#      --clip_ckpt_path "${clip_ckpt_path}" \
#      --level "${level}" \
#      --thresholds "${EVAL_THRESHOLDS}" \
#      --vlm fine
#  done
}

# ----------------------------------------
# cholecseg_sub
# ----------------------------------------
#for name in 01_00080 01_00240 01_15019 12_15750 17_01803
#do
#  run_pipeline \
#    "/home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video${name}" \
#    "cholecseg_sub/video${name}"
#done

# ----------------------------------------
# endovis_2018
# ----------------------------------------
for name in seq_5_sub seq_9_sub
do
  run_pipeline \
    "/home/jihun/PycharmProjects/SurgTPGS/data/endovis_2018/${name}" \
    "endovis_2018/${name}"
done
