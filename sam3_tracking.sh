cd /home/jihun/PycharmProjects/SurgTPGS

# 첫번째, 마지막 이미지에서 segmentation후 mask를 저장 (image_sam3_seg)
conda activate sam3
python sam3_seg_mask.py

# 저장된 mask를 읽고 sam3를 사용하여 tracking후 결과 저장 (samgeo_track/forward )
conda activate samgeo_track
python sam3_samgeo_tracking.py

# tracking결과를 보고  프레임별 mask로 crop후 CLIP으로 임베딩 생성, 같은 instance끼리는 임베딩 평균내어 semantic map 저장 (language_features_fine)
conda activate SurgTPGS
python sam3_tracking_feature_save.py \
    --dataset_path /home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video01_00080 \
    --image_folder images \
    --clip_ckpt_path ckpts/model_final_cholecseg.pth \
    --save_name language_features_fine_mean \
    --save_4ch
