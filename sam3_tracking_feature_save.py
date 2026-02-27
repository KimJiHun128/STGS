# track_clip_features.py
# 실행 인자 4개: --dataset_path --image_folder --sam_ckpt_path --clip_ckpt_path
# (sam_ckpt_path는 형식 맞추려고 남겨둠. 여기선 SAM 자동분할은 안 씀)

import os
import random
import argparse
import glob
import re

import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type, Dict, List
from scene import clip
import torchvision
from torch import nn

def parse_start_frame_from_dataset_path(dataset_path: str, image_dir: str = None) -> int:
    # video01_00080 -> 80
    base = os.path.basename(os.path.normpath(dataset_path))
    m = re.search(r"(\d{5})$", base)
    if m is not None:
        return int(m.group(1))

    # endovis_2018/seq_x_sub 같이 5자리 suffix가 없는 경우:
    # images의 첫 프레임 번호를 사용하고, 없으면 0으로 fallback
    if image_dir is not None and os.path.isdir(image_dir):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
        if len(files) > 0:
            files = sorted(files)
            nums = re.findall(r"\d+", files[0])
            if len(nums) > 0:
                return int(nums[0])
    return 0

# =========================
# CLIP (원 코드 그대로)
# =========================
@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B/16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        model, preprocess = clip.load(
            self.config.clip_model_type,
            pretrained=pretrained_path,
            device="cuda",
            jit=False,
            prompt_depth=0,
            prompt_length=0,
        )
        print("Using CLIP model with CAT-Seg finetuned")
        model.eval()
        self.tokenizer = clip.tokenize
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(p) for p in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(p) for p in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        assert self.pos_embeds.shape[1] == self.clip_n_dims

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

# =========================
# 유틸 (원 코드 crop 방식 유지)
# =========================
def pad_img(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w)//2:(h - w)//2 + w, :] = img
    else:
        pad[(w - h)//2:(w - h)//2 + h, :, :] = img
    return pad

def get_seg_img_from_id(mask_bin: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """
    원 코드 get_seg_img와 동일한 역할:
    - mask 밖은 0으로
    - bbox로 crop
    """
    img = image_rgb.copy()
    img[mask_bin == 0] = np.array([0, 0, 0], dtype=np.uint8)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # inclusive->exclusive
    x2 += 1
    y2 += 1
    return img[y1:y2, x1:x2, :]

def expand_bbox_from_mask(mask_bin: np.ndarray, H: int, W: int, scale: float = 1.35, min_margin: int = 12):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nw = max(bw * scale, bw + 2 * min_margin)
    nh = max(bh * scale, bh + 2 * min_margin)

    nx1 = int(round(cx - nw / 2))
    nx2 = int(round(cx + nw / 2))
    ny1 = int(round(cy - nh / 2))
    ny2 = int(round(cy + nh / 2))

    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(W, nx2); ny2 = min(H, ny2)
    return nx1, ny1, nx2, ny2

def make_tile_from_id(image_rgb: np.ndarray, id_map: np.ndarray, oid: int,
                      scale: float = 1.35, min_margin: int = 12) -> Tuple[np.ndarray, int]:
    """
    원 코드 흐름을 최대한 맞춤:
    - mask 밖 0
    - bbox crop
    - pad -> 224 resize
    return: (224,224,3) uint8 tile, area
    """
    mask = (id_map == oid).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return None, 0

    H, W = id_map.shape
    bb = expand_bbox_from_mask(mask, H, W, scale=scale, min_margin=min_margin)
    if bb is None:
        return None, 0
    x1, y1, x2, y2 = bb

    crop_img = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # mask 밖은 0
    crop_img[crop_mask == 0] = 0

    pad_crop = pad_img(crop_img)
    tile = cv2.resize(pad_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    return tile, area

def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18

def sorted_frame_files(image_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    return sorted(files, key=extract_first_int)

def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# =========================
# 1) Pass1: instance_id별 평균 feature 만들기
# =========================
@torch.no_grad()
def pass1_build_instance_mean_features(
    dataset_path: str,
    image_dir: str,
    idmap_dir: str,
    model: OpenCLIPNetwork,
    resolution: int = -1,
    scale: float = 1.35,
    min_margin: int = 12,
    area_weighted: bool = True,
) -> Dict[int, np.ndarray]:
    frame_files = sorted_frame_files(image_dir)

    sum_feat: Dict[int, np.ndarray] = {}
    sum_w: Dict[int, float] = {}

    for fname in tqdm(frame_files, desc="Pass1: build mean features"):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        id_path = os.path.join(idmap_dir, f"{stem}.npy")
        if not os.path.exists(id_path):
            continue

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # 원 코드처럼 리사이즈 정책 적용
        orig_h, orig_w = img_bgr.shape[:2]
        if resolution == -1:
            global_down = (orig_h / 1080) if orig_h > 1080 else 1.0
        else:
            global_down = (orig_w / resolution)
        scale_down = float(global_down)
        new_w, new_h = int(orig_w / scale_down), int(orig_h / scale_down)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        id_map = np.load(id_path).astype(np.int32)
        if id_map.shape != (orig_h, orig_w):
            # id_map이 이미 원본 크기와 다를 수도 있으니 일단 원본 기준으로 맞추고,
            # 이후 img_bgr 리사이즈에 맞춰 다시 맞춤
            id_map = cv2.resize(id_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        id_map = cv2.resize(id_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # BGR->RGB (원 코드 sam_encoder에서 RGB로 바꾸는 방식과 정합)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ids = np.unique(id_map)
        ids = ids[ids > 0]

        tiles = []
        oids = []
        weights = []

        for oid in ids:
            tile, area = make_tile_from_id(img_rgb, id_map, int(oid), scale=scale, min_margin=min_margin)
            if tile is None:
                continue
            tiles.append(tile)
            oids.append(int(oid))
            weights.append(float(area) if area_weighted else 1.0)

        if len(tiles) == 0:
            continue

        # (B,3,224,224) float32 /255 -> cuda
        tiles_np = np.stack(tiles, axis=0).astype("float32") / 255.0
        tiles_t = torch.from_numpy(tiles_np).permute(0, 3, 1, 2).to("cuda")

        clip_embed = model.encode_image(tiles_t)  # (B,512)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embed = clip_embed.detach().cpu().float().numpy()  # float32로 누적 (평균 안정)

        for emb, oid, w in zip(clip_embed, oids, weights):
            if oid not in sum_feat:
                sum_feat[oid] = emb * w
                sum_w[oid] = w
            else:
                sum_feat[oid] += emb * w
                sum_w[oid] += w

    mean_feat: Dict[int, np.ndarray] = {}
    for oid in sum_feat.keys():
        v = sum_feat[oid] / max(sum_w[oid], 1e-8)
        v = v / (np.linalg.norm(v) + 1e-12)  # 최종 L2 normalize
        mean_feat[int(oid)] = v.astype(np.float32)

    return mean_feat


# =========================
# 2) Pass2: 프레임별 *_f.npy, *_s.npy 저장 (feature는 mean 사용)
# =========================
def pass2_save_per_frame_outputs(
    image_dir: str,
    idmap_dir: str,
    save_folder: str,
    mean_feat: Dict[int, np.ndarray],
    resolution: int = -1,
    save_4ch: bool = True,
    file_prefix: str = "",
    start_frame: int = 0,          # ✅ 추가
):

    os.makedirs(save_folder, exist_ok=True)
    frame_files = sorted_frame_files(image_dir)

    for fname in tqdm(frame_files, desc="Pass2: save per-frame"):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        id_path = os.path.join(idmap_dir, f"{stem}.npy")
        if not os.path.exists(id_path):
            continue

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # 원 코드 리사이즈 정책 동일하게 적용 (seg_map 크기 맞추기 위해)
        orig_h, orig_w = img_bgr.shape[:2]
        if resolution == -1:
            global_down = (orig_h / 1080) if orig_h > 1080 else 1.0
        else:
            global_down = (orig_w / resolution)
        scale_down = float(global_down)
        new_w, new_h = int(orig_w / scale_down), int(orig_h / scale_down)

        id_map = np.load(id_path).astype(np.int32)
        if id_map.shape != (orig_h, orig_w):
            id_map = cv2.resize(id_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        id_map = cv2.resize(id_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        H, W = id_map.shape

        ids = np.unique(id_map)
        ids = ids[ids > 0]
        ids = [int(x) for x in ids if int(x) in mean_feat]

        # 프레임 내 row index 매핑
        oid_to_row = {oid: i for i, oid in enumerate(ids)}

        # preprocess_fine 정합: feature/seg_map 모두 float32로 저장
        if len(ids) == 0:
            feat = np.zeros((0, 512), dtype=np.float32)
            seg = -np.ones((H, W), dtype=np.float32)
        else:
            feat_f32 = np.stack([mean_feat[oid] for oid in ids], axis=0).astype(np.float32)
            feat = feat_f32.astype(np.float32)
            seg = -np.ones((H, W), dtype=np.float32)
            for oid, row in oid_to_row.items():
                seg[id_map == oid] = row

        if save_4ch:
            seg_out = np.stack([seg, seg, seg, seg], axis=0).astype(np.float32)  # (4,H,W)
        else:
            seg_out = seg[None, ...].astype(np.float32)  # (1,H,W)

        i = extract_first_int(stem)  # 0,1,2...
        save_base = f"frame_{start_frame + i:06d}_endo"  # ✅ 원하는 규칙
        np.save(os.path.join(save_folder, f"{file_prefix}{save_base}_s.npy"), seg_out)
        np.save(os.path.join(save_folder, f"{file_prefix}{save_base}_f.npy"), feat)



def save_video_tables(save_folder: str, mean_feat: Dict[int, np.ndarray], file_prefix: str = ""):
    ids = sorted(mean_feat.keys())
    feats = np.stack([mean_feat[i] for i in ids], axis=0).astype(np.float32)
    np.save(os.path.join(save_folder, f"{file_prefix}video_instance_ids.npy"), np.array(ids, dtype=np.int32))
    np.save(os.path.join(save_folder, f"{file_prefix}video_instance_f.npy"), feats)


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--sam_ckpt_path", type=str, default="ckpts/sam_vit_h_4b8939.pth")  # unused here
    parser.add_argument("--clip_ckpt_path", type=str, default="ckpts/model_final_cholecseg.pth")
    parser.add_argument("--resolution", type=int, default=-1)

    # crop 넉넉하게
    parser.add_argument("--bbox_scale", type=float, default=1.35)
    parser.add_argument("--bbox_margin", type=int, default=12)

    # 평균 방식
    parser.add_argument("--area_weighted", action="store_true", help="mask area로 가중 평균")
    parser.add_argument("--save_4ch", action="store_true", default=True,
                        help="seg_map을 (4,H,W)로 저장 (preprocess_fine 형태, 기본값=True)")
    parser.add_argument("--save_1ch", dest="save_4ch", action="store_false",
                        help="seg_map을 (1,H,W)로 저장")

    # tracking output 위치
    parser.add_argument("--track_out_name", type=str, default="samgeo_track_out_1")
    parser.add_argument("--idmap_subdir", type=str, default="id_maps")

    # 저장 폴더 이름
    parser.add_argument("--save_name", type=str, default="language_features_fine_mean")
    parser.add_argument("--file_prefix", type=str, default="")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    image_dir = os.path.join(dataset_path, args.image_folder)

    # tracking 결과 위치 (네 tracking 코드 출력 구조 그대로)
    # dataset_path = .../videoXX (라고 가정하면)
    # image_dir = .../videoXX/images
    # out_dir = .../videoXX/samgeo_track_out
    video_root = os.path.dirname(image_dir)
    track_out = os.path.join(video_root, args.track_out_name)
    idmap_dir = os.path.join(track_out, args.idmap_subdir)

    if not os.path.isdir(idmap_dir):
        raise FileNotFoundError(f"idmap_dir not found: {idmap_dir}")

    # CLIP 로드 (원 코드 그대로)
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig, args.clip_ckpt_path)

    # Pass1: instance mean feature
    mean_feat = pass1_build_instance_mean_features(
        dataset_path=dataset_path,
        image_dir=image_dir,
        idmap_dir=idmap_dir,
        model=model,
        resolution=args.resolution,
        scale=args.bbox_scale,
        min_margin=args.bbox_margin,
        area_weighted=args.area_weighted,
    )

    # Pass2: 프레임별 저장
    save_folder = os.path.join(dataset_path, args.save_name)

    start_frame = parse_start_frame_from_dataset_path(dataset_path, image_dir=image_dir)

    pass2_save_per_frame_outputs(
        image_dir=image_dir,
        idmap_dir=idmap_dir,
        save_folder=save_folder,
        mean_feat=mean_feat,
        resolution=args.resolution,
        save_4ch=args.save_4ch,
        file_prefix=args.file_prefix,
        start_frame=start_frame,  # ✅ 추가
    )


    # # (추천) 비디오 단위 테이블도 함께 저장
    # save_video_tables(save_folder, mean_feat, file_prefix=args.file_prefix)

    print("[Done]")
    print("  save_folder:", save_folder)
    print("  mean instances:", len(mean_feat))
