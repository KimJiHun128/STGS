# sam3_samgeo_tracking.py
"""
입력
- image_dir: 비디오 프레임 폴더 (.../videoXX/images)
- 초기 마스크:
  .../videoXX/image_sam3_seg/<첫프레임이름>/masks_stack.npy  (N,H,W)

처리
1) 첫 프레임 N개 mask를 obj_id(기본 100~)에 매핑
2) SamGeo3Video tracker 초기화 + frame0 mask prompt 입력
   - add_mask_prompt: mask 내부에서 positive points 샘플링
   - add_point_prompts: mask 바깥 '가까운 링'에서 negative points 추가(옵션)
3) propagate/track 1회
4) tracker.save_masks 결과 저장
5) (옵션) WTA 없이 raw_instances/obj_XXXX/... 저장 (가능한 경우 score stack 기반)
6) save_masks 결과를 frame별 id_map으로 변환
   - frame0은 초기 prompt 강제 사용
   - (stack이면) WTA로 id_map 생성
7) 공간 후처리 + 시간 후처리 + 단일컴포넌트 강제
8) id_map / instance / color 저장
9) 후처리 id_map 기반 overlay 비디오 저장(옵션)

출력
- base_dir = .../videoXX/samgeo_track/
  - forward/   (PROMPT_FRAME_MODE="first")
  - backward/  (PROMPT_FRAME_MODE="last")
- 각 하위 폴더 내부 저장 구조:
  - masks/                                   (tracker.save_masks raw)
  - raw_instances/obj_XXXX/masks/*.png        (옵션, WTA 없이 채널별)
  - raw_instances/obj_XXXX/overlays/*.png     (옵션)
  - id_maps/*.npy
  - instance_masks/*.npz
  - id_maps_color/*.png
  - tracked_postprocessed.mp4 (옵션)
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import re
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from samgeo.samgeo3 import SamGeo3Video

# =========================
# 사용자 설정
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
image_dir = str(PROJECT_ROOT / "data" / "cholecseg_sub" / "video01_00080" / "images")

# tracking direction is auto-selected by prompt frame mode:
# - "first" -> forward
# - "last"  -> backward
PROMPT_FRAME_MODE = "last"          # "first" or "last"
MAX_FRAME_NUM_TO_TRACK = None       # int or None (None: full length)


NUM_POINTS = 5
# ✅ negative prompt (prompt frame only)
USE_NEGATIVE_PROMPTS = True
NUM_NEG_POINTS = 5
NEG_RING_WIDTH = 6     # mask 바깥 링 두께(px)
NEG_RING_GAP = 10        # mask에서 약간 띄운 뒤 링 시작
NEG_SAMPLE_MODE = "linspace"  # "random" or "linspace"
NEG_RANDOM_SEED = 0


OBJ_ID_START = 100

# ✅ WTA 없이 raw(채널별) 저장 (stack이 있을 때만 의미 있음)
SAVE_RAW_INSTANCES = True
RAW_SAVE_OVERLAY = True
RAW_OVERLAY_ALPHA = 0.55
RAW_STACK_THRESHOLD = 0.0   # score > thresh를 foreground로 저장 (미세하면 -0.1 등도 실험)

# 공간 후처리
MIN_OBJ_AREA = 2000
MIN_HOLE_AREA = 500
OPEN_KERNEL = 10
CLOSE_KERNEL = 10
MIN_FRACTION_REMAIN = 0.1
MIN_THICKNESS = 10

# 시간 후처리
IOU_KEEP_THRESH = 0.50
MAX_CENTER_SHIFT = 80.0
ALLOW_NEW_IDS = True

# 비디오
VIDEO_FPS = 25
VIDEO_ALPHA = 0.45
SAVE_VIDEO = True
SAVE_PROMPT_POINTS_DEBUG = True
PROMPT_POINTS_DEBUG_NAME = "frame0_prompt_points.png"
POINT_RADIUS = 4
POINT_THICKNESS = -1  # filled

# ✅ raw 저장 결과 점검 (tracker가 frame1부터 id를 내는지 확인)
DEBUG_RAW_INSPECT = True
RAW_INSPECT_FRAME_IDX = 1
RAW_INSPECT_OBJ_IDS = [100, 103]
RAW_INSPECT_MAX_UNIQUE = 50

# =========================
# ✅ Ablation switches (후처리 원흉 후보 3개 on/off)
# =========================
ENABLE_SPLIT_FILTER = False   # split_and_filter_components (MIN_OBJ_AREA, MIN_FRACTION_REMAIN)
ENABLE_CUT_THIN = False       # cut_thin_bridges (MIN_THICKNESS)
ENABLE_TEMPORAL = False       # temporal_id_consistency_step (프레임간 유지/삭제)

# (옵션) 디버그: 매 프레임마다 죽은 id 출력
PRINT_DEAD_IDS_EACH_FRAME = True

# -------------------------
# 시각화
# -------------------------

def draw_neg_ring_polyline(
    vis: np.ndarray,
    neg_pts: List[List[int]],
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    draw_points: bool = False,
    point_radius: int = 3,
):
    """
    negative points를 centroid 기준 각도 정렬해서 닫힌 폴리라인으로 그려줌.
    - neg_pts가 3개 미만이면 아무것도 안 그리거나(옵션) 점만 표시.
    """
    if len(neg_pts) == 0:
        return

    pts = np.array(neg_pts, dtype=np.int32)  # (N,2) [x,y]

    if len(neg_pts) < 3:
        # 점이 너무 적으면 원(폴리라인) 의미가 약함: 필요하면 점만
        if draw_points:
            for x, y in pts:
                cv2.circle(vis, (int(x), int(y)), point_radius, color_bgr, -1)
        return

    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(ang)
    pts_sorted = pts[order].reshape((-1, 1, 2))  # polyline 포맷

    # 닫힌 선(원처럼)
    cv2.polylines(vis, [pts_sorted], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    if draw_points:
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), point_radius, color_bgr, -1)


def colorize_id_map(id_map: np.ndarray) -> np.ndarray:
    h, w = id_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(id_map)
    ids = ids[ids > 0]
    for oid in ids:
        oid_i = int(oid)
        b = (37 * oid_i) % 256
        g = (17 * oid_i) % 256
        r = (97 * oid_i) % 256
        color[id_map == oid_i] = (b, g, r)
    return color


def sample_positive_points_from_mask_like_add_mask_prompt(mask_np: np.ndarray, num_points: int) -> List[List[int]]:
    """add_mask_prompt 구현과 동일하게: mask 내부 픽셀 리스트에서 linspace 인덱스로 샘플"""
    m = (mask_np > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or num_points <= 0:
        return []
    n = min(num_points, len(xs))
    if n == len(xs):
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs) - 1, n, dtype=int)
    return [[int(xs[i]), int(ys[i])] for i in idx]


def draw_points_on_image(img_bgr: np.ndarray, pts: List[List[int]], color_bgr: Tuple[int, int, int],
                         radius: int = 4, thickness: int = -1):
    for x, y in pts:
        cv2.circle(img_bgr, (int(x), int(y)), radius, color_bgr, thickness)


def save_frame0_prompt_points_debug(
    image_dir_: str,
    frame0_name: str,
    out_dir: str,
    per_obj_points: Dict[int, Dict[str, List[List[int]]]],
    radius: int = 4,
    thickness: int = -1,
    out_name: str = "frame0_prompt_points.png",
):
    fp = os.path.join(image_dir_, frame0_name)
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read frame0 for debug: {fp}")

    vis = img.copy()

    for oid, d in per_obj_points.items():
        pos_pts = d.get("pos", [])
        neg_pts = d.get("neg", [])

        # ✅ negative는 점 대신 "원처럼 연결" (겹쳐도 구분 잘 됨)
        draw_neg_ring_polyline(
            vis,
            neg_pts,
            color_bgr=(0, 0, 255),
            thickness=2,
            draw_points=False,   # 원만
            point_radius=3,
        )

        # positive는 초록 점
        for x, y in pos_pts:
            cv2.circle(vis, (int(x), int(y)), radius, (0, 255, 0), thickness)

        # 텍스트
        if len(pos_pts) > 0:
            x0, y0 = pos_pts[0]
            cv2.putText(vis, f"id {oid}", (x0 + 6, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, vis)
    print(f"[Saved] prompt points debug: {out_path}")


def overlay_id_map_on_bgr(frame_bgr: np.ndarray, id_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_id_map(id_map)
    out = frame_bgr.copy()
    fg = (id_map > 0)
    if np.any(fg):
        out_f = out.astype(np.float32)
        color_f = color.astype(np.float32)
        out_f[fg] = (1.0 - alpha) * out_f[fg] + alpha * color_f[fg]
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def overlay_single_mask(frame_bgr: np.ndarray, m: np.ndarray, oid: int, alpha: float = 0.55) -> np.ndarray:
    out = frame_bgr.copy()
    m = (m > 0).astype(np.uint8)
    if m.sum() == 0:
        return out
    color = np.zeros_like(out, dtype=np.uint8)
    b = (37 * int(oid)) % 256
    g = (17 * int(oid)) % 256
    r = (97 * int(oid)) % 256
    color[m > 0] = (b, g, r)
    out_f = out.astype(np.float32)
    c_f = color.astype(np.float32)
    fg = (m > 0)
    out_f[fg] = (1 - alpha) * out_f[fg] + alpha * c_f[fg]
    out = np.clip(out_f, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)
    return out


def save_postprocessed_video_from_id_maps(
    image_dir_: str,
    frame_files: List[str],
    idmap_npy_dir: str,
    video_out_path: str,
    fps: int = 25,
    alpha: float = 0.45,
):
    if len(frame_files) == 0:
        raise RuntimeError("No frames to render video.")

    first_frame_path = os.path.join(image_dir_, frame_files[0])
    first_frame = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {video_out_path}")

    try:
        for fname in frame_files:
            stem = os.path.splitext(fname)[0]
            frame_path = os.path.join(image_dir_, fname)
            id_map_path = os.path.join(idmap_npy_dir, f"{stem}.npy")

            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[Warn] skip frame read fail: {frame_path}")
                continue

            if not os.path.exists(id_map_path):
                vw.write(frame)
                continue

            id_map = np.load(id_map_path).astype(np.int32)
            if id_map.shape != (h, w):
                id_map = cv2.resize(id_map, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

            vis = overlay_id_map_on_bgr(frame, id_map, alpha=alpha)
            vw.write(vis)
    finally:
        vw.release()

    print(f"[Saved] postprocessed video: {video_out_path}")


# -------------------------
# 유틸
# -------------------------
def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18


def sorted_frame_files(image_dir_: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(image_dir_) if f.lower().endswith(exts)]
    return sorted(files, key=extract_first_int)


def get_first_frame_hw(image_dir_: str) -> Tuple[int, int]:
    files = sorted_frame_files(image_dir_)
    if not files:
        raise FileNotFoundError(f"No image files found in: {image_dir_}")
    first_path = os.path.join(image_dir_, files[0])
    img = cv2.imread(first_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read first frame: {first_path}")
    return img.shape[:2]


def find_masks_npy_from_image_dir(image_dir_: str, prompt_frame_mode: str = "first") -> Tuple[str, str, int]:
    files = sorted_frame_files(image_dir_)
    if not files:
        raise FileNotFoundError(f"No image files found in: {image_dir_}")

    mode = prompt_frame_mode.lower().strip()
    if mode == "first":
        prompt_idx = 0
    elif mode == "last":
        prompt_idx = len(files) - 1
    else:
        raise ValueError(f"prompt_frame_mode must be 'first' or 'last', got: {prompt_frame_mode}")

    prompt_image_name = files[prompt_idx]
    prompt_stem = os.path.splitext(prompt_image_name)[0]

    video_root = os.path.dirname(image_dir_)
    masks_npy = os.path.join(video_root, "image_sam3_seg", prompt_stem, "masks_stack.npy")
    if not os.path.exists(masks_npy):
        raise FileNotFoundError(f"masks_stack.npy not found: {masks_npy}")

    return masks_npy, prompt_image_name, prompt_idx


def safe_to_uint8_binary(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    if m.ndim == 3:
        m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"mask ndim must be 2, got {m.ndim}")
    return (m > 0).astype(np.uint8)


def build_obj_masks_from_npy(
    masks_npy: str,
    h: int,
    w: int,
    obj_id_start: int = 100,
) -> Dict[int, np.ndarray]:
    masks = np.load(masks_npy)  # (N,H,W)
    if masks.ndim != 3:
        raise ValueError(f"masks_stack.npy shape must be (N,H,W), got {masks.shape}")
    if masks.shape[0] == 0:
        raise RuntimeError("No masks in masks_stack.npy (N=0).")

    obj_masks: Dict[int, np.ndarray] = {}
    for i in range(masks.shape[0]):
        m = safe_to_uint8_binary(masks[i])

        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)

        if m.sum() == 0:
            continue

        obj_masks[obj_id_start + i] = m

    if not obj_masks:
        raise RuntimeError("All prompt masks became empty after preprocessing.")
    return obj_masks


def collect_saved_mask_paths(masks_out_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(masks_out_dir, e)))
    return sorted(paths, key=lambda p: extract_first_int(os.path.basename(p)))


def rebuild_instance_from_id_map(id_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.unique(id_map)
    ids = ids[ids > 0]

    ids_list = []
    masks_list = []
    for oid in ids:
        m = (id_map == int(oid)).astype(np.uint8)
        if m.sum() == 0:
            continue
        ids_list.append(int(oid))
        masks_list.append(m)

    if len(masks_list) == 0:
        return np.array([], dtype=np.int32), np.zeros((0, id_map.shape[0], id_map.shape[1]), dtype=np.uint8)

    return np.array(ids_list, dtype=np.int32), np.stack(masks_list, axis=0).astype(np.uint8)


# -------------------------
# negative point 샘플링 (mask 바깥 가까운 링)
# -------------------------
def sample_negative_points_near_mask(
    mask_np: np.ndarray,
    num_neg: int = 10,
    ring_width: int = 25,
    ring_gap: int = 2,
    mode: str = "random",
    seed: int = 0,
) -> List[List[int]]:
    """
    ring = dilate(mask, gap+width) - dilate(mask, gap)
    -> mask 바깥 '근처'에서만 negative를 찍음(너무 멀리 안 감)
    """
    m = (mask_np > 0).astype(np.uint8)
    if m.sum() == 0 or num_neg <= 0:
        return []

    gap = max(0, int(ring_gap))
    width = max(1, int(ring_width))

    k1 = max(3, (2 * gap + 1) | 1)
    k2 = max(3, (2 * (gap + width) + 1) | 1)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    dil1 = cv2.dilate(m, ker1, iterations=1) if gap > 0 else m
    dil2 = cv2.dilate(m, ker2, iterations=1)

    ring = ((dil2 > 0) & (dil1 == 0)).astype(np.uint8)

    ys, xs = np.where(ring > 0)
    if len(xs) == 0:
        return []

    n = min(num_neg, len(xs))
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xs), size=n, replace=False)
    else:
        idx = np.linspace(0, len(xs) - 1, n, dtype=int)

    return [[int(xs[i]), int(ys[i])] for i in idx]


# -------------------------
# stack -> (N,H,W) float32 정규화
# -------------------------
def to_stack_nhw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.ndim == 2:
        s = arr.astype(np.float32)
        if s.shape != (h, w):
            s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
        return s[None, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported arr ndim: {arr.ndim}, shape={arr.shape}")

    if arr.shape[1:] == (h, w):  # (N,H,W)
        return arr.astype(np.float32)

    if arr.shape[:2] == (h, w):  # (H,W,N)
        c = arr.shape[2]
        if c in (3, 4):
            raise ValueError(f"Color-like 3/4ch not a score stack: {arr.shape}")
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)

    if arr.shape[0] < 4096 and arr.shape[1] > 16 and arr.shape[2] > 16:
        resized = []
        for i in range(arr.shape[0]):
            s = cv2.resize(arr[i].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(s)
        return np.stack(resized, axis=0).astype(np.float32)

    raise ValueError(f"Cannot infer stack layout from shape={arr.shape}")


# -------------------------
# ✅ WTA 없이 raw(채널별) obj 저장
# -------------------------
def save_raw_instances_without_wta(
    image_dir_: str,
    frame_files: List[str],
    i: int,
    saved_mask_path: str,
    out_dir: str,
    obj_ids_all: List[int],
    h: int,
    w: int,
    bin_thresh: float = 0.0,
    save_overlay: bool = True,
    overlay_alpha: float = 0.55,
):
    stem = os.path.splitext(frame_files[i])[0]
    root = os.path.join(out_dir, "raw_instances")
    os.makedirs(root, exist_ok=True)

    frame = None
    if save_overlay:
        frame_path = os.path.join(image_dir_, frame_files[i])
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {frame_path}")
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    if saved_mask_path.lower().endswith(".npy"):
        arr = np.load(saved_mask_path)
    else:
        arr = cv2.imread(saved_mask_path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Cannot read mask file: {saved_mask_path}")

    if arr.ndim == 3 and not (arr.shape[:2] == (h, w) and arr.shape[-1] in (3, 4)):
        stack = to_stack_nhw(arr, h, w)  # (N,H,W)
        n_ch = stack.shape[0]
        n_obj = len(obj_ids_all)
        use_n = min(n_ch, n_obj)

        if n_ch != n_obj:
            print(f"[Warn] raw stack channels({n_ch}) != num objs({n_obj}). save min={use_n}")

        for k in range(use_n):
            oid = int(obj_ids_all[k])
            score = stack[k]
            m = (score > bin_thresh).astype(np.uint8)

            obj_dir = os.path.join(root, f"obj_{oid:04d}")
            mdir = os.path.join(obj_dir, "masks")
            os.makedirs(mdir, exist_ok=True)
            cv2.imwrite(os.path.join(mdir, f"{stem}.png"), (m * 255).astype(np.uint8))

            if save_overlay:
                odir = os.path.join(obj_dir, "overlays")
                os.makedirs(odir, exist_ok=True)
                ov = overlay_single_mask(frame, m, oid=oid, alpha=overlay_alpha)
                cv2.imwrite(os.path.join(odir, f"{stem}.png"), ov)
        return

    if arr.ndim == 2:
        id_map = arr.astype(np.int32)
        if id_map.shape != (h, w):
            id_map = cv2.resize(id_map, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        for oid in obj_ids_all:
            m = (id_map == int(oid)).astype(np.uint8)

            obj_dir = os.path.join(root, f"obj_{int(oid):04d}")
            mdir = os.path.join(obj_dir, "masks")
            os.makedirs(mdir, exist_ok=True)
            cv2.imwrite(os.path.join(mdir, f"{stem}.png"), (m * 255).astype(np.uint8))

            if save_overlay:
                odir = os.path.join(obj_dir, "overlays")
                os.makedirs(odir, exist_ok=True)
                ov = overlay_single_mask(frame, m, oid=int(oid), alpha=overlay_alpha)
                cv2.imwrite(os.path.join(odir, f"{stem}.png"), ov)
        return

    raise RuntimeError(f"Unsupported mask format for raw-without-wta: shape={arr.shape}, path={saved_mask_path}")


# -------------------------
# raw 저장 점검
# -------------------------
def inspect_raw_saved_mask(
    masks_out_dir: str,
    frame_idx: int,
    num_frames: int,
    obj_ids: Optional[List[int]] = None,
    max_unique: int = 50,
) -> None:
    num_digits = len(str(num_frames))
    base = str(frame_idx).zfill(num_digits)
    candidates = [
        os.path.join(masks_out_dir, f"{base}.png"),
        os.path.join(masks_out_dir, f"{base}.npy"),
        os.path.join(masks_out_dir, f"{base}.tif"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f"[RAW INSPECT] no mask file for frame {frame_idx} in {masks_out_dir}")
        return

    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            print(f"[RAW INSPECT] failed to read: {path}")
            return

    print(f"[RAW INSPECT] frame={frame_idx} path={path}")
    print(f"[RAW INSPECT] shape={arr.shape} dtype={arr.dtype} min/max={arr.min()}/{arr.max()}")

    if arr.ndim == 2:
        uniq = np.unique(arr)
        if len(uniq) <= max_unique:
            print(f"[RAW INSPECT] unique({len(uniq)}): {uniq.tolist()}")
        else:
            head = uniq[:max_unique].tolist()
            print(f"[RAW INSPECT] unique({len(uniq)}): {head} ...")

        if obj_ids:
            for oid in obj_ids:
                area = int((arr == oid).sum())
                print(f"[RAW INSPECT] id={oid} area={area}")
    else:
        print("[RAW INSPECT] non-2D mask; expected id map. Check channel layout.")

# -------------------------
# WTA (후처리용 id_map)
# -------------------------
def build_raw_id_map_winner_takes_all_from_stack(
    stack: np.ndarray,
    obj_id_start: int,
    h: int,
    w: int,
) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3D (N,H,W), got {stack.shape}")

    if stack.shape[1:] != (h, w):
        resized = []
        for i in range(stack.shape[0]):
            s = cv2.resize(stack[i].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(s)
        stack = np.stack(resized, axis=0)

    scores = stack.astype(np.float32)
    max_scores = scores.max(axis=0)
    max_inds = np.argmax(scores, axis=0)

    id_map_raw = np.zeros((h, w), dtype=np.int32)
    fg = max_scores > 0
    id_map_raw[fg] = (obj_id_start + max_inds[fg]).astype(np.int32)
    return id_map_raw


def build_raw_id_map_from_frame0_prompt(first_obj_masks: Dict[int, np.ndarray], h: int, w: int) -> np.ndarray:
    if len(first_obj_masks) == 0:
        return np.zeros((h, w), dtype=np.int32)

    sorted_oids = sorted(first_obj_masks.keys())
    stack = []
    for oid in sorted_oids:
        m = first_obj_masks[oid]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        stack.append((m > 0).astype(np.float32))

    stack = np.stack(stack, axis=0)
    max_scores = stack.max(axis=0)
    max_inds = np.argmax(stack, axis=0)

    id_map_raw = np.zeros((h, w), dtype=np.int32)
    fg = max_scores > 0
    oid_arr = np.array(sorted_oids, dtype=np.int32)
    id_map_raw[fg] = oid_arr[max_inds[fg]]
    return id_map_raw


# -------------------------
# 후처리(공간)
# -------------------------
def cut_thin_bridges(mask: np.ndarray, min_thickness: int = 5) -> np.ndarray:
    if min_thickness <= 1:
        return (mask > 0).astype(np.uint8)

    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
    keep = (dist >= (min_thickness / 2.0)).astype(np.uint8)

    if keep.sum() < max(20, int(m.sum() * 0.15)):
        return m
    return keep


def split_and_filter_components(m: np.ndarray, min_obj_area: int, min_fraction_remain: float, orig_area: int) -> np.ndarray:
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)

    for cc in range(1, num):
        area = int(stats[cc, cv2.CC_STAT_AREA])
        if area < min_obj_area:
            continue
        out[lbl == cc] = 1

    remain = int(out.sum())
    min_allow = max(min_obj_area, int(orig_area * min_fraction_remain))
    if remain < min_allow:
        return np.zeros_like(m, dtype=np.uint8)

    return out


def fill_small_holes(m: np.ndarray, min_hole_area: int) -> np.ndarray:
    h, w = m.shape
    inv = (1 - (m > 0).astype(np.uint8)).astype(np.uint8)
    num2, lbl2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    out = (m > 0).astype(np.uint8).copy()
    for cc in range(1, num2):
        x = int(stats2[cc, cv2.CC_STAT_LEFT])
        y = int(stats2[cc, cv2.CC_STAT_TOP])
        ww = int(stats2[cc, cv2.CC_STAT_WIDTH])
        hh = int(stats2[cc, cv2.CC_STAT_HEIGHT])
        area = int(stats2[cc, cv2.CC_STAT_AREA])

        touches_border = (x == 0) or (y == 0) or (x + ww == w) or (y + hh == h)
        if (not touches_border) and (area <= min_hole_area):
            out[lbl2 == cc] = 1
    return out


def refine_id_map_tracking_v2(
    id_map: np.ndarray,
    min_obj_area: int = 3000,
    min_fraction_remain: float = 0.1,
    open_kernel: int = 3,
    close_kernel: int = 7,
    min_hole_area: int = 500,
    min_thickness: int = 5,
) -> np.ndarray:
    h, w = id_map.shape
    out = np.zeros((h, w), dtype=np.int32)

    ids = np.unique(id_map)
    ids = ids[ids > 0]

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)) if open_kernel and open_kernel > 1 else None
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)) if close_kernel and close_kernel > 1 else None

    for oid in ids:
        m0 = (id_map == oid).astype(np.uint8)
        orig_area = int(m0.sum())
        if orig_area == 0:
            continue

        m = m0.copy()
        if k_open is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
        if k_close is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)

        # ✅ ablation: 얇은 다리 끊기
        if ENABLE_CUT_THIN:
            m = cut_thin_bridges(m, min_thickness=min_thickness)

        # ✅ ablation: 컴포넌트 분리 + 필터
        if ENABLE_SPLIT_FILTER:
            m = split_and_filter_components(m, min_obj_area, min_fraction_remain, orig_area)

        if m.sum() == 0:
            continue

        m = fill_small_holes(m, min_hole_area=min_hole_area)
        out[m > 0] = int(oid)

    return out


# -------------------------
# 후처리(시간)
# -------------------------
def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_centroid(m: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def temporal_id_consistency_step(
    curr_id_map: np.ndarray,
    prev_id_map: Optional[np.ndarray],
    iou_keep_thresh: float = 0.15,
    max_center_shift: float = 80.0,
    allow_new_ids: bool = True,
) -> np.ndarray:
    if prev_id_map is None:
        return curr_id_map

    out = curr_id_map.copy()
    ids = np.unique(out)
    ids = ids[ids > 0]

    for oid in ids:
        cur = (out == oid).astype(np.uint8)
        prev = (prev_id_map == oid).astype(np.uint8)

        if cur.sum() == 0:
            continue

        if prev.sum() == 0:
            if not allow_new_ids:
                out[cur > 0] = 0
            continue

        ov = iou_binary(cur, prev)
        if ov >= iou_keep_thresh:
            continue

        c_cur = mask_centroid(cur)
        c_prev = mask_centroid(prev)
        if (c_cur is None) or (c_prev is None):
            out[cur > 0] = 0
            continue

        dx = c_cur[0] - c_prev[0]
        dy = c_cur[1] - c_prev[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist > max_center_shift:
            out[cur > 0] = 0

    return out


def enforce_single_component_per_id(
    curr_id_map: np.ndarray,
    prev_id_map: Optional[np.ndarray] = None,
    min_keep_area: int = 300,
    iou_pick_thresh: float = 0.01,
) -> np.ndarray:
    out = np.zeros_like(curr_id_map, dtype=np.int32)
    ids = np.unique(curr_id_map)
    ids = ids[ids > 0]

    for oid in ids:
        m = (curr_id_map == oid).astype(np.uint8)
        if m.sum() == 0:
            continue

        num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            out[m > 0] = int(oid)
            continue

        comps = []
        for cc in range(1, num):
            area = int(stats[cc, cv2.CC_STAT_AREA])
            if area < min_keep_area:
                continue
            c = (lbl == cc).astype(np.uint8)
            comps.append((cc, area, c))

        if len(comps) == 0:
            best_cc = 1 + np.argmax([int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, num)])
            out[lbl == best_cc] = int(oid)
            continue

        chosen_mask = None
        if prev_id_map is not None:
            prev = (prev_id_map == oid).astype(np.uint8)
            if prev.sum() > 0:
                best_iou = -1.0
                best_c = None
                for _, _, c in comps:
                    inter = np.logical_and(c > 0, prev > 0).sum()
                    union = np.logical_or(c > 0, prev > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_c = c
                if best_c is not None and best_iou >= iou_pick_thresh:
                    chosen_mask = best_c

        if chosen_mask is None:
            comps_sorted = sorted(comps, key=lambda x: x[1], reverse=True)
            chosen_mask = comps_sorted[0][2]

        out[chosen_mask > 0] = int(oid)

    return out


# -------------------------
# 변환/저장
# -------------------------
def build_id_outputs_from_saved_masks(
    image_dir_: str,
    masks_out_dir: str,
    frame_files: List[str],
    out_dir: str,
    force_hw: Tuple[int, int],
    first_obj_masks: Dict[int, np.ndarray],
    prompt_frame_idx: int = 0,
    save_raw_instances: bool = True,
    raw_save_overlay: bool = True,
    raw_overlay_alpha: float = 0.55,
    raw_stack_threshold: float = 0.0,
    min_obj_area: int = 3000,
    min_hole_area: int = 500,
    close_kernel: int = 7,
    open_kernel: int = 5,
    min_fraction_remain: float = 0.1,
    min_thickness: int = 5,
    iou_keep_thresh: float = 0.15,
    max_center_shift: float = 80.0,
    allow_new_ids: bool = True,
):
    h, w = force_hw

    idmap_npy_dir = os.path.join(out_dir, "id_maps")
    inst_npz_dir = os.path.join(out_dir, "instance_masks")
    idmap_color_dir = os.path.join(out_dir, "id_maps_color")
    os.makedirs(idmap_npy_dir, exist_ok=True)
    os.makedirs(inst_npz_dir, exist_ok=True)
    os.makedirs(idmap_color_dir, exist_ok=True)

    saved_paths = collect_saved_mask_paths(masks_out_dir)
    if len(saved_paths) == 0:
        raise RuntimeError(f"No mask files found in: {masks_out_dir}")

    saved_path_by_frame_idx: Dict[int, str] = {}
    for p in saved_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            idx = int(base)
        except ValueError:
            idx = extract_first_int(base)
        saved_path_by_frame_idx[int(idx)] = p

    n = len(frame_files)
    prev_id_map = None
    obj_ids_all = sorted([int(k) for k in first_obj_masks.keys()])

    for i in range(n):
        stem = os.path.splitext(frame_files[i])[0]
        p = saved_path_by_frame_idx.get(i)

        if i == 0:
            if p is None:
                print("[DEBUG] no saved raw mask file for frame index 0")
            elif p.lower().endswith(".npy"):
                print("[DEBUG] saved mask path:", p)
                arr = np.load(p)
                print(
                    "[DEBUG] npy shape:",
                    arr.shape,
                    "dtype:",
                    arr.dtype,
                    "min/max:",
                    float(arr.min()),
                    float(arr.max()),
                )
            else:
                print("[DEBUG] saved mask path:", p)
                arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                print(
                    "[DEBUG] img shape:",
                    arr.shape,
                    "dtype:",
                    arr.dtype,
                    "min/max:",
                    int(arr.min()),
                    int(arr.max()),
                )

        # ✅ raw_instances (WTA 없이)
        if save_raw_instances:
            if i == prompt_frame_idx:
                root = os.path.join(out_dir, "raw_instances")
                os.makedirs(root, exist_ok=True)

                frame = None
                if raw_save_overlay:
                    fp = os.path.join(image_dir_, frame_files[i])
                    frame = cv2.imread(fp, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError(f"Cannot read frame: {fp}")
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                for oid in obj_ids_all:
                    m = (first_obj_masks[int(oid)] > 0).astype(np.uint8)
                    obj_dir = os.path.join(root, f"obj_{int(oid):04d}")
                    mdir = os.path.join(obj_dir, "masks")
                    os.makedirs(mdir, exist_ok=True)
                    cv2.imwrite(os.path.join(mdir, f"{stem}.png"), (m * 255).astype(np.uint8))

                    if raw_save_overlay:
                        odir = os.path.join(obj_dir, "overlays")
                        os.makedirs(odir, exist_ok=True)
                        ov = overlay_single_mask(frame, m, oid=int(oid), alpha=raw_overlay_alpha)
                        cv2.imwrite(os.path.join(odir, f"{stem}.png"), ov)
            elif p is not None:
                save_raw_instances_without_wta(
                    image_dir_=image_dir_,
                    frame_files=frame_files,
                    i=i,
                    saved_mask_path=p,
                    out_dir=out_dir,
                    obj_ids_all=obj_ids_all,
                    h=h,
                    w=w,
                    bin_thresh=raw_stack_threshold,
                    save_overlay=raw_save_overlay,
                    overlay_alpha=raw_overlay_alpha,
                )
            else:
                print(f"[Warn] raw mask not found for frame index {i}; skip RAW export.")

        # ---- 후처리 파이프용 raw id_map (stack이면 WTA) ----
        if i == prompt_frame_idx:
            id_map_raw = build_raw_id_map_from_frame0_prompt(first_obj_masks, h, w)
        else:
            if p is None:
                if prev_id_map is not None:
                    id_map_raw = prev_id_map.copy()
                    print(f"[Warn] mask not found for frame index {i}; use previous id_map.")
                else:
                    id_map_raw = np.zeros((h, w), dtype=np.int32)
                    print(f"[Warn] mask not found for frame index {i}; use empty id_map.")
            elif p.lower().endswith(".npy"):
                arr = np.load(p)
                if arr.ndim == 2:
                    if arr.shape != (h, w):
                        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
                    id_map_raw = arr.astype(np.int32)
                elif arr.ndim == 3:
                    try:
                        stack = to_stack_nhw(arr, h, w)
                        id_map_raw = build_raw_id_map_winner_takes_all_from_stack(
                            stack, obj_id_start=OBJ_ID_START, h=h, w=w
                        )
                    except Exception:
                        if arr.shape[:2] == (h, w) and arr.shape[-1] in (3, 4):
                            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
                            id_map_raw = gray.astype(np.int32)
                        else:
                            raise RuntimeError(f"Unsupported 3D mask shape: {arr.shape} from {p}")
                else:
                    raise RuntimeError(f"Unsupported saved mask shape: {arr.shape} from {p}")
            else:
                arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise RuntimeError(f"Cannot read mask file: {p}")
                if arr.ndim == 2:
                    if arr.shape != (h, w):
                        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
                    id_map_raw = arr.astype(np.int32)
                elif arr.ndim == 3:
                    try:
                        stack = to_stack_nhw(arr, h, w)
                        id_map_raw = build_raw_id_map_winner_takes_all_from_stack(
                            stack, obj_id_start=OBJ_ID_START, h=h, w=w
                        )
                    except Exception:
                        if arr.shape[:2] == (h, w) and arr.shape[-1] in (3, 4):
                            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
                            id_map_raw = gray.astype(np.int32)
                        else:
                            raise RuntimeError(f"Unsupported 3D mask shape: {arr.shape} from {p}")
                else:
                    raise RuntimeError(f"Unsupported saved mask shape: {arr.shape} from {p}")

        # 공간
        id_map = refine_id_map_tracking_v2(
            id_map_raw,
            min_obj_area=min_obj_area,
            min_hole_area=min_hole_area,
            close_kernel=close_kernel,
            open_kernel=open_kernel,
            min_fraction_remain=min_fraction_remain,
            min_thickness=min_thickness,
        )

        # 시간 (✅ ablation)
        if ENABLE_TEMPORAL:
            id_map = temporal_id_consistency_step(
                curr_id_map=id_map,
                prev_id_map=prev_id_map,
                iou_keep_thresh=iou_keep_thresh,
                max_center_shift=max_center_shift,
                allow_new_ids=allow_new_ids,
            )

        # 단일 컴포넌트(이건 너가 원흉 후보에 안 넣었으니 그대로 둠)
        id_map = enforce_single_component_per_id(
            curr_id_map=id_map,
            prev_id_map=prev_id_map,
            min_keep_area=300,
            iou_pick_thresh=0.01,
        )

        # (옵션) 프레임별 dead id 출력
        if PRINT_DEAD_IDS_EACH_FRAME:
            dead = [oid for oid in obj_ids_all if int((id_map == oid).sum()) == 0]
            if len(dead) > 0:
                print(f"[dead @ {stem}] {len(dead)}/{len(obj_ids_all)} ids dead. ex) {dead[:10]}")

        prev_id_map = id_map.copy()

        # instance
        ids_arr, masks_arr = rebuild_instance_from_id_map(id_map)

        # 저장
        np.save(os.path.join(idmap_npy_dir, f"{stem}.npy"), id_map)
        np.savez_compressed(os.path.join(inst_npz_dir, f"{stem}.npz"), ids=ids_arr, masks=masks_arr)
        cv2.imwrite(os.path.join(idmap_color_dir, f"{stem}.png"), colorize_id_map(id_map))

    print(f"[Saved] converted {n} frames")
    if save_raw_instances:
        print(f"  - RAW(no WTA): {os.path.join(out_dir, 'raw_instances')}")
    print(f"  - POST: {idmap_npy_dir}, {inst_npz_dir}, {idmap_color_dir}")


def main():
    print(f"[image_dir] {image_dir}")
    prompt_mode = PROMPT_FRAME_MODE.lower().strip()
    if prompt_mode == "first":
        propagation_direction = "forward"
        run_subdir = "forward"
    elif prompt_mode == "last":
        propagation_direction = "backward"
        run_subdir = "backward"
    else:
        raise ValueError(f"PROMPT_FRAME_MODE must be 'first' or 'last', got: {PROMPT_FRAME_MODE}")

    masks_npy, prompt_image_name, prompt_frame_idx = find_masks_npy_from_image_dir(
        image_dir,
        prompt_frame_mode=prompt_mode,
    )
    print(f"[masks_npy] {masks_npy}")
    print(f"[prompt_image] {prompt_image_name}")
    print(f"[prompt_frame_idx] {prompt_frame_idx}")
    print(f"[propagation_direction] {propagation_direction}")

    video_root = os.path.dirname(image_dir)
    track_root = os.path.join(video_root, "samgeo_track")
    out_dir = os.path.join(track_root, run_subdir)
    os.makedirs(track_root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out_dir] {out_dir}")

    frame_files = sorted_frame_files(image_dir)
    if len(frame_files) == 0:
        raise RuntimeError("No frames found.")
    h, w = get_first_frame_hw(image_dir)

    obj_masks = build_obj_masks_from_npy(masks_npy, h, w, obj_id_start=OBJ_ID_START)
    print(f"[prompt objects] {len(obj_masks)}")

    tracker = SamGeo3Video()

    if hasattr(tracker, "set_video"):
        tracker.set_video(image_dir)
    elif hasattr(tracker, "set_video_path"):
        tracker.set_video_path(image_dir)
    else:
        raise AttributeError("No set_video/set_video_path method found.")

    if hasattr(tracker, "init_tracker"):
        tracker.init_tracker(frame_idx=prompt_frame_idx)
    elif hasattr(tracker, "initialize_tracker"):
        tracker.initialize_tracker(frame_idx=prompt_frame_idx)
    else:
        print("[Warn] init API not found (version-dependent).")

    # -------------------------
    # frame0 prompts: pos(mask) + neg(ring points) + debug image
    # -------------------------
    per_obj_points = {}  # debug용

    if hasattr(tracker, "add_mask_prompt"):
        for oid in sorted(obj_masks.keys()):
            oid = int(oid)

            pos_pts = sample_positive_points_from_mask_like_add_mask_prompt(obj_masks[oid], NUM_POINTS)

            neg_pts = []
            if USE_NEGATIVE_PROMPTS and hasattr(tracker, "add_point_prompts"):
                neg_pts = sample_negative_points_near_mask(
                    mask_np=obj_masks[oid],
                    num_neg=NUM_NEG_POINTS,
                    ring_width=NEG_RING_WIDTH,
                    ring_gap=NEG_RING_GAP,
                    mode=NEG_SAMPLE_MODE,
                    seed=NEG_RANDOM_SEED + oid,
                )

            bad = sum(obj_masks[oid][y, x] > 0 for x, y in neg_pts)
            if bad > 0:
                print(f"[BUG] obj {oid}: negative points fell inside mask: {bad}/{len(neg_pts)}")

            per_obj_points[oid] = {"pos": pos_pts, "neg": neg_pts}

            # 1) positive
            tracker.add_mask_prompt(
                frame_idx=prompt_frame_idx,
                obj_id=oid,
                mask=obj_masks[oid],
                num_points=NUM_POINTS,
            )

            # 2) negative (same obj_id)
            if USE_NEGATIVE_PROMPTS and hasattr(tracker, "add_point_prompts") and len(neg_pts) > 0:
                tracker.add_point_prompts(
                    points=neg_pts,
                    labels=[0] * len(neg_pts),
                    obj_id=oid,
                    frame_idx=prompt_frame_idx,
                )

        if SAVE_PROMPT_POINTS_DEBUG:
            frame0_name = frame_files[prompt_frame_idx]
            save_frame0_prompt_points_debug(
                image_dir_=image_dir,
                frame0_name=frame0_name,
                out_dir=out_dir,
                per_obj_points=per_obj_points,
                radius=POINT_RADIUS,
                thickness=POINT_THICKNESS,
                out_name=PROMPT_POINTS_DEBUG_NAME,
            )

    elif hasattr(tracker, "add_masks_prompt"):
        tracker.add_masks_prompt(frame_idx=prompt_frame_idx, masks=obj_masks)
        print("[Warn] add_masks_prompt 사용 중: point debug/negative는 이 경로에서 적용하지 않음.")
    else:
        raise AttributeError("No add_mask_prompt/add_masks_prompt method found.")

    # propagate
    if hasattr(tracker, "predictor") and hasattr(tracker.predictor, "handle_stream_request"):
        outputs_per_frame = {}
        for response in tracker.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=tracker.session_id,
                propagation_direction=propagation_direction,
                start_frame_index=prompt_frame_idx,
                max_frame_num_to_track=MAX_FRAME_NUM_TO_TRACK,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        tracker.outputs_per_frame = outputs_per_frame
        print(
            f"Propagated masks to {len(outputs_per_frame)} frames "
            f"(direction={propagation_direction}, start={prompt_frame_idx})."
        )
    elif hasattr(tracker, "propagate"):
        tracker.propagate()
    elif hasattr(tracker, "track"):
        tracker.track()
    else:
        raise AttributeError("No propagate/track method found.")

    # raw masks 저장
    masks_out = os.path.join(out_dir, "masks")
    os.makedirs(masks_out, exist_ok=True)
    if hasattr(tracker, "save_masks"):
        tracker.save_masks(masks_out)
    else:
        raise AttributeError("save_masks not available in this version.")

    if DEBUG_RAW_INSPECT:
        inspect_raw_saved_mask(
            masks_out_dir=masks_out,
            frame_idx=RAW_INSPECT_FRAME_IDX,
            num_frames=len(frame_files),
            obj_ids=RAW_INSPECT_OBJ_IDS,
            max_unique=RAW_INSPECT_MAX_UNIQUE,
        )

    # outputs
    build_id_outputs_from_saved_masks(
        image_dir_=image_dir,
        masks_out_dir=masks_out,
        frame_files=frame_files,
        out_dir=out_dir,
        force_hw=(h, w),
        first_obj_masks=obj_masks,
        prompt_frame_idx=prompt_frame_idx,
        save_raw_instances=SAVE_RAW_INSTANCES,
        raw_save_overlay=RAW_SAVE_OVERLAY,
        raw_overlay_alpha=RAW_OVERLAY_ALPHA,
        raw_stack_threshold=RAW_STACK_THRESHOLD,
        min_obj_area=MIN_OBJ_AREA,
        min_hole_area=MIN_HOLE_AREA,
        close_kernel=CLOSE_KERNEL,
        open_kernel=OPEN_KERNEL,
        min_fraction_remain=MIN_FRACTION_REMAIN,
        min_thickness=MIN_THICKNESS,
        iou_keep_thresh=IOU_KEEP_THRESH,
        max_center_shift=MAX_CENTER_SHIFT,
        allow_new_ids=ALLOW_NEW_IDS,
    )

    # video (postprocessed)
    if SAVE_VIDEO:
        idmap_npy_dir = os.path.join(out_dir, "id_maps")
        video_out = os.path.join(out_dir, "tracked_postprocessed.mp4")
        save_postprocessed_video_from_id_maps(
            image_dir_=image_dir,
            frame_files=frame_files,
            idmap_npy_dir=idmap_npy_dir,
            video_out_path=video_out,
            fps=VIDEO_FPS,
            alpha=VIDEO_ALPHA,
        )

    print("[Done]")
    print(f"  out_dir: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset root path (.../videoXX_YYYYY). If set, image_dir becomes dataset_path/images.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Direct images folder path. If set, this overrides --dataset_path.",
    )
    parser.add_argument(
        "--run_both_directions",
        action="store_true",
        default=True,
        help="Run both forward(first prompt) and backward(last prompt) in one command (default: True).",
    )
    parser.add_argument(
        "--single_direction",
        dest="run_both_directions",
        action="store_false",
        help="Run only one direction using current PROMPT_FRAME_MODE.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["first", "last"],
        default=None,
        help="Prompt frame mode for single-direction run: first(forward) or last(backward).",
    )
    args = parser.parse_args()

    if args.image_dir:
        image_dir = args.image_dir
    elif args.dataset_path:
        image_dir = os.path.join(args.dataset_path, "images")

    if args.run_both_directions:
        for mode in ["first", "last"]:
            PROMPT_FRAME_MODE = mode
            print(f"\n[Run] PROMPT_FRAME_MODE={PROMPT_FRAME_MODE}")
            main()
    else:
        if args.prompt_mode is not None:
            PROMPT_FRAME_MODE = args.prompt_mode
            print(f"\n[Run] PROMPT_FRAME_MODE={PROMPT_FRAME_MODE}")
        main()
