# sam3_seg_first_last.py
"""
입력
- image_dir: 비디오 프레임 폴더 (.../videoXX/images)
- fov mask dir: image_dir의 "/images"를 "/masks"로 바꾼 폴더
  예) .../videoXX/masks/00000.png  (FOV/가시영역 마스크, 0~255)

처리(프레임별: 첫 프레임 + 마지막 프레임)
1) SAM3 image model 로드 + Sam3Processor 준비
2) grid points 생성 → predict_inst_batch로 point prompt segmentation
3) 후보 마스크 score/area 필터
4) 경계 roughness 기반 정제 + bad mask 제거
5) mask IoU NMS로 중복 제거
6) make_masks_disjoint로 서로 겹치지 않게 정리
7) FOV 마스크 적용(보이는 영역 밖 제거)
8) FOV 내부에서 큰 hole(빈 영역) 탐지하여 마스크로 추가
9) 마스크/시각화 저장

출력(프레임별 저장 경로)
- parent_dir = dirname(image_dir)  # .../videoXX
- out_dir = parent_dir/image_sam3_seg/<frame_stem>/
  예) .../video01_00080/image_sam3_seg/00000/

저장 파일
- mask_000.png, mask_001.png, ... : 각 마스크 (uint8 binary, 0/255)
- masks_stack.npy : (N,H,W) uint8 마스크 스택
- vis_final_masks_inside_fov.png : 최종 마스크 시각화 (FOV 적용 + holes 포함)
  ※ vis_all_masks_inside_fov.png는 저장하지 않음
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ===== SAM3 import =====
sam3_root = "/home/jihun/PycharmProjects/SurgTPGS/submodules/sam3"
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)

import sam3  # noqa: E402
from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

# ==== 사용자 설정 ====
image_dir = "/home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video01_00080/images"


# ======================
# 유틸
# ======================
def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18


def sorted_frame_files(image_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    return sorted(files, key=extract_first_int)


def infer_fov_mask_path(image_dir: str, image_name: str) -> str:
    """
    images/XXXX.jpg  -> masks/XXXXX.png 형태로 가정.
    - 파일명이 00000.jpg면 frame_idx=0
    - 파일명이 frame_000080_endo.jpg여도 첫 숫자 80 사용
    """
    mask_dir = image_dir.replace("/images", "/masks")
    frame_idx = extract_first_int(os.path.splitext(image_name)[0])
    return os.path.join(mask_dir, f"{frame_idx:05d}.png")


def mask_to_box_xyxy(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1), float(y1)]


def box_xyxy_to_xywh(box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    w = x1 - x0
    h = y1 - y0
    return [x0, y0, w, h]


def _to_u8_binary(mask: np.ndarray) -> np.ndarray:
    m = mask
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    return m


def save_extracted_masks_minimal(
    image_dir: str,
    image_name: str,
    anns: List[Dict[str, Any]],
) -> str:
    parent_dir = os.path.dirname(image_dir)            # .../video01_00080
    image_stem = os.path.splitext(image_name)[0]       # 00000 or frame_000080_endo
    out_dir = os.path.join(parent_dir, "image_sam3_seg", image_stem)
    os.makedirs(out_dir, exist_ok=True)

    masks_u8 = []
    for i, ann in enumerate(anns):
        m = _to_u8_binary(ann["segmentation"])
        masks_u8.append(m)
        cv2.imwrite(os.path.join(out_dir, f"mask_{i:03d}.png"), m)

    if len(masks_u8) > 0:
        masks_stack = np.stack(masks_u8, axis=0)       # (N,H,W)
    else:
        masks_stack = np.zeros((0, 0, 0), dtype=np.uint8)

    np.save(os.path.join(out_dir, "masks_stack.npy"), masks_stack)

    print(f"[Save] dir={out_dir}")
    print(f"[Save] num_masks={len(masks_u8)}")
    return out_dir


def show_mask(mask, ax, random_color=False, borders=True, border_thickness: float = 1.0):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.35])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.35])

    h, w = mask.shape[-2:]
    mask_u8 = mask.astype(np.uint8)
    mask_img = mask_u8.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)

    if borders:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.squeeze(1)
            if c.ndim != 2 or c.shape[0] < 2:
                continue
            ax.plot(c[:, 0], c[:, 1], linewidth=border_thickness, color="yellow")


def save_final_seg_visualization(
    image_dir: str,
    image_name: str,
    img_np: np.ndarray,
    anns_final: List[Dict[str, Any]],
) -> str:
    """
    저장:
    .../videoXX/image_sam3_seg/<frame_stem>/vis_final_masks_inside_fov.png
    (vis_all_masks_inside_fov 는 저장 안 함)
    """
    parent_dir = os.path.dirname(image_dir)
    image_stem = os.path.splitext(image_name)[0]
    out_dir = os.path.join(parent_dir, "image_sam3_seg", image_stem)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.imshow(img_np)
    for ann in anns_final:
        show_mask(ann["segmentation"], ax, random_color=True, borders=True, border_thickness=1.0)
    ax.set_title("Masks INSIDE FOV (after NMS + holes)")
    ax.axis("off")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "vis_final_masks_inside_fov.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Save] final vis: {save_path}")
    return save_path


# ======================
# 마스크 후처리
# ======================
def make_masks_disjoint(
    anns: List[Dict[str, Any]],
    min_area_remain: int = 500,
    min_fraction_remain: float = 0.1,
) -> List[Dict[str, Any]]:
    if len(anns) == 0:
        return anns

    H, W = anns[0]["segmentation"].shape
    taken = np.zeros((H, W), dtype=bool)

    areas = [a["area"] for a in anns]
    order = np.argsort(areas)

    new_anns = []
    for idx in order:
        ann = anns[idx]
        m = ann["segmentation"].astype(bool)
        orig_area = int(ann.get("area", m.sum()))

        m_new = m & (~taken)
        remain_area_total = int(m_new.sum())
        if remain_area_total == 0:
            continue

        min_allow_area = max(min_area_remain, int(orig_area * min_fraction_remain))
        if remain_area_total < min_allow_area:
            continue

        num_cc, labels = cv2.connectedComponents(m_new.astype(np.uint8))
        for lab in range(1, num_cc):
            part = (labels == lab)
            area = int(part.sum())
            if area < min_area_remain:
                continue

            box_xyxy = mask_to_box_xyxy(part)
            if box_xyxy is None:
                continue
            bbox_xywh = box_xyxy_to_xywh(box_xyxy)

            ann2 = ann.copy()
            ann2["segmentation"] = part
            ann2["area"] = area
            ann2["bbox"] = bbox_xywh
            new_anns.append(ann2)

            taken |= part

    return new_anns


def smooth_mask_bool(mask_bool: np.ndarray,
                     ksize: int = 7,
                     min_island_area: int = 300,
                     min_hole_area: int = 300) -> np.ndarray:
    m = mask_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_cc, labels = cv2.connectedComponents(m)
    for i in range(1, num_cc):
        area = (labels == i).sum()
        if area < min_island_area:
            m[labels == i] = 0

    inv = 1 - m
    num_cc_inv, labels_inv = cv2.connectedComponents(inv)
    for i in range(1, num_cc_inv):
        area = (labels_inv == i).sum()
        if area < min_hole_area:
            inv[labels_inv == i] = 0
    m = 1 - inv
    return m.astype(bool)


def mask_edge_roughness_and_refine(mask_bool: np.ndarray, ksize: int = 7) -> Tuple[float, np.ndarray]:
    m = mask_bool.astype(np.uint8)
    area = m.sum()
    if area == 0:
        return 1.0, mask_bool

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m_smooth = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m_smooth = cv2.morphologyEx(m_smooth, cv2.MORPH_CLOSE, kernel, iterations=1)

    diff = cv2.bitwise_xor(m, m_smooth)
    noisy = diff.sum()
    rough = float(noisy / (area + 1e-6))
    rough = float(max(0.0, min(1.0, rough)))
    return rough, m_smooth.astype(bool)


def _remove_small_connected_components(mask_bool: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask_bool
    m = mask_bool.astype(np.uint8)
    num_cc, labels = cv2.connectedComponents(m)
    if num_cc <= 1:
        return mask_bool
    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_cc):
        part = (labels == lab)
        if int(part.sum()) >= min_component_area:
            out[part] = 1
    return out.astype(bool)


def _is_too_thin_by_erosion(
    mask_bool: np.ndarray,
    erosion_ksize: int,
    min_eroded_area: int,
    min_survival_ratio: float,
) -> bool:
    if erosion_ksize <= 1:
        return False
    area = int(mask_bool.sum())
    if area == 0:
        return True

    m = mask_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_ksize, erosion_ksize))
    eroded = cv2.erode(m, kernel, iterations=1)
    eroded_area = int(eroded.sum())
    survival_ratio = eroded_area / float(area + 1e-6)

    if eroded_area < min_eroded_area:
        return True
    if survival_ratio < min_survival_ratio:
        return True
    return False


def apply_visible_fov_mask(
    anns: List[Dict[str, Any]],
    fov_mask_path: str,
    img_shape,
    min_area_after_fov: int = 0,
    min_component_area_after_fov: int = 0,
    thin_erosion_ksize: int = 0,
    thin_min_eroded_area: int = 0,
    thin_min_survival_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    if not os.path.exists(fov_mask_path):
        raise FileNotFoundError(f"FOV mask path not found: {fov_mask_path}")

    fov_pil = Image.open(fov_mask_path).convert("L")
    fov = np.array(fov_pil)

    H, W = img_shape[:2]
    if fov.shape[:2] != (H, W):
        fov = cv2.resize(fov, (W, H), interpolation=cv2.INTER_NEAREST)

    fov_bool = fov > 127

    filtered = []
    dropped_small = 0
    dropped_thin = 0
    for ann in anns:
        m = ann["segmentation"].astype(bool)
        m_in = m & fov_bool
        if min_component_area_after_fov > 0:
            m_in = _remove_small_connected_components(m_in, min_component_area_after_fov)

        area_in = int(m_in.sum())
        if area_in == 0:
            continue
        if min_area_after_fov > 0 and area_in < min_area_after_fov:
            dropped_small += 1
            continue
        if _is_too_thin_by_erosion(
            m_in,
            erosion_ksize=thin_erosion_ksize,
            min_eroded_area=thin_min_eroded_area,
            min_survival_ratio=thin_min_survival_ratio,
        ):
            dropped_thin += 1
            continue

        ann2 = ann.copy()
        ann2["segmentation"] = m_in
        ann2["area"] = area_in

        box_xyxy = mask_to_box_xyxy(m_in)
        if box_xyxy is None:
            continue
        ann2["bbox"] = box_xyxy_to_xywh(box_xyxy)
        filtered.append(ann2)

    print(
        f"[FOV post] kept={len(filtered)} "
        f"(dropped_small={dropped_small}, dropped_thin={dropped_thin})"
    )
    return filtered


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_nms_np(masks_bool: np.ndarray, scores: np.ndarray, iou_thr: float = 0.7) -> np.ndarray:
    N = masks_bool.shape[0]
    if N == 0:
        return np.array([], dtype=np.int32)

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([mask_iou(masks_bool[i], masks_bool[j]) for j in rest], dtype=np.float32)
        rest = rest[ious <= iou_thr]
        order = rest

    return np.array(keep, dtype=np.int32)


def find_big_holes_in_fov(
    fov_bool: np.ndarray,
    anns: List[Dict[str, Any]],
    area_thr: int = 5000,
    radius_thr: int = 10,
) -> List[np.ndarray]:
    H, W = fov_bool.shape
    covered = np.zeros((H, W), dtype=bool)
    for ann in anns:
        covered |= ann["segmentation"].astype(bool)

    holes_raw = fov_bool & (~covered)
    if holes_raw.sum() == 0:
        return []

    dist = cv2.distanceTransform(holes_raw.astype(np.uint8), cv2.DIST_L2, 3)
    core = dist >= radius_thr
    if core.sum() == 0:
        return []

    num_labels, labels = cv2.connectedComponents(core.astype(np.uint8))
    big_holes = []
    for lab in range(1, num_labels):
        core_comp = (labels == lab)
        dilated = cv2.dilate(core_comp.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        mask_hole = holes_raw & dilated
        if mask_hole.sum() < area_thr:
            continue
        big_holes.append(mask_hole)

    return big_holes


# ======================
# SAM3 Grid 자동 분할기
# ======================
def generate_grid_points(width: int, height: int, points_per_side: int = 32, margin: int = 0) -> np.ndarray:
    xs = np.linspace(margin, width - 1 - margin, points_per_side, dtype=np.float32)
    ys = np.linspace(margin, height - 1 - margin, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    return pts


class Sam3GridMaskGenerator:
    def __init__(
        self,
        model,
        processor: Sam3Processor,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        score_thresh: float = 0.0,
        iou_nms_thresh: float = 0.7,
        min_mask_area: int = 0,
        smooth_ksize: int = 10,
        smooth_min_island: int = 3000,
        smooth_min_hole: int = 3000,
        rough_ksize: int = 15,
        rough_thr: float = 0.2,
        disjoint_min_area: int = 2000,
    ):
        self.model = model
        self.processor = processor
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.score_thresh = score_thresh
        self.iou_nms_thresh = iou_nms_thresh
        self.min_mask_area = min_mask_area

        self.smooth_ksize = smooth_ksize
        self.smooth_min_island = smooth_min_island
        self.smooth_min_hole = smooth_min_hole

        self.rough_ksize = rough_ksize
        self.rough_thr = rough_thr

        self.disjoint_min_area = disjoint_min_area

    @torch.no_grad()
    def generate(self, img_np: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
        pil_img = Image.fromarray(img_np).convert("RGB")
        H, W = img_np.shape[:2]

        inference_state = self.processor.set_image_batch([pil_img])

        grid_points = generate_grid_points(W, H, self.points_per_side, margin=0)
        N = len(grid_points)
        print(f"[Sam3GridMaskGenerator] total grid points: {N}")

        all_masks, all_scores, all_points = [], [], []
        for start in range(0, N, self.points_per_batch):
            end = min(start + self.points_per_batch, N)
            pts_chunk = grid_points[start:end]  # (M,2)

            labels_chunk = np.ones((len(pts_chunk), 1), dtype=np.int32)
            pts_chunk_batched = pts_chunk.reshape(-1, 1, 2)

            pts_batch = [pts_chunk_batched]
            labels_batch = [labels_chunk]

            masks_batch, scores_batch, _ = self.model.predict_inst_batch(
                inference_state,
                pts_batch,
                labels_batch,
                box_batch=None,
                multimask_output=True,
            )

            masks = masks_batch[0]   # (M,K,H,W)
            scores = np.asarray(scores_batch[0])  # (M,K)

            masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)

            best_idx = np.argmax(scores, axis=-1)
            best_masks = masks_np[np.arange(len(masks_np)), best_idx]  # (M,H,W)
            best_scores = scores[np.arange(len(scores)), best_idx]     # (M,)

            best_masks = best_masks > 0.3

            # smoothing
            best_masks = np.stack(
                [
                    smooth_mask_bool(
                        m,
                        ksize=self.smooth_ksize,
                        min_island_area=self.smooth_min_island,
                        min_hole_area=self.smooth_min_hole,
                    )
                    for m in best_masks
                ],
                axis=0,
            )

            all_masks.append(best_masks)
            all_scores.append(best_scores)
            all_points.append(pts_chunk)

            print(f"  processed points {start}-{end} / {N}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        masks_all = np.concatenate(all_masks, axis=0)
        scores_all = np.concatenate(all_scores, axis=0)
        points_all = np.concatenate(all_points, axis=0)

        # score/area filter
        keep = np.ones(len(masks_all), dtype=bool)
        if self.score_thresh > 0.0:
            keep &= scores_all >= self.score_thresh
        areas = masks_all.reshape(len(masks_all), -1).sum(axis=1)
        if self.min_mask_area > 0:
            keep &= areas >= self.min_mask_area

        masks_f = masks_all[keep]
        scores_f = scores_all[keep]
        points_f = points_all[keep]
        areas_f = areas[keep]
        print(f"after score/area filter: {len(masks_f)} masks")

        # roughness refine + filter
        refined = [mask_edge_roughness_and_refine(m, ksize=self.rough_ksize) for m in masks_f]
        roughs = np.array([r[0] for r in refined], dtype=np.float32)
        masks_ref = np.array([r[1] for r in refined], dtype=bool)

        good = roughs <= self.rough_thr
        masks_ref = masks_ref[good]
        scores_ref = scores_f[good]
        points_ref = points_f[good]
        areas_ref = areas_f[good]
        print(f"after roughness filter: {len(masks_ref)} masks")

        # NMS
        keep_nms = mask_nms_np(masks_ref, scores_ref, iou_thr=self.iou_nms_thresh)
        masks_nms = masks_ref[keep_nms]
        scores_nms = scores_ref[keep_nms]
        points_nms = points_ref[keep_nms]
        areas_nms = areas_ref[keep_nms]
        print(f"after NMS: {len(masks_nms)} masks")

        def build_anns(masks_arr, scores_arr, points_arr, areas_arr):
            out = []
            for mask, score, pt, area in zip(masks_arr, scores_arr, points_arr, areas_arr):
                box_xyxy = mask_to_box_xyxy(mask)
                if box_xyxy is None:
                    continue
                out.append(
                    {
                        "segmentation": mask,
                        "area": int(area),
                        "bbox": box_xyxy_to_xywh(box_xyxy),
                        "score": float(score),
                        "point_coords": [pt.tolist()],
                    }
                )
            return out

        anns_all = build_anns(masks_f, scores_f, points_f, areas_f)
        anns_nms = build_anns(masks_nms, scores_nms, points_nms, areas_nms)

        # disjoint
        anns_nms = make_masks_disjoint(anns_nms, min_area_remain=self.disjoint_min_area)
        return anns_all, anns_nms, grid_points


# ======================
# 메인: first + last 저장
# ======================
def main():
    # generator hyperparams (원 코드 값 기반)
    points_per_side = 25
    points_per_batch = 64
    score_thresh = 0.3
    iou_nms_thresh = 0.3
    min_mask_area = 3000
    min_area_after_fov = 2000
    min_component_area_after_fov = 500
    thin_erosion_ksize = 7
    thin_min_eroded_area = 300
    thin_min_survival_ratio = 0.08

    # hole params
    hole_area_thr = 3000
    hole_radius_thr = 10

    # ==== device/autocast ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ==== SAM3 model load ====
    sam3_root_local = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root_local}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)

    # ==== files ====
    files_sorted = sorted_frame_files(image_dir)
    if len(files_sorted) == 0:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    targets = [files_sorted[0], files_sorted[-1]]
    print("[Targets]", targets)

    # ==== generator ====
    gen = Sam3GridMaskGenerator(
        model=model,
        processor=processor,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        score_thresh=score_thresh,
        iou_nms_thresh=iou_nms_thresh,
        min_mask_area=min_mask_area,
    )

    # ==== run each target ====
    for image_name in targets:
        image_path = os.path.join(image_dir, image_name)
        print("\n==============================")
        print(f"[Folder] {image_dir}")
        print(f"[Target Image] {image_name}")
        print(f"[Target Image Path] {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        anns_all, anns_nms, _ = gen.generate(img_np)
        print("before FOV:", len(anns_nms))

        fov_mask_path = infer_fov_mask_path(image_dir, image_name)
        print("[FOV Mask Path]", fov_mask_path)

        # vis_all은 더 이상 저장 안 하지만, 파이프라인상 계산은 필요없어서 생략 가능
        # 여기선 최종 결과만 필요하니 anns_nms만 FOV 적용
        anns_nms_fov = apply_visible_fov_mask(
            anns_nms,
            fov_mask_path,
            img_np.shape,
            min_area_after_fov=min_area_after_fov,
            min_component_area_after_fov=min_component_area_after_fov,
            thin_erosion_ksize=thin_erosion_ksize,
            thin_min_eroded_area=thin_min_eroded_area,
            thin_min_survival_ratio=thin_min_survival_ratio,
        )
        print("after  FOV:", len(anns_nms_fov))

        # FOV bool
        fov_pil = Image.open(fov_mask_path).convert("L")
        fov_np = np.array(fov_pil)
        H, W = img_np.shape[:2]
        if fov_np.shape[:2] != (H, W):
            fov_np = cv2.resize(fov_np, (W, H), interpolation=cv2.INTER_NEAREST)
        fov_bool = fov_np > 127

        # holes
        big_holes = find_big_holes_in_fov(
            fov_bool=fov_bool,
            anns=anns_nms_fov,
            area_thr=hole_area_thr,
            radius_thr=hole_radius_thr,
        )
        print("num big holes:", len(big_holes))

        hole_anns = []
        for h in big_holes:
            box_xyxy = mask_to_box_xyxy(h)
            if box_xyxy is None:
                continue
            hole_anns.append(
                {
                    "segmentation": h,
                    "area": int(h.sum()),
                    "bbox": box_xyxy_to_xywh(box_xyxy),
                    "score": 0.0,
                    "point_coords": [[]],
                }
            )

        anns_final = anns_nms_fov + hole_anns
        print(f"final masks (RRMD + holes): {len(anns_final)}")

        # 1) 마스크 저장
        saved_dir = save_extracted_masks_minimal(
            image_dir=image_dir,
            image_name=image_name,
            anns=anns_final,
        )
        print("[Saved Dir]", saved_dir)

        # 2) 최종 시각화 저장 (first/last 둘 다 저장됨)
        save_final_seg_visualization(
            image_dir=image_dir,
            image_name=image_name,
            img_np=img_np,
            anns_final=anns_final,
        )

    print("\n[Done] first + last segmentation saved.")


if __name__ == "__main__":
    main()
