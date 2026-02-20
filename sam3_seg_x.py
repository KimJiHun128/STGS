import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from matplotlib.patches import Circle
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
sam3_root = "/home/jihun/PycharmProjects/SurgTPGS/submodules/sam3"
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)
import inspect
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_model


# ======================
# 도우미 함수들
# ======================
def make_masks_disjoint(
    anns: List[Dict[str, Any]],
    min_area_remain: int = 500,
    min_fraction_remain: float = 0.1,   # 원래 area의 최소 10%는 남아야 유지
) -> List[Dict[str, Any]]:
    """
    anns 리스트 안의 segmentation 들이 서로 겹치지 않도록 정리.

    - 작은 마스크부터 순서대로 배치하고
    - 이미 사용된 픽셀은 잘라낸 뒤,
    - 잘린 결과가 여러 조각이면 각 조각을 '서로 다른 마스크'로 분리
    - 남은 전체 영역이 원래 area의 min_fraction_remain(기본 10%) 보다 작으면
      그 마스크는 통째로 버림(= 90% 이상 잘려 나간 애는 제거)
    - 각 조각의 면적이 min_area_remain 보다 작으면 노이즈로 버림
    """
    if len(anns) == 0:
        return anns

    H, W = anns[0]["segmentation"].shape
    taken = np.zeros((H, W), dtype=bool)   # 이미 다른 마스크가 차지한 픽셀

    # 작은 마스크부터 처리 (area 오름차순)
    areas = [a["area"] for a in anns]
    order = np.argsort(areas)  # 작은 것 → 큰 것

    new_anns = []
    for idx in order:
        ann = anns[idx]
        m = ann["segmentation"].astype(bool)

        # 원래 area
        orig_area = int(ann.get("area", m.sum()))

        # 이미 다른 마스크가 차지한 픽셀 제거
        m_new = m & (~taken)
        remain_area_total = int(m_new.sum())
        if remain_area_total == 0:
            continue

        # 🔴 90% 이상 잘려나갔으면 통째로 버리기
        #    (즉, 남은게 원래의 10% 미만이면 skip)
        min_allow_area = max(
            min_area_remain,
            int(orig_area * min_fraction_remain)
        )
        if remain_area_total < min_allow_area:
            # 아예 이 마스크는 의미 없다고 판단하고 넘어감
            continue

        # ⚠️ 남은 m_new 내부에서 disconnected component들을
        #     전부 '서로 다른 마스크'로 쪼갬
        num_cc, labels = cv2.connectedComponents(m_new.astype(np.uint8))
        for lab in range(1, num_cc):
            part = (labels == lab)
            area = int(part.sum())
            if area < min_area_remain:
                continue  # 너무 작은 조각은 버림(노이즈)

            box_xyxy = mask_to_box_xyxy(part)
            if box_xyxy is None:
                continue
            bbox_xywh = box_xyxy_to_xywh(box_xyxy)

            ann2 = ann.copy()
            ann2["segmentation"] = part
            ann2["area"] = area
            ann2["bbox"] = bbox_xywh
            new_anns.append(ann2)

            # 이 조각이 차지한 픽셀은 다른 마스크가 못 쓰게 막기
            taken |= part

    return new_anns

def mask_edge_roughness_and_refine(mask_bool: np.ndarray,
                                   ksize: int = 7) -> Tuple[float, np.ndarray]:
    """
    경계 노이즈 점수를 계산함과 동시에, 노이즈가 제거된 정제된 마스크를 반환합니다.
    """
    m = mask_bool.astype(np.uint8)
    area = m.sum()
    if area == 0:
        return 1.0, mask_bool  # 비어있으면 최악 점수와 원본 반환

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # 1. 부드러운 버전 만들기 (이미 구현하신 강력한 노이즈 제거 로직)
    # Opening: 바깥쪽 튀어나온 찌꺼기 제거
    # Closing: 안쪽 미세한 구멍 메우기
    m_smooth = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m_smooth = cv2.morphologyEx(m_smooth, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2. 원본과 부드러운 마스크의 차이로 roughness 계산
    diff = cv2.bitwise_xor(m, m_smooth)
    noisy = diff.sum()
    rough = noisy / (area + 1e-6)
    rough = float(max(0.0, min(1.0, rough)))

    # 3. 점수와 함께 '정제된 마스크'를 bool 타입으로 반환
    return rough, m_smooth.astype(bool)


def find_big_holes_in_fov(
    fov_bool: np.ndarray,
    anns: List[Dict[str, Any]],
    area_thr: int = 5000,
    radius_thr: int = 10,
) -> List[np.ndarray]:
    """
    FOV 안에서 아직 어떤 mask도 덮지 않은 '큰 빈 영역'만 찾아서 리턴.
    - fov_bool: (H,W) 내시경 FOV (True=보이는 영역)
    - anns: NMS 후 마스크 리스트 (segmentation: bool HxW)
    - area_thr: hole 최소 면적 (픽셀 수)
    - radius_thr: distance transform 반경 기준 (얇은 틈은 제외)
    return: [HxW bool] 리스트 (각각 하나의 hole)
    """
    H, W = fov_bool.shape

    # 1) 이미 마스크가 덮고 있는 영역의 union
    covered = np.zeros((H, W), dtype=bool)
    for ann in anns:
        covered |= ann["segmentation"].astype(bool)

    # 2) FOV 안이면서 아직 덮이지 않은 영역
    holes_raw = fov_bool & (~covered)
    if holes_raw.sum() == 0:
        return []

    # 3) distance transform으로 '두께'가 충분한 부분만 남기기
    dist = cv2.distanceTransform(holes_raw.astype(np.uint8), cv2.DIST_L2, 3)
    core = dist >= radius_thr  # 중심부 반경이 radius_thr 이상인 픽셀만

    if core.sum() == 0:
        return []

    # 4) core에서 connected component → 각 컴포넌트별로 holes_raw 영역 복원
    num_labels, labels = cv2.connectedComponents(core.astype(np.uint8))
    big_holes = []
    for lab in range(1, num_labels):
        core_comp = (labels == lab)

        # core를 살짝 dilate해서 원래 holes_raw와 AND
        dilated = cv2.dilate(core_comp.astype(np.uint8),
                             np.ones((3, 3), np.uint8),
                             iterations=1).astype(bool)
        mask_hole = holes_raw & dilated

        if mask_hole.sum() < area_thr:
            continue

        big_holes.append(mask_hole)

    return big_holes

def smooth_mask_bool(mask_bool,
                     ksize: int = 7,
                     min_island_area: int = 300,   # 너무 작은 섬 제거
                     min_hole_area: int = 300):    # 너무 작은 구멍 메우기
    """
    - closing으로 경계 다듬고
    - mask 안쪽의 작은 섬(튀는 픽셀) 제거
    - mask 내부의 작은 구멍을 메움
    """
    m = mask_bool.astype(np.uint8)

    # 1) 경계 부드럽게 (dilation 후 erosion = closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) mask 안의 작은 섬 제거 (너무 작게 튀어나온 blob 날리기)
    num_cc, labels = cv2.connectedComponents(m)
    for i in range(1, num_cc):
        area = (labels == i).sum()
        if area < min_island_area:
            m[labels == i] = 0

    # 3) mask 내부의 작은 구멍 메우기
    inv = 1 - m  # 0/1 반전
    num_cc_inv, labels_inv = cv2.connectedComponents(inv)
    for i in range(1, num_cc_inv):
        area = (labels_inv == i).sum()
        if area < min_hole_area:
            inv[labels_inv == i] = 0
    m = 1 - inv

    return m.astype(bool)


def generate_grid_points(width, height, points_per_side=32, margin=0):
    """
    width, height: 이미지 크기 (W, H)
    points_per_side: 한 변당 포인트 개수 → 총 points_per_side^2 개
    return: (N, 2) [x, y]
    """
    xs = np.linspace(margin, width - 1 - margin, points_per_side, dtype=np.float32)
    ys = np.linspace(margin, height - 1 - margin, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    return pts


def mask_to_box_xyxy(mask: np.ndarray):
    """bool mask → [x0, y0, x1, y1] (xyxy)"""
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

def apply_visible_fov_mask(anns, fov_mask_path, img_shape):
    # 1) 파일 자체 존재 여부 먼저 체크
    import os
    if not os.path.exists(fov_mask_path):
        raise FileNotFoundError(f"FOV mask path not found on disk: {fov_mask_path}")

    # 2) PIL로 로드 후 그레이스케일로 변환
    try:
        fov_pil = Image.open(fov_mask_path).convert("L")  # 1채널
    except Exception as e:
        raise RuntimeError(f"Cannot open FOV mask with PIL: {fov_mask_path}, error: {e}")

    fov = np.array(fov_pil)  # HxW, uint8

    H, W = img_shape[:2]
    if fov.shape[:2] != (H, W):
        # 내시경 마스크는 최근접 보간으로 리사이즈
        fov = cv2.resize(fov, (W, H), interpolation=cv2.INTER_NEAREST)

    # 흰 부분(255 근처)만 남기기
    fov_bool = fov > 127

    filtered = []
    for ann in anns:
        m = ann["segmentation"].astype(bool)
        m_in = m & fov_bool

        # 완전히 잘려 나간 마스크는 제거
        if m_in.sum() == 0:
            continue

        ann2 = ann.copy()
        ann2["segmentation"] = m_in
        ann2["area"] = int(m_in.sum())

        box_xyxy = mask_to_box_xyxy(m_in)
        if box_xyxy is None:
            continue
        ann2["bbox"] = box_xyxy_to_xywh(box_xyxy)

        filtered.append(ann2)

    return filtered

def mask_iou(m1: np.ndarray, m2: np.ndarray):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return inter / union


def mask_nms_np(masks_bool: np.ndarray, scores: np.ndarray, iou_thr: float = 0.7):
    """
    SAM v1 box NMS 대신, 그냥 mask IoU 기반 NMS 간단 구현
    masks_bool: (N, H, W)
    scores: (N,)
    """
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
        ious = np.array(
            [mask_iou(masks_bool[i], masks_bool[j]) for j in rest],
            dtype=np.float32,
        )
        rest = rest[ious <= iou_thr]
        order = rest

    return np.array(keep, dtype=np.int32)


def show_mask(mask, ax, random_color=False, borders=True,
              border_thickness: int = 3):
    """
    mask: HxW bool / 0-1
    border_thickness: 경계선 굵기 (3~5 정도 추천)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])  # 조금 더 투명

    h, w = mask.shape[-2:]
    mask_uint8 = mask.astype(np.uint8)

    # 내부는 기존처럼 반투명으로 채우기
    mask_image = mask_uint8.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    if borders:
        # 외곽 컨투어 찾기
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 테두리는 두꺼운 노란 선으로
        for c in contours:
            c = c.squeeze(1)  # (N,1,2) -> (N,2)
            if c.ndim != 2 or c.shape[0] < 2:
                continue
            xs = c[:, 0]
            ys = c[:, 1]
            ax.plot(xs, ys,
                    linewidth=border_thickness,
                    color='yellow')


# ======================
# SAM3 Grid 기반 자동 마스크 제너레이터
# ======================

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
    ):
        """
        SAM v1 SamAutomaticMaskGenerator 를 SAM3 point prompt 로 대충 흉내낸 버전.
        - crop 계층, stability score, RLE 같은 건 생략
        - grid point → SAM3 → best mask per point → IoU NMS 구조

        model: build_sam3_image_model(...) 로 로드한 이미지 모델
        processor: Sam3Processor(model)
        """
        self.model = model
        self.processor = processor
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.score_thresh = score_thresh
        self.iou_nms_thresh = iou_nms_thresh
        self.min_mask_area = min_mask_area

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]],
                                                   List[Dict[str, Any]],
                                                   np.ndarray]:
        """
        image: HWC uint8 (np.ndarray) 또는 PIL.Image 를 np.array 로 변환해서 넣어도 됨.
        return: [
          {
            "segmentation": HxW bool mask,
            "area": int,
            "bbox": [x, y, w, h],
            "score": float,           # SAM3 score (SAM v1 predicted_iou 비슷하게 사용)
            "point_coords": [[x, y]], # 이 마스크를 생성한 grid point
          }, ...
        ]
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            img_np = np.array(pil_img)
        else:
            img_np = image
            pil_img = Image.fromarray(img_np)

        H, W = img_np.shape[:2]

        # 1) 이미지 임베딩
        inference_state = self.processor.set_image_batch([pil_img])

        # 2) grid point 생성
        grid_points = generate_grid_points(W, H, self.points_per_side, margin=0)
        N = len(grid_points)
        print(f"[Sam3GridMaskGenerator] total grid points: {N}")

        # 3) batch 로 나눠서 SAM3 호출
        all_masks = []
        all_scores = []
        all_points = []

        for start in range(0, N, self.points_per_batch):
            end = min(start + self.points_per_batch, N)
            pts_chunk = grid_points[start:end]          # (M, 2)
            labels_chunk = np.ones((len(pts_chunk), 1), dtype=np.int32)  # (M, 1)
            pts_chunk_batched = pts_chunk.reshape(-1, 1, 2)              # (M,1,2)

            pts_batch = [pts_chunk_batched]
            labels_batch = [labels_chunk]

            masks_batch, scores_batch, _ = self.model.predict_inst_batch(
                inference_state,
                pts_batch,
                labels_batch,
                box_batch=None,
                multimask_output=True,
            )

            # 한 장짜리 이미지라 0번째
            masks = masks_batch[0]  # (M, K, H, W)  (torch or np)
            scores = scores_batch[0]  # (M, K)        (np.ndarray)

            # 둘 다 numpy로 통일
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = masks


            scores_np = np.asarray(scores)  # (M, K)

            # # 🔍 디버깅: 첫 배치에서 한 번만 찍기
            # if start == 0:
            #     print("masks_np shape:", masks_np.shape, "dtype:", masks_np.dtype)
            #     print("masks_np min/max:", masks_np.min(), masks_np.max())
            #     print("masks_np[0,0] unique:", np.unique(masks_np[0, 0]))
            #
            #     print("scores_np shape:", scores_np.shape,
            #           "min/max:", scores_np.min(), scores_np.max())

            # SAM3가 한 점(point)에 대해 여러 candidate mask (K개) 를 뱉은 상태
            # scores_np: 각 candidate mask의 점수 (일종의 quality / IoU-like score)
            # 각 point마다 score 최대 mask 하나만 사용
            best_idx = np.argmax(scores_np, axis=-1)  # (M,) 각 point마다 가장 score가 높은 mask 한 개
            best_masks_np = masks_np[np.arange(len(masks_np)), best_idx]  # (M, H, W)
            best_scores = scores_np[np.arange(len(scores_np)), best_idx]  # (M,)
            #
            # if start == 0:
            #     print("best_idx[0:10]:", best_idx[:10])
            #     print("best_masks_np shape:", best_masks_np.shape,
            #           "min/max:", best_masks_np.min(), best_masks_np.max())
            #     print("best_masks_np[0] unique:", np.unique(best_masks_np[0]))
            #

            # bool 마스크로 변환
            best_masks_np = best_masks_np > 0.3

            smooth_list = [
                smooth_mask_bool(
                    m,
                    ksize=10,  # 경계 부드럽게 정도
                    min_island_area=3000,  # 너무 작은 섬 제거 # <--------------------------------------------
                    min_hole_area=3000  # 너무 작은 구멍 메우기
                )
                for m in best_masks_np
            ]
            best_masks_np = np.stack(smooth_list, axis=0)




            all_masks.append(best_masks_np)  # (M, H, W)
            all_scores.append(best_scores)  # (M,)
            all_points.append(pts_chunk)  # (M, 2)

            print(f"  processed points {start}-{end} / {N}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        masks_all = np.concatenate(all_masks, axis=0)         # (N, H, W)
        scores_all = np.concatenate(all_scores, axis=0)       # (N,)
        points_all = np.concatenate(all_points, axis=0)       # (N, 2)

        # 4) score / area 필터링
        keep = np.ones(len(masks_all), dtype=bool)

        if self.score_thresh > 0.0:
            keep &= scores_all >= self.score_thresh

        areas = masks_all.reshape(len(masks_all), -1).sum(axis=1)
        if self.min_mask_area > 0: # 면적(픽셀 수)이 min_mask_area 이상인가
            keep &= areas >= self.min_mask_area

        masks_f = masks_all[keep]
        scores_f = scores_all[keep]
        points_f = points_all[keep]
        areas_f = areas[keep]

        print(f"after score/area filter: {len(masks_f)} masks")

        # NMS 하기 전 전체 마스크(겹침 포함)를 따로 보관
        masks_before = masks_f
        scores_before = scores_f
        points_before = points_f
        areas_before = areas_f

        # --- 경계 자글자글한 정도 계산 및 마스크 데이터 정제 ---
        refined_results = [mask_edge_roughness_and_refine(m, ksize=15) for m in masks_f]

        # 1. 계산된 점수들과 정제된 마스크들을 각각 분리하여 추출합니다.
        roughs = np.array([res[0] for res in refined_results])
        masks_f = np.array([res[1] for res in refined_results])  # 여기서 마스크가 정제된 버전으로 교체됨!

        # 2. 정제된 마스크들 중에서도 여전히 기준(0.2)보다 지저분한 놈들을 골라냅니다.
        bad_shape = roughs > 0.2                                         # <--------------------------------------------

        # # --- 경계 자글자글한 정도 계산 ---
        # roughs = np.array([mask_edge_roughness(m, ksize=15) for m in masks_f])
        #
        # # 너무 자글자글한 애들만 제거 (threshold는 0.2~0.4 정도에서 튜닝)
        # bad_shape = roughs > 0.2            # <--------------------------------------------

        keep_shape = ~bad_shape
        masks_f = masks_f[keep_shape]
        scores_f = scores_f[keep_shape]
        points_f = points_f[keep_shape]
        areas_f = areas_f[keep_shape]
        roughs = roughs[keep_shape]


        scores_for_nms = scores_f
        # -------------------------------------------

        # 5) IoU NMS (겹치는 것 제거한 버전)
        keep_nms = mask_nms_np(masks_f, scores_for_nms, iou_thr=self.iou_nms_thresh)
        masks_nms = masks_f[keep_nms]
        scores_nms = scores_f[keep_nms]
        points_nms = points_f[keep_nms]
        areas_nms = areas_f[keep_nms]

        print(f"after NMS: {len(masks_nms)} masks")

        def build_anns(masks, scores, points, areas):
            out = []
            for mask, score, pt, area in zip(masks, scores, points, areas):
                box_xyxy = mask_to_box_xyxy(mask)
                if box_xyxy is None:
                    continue
                bbox_xywh = box_xyxy_to_xywh(box_xyxy)
                out.append(
                    {
                        "segmentation": mask,
                        "area": int(area),
                        "bbox": bbox_xywh,
                        "score": float(score),
                        "point_coords": [pt.tolist()],
                    }
                )
            return out

        anns_all = build_anns(masks_before, scores_before, points_before, areas_before)  # 전체
        anns_nms = build_anns(masks_nms, scores_nms, points_nms, areas_nms)  # NMS 후

        # 🔹 NMS까지 끝난 후, 서로 겹치지 않게 정리
        anns_nms = make_masks_disjoint(anns_nms, min_area_remain=2000) # <--------------------------------------------

        return anns_all, anns_nms, grid_points


        # # 6) record 형태로 정리
        # anns: List[Dict[str, Any]] = []
        # for mask, score, pt, area in zip(masks_nms, scores_nms, points_nms, areas_nms):
        #     box_xyxy = mask_to_box_xyxy(mask)
        #     if box_xyxy is None:
        #         continue
        #     bbox_xywh = box_xyxy_to_xywh(box_xyxy)
        #     anns.append(
        #         {
        #             "segmentation": mask,
        #             "area": int(area),
        #             "bbox": bbox_xywh,
        #             "score": float(score),
        #             "point_coords": [pt.tolist()],
        #         }
        #     )
        #
        # return anns


# ======================
# 사용 예시 & 시각화
# ======================

if __name__ == "__main__":
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # SAM3 모델 로드
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)

    # 이미지 경로
    image_dir = "/home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video01_00080/images" # <--------------------------------------------
    # 폴더 내 이미지 파일 리스트(확장자 필터)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(img_exts)]
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {image_dir}")


    # 파일명에서 숫자(프레임 번호) 기준 정렬: frame_000080, frame80, img_12 등 대응
    def extract_first_int(name: str) -> int:
        nums = re.findall(r"\d+", name)
        return int(nums[0]) if nums else 10 ** 18  # 숫자 없으면 맨 뒤로


    files_sorted = sorted(files, key=extract_first_int)
    # files_sorted = sorted(files, key=extract_first_int, reverse=True) # reverse

    first_image_name = files_sorted[0]
    image_path = os.path.join(image_dir, first_image_name)

    print(f"[Folder] {image_dir}")
    print(f"[First Image] {first_image_name}")
    print(f"[First Image Path] {image_path}")

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    # 제너레이터 생성 (Hyperparam은 필요에 따라 조절)
    gen = Sam3GridMaskGenerator(
        model=model,
        processor=processor,
        points_per_side=25,      # point prompt를 넣기 16x16       # <--------------------------------------------
        points_per_batch=64,     # 메모리 보고 조절
        score_thresh=0.3,        # 너무 낮으면 많아짐. 0.2~0.5도 시도 가능
        iou_nms_thresh=0.3,      #  0.3~0.4 로 낮추기 (더 강한 NMS)
        min_mask_area=3000,       # 너무 작은 잡음 제거용
    )

    anns_all, anns_nms, grid_points  = gen.generate(img_np)
    print("before FOV: ", len(anns_nms))
    # FOV 마스크 경로 만들기: images -> masks, 확장자는 무조건 .png로
    # images/00000.jpg -> masks/00000.png (인덱스 기반)
    mask_dir = image_dir.replace("/images", "/masks")  # .../video01_00080/masks
    # frame_idx = 0  # 지금은 first frame을 쓰니 0
    frame_idx = int(os.path.splitext(first_image_name)[0])

    fov_mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
    print("[FOV Mask Path]", fov_mask_path)

    anns_all_fov = apply_visible_fov_mask(anns_all, fov_mask_path, img_np.shape)
    anns_nms_fov = apply_visible_fov_mask(anns_nms, fov_mask_path, img_np.shape)
    print("after  FOV: ", len(anns_nms_fov))

    # --- FOV mask를 bool로 로드 ---
    fov_pil = Image.open(fov_mask_path).convert("L")
    fov_np = np.array(fov_pil)
    H, W = img_np.shape[:2]
    if fov_np.shape[:2] != (H, W):
        fov_np = cv2.resize(fov_np, (W, H), interpolation=cv2.INTER_NEAREST)
    fov_bool = fov_np > 127

    # --- FOV 안에서 아직 덮이지 않은 '큰 빈 영역' 찾기 ---
    big_holes = find_big_holes_in_fov(
        fov_bool=fov_bool,
        anns=anns_nms_fov,  # NMS + FOV 적용된 마스크 기준
        area_thr=3000,  # hole 픽셀 수가 이 값 이상일 때만 “큰 hole”로 인정        # <--------------------------------------------
        radius_thr=10  # 빈 영역 두께 기준(픽셀 단위)
    )
    print("num big holes:", len(big_holes))

    # big_holes를 실제 ann으로 추가해서 같이 쓰고 싶으면:
    hole_anns = []
    for h in big_holes:
        box_xyxy = mask_to_box_xyxy(h)
        if box_xyxy is None:
            continue
        bbox_xywh = box_xyxy_to_xywh(box_xyxy)
        hole_anns.append({
            "segmentation": h,
            "area": int(h.sum()),
            "bbox": bbox_xywh,
            "score": 0.0,  # 나중에 필요하면 정의
            "point_coords": [[]],  # 일단 비워둠
        })

    # 이후 파이프라인에서 '비어 있던 영역도 하나의 mask로 취급'하고 싶으면
    # anns_nms_fov에 붙여서 사용하면 됨
    anns_nms_fov_extended = anns_nms_fov + hole_anns
    print(f"final masks (RRMD + holes): {len(anns_nms_fov_extended)}")

    # # -------------------------------------------------------------------------------------------
    # # Segmentation 시각화
    # fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    #
    # # (0) 원본
    # axes[0].imshow(img_np)
    # axes[0].set_title("Original")
    # axes[0].axis("off")
    #
    #
    # # (1) NMS 전: FOV 안의 모든 마스크
    # axes[1].imshow(img_np)
    # for ann in anns_all_fov:  # ✅ 필터링 된 리스트 사용
    #     show_mask(ann["segmentation"], axes[1], random_color=True, borders=True,  border_thickness=0.5)
    # axes[1].set_title("All masks INSIDE FOV (no NMS)")
    # axes[1].axis("off")
    #
    # # (2) NMS 후: FOV 안의 최종 후보 마스크 + 비어 있던 큰 홀
    # axes[2].imshow(img_np)
    # for ann in anns_nms_fov_extended:
    #     show_mask(ann["segmentation"], axes[2], random_color=True, borders=True,  border_thickness=1)
    # axes[2].set_title("Masks INSIDE FOV (after NMS + holes)")
    # axes[2].axis("off")
    #
    # plt.tight_layout()
    # plt.show()


    #-------------------------------------------------------------------------------------------
    # ======================
    # 1. 비디오 경로 및 프레임 정렬
    # ======================
    video_dir = image_dir  # .../images

    all_frame_names = sorted(
        [
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".jpg", ".jpeg")) and os.path.splitext(f)[0].isdigit()
        ],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if len(all_frame_names) == 0:
        raise RuntimeError(f"No numeric jpg/jpeg frames in {video_dir}")

    # segmentation에 실제 사용한 image_path 기준
    current_file_name = os.path.basename(image_path)
    first_frame_idx = all_frame_names.index(current_file_name)

    print(f"--- Path & Frame Info ---")
    print(f"Video Directory: {video_dir}")
    print(f"Total images in folder: {len(all_frame_names)}")
    print(f"Tracking Start Frame: {current_file_name} (Index: {first_frame_idx})")

    # ======================
    # 2. 비디오 모델 로드 (wrapper + tracker)
    # ======================
    video_model = build_sam3_video_model(
        device="cuda",
        load_from_HF=True,
        checkpoint_path=None,
        apply_temporal_disambiguation=True,
    )

    wrapper = video_model
    tracker = video_model.tracker
    predictor = tracker  # 추적/프롬프트 API는 tracker로 통일

    print("VIDEO MODEL TYPE:", type(wrapper))
    print("TRACKER TYPE:", type(tracker))
    print("TRACKER HAS add_new_mask?:", hasattr(tracker, "add_new_mask"))


    def init_tracker_state_flexible(tracker_obj, video_dir_path):
        sig = inspect.signature(tracker_obj.init_state)
        p = sig.parameters
        kwargs = {}

        # 경로 인자명 자동 매핑
        if "video_path" in p:
            kwargs["video_path"] = video_dir_path
        elif "resource_path" in p:
            kwargs["resource_path"] = video_dir_path
        elif "path" in p:
            kwargs["path"] = video_dir_path
        else:
            raise RuntimeError(f"tracker.init_state 인자 불일치: signature={sig}")

        # 옵션은 해당 인자가 있을 때만 주입
        optional_args = {
            "offload_video_to_cpu": False,
            "async_loading_frames": False,
            "video_loader_type": "cv2",
            "cache_image_features": True,
            "cache_all_image_features": True,
            "prefetch_image_features": True,
            "preload_image_features": True,
        }
        for k, v in optional_args.items():
            if k in p:
                kwargs[k] = v

        return tracker_obj.init_state(**kwargs)


    state = init_tracker_state_flexible(tracker, video_dir)

    print("tracker init_state signature:", inspect.signature(tracker.init_state))
    print("STATE KEYS:", state.keys())
    print("num_frames:", state.get("num_frames", None))
    print("feature_cache type:", type(state.get("feature_cache", None)))
    print("cached_features type:", type(state.get("cached_features", None)))

    if len(anns_nms_fov_extended) == 0:
        raise RuntimeError("No masks to inject after FOV filtering.")

    if not (0 <= first_frame_idx < state["num_frames"]):
        raise RuntimeError(
            f"first_frame_idx={first_frame_idx} out of range (num_frames={state['num_frames']})"
        )

    # state 키 호환 처리
    video_H = int(state["video_height"] if "video_height" in state else state["orig_height"])
    video_W = int(state["video_width"] if "video_width" in state else state["orig_width"])

    print("VIDEO H/W:", video_H, video_W, "num_frames:", state["num_frames"])
    print(f"--- Injecting {len(anns_nms_fov_extended)} masks via add_new_mask ---")

    if not hasattr(tracker, "add_new_mask"):
        raise RuntimeError("현재 tracker에 add_new_mask가 없습니다. sam3 버전 불일치 가능성이 큽니다.")

    print("add_new_mask sig:", inspect.signature(tracker.add_new_mask))
    print("propagate_in_video sig:", inspect.signature(tracker.propagate_in_video))


    # ===== cache priming: add_new_mask 전에 반드시 1회 실행 =====
    def _prime_cache_with_dummy_point(tracker, state, frame_idx, dummy_obj_id=999999):
        dev = state["device"]
        h, w = state["video_height"], state["video_width"]

        # 중앙 점 1개 (pixel coords)
        pts = torch.tensor([[[w * 0.5, h * 0.5]]], dtype=torch.float32, device=dev)  # (1,1,2)
        lbs = torch.tensor([[1]], dtype=torch.int64, device=dev)  # (1,1)

        tracker.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=dummy_obj_id,
            points=pts,
            labels=lbs,
            box=None,
            normalize_coords=False,
        )

        # dummy 흔적 제거 (point-prompt 의도 오염 방지)
        if "point_inputs_per_obj" in state and isinstance(state["point_inputs_per_obj"], dict):
            state["point_inputs_per_obj"].pop(dummy_obj_id, None)

        # 매핑 dict도 정리
        if "obj_id_to_idx" in state and isinstance(state["obj_id_to_idx"], dict):
            rm_idx = state["obj_id_to_idx"].pop(dummy_obj_id, None)
            if rm_idx is not None and "obj_idx_to_id" in state and isinstance(state["obj_idx_to_id"], dict):
                state["obj_idx_to_id"].pop(rm_idx, None)

        if "obj_ids" in state and isinstance(state["obj_ids"], list):
            state["obj_ids"] = [x for x in state["obj_ids"] if x != dummy_obj_id]


    # 실제 priming 호출
    _prime_cache_with_dummy_point(tracker, state, first_frame_idx)

    # 확인 로그
    print("cached frame keys:", list(state.get("cached_features", {}).keys())[:10], "target:", first_frame_idx)
    if first_frame_idx not in state.get("cached_features", {}):
        raise RuntimeError(f"cache priming failed at frame {first_frame_idx}")
    # ===== cache priming end =====

    # ======================
    # 3. 첫 프레임 마스크 주입
    # ======================
    for obj_id, ann in enumerate(anns_nms_fov_extended, start=1):
        mask_np = ann["segmentation"]
        if mask_np is None:
            continue

        if mask_np.dtype != np.bool_:
            mask_np = (mask_np > 0)

        if mask_np.shape[:2] != (video_H, video_W):
            mask_np = cv2.resize(
                mask_np.astype(np.uint8),
                (video_W, video_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        mask_np = np.ascontiguousarray(mask_np.astype(np.float32))
        # device 키가 없는 버전도 있으므로 CPU 텐서로 전달
        mask_t = torch.from_numpy(mask_np).float().contiguous()

        tracker.add_new_mask(
            inference_state=state,
            frame_idx=first_frame_idx,
            obj_id=obj_id,
            mask=mask_t,
        )
    print("cached keys before injection:", list(state.get("cached_features", {}).keys())[:10])
    print("has first frame cache?:", first_frame_idx in state.get("cached_features", {}))

    print("Mask injection done.")

    # ======================
    # 4. 순방향 트래킹 실행
    # ======================
    video_segments = {}

    for out in tracker.propagate_in_video(
            inference_state=state,
            start_frame_idx=first_frame_idx,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
    ):
        # 버전별 반환 길이 호환
        if len(out) == 5:
            out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores = out
        elif len(out) == 4:
            out_frame_idx, out_obj_ids, video_res_masks, obj_scores = out
        else:
            raise RuntimeError(f"Unexpected propagate output length: {len(out)}")

        video_segments[out_frame_idx] = {}
        masks = (video_res_masks > 0).cpu().numpy()

        for i, out_obj_id in enumerate(out_obj_ids):
            video_segments[out_frame_idx][out_obj_id] = masks[i]

    print("--- Starting Forward Tracking with Advanced Settings ---")
