"""
tracking_merge.py

입력
- image_dir: 원본 프레임 폴더 (.../videoXX/images)
- forward tracking 결과: .../videoXX/samgeo_track/forward/id_maps/*.npy
- backward tracking 결과: .../videoXX/samgeo_track/backward/id_maps/*.npy

핵심 동작
1) 전체 프레임(= 전 프레임: 0 ~ 마지막)에서 forward/backward 트랙 겹침 통계 누적
2) 누적 IoU 기준으로 서로 많이 겹치는 트랙 쌍을 같은 instance로 매핑
3) 프레임별 병합 id_map 생성 (같은 instance는 forward/backward 합집합)
4) 후처리: 전체 프레임 누적 면적이 작은 instance(track-level)만 주변 인접 instance로 병합
5) 후처리: 한 track이 다른 track에 일정 비율(기본 50%) 이상 포함되면 흡수 병합
6) FOV 기반 후처리: FOV 밖 제거 + 안겹치게 dilation으로 빈틈 메우기 + 큰 hole 신규 instance 생성
7) tracking과 동일한 출력 형식으로 저장

출력
- .../videoXX/samgeo_track/merged/
  - id_maps/*.npy
  - instance_masks/*.npz  (ids, masks)
  - id_maps_color/*.png
  - masks/*.png           (tracker raw와 유사한 id PNG; uint8)
  - tracked_postprocessed.mp4

적용 순서(실행 파이프라인)
1) forward/backward id_maps를 읽어 전 프레임 누적으로 IoU 매칭
2) 매칭된 ID를 하나의 merged ID로 remap
3) 프레임별로 같은 merged ID의 forward/backward 마스크를 합집합(union)
4) 포함(겹침) 기반 흡수:
   - 한 ID가 다른 ID에 일정 비율 이상 포함되면 큰 쪽으로 흡수
5) 작은 트랙(track-level) 흡수:
   - 전체 프레임 누적 면적이 작은 ID만 인접/nearest 큰 ID로 흡수
6) FOV 기반 gap/hole 보정
7) 최종 id_map/instance/color/mp4 저장

로그 해석
- [Match] ... merged_ids=K
  : IoU 매칭 직후(포함/소트랙 흡수 전) merged ID 개수
- [Containment Absorb] ... absorbed_ids=A
  : 포함 기반으로 흡수된 ID 개수(매핑상 child->parent)
- [ID Count] after_containment=C, after_small_track=S
  : 포함 흡수 후 / 소트랙 흡수 후 실제 최종 ID 개수
- [ID Count] after_fov_gap_hole=F
  : FOV + dilation gap fill + 큰 hole 생성 후 최종 ID 개수
"""

import os
import re
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pathlib import Path

# =========================
# User config (수정 가능한 하이퍼파라미터)
# =========================
# 입력 비디오(images) 위치
PROJECT_ROOT = Path(__file__).resolve().parent
image_dir = str(PROJECT_ROOT / "data" / "cholecseg_sub" / "video01_00080" / "images")

# tracking 결과 루트/하위 폴더 이름
track_root_name = "samgeo_track"
forward_subdir = "forward"
backward_subdir = "backward"
merged_subdir = "merged"

# 트랙 병합 민감도
# - merge_iou_thr ↑ : 더 엄격(정말 많이 겹칠 때만 같은 트랙으로 병합)
# - merge_iou_thr ↓ : 더 완화(대충 겹쳐도 병합)
merge_iou_thr = 0.45

# 작은 instance 후처리 (track-level)
# - 전체 프레임 누적 면적이 small_track_total_area_thr 미만인 ID만 흡수 대상
# - 특정 프레임에서만 작아지는 정상 객체는 보존됨
# - enable_small_track_absorb:
#   True면 작은 track 흡수 단계 활성화, False면 건너뜀
enable_small_track_absorb = True
# - small_track_total_area_thr (pixel):
#   이 값보다 "전체 프레임 누적 면적"이 작은 ID를 작은 track으로 간주
#   값을 올리면 더 많은 track이 흡수되고, 내리면 덜 흡수됨
small_track_total_area_thr = 2000
# - small_track_contact_kernel (odd int):
#   인접(접촉) 이웃 탐색을 위한 dilation 커널 크기
#   값을 올리면 더 멀리 있는 이웃도 접촉으로 간주
small_track_contact_kernel = 3
# - small_track_min_contact (pixel):
#   접촉 이웃으로 흡수할 때 필요한 최소 접촉 픽셀 수
#   값을 올리면 흡수가 보수적(엄격)이고, 내리면 적극적으로 흡수
small_track_min_contact = 5

# 겹침(포함) 기반 흡수 (track-level)
# - child track의 누적 픽셀 중 containment_ratio_thr 이상이 parent와 겹치면 child를 parent로 흡수
# - enable_containment_absorb:
#   True면 포함(겹침) 기반 흡수 단계 활성화
enable_containment_absorb = True
# - containment_ratio_thr (0~1):
#   child의 누적 픽셀 중 parent와 겹친 비율 임계값
#   예) 0.50이면 child의 50% 이상이 parent와 겹칠 때 흡수
#   값을 올리면 더 엄격, 내리면 더 쉽게 흡수
containment_ratio_thr = 0.50
# - containment_min_intersection (pixel):
#   우연한 소규모 겹침을 무시하기 위한 최소 교집합 픽셀 수
#   값을 올리면 작은 겹침은 무시, 내리면 작은 겹침도 흡수 후보
containment_min_intersection = 1000

# FOV + gap/hole 후처리
# - enable_fov_gap_hole_postprocess:
#   True면 FOV 마스크를 읽어 영역 밖 제거 + 빈틈 보정 + 큰 hole 신규 instance 생성 수행
enable_fov_gap_hole_postprocess = True
# - fov_masks_dir:
#   None이면 image_dir 기준 자동 추정(.../images -> .../masks)
fov_masks_dir = None
# - gap_dilation_iters / gap_dilation_kernel:
#   안겹치게 dilation으로 빈틈을 메우는 강도
# - gap_dilation_iters:
#   dilation 반복 횟수. 값을 올리면 빈틈을 더 멀리까지 메우고, 내리면 보수적으로 메움.
# - gap_dilation_kernel:
#   dilation 커널 크기(홀수 권장). 값을 올리면 한 번에 확장되는 폭이 커짐.
#   너무 크게 주면 인접 경계가 과도하게 퍼질 수 있음.
gap_dilation_iters = 5
gap_dilation_kernel = 5
# - new_hole_min_area:
#   보정 후 남은 hole 중 이 값 이상은 신규 instance로 생성
new_hole_min_area = 3000
# - hole_id_start:
#   hole 전용 instance ID 시작값 (평균 feature 제외 처리를 위해 별도 대역 사용)
hole_id_start = 10000

# MP4 시각화 저장 옵션
save_video = True
video_fps = 25
video_alpha = 0.45
video_name = "merged_track.mp4"


def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18


def sorted_frame_files(image_dir_: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(image_dir_) if f.lower().endswith(exts)]
    return sorted(files, key=extract_first_int)


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


def overlay_id_map_on_bgr(frame_bgr: np.ndarray, id_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_id_map(id_map)
    out = frame_bgr.copy()
    fg = id_map > 0
    out[fg] = (alpha * color[fg] + (1 - alpha) * out[fg]).astype(np.uint8)
    return out


def rebuild_instance_from_id_map(id_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.unique(id_map)
    ids = ids[ids > 0].astype(np.int32)
    if len(ids) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, *id_map.shape), dtype=np.uint8)
    masks = np.stack([(id_map == oid).astype(np.uint8) for oid in ids], axis=0)
    return ids, masks


def load_id_map(id_dir: str, stem: str) -> np.ndarray:
    p = os.path.join(id_dir, f"{stem}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"id_map not found: {p}")
    return np.load(p).astype(np.int32)


def infer_fov_mask_dir(image_dir_: str, configured_dir: str = None) -> str:
    if configured_dir is not None and len(configured_dir) > 0:
        return configured_dir
    if "/images" in image_dir_:
        return image_dir_.replace("/images", "/masks")
    return os.path.join(os.path.dirname(image_dir_), "masks")


def load_fov_bool_for_stem(
    frame_stem: str,
    fov_dir: str,
    shape_hw: Tuple[int, int],
) -> np.ndarray:
    idx = extract_first_int(frame_stem)
    fov_path = os.path.join(fov_dir, f"{idx:05d}.png")
    if not os.path.exists(fov_path):
        raise FileNotFoundError(f"FOV mask not found: {fov_path}")
    fov = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
    if fov is None:
        raise RuntimeError(f"Cannot read FOV mask: {fov_path}")
    h, w = shape_hw
    if fov.shape[:2] != (h, w):
        fov = cv2.resize(fov, (w, h), interpolation=cv2.INTER_NEAREST)
    return fov > 127


def accumulate_overlap_stats(
    frame_stems: List[str],
    fwd_id_dir: str,
    bwd_id_dir: str,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], int]]:
    area_f: Dict[int, int] = {}
    area_b: Dict[int, int] = {}
    inter_fb: Dict[Tuple[int, int], int] = {}

    for stem in frame_stems:
        f = load_id_map(fwd_id_dir, stem)
        b = load_id_map(bwd_id_dir, stem)
        if f.shape != b.shape:
            raise RuntimeError(f"shape mismatch at {stem}: forward={f.shape}, backward={b.shape}")

        f_ids, f_cnt = np.unique(f[f > 0], return_counts=True)
        for oid, c in zip(f_ids.tolist(), f_cnt.tolist()):
            area_f[int(oid)] = area_f.get(int(oid), 0) + int(c)

        b_ids, b_cnt = np.unique(b[b > 0], return_counts=True)
        for oid, c in zip(b_ids.tolist(), b_cnt.tolist()):
            area_b[int(oid)] = area_b.get(int(oid), 0) + int(c)

        both = (f > 0) & (b > 0)
        if both.any():
            pairs = np.stack([f[both], b[both]], axis=1)
            uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
            for (foid, boid), c in zip(uniq.tolist(), cnt.tolist()):
                key = (int(foid), int(boid))
                inter_fb[key] = inter_fb.get(key, 0) + int(c)

    return area_f, area_b, inter_fb


def build_track_mapping(
    area_f: Dict[int, int],
    area_b: Dict[int, int],
    inter_fb: Dict[Tuple[int, int], int],
    iou_thr: float,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    candidates = []
    for (foid, boid), inter in inter_fb.items():
        union = area_f.get(foid, 0) + area_b.get(boid, 0) - inter
        if union <= 0:
            continue
        iou = inter / float(union)
        if iou >= iou_thr:
            candidates.append((iou, inter, foid, boid))

    candidates.sort(reverse=True)

    used_f, used_b = set(), set()
    f_to_m: Dict[int, int] = {}
    b_to_m: Dict[int, int] = {}
    next_mid = 1

    for iou, inter, foid, boid in candidates:
        if foid in used_f or boid in used_b:
            continue
        used_f.add(foid)
        used_b.add(boid)
        f_to_m[foid] = next_mid
        b_to_m[boid] = next_mid
        next_mid += 1

    for foid in sorted(area_f.keys()):
        if foid not in f_to_m:
            f_to_m[foid] = next_mid
            next_mid += 1

    for boid in sorted(area_b.keys()):
        if boid not in b_to_m:
            b_to_m[boid] = next_mid
            next_mid += 1

    return f_to_m, b_to_m


def remap_id_map(id_map: np.ndarray, old_to_new: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(id_map, dtype=np.int32)
    ids = np.unique(id_map)
    ids = ids[ids > 0]
    for oid in ids.tolist():
        out[id_map == int(oid)] = int(old_to_new[int(oid)])
    return out


def merge_frame_maps(
    fmap: np.ndarray,
    bmap: np.ndarray,
    f_to_m: Dict[int, int],
    b_to_m: Dict[int, int],
) -> np.ndarray:
    mf = remap_id_map(fmap, f_to_m)
    mb = remap_id_map(bmap, b_to_m)

    # Union per merged instance ID:
    # Each merged ID first gets (forward OR backward) as its mask.
    mids = np.unique(np.concatenate([mf[mf > 0], mb[mb > 0]]))
    if len(mids) == 0:
        return np.zeros_like(mf, dtype=np.int32)

    union_masks: Dict[int, np.ndarray] = {}
    for mid in mids.tolist():
        mid_i = int(mid)
        union_masks[mid_i] = (mf == mid_i) | (mb == mid_i)

    # Single-label id_map cannot store two IDs on one pixel.
    # If union masks overlap across different IDs, assign in area-descending order.
    order = sorted(union_masks.keys(), key=lambda x: int(union_masks[x].sum()), reverse=True)
    out = np.zeros_like(mf, dtype=np.int32)
    for mid in order:
        m = union_masks[mid]
        out[(out == 0) & m] = int(mid)

    return out


def build_union_masks_from_maps(
    fmap: np.ndarray,
    bmap: np.ndarray,
    f_to_m: Dict[int, int],
    b_to_m: Dict[int, int],
) -> Dict[int, np.ndarray]:
    mf = remap_id_map(fmap, f_to_m)
    mb = remap_id_map(bmap, b_to_m)
    mids = np.unique(np.concatenate([mf[mf > 0], mb[mb > 0]]))
    union_masks: Dict[int, np.ndarray] = {}
    for mid in mids.tolist():
        mid_i = int(mid)
        union_masks[mid_i] = (mf == mid_i) | (mb == mid_i)
    return union_masks


def rasterize_union_masks(union_masks: Dict[int, np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    if len(union_masks) == 0:
        return np.zeros(shape_hw, dtype=np.int32)
    order = sorted(union_masks.keys(), key=lambda x: int(union_masks[x].sum()), reverse=True)
    out = np.zeros(shape_hw, dtype=np.int32)
    for mid in order:
        m = union_masks[mid]
        out[(out == 0) & m] = int(mid)
    return out


def accumulate_union_overlap_stats(
    frame_stems: List[str],
    fwd_id_dir: str,
    bwd_id_dir: str,
    f_to_m: Dict[int, int],
    b_to_m: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    area_u: Dict[int, int] = {}
    inter_u: Dict[Tuple[int, int], int] = {}

    for stem in frame_stems:
        fmap = load_id_map(fwd_id_dir, stem)
        bmap = load_id_map(bwd_id_dir, stem)
        union_masks = build_union_masks_from_maps(fmap, bmap, f_to_m, b_to_m)
        mids = sorted(union_masks.keys())
        for mid in mids:
            area_u[mid] = area_u.get(mid, 0) + int(union_masks[mid].sum())
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                a, b = mids[i], mids[j]
                inter = int((union_masks[a] & union_masks[b]).sum())
                if inter > 0:
                    inter_u[(a, b)] = inter_u.get((a, b), 0) + inter

    return area_u, inter_u


def build_containment_absorb_map(
    area_u: Dict[int, int],
    inter_u: Dict[Tuple[int, int], int],
    ratio_thr: float,
    min_inter: int,
) -> Dict[int, int]:
    if len(area_u) == 0:
        return {}

    parent: Dict[int, int] = {int(k): int(k) for k in area_u.keys()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def absorb(child: int, par: int):
        rc = find(child)
        rp = find(par)
        if rc == rp:
            return
        parent[rc] = rp

    candidates = []
    for (a, b), inter in inter_u.items():
        if inter < min_inter:
            continue
        area_a = area_u.get(a, 0)
        area_b = area_u.get(b, 0)
        if area_a <= 0 or area_b <= 0:
            continue
        ratio_a_in_b = inter / float(area_a)
        ratio_b_in_a = inter / float(area_b)
        if ratio_a_in_b >= ratio_thr and area_a <= area_b:
            candidates.append((ratio_a_in_b, inter, a, b))
        if ratio_b_in_a >= ratio_thr and area_b <= area_a:
            candidates.append((ratio_b_in_a, inter, b, a))

    candidates.sort(reverse=True)
    for ratio, inter, child, par in candidates:
        absorb(int(child), int(par))

    out = {oid: find(oid) for oid in area_u.keys()}
    return out


def apply_absorb_map_to_union_masks(
    union_masks: Dict[int, np.ndarray],
    absorb_map: Dict[int, int],
) -> Dict[int, np.ndarray]:
    if len(absorb_map) == 0:
        return union_masks
    merged: Dict[int, np.ndarray] = {}
    for oid, m in union_masks.items():
        tgt = int(absorb_map.get(int(oid), int(oid)))
        if tgt not in merged:
            merged[tgt] = m.copy()
        else:
            merged[tgt] |= m
    return merged


def compute_global_id_areas(merged_maps: Dict[str, np.ndarray]) -> Dict[int, int]:
    area: Dict[int, int] = {}
    for m in merged_maps.values():
        ids, cnt = np.unique(m[m > 0], return_counts=True)
        for oid, c in zip(ids.tolist(), cnt.tolist()):
            area[int(oid)] = area.get(int(oid), 0) + int(c)
    return area


def count_unique_ids_in_maps(merged_maps: Dict[str, np.ndarray]) -> int:
    s = set()
    for m in merged_maps.values():
        ids = np.unique(m)
        ids = ids[ids > 0]
        s.update([int(x) for x in ids.tolist()])
    return len(s)


def _assign_component_to_nearest_non_small(
    id_map: np.ndarray,
    comp_mask: np.ndarray,
    small_ids: set,
) -> Tuple[np.ndarray, bool]:
    out = id_map.copy()
    seed_mask = (out > 0) & (~np.isin(out, list(small_ids)))
    if not seed_mask.any():
        seed_mask = out > 0
    if not seed_mask.any():
        return out, False

    bg = (~seed_mask).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        bg,
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    seed_coords = np.column_stack(np.where(seed_mask))
    if len(seed_coords) == 0:
        return out, False
    seed_ids = np.array([out[y, x] for y, x in seed_coords], dtype=np.int32)

    ys, xs = np.where(comp_mask)
    picked = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        li = int(labels[y, x]) - 1
        if 0 <= li < len(seed_ids):
            sid = int(seed_ids[li])
            if sid > 0:
                picked.append(sid)
    if len(picked) == 0:
        return out, False

    uniq, cnt = np.unique(np.array(picked, dtype=np.int32), return_counts=True)
    target = int(uniq[int(np.argmax(cnt))])
    out[comp_mask] = target
    return out, True


def absorb_small_track_ids(
    id_map: np.ndarray,
    small_ids: set,
    contact_kernel: int,
    min_contact: int,
) -> np.ndarray:
    if len(small_ids) == 0:
        return id_map

    out = id_map.copy()
    k = max(1, int(contact_kernel))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for sid in sorted(list(small_ids)):
        m = (out == int(sid)).astype(np.uint8)
        if m.sum() == 0:
            continue
        n_comp, labels, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for comp_idx in range(1, n_comp):
            comp = labels == comp_idx
            dil = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool)
            border = dil & (~comp)

            neigh = out[border]
            neigh = neigh[(neigh > 0) & (neigh != int(sid))]
            if len(neigh) > 0:
                # Prefer non-small neighbors first.
                major = neigh[~np.isin(neigh, list(small_ids))]
                if len(major) == 0:
                    major = neigh
                uniq, cnt = np.unique(major, return_counts=True)
                best_idx = int(np.argmax(cnt))
                best_id = int(uniq[best_idx])
                best_contact = int(cnt[best_idx])
                if best_contact >= int(min_contact):
                    out[comp] = best_id
                    continue

            # No reliable touching neighbor: absorb to nearest existing instance.
            out2, ok = _assign_component_to_nearest_non_small(out, comp, small_ids=small_ids)
            if ok:
                out = out2

    return out


def fill_gaps_by_nonoverlap_dilation(
    id_map: np.ndarray,
    fov_bool: np.ndarray,
    iters: int,
    kernel_size: int,
) -> np.ndarray:
    out = id_map.copy()
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for _ in range(max(0, int(iters))):
        empty = (out == 0) & fov_bool
        if not empty.any():
            break
        ids = np.unique(out)
        ids = ids[ids > 0]
        if len(ids) == 0:
            break
        id_order = sorted(ids.tolist(), key=lambda oid: int((out == int(oid)).sum()), reverse=True)
        for oid in id_order:
            m = out == int(oid)
            if not m.any():
                continue
            dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
            frontier = dil & empty
            if frontier.any():
                out[frontier] = int(oid)
                empty[frontier] = False
    return out


def add_large_holes_as_new_instances(
    id_map: np.ndarray,
    fov_bool: np.ndarray,
    min_hole_area: int,
    next_id: int,
) -> Tuple[np.ndarray, int]:
    out = id_map.copy()
    holes = (out == 0) & fov_bool
    if not holes.any():
        return out, next_id
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    for comp_idx in range(1, n_cc):
        area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if area < int(min_hole_area):
            continue
        comp = labels == comp_idx
        out[comp] = int(next_id)
        next_id += 1
    return out, next_id


def apply_fov_gap_hole_postprocess(
    merged_maps: Dict[str, np.ndarray],
    frame_stems: List[str],
    fov_dir: str,
    dilation_iters: int,
    dilation_kernel: int,
    hole_min_area: int,
    hole_id_start_: int,
) -> Dict[str, np.ndarray]:
    out_maps = {k: v.copy() for k, v in merged_maps.items()}
    max_id = 0
    for m in out_maps.values():
        if m.size > 0:
            max_id = max(max_id, int(m.max()))
    next_new_id = max(int(hole_id_start_), max_id + 1)

    for stem in frame_stems:
        m = out_maps[stem]
        fov_bool = load_fov_bool_for_stem(stem, fov_dir=fov_dir, shape_hw=m.shape)
        # FOV 밖 제거
        m[~fov_bool] = 0
        # 안겹치게 dilation으로 빈틈 보정
        m = fill_gaps_by_nonoverlap_dilation(
            m,
            fov_bool=fov_bool,
            iters=dilation_iters,
            kernel_size=dilation_kernel,
        )
        # 남은 큰 hole은 신규 instance 생성
        m, next_new_id = add_large_holes_as_new_instances(
            m,
            fov_bool=fov_bool,
            min_hole_area=hole_min_area,
            next_id=next_new_id,
        )
        out_maps[stem] = m
    return out_maps


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
            id_path = os.path.join(idmap_npy_dir, f"{stem}.npy")

            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            if not os.path.exists(id_path):
                vw.write(frame)
                continue
            id_map = np.load(id_path).astype(np.int32)
            vis = overlay_id_map_on_bgr(frame, id_map, alpha=alpha)
            vw.write(vis)
    finally:
        vw.release()


def main():
    frame_files = sorted_frame_files(image_dir)
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    frame_stems = [os.path.splitext(f)[0] for f in frame_files]
    video_root = os.path.dirname(image_dir)
    track_root = os.path.join(video_root, track_root_name)
    fwd_id_dir = os.path.join(track_root, forward_subdir, "id_maps")
    bwd_id_dir = os.path.join(track_root, backward_subdir, "id_maps")

    if not os.path.isdir(fwd_id_dir):
        raise FileNotFoundError(f"forward id_maps not found: {fwd_id_dir}")
    if not os.path.isdir(bwd_id_dir):
        raise FileNotFoundError(f"backward id_maps not found: {bwd_id_dir}")

    out_dir = os.path.join(track_root, merged_subdir)
    idmap_npy_dir = os.path.join(out_dir, "id_maps")
    inst_npz_dir = os.path.join(out_dir, "instance_masks")
    idmap_color_dir = os.path.join(out_dir, "id_maps_color")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(idmap_npy_dir, exist_ok=True)
    os.makedirs(inst_npz_dir, exist_ok=True)
    os.makedirs(idmap_color_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    area_f, area_b, inter_fb = accumulate_overlap_stats(frame_stems, fwd_id_dir, bwd_id_dir)
    f_to_m, b_to_m = build_track_mapping(
        area_f=area_f,
        area_b=area_b,
        inter_fb=inter_fb,
        iou_thr=merge_iou_thr,
    )
    print(
        f"[Match] forward_ids={len(area_f)}, backward_ids={len(area_b)}, "
        f"merged_ids={len(set(f_to_m.values()) | set(b_to_m.values()))}"
    )
    print(f"[Merge Hyperparams] merge_iou_thr={merge_iou_thr}")

    contain_map: Dict[int, int] = {}
    if enable_containment_absorb:
        area_u, inter_u = accumulate_union_overlap_stats(
            frame_stems=frame_stems,
            fwd_id_dir=fwd_id_dir,
            bwd_id_dir=bwd_id_dir,
            f_to_m=f_to_m,
            b_to_m=b_to_m,
        )
        contain_map = build_containment_absorb_map(
            area_u=area_u,
            inter_u=inter_u,
            ratio_thr=containment_ratio_thr,
            min_inter=containment_min_intersection,
        )
        n_absorbed = sum(1 for k, v in contain_map.items() if int(k) != int(v))
        print(
            f"[Containment Absorb] ratio_thr={containment_ratio_thr}, "
            f"min_inter={containment_min_intersection}, absorbed_ids={n_absorbed}"
        )
    else:
        print("[Containment Absorb] disabled")

    merged_maps: Dict[str, np.ndarray] = {}
    for stem in frame_stems:
        fmap = load_id_map(fwd_id_dir, stem)
        bmap = load_id_map(bwd_id_dir, stem)
        union_masks = build_union_masks_from_maps(fmap, bmap, f_to_m, b_to_m)
        union_masks = apply_absorb_map_to_union_masks(union_masks, contain_map)
        merged_maps[stem] = rasterize_union_masks(union_masks, shape_hw=fmap.shape)

    count_after_containment = count_unique_ids_in_maps(merged_maps)
    print(f"[ID Count] after_containment={count_after_containment}")

    if enable_small_track_absorb:
        global_area = compute_global_id_areas(merged_maps)
        small_ids = {oid for oid, a in global_area.items() if int(a) < int(small_track_total_area_thr)}
        print(
            f"[Small Track Absorb] total_ids={len(global_area)}, "
            f"small_ids={len(small_ids)}, area_thr={small_track_total_area_thr}"
        )
        if len(small_ids) > 0:
            for stem in frame_stems:
                merged_maps[stem] = absorb_small_track_ids(
                    merged_maps[stem],
                    small_ids=small_ids,
                    contact_kernel=small_track_contact_kernel,
                    min_contact=small_track_min_contact,
                )
            count_after_small = count_unique_ids_in_maps(merged_maps)
            print(f"[ID Count] after_small_track={count_after_small}")
        else:
            print(f"[ID Count] after_small_track={count_after_containment} (no small IDs)")
    else:
        print("[Small Track Absorb] disabled")
        print(f"[ID Count] after_small_track={count_after_containment} (stage disabled)")

    if enable_fov_gap_hole_postprocess:
        resolved_fov_dir = infer_fov_mask_dir(image_dir, configured_dir=fov_masks_dir)
        first_idx = extract_first_int(frame_stems[0]) if len(frame_stems) > 0 else 0
        first_fov_path = os.path.join(resolved_fov_dir, f"{first_idx:05d}.png")
        can_use_fov = os.path.isdir(resolved_fov_dir) and os.path.exists(first_fov_path)
        if can_use_fov:
            print(
                f"[FOV Gap/Hole] fov_dir={resolved_fov_dir}, "
                f"dilation_iters={gap_dilation_iters}, dilation_kernel={gap_dilation_kernel}, "
                f"new_hole_min_area={new_hole_min_area}, hole_id_start={hole_id_start}"
            )
            merged_maps = apply_fov_gap_hole_postprocess(
                merged_maps=merged_maps,
                frame_stems=frame_stems,
                fov_dir=resolved_fov_dir,
                dilation_iters=gap_dilation_iters,
                dilation_kernel=gap_dilation_kernel,
                hole_min_area=new_hole_min_area,
                hole_id_start_=hole_id_start,
            )
            count_after_fov_gap_hole = count_unique_ids_in_maps(merged_maps)
            print(f"[ID Count] after_fov_gap_hole={count_after_fov_gap_hole}")
        else:
            cur_count = count_unique_ids_in_maps(merged_maps)
            print(
                "[FOV Gap/Hole] masks not found -> skip this stage "
                f"(dir={resolved_fov_dir})"
            )
            print(f"[ID Count] after_fov_gap_hole={cur_count} (stage skipped)")
    else:
        print("[FOV Gap/Hole] disabled")

    for stem in frame_stems:
        merged = merged_maps[stem]

        np.save(os.path.join(idmap_npy_dir, f"{stem}.npy"), merged)
        ids_arr, masks_arr = rebuild_instance_from_id_map(merged)
        np.savez_compressed(os.path.join(inst_npz_dir, f"{stem}.npz"), ids=ids_arr, masks=masks_arr)
        cv2.imwrite(os.path.join(idmap_color_dir, f"{stem}.png"), colorize_id_map(merged))
        # keep raw-like mask format aligned with tracker output (PNG uint8 id map).
        cv2.imwrite(os.path.join(masks_dir, f"{stem}.png"), np.clip(merged, 0, 255).astype(np.uint8))

    if save_video:
        save_postprocessed_video_from_id_maps(
            image_dir_=image_dir,
            frame_files=frame_files,
            idmap_npy_dir=idmap_npy_dir,
            video_out_path=os.path.join(out_dir, video_name),
            fps=video_fps,
            alpha=video_alpha,
        )

    print("[Done]")
    print(f"  out_dir: {out_dir}")
    print(f"  - id_maps: {idmap_npy_dir}")
    print(f"  - instance_masks: {inst_npz_dir}")
    print(f"  - id_maps_color: {idmap_color_dir}")


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
    args = parser.parse_args()

    if args.image_dir:
        image_dir = args.image_dir
    elif args.dataset_path:
        image_dir = os.path.join(args.dataset_path, "images")

    main()
