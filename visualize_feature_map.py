import argparse
import glob
import os
import re
from typing import List, Tuple

import cv2
import numpy as np


def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18


def list_feature_pairs(feature_dir: str) -> List[Tuple[str, str, str]]:
    f_files = sorted(glob.glob(os.path.join(feature_dir, "*_f.npy")), key=extract_first_int)
    pairs = []
    for fp in f_files:
        base = os.path.basename(fp)
        stem = base[:-6]  # remove "_f.npy"
        sp = os.path.join(feature_dir, f"{stem}_s.npy")
        if os.path.exists(sp):
            pairs.append((stem, fp, sp))
    return pairs


def load_seg_map(seg_path: str) -> np.ndarray:
    s = np.load(seg_path)
    if s.ndim == 3:
        s = s[0]
    if s.ndim != 2:
        raise ValueError(f"Unsupported seg shape: {s.shape} at {seg_path}")
    return s.astype(np.int32)


def pca_basis_from_feature_tables(
    feature_paths: List[str],
    sample_cap: int = 200000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    tables = []
    for fp in feature_paths:
        feat = np.load(fp)
        if feat.ndim != 2 or feat.shape[0] == 0:
            continue
        tables.append(feat.astype(np.float32))
    if len(tables) == 0:
        raise RuntimeError("No valid feature tables found for PCA.")

    x = np.concatenate(tables, axis=0)
    if len(x) > sample_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=sample_cap, replace=False)
        x = x[idx]

    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # PCA via SVD
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:3].T  # (D,3)
    return basis.astype(np.float32), mean.reshape(-1).astype(np.float32)


def compute_global_proj_minmax(
    feature_paths: List[str],
    basis: np.ndarray,
    mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pmin = None
    pmax = None
    for fp in feature_paths:
        feat = np.load(fp).astype(np.float32)
        if feat.ndim != 2 or feat.shape[0] == 0:
            continue
        proj = (feat - mean[None, :]) @ basis
        cur_min = proj.min(axis=0)
        cur_max = proj.max(axis=0)
        if pmin is None:
            pmin = cur_min
            pmax = cur_max
        else:
            pmin = np.minimum(pmin, cur_min)
            pmax = np.maximum(pmax, cur_max)
    if pmin is None:
        pmin = np.zeros((3,), dtype=np.float32)
        pmax = np.ones((3,), dtype=np.float32)
    return pmin.astype(np.float32), pmax.astype(np.float32)


def feature_table_to_colors(
    feat: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> np.ndarray:
    if feat.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    proj = (feat - mean[None, :]) @ basis  # (N,3)
    denom = np.maximum(global_max[None, :] - global_min[None, :], 1e-8)
    norm = (proj - global_min[None, :]) / denom
    rgb = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return rgb


def render_frame(seg: np.ndarray, row_colors: np.ndarray, bg_color=(0, 0, 0)) -> np.ndarray:
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = np.array(bg_color, dtype=np.uint8)
    valid = seg >= 0
    if np.any(valid):
        rows = seg[valid]
        keep = (rows >= 0) & (rows < len(row_colors))
        if np.any(keep):
            ys, xs = np.where(valid)
            out[ys[keep], xs[keep]] = row_colors[rows[keep]]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--feature_dir_name", type=str, default="language_features_fine_mean")
    parser.add_argument("--out_dir_name", type=str, default="feature_map_vis")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample_cap", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", dest="save_video", action="store_false")
    args = parser.parse_args()

    feature_dir = os.path.join(args.dataset_path, args.feature_dir_name)
    out_dir = os.path.join(args.dataset_path, args.out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    pairs = list_feature_pairs(feature_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No *_f.npy/*_s.npy pairs found in: {feature_dir}")

    basis, mean = pca_basis_from_feature_tables(
        [fp for _, fp, _ in pairs],
        sample_cap=args.sample_cap,
        seed=args.seed,
    )
    global_min, global_max = compute_global_proj_minmax(
        [fp for _, fp, _ in pairs],
        basis=basis,
        mean=mean,
    )
    print(f"[PCA] basis_shape={basis.shape}, frames={len(pairs)}")
    print(f"[PCA] global_min={global_min.tolist()}, global_max={global_max.tolist()}")

    first_seg = load_seg_map(pairs[0][2])
    h, w = first_seg.shape
    video_path = os.path.join(out_dir, "feature_map_vis.mp4")
    vw = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {video_path}")

    try:
        for stem, fp, sp in pairs:
            feat = np.load(fp).astype(np.float32)
            seg = load_seg_map(sp)
            row_colors = feature_table_to_colors(
                feat,
                basis=basis,
                mean=mean,
                global_min=global_min,
                global_max=global_max,
            )
            img = render_frame(seg, row_colors=row_colors, bg_color=(0, 0, 0))
            out_png = os.path.join(out_dir, f"{stem}.png")
            cv2.imwrite(out_png, img)
            if vw is not None:
                vw.write(img)
    finally:
        if vw is not None:
            vw.release()

    print("[Done]")
    print(f"  feature_dir: {feature_dir}")
    print(f"  out_dir: {out_dir}")
    if args.save_video:
        print(f"  video: {video_path}")


if __name__ == "__main__":
    main()
