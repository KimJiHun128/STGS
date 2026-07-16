#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


def stretch_channel(ch: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = np.percentile(ch, p_low)
    hi = np.percentile(ch, p_high)
    if hi <= lo + 1e-6:
        return ch
    out = (ch - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def enhance(img_bgr: np.ndarray, gamma: float, sat_boost: float, edge_alpha: float) -> np.ndarray:
    f = img_bgr.astype(np.float32) / 255.0

    # Per-channel contrast stretch
    b, g, r = cv2.split(f)
    b = stretch_channel(b, 1, 99)
    g = stretch_channel(g, 1, 99)
    r = stretch_channel(r, 1, 99)
    f = cv2.merge([b, g, r])

    # Gamma correction (<1 brightens)
    f = np.power(np.clip(f, 0.0, 1.0), gamma)

    # Saturation boost for color separation
    hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
    f2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Edge emphasis to separate adjacent dark regions
    gray = cv2.cvtColor((f2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_mask = (edges > 0)[..., None]
    edge_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # white edges
    out = f2.copy()
    out[edge_mask[..., 0]] = (1 - edge_alpha) * out[edge_mask[..., 0]] + edge_alpha * edge_color

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='')
    p.add_argument('--gamma', type=float, default=0.75)
    p.add_argument('--sat_boost', type=float, default=1.6)
    p.add_argument('--edge_alpha', type=float, default=0.75)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir.parent / f"{in_dir.name}_contrast"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([x for x in in_dir.iterdir() if x.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    for fp in files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = enhance(img, args.gamma, args.sat_boost, args.edge_alpha)
        cv2.imwrite(str(out_dir / fp.name), out)

    print('[Done]')
    print(' input_dir :', in_dir)
    print(' output_dir:', out_dir)
    print(' files     :', len(files))


if __name__ == '__main__':
    main()
