#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PNG 폴더(images_png)를 입력으로 받아:
- 같은 레벨에 images/ 폴더를 생성
- images_png/ 안의 *.png 를 *.jpg 로 변환해 images/에 저장
- 저장 파일명은 SAM3 tracker가 요구하는 형태로: 00000.jpg, 00001.jpg, ...
- 원본 PNG는 그대로 둠(이동/삭제 안 함)

사용:
1) 아래 INPUT_DIR만 바꾸고
2) python /home/jihun/PycharmProjects/SurgTPGS/png2jpg_move.py 실행
"""

import re
from pathlib import Path
from PIL import Image

# =========================
# 여기만 수정하면 됨
# =========================
INPUT_DIR = Path("/home/jihun/PycharmProjects/SurgTPGS/data/endovis_2018/seq_9_sub/images_png")

# JPG 품질(0~100)
JPG_QUALITY = 100
# =========================


def extract_first_int(name: str) -> int:
    """
    파일명에서 첫 번째 숫자 덩어리를 추출해 정렬 키로 사용.
    예: frame_000080_endo.png -> 80
        frame059.png -> 59
    """
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18  # 숫자 없으면 맨 뒤


def png_to_jpg(src_png: Path, dst_jpg: Path, quality: int = 100) -> None:
    """PNG -> JPG 변환. 알파 채널이 있으면 제거 후 RGB로 저장."""
    with Image.open(src_png) as im:
        im = im.convert("RGB")
        # optimize=True는 파일 용량만 줄이는 옵션(해상도/크기 변경 없음)
        im.save(dst_jpg, format="JPEG", quality=quality, optimize=True)


def main():
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"INPUT_DIR not found or not a directory: {INPUT_DIR}")

    parent = INPUT_DIR.parent
    output_dir = parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = [p for p in INPUT_DIR.glob("*.png")]
    if not png_files:
        raise RuntimeError(f"No .png files found in: {INPUT_DIR}")

    # 숫자 기준으로 정렬
    png_files_sorted = sorted(png_files, key=lambda p: extract_first_int(p.name))

    print(f"[INPUT ] {INPUT_DIR}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[COUNT ] {len(png_files_sorted)} png files")
    print(f"[NAMING] 00000.jpg ...")

    converted = 0
    skipped = 0

    for i, src in enumerate(png_files_sorted):
        dst = output_dir / f"{i:05d}.jpg"
        if dst.exists():
            skipped += 1
            continue
        png_to_jpg(src, dst, quality=JPG_QUALITY)
        converted += 1

    print(f"[DONE] converted={converted}, skipped(existing)={skipped}")

    # 샘플 출력
    sample = list(sorted(output_dir.glob("*.jpg")))[:5]
    if sample:
        print("[SAMPLE OUTPUT FILES]")
        for p in sample:
            print(" ", p.name)


if __name__ == "__main__":
    main()
