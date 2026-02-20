#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
masks_png/ 폴더를 입력으로 받아:
- 같은 레벨에 masks/ 폴더 생성
- masks_png 안의 마스크 파일들을 "내용은 그대로" 복사하면서
  이름만 00000.png, 00001.png ... 로 변경하여 masks/에 저장
- 원본 masks_png는 변경/이동/삭제하지 않음

사용:
1) 아래 MASKS_PNG_DIR만 수정
2) python /home/jihun/PycharmProjects/SurgTPGS/rename_masks_only.py 실행
"""

import re
import shutil
from pathlib import Path

# =========================
# 여기만 수정하면 됨
# =========================
MASKS_PNG_DIR = Path("/home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video01_00080/masks_png")
# =========================


def extract_first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**18


def main():
    if not MASKS_PNG_DIR.exists() or not MASKS_PNG_DIR.is_dir():
        raise FileNotFoundError(f"MASKS_PNG_DIR not found or not a directory: {MASKS_PNG_DIR}")

    parent = MASKS_PNG_DIR.parent
    out_dir = parent / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 확장자 필터 (보통 마스크는 png)
    exts = {".png", ".jpg", ".jpeg"}  # 혹시 섞여 있어도 처리
    files = [p for p in MASKS_PNG_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"No mask files found in: {MASKS_PNG_DIR}")

    # 파일명에 포함된 숫자로 정렬 (frame_000080_endo.png 등 대응)
    files.sort(key=lambda p: extract_first_int(p.name))

    print(f"[INPUT ] {MASKS_PNG_DIR}")
    print(f"[OUTPUT] {out_dir}")
    print(f"[COUNT ] {len(files)} files")

    converted = 0
    skipped = 0

    for i, src in enumerate(files):
        # 출력 확장자는 원본 확장자를 그대로 사용 (대부분 .png)
        dst = out_dir / f"{i:05d}{src.suffix.lower()}"

        if dst.exists():
            skipped += 1
            continue

        # 내용 그대로 복사 (포맷/픽셀/압축 재인코딩 없음)
        shutil.copy2(src, dst)
        converted += 1

    print(f"[DONE] copied={converted}, skipped(existing)={skipped}")

    sample = sorted(out_dir.iterdir())[:5]
    if sample:
        print("[SAMPLE OUTPUT FILES]")
        for p in sample:
            print(" ", p.name)


if __name__ == "__main__":
    main()
