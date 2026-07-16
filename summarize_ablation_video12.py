#!/usr/bin/env python3
import csv
import json
from pathlib import Path

OUTPUT_ROOT = Path("output")
DATASET_NAME = "cholecseg_sub/video12_15750_0"
LEVEL = "0"
THR_TAG = "0p40"

EXPERIMENTS = {
    "all_off": "language_features_ablation_all_off_none_dim3",
    "snid_rrmd_on_only": "language_features_ablation_snid_rrmd_on_only_none_dim3",
}


def pick_result_json(exp_dir: str) -> Path:
    base = OUTPUT_ROOT / exp_dir / DATASET_NAME / "test" / "ours_3000"
    p_thr = base / f"result_thr_{THR_TAG}.json"
    if p_thr.exists():
        return p_thr
    p_default = base / "result.json"
    return p_default


def load_json(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing result file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    save_dir = OUTPUT_ROOT / "ablation_video12_15750"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "ablation_summary.csv"

    rows = []
    all_classes = set()
    raw = {}
    for name, exp_dir in EXPERIMENTS.items():
        p = pick_result_json(exp_dir)
        d = load_json(p)
        raw[name] = d
        all_classes.update(d.get("Total Avg Per Class", {}).keys())

    headers = ["experiment", "total_average"] + sorted(all_classes)

    for name in EXPERIMENTS.keys():
        d = raw[name]
        per_cls = d.get("Total Avg Per Class", {})
        row = {
            "experiment": name,
            "total_average": d.get("Total Average", "")
        }
        for c in sorted(all_classes):
            row[c] = per_cls.get(c, "")
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    print("[Done]")
    print(f"  csv: {csv_path}")


if __name__ == "__main__":
    main()
