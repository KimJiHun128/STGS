#!/usr/bin/env python3
"""
summarize_cholecseg_eval.py

무엇을 하는 스크립트인가?
- cholecseg_sub 실험 결과(result.json / result_thr_*.json)를 모아서
  threshold별로
  1) 데이터셋별 Total Average 비교표
  2) "등장한 클래스만" 포함한 Per-class 비교표
  를 생성한다.

비교 대상(고정):
- origin        : output/surgTPGS_origin/...
- none          : output/language_features_fine_none_dim3/...
- mean          : output/language_features_fine_mean_dim3/...
- area_weighted : output/language_features_fine_area_weighted_dim3/...

입력:
- --output_root (기본: output)
  아래에 각 실험의 result 파일이 존재해야 함.
  예) output/<method_dir>/cholecseg_sub/<video_name>/test/ours_3000/result_thr_0p40.json
      output/<method_dir>/cholecseg_sub/<video_name>/test/ours_3000/result.json

출력:
- --save_dir (기본: output/cholecseg_eval_compare)
  threshold마다 아래 파일 생성:
  - total_average_comparison_thr_<tag>.md/.csv
  - per_class_comparison_appeared_only_thr_<tag>.md/.csv

주의:
- eval_fine.py 포맷 기준으로,
  int 0 은 "해당 클래스가 GT에 등장하지 않음"으로 간주한다.
  float 0.0 은 "등장했지만 성능이 0"일 수 있으므로 등장 클래스로 유지한다.
"""

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
from xml.sax.saxutils import escape


METHOD_TO_DIR = {
    "origin": "surgTPGS_origin",
    "none": "language_features_fine_none_dim3",
    "mean": "language_features_fine_mean_dim3",
    "area_weighted": "language_features_fine_area_weighted_dim3",
}

DATASETS = [
    "video01_00080_0",
    "video01_00240_0",
    "video01_15019_0",
    "video12_15750_0",
    "video17_01803_0",
]


def load_result(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_present_value(v: Any) -> bool:
    # In eval_fine.py:
    # - absent class => int 0
    # - present but failed can be float 0.0
    return not (isinstance(v, int) and v == 0)


def short_dataset_label(dataset_name: str) -> str:
    # video01_00080_0 -> 00080
    m = re.match(r"video\d+_(\d{5})_\d+$", dataset_name)
    if m:
        return m.group(1)
    return dataset_name


def method_result_dir(output_root: Path, method_dir: str, dataset_name: str) -> Path:
    return output_root / method_dir / "cholecseg_sub" / dataset_name / "test" / "ours_3000"


def discover_threshold_tags(output_root: Path, method_dir: str = "language_features_fine_none_dim3") -> List[str]:
    # Use one threshold-sweep method as discovery source.
    tags = set()
    for ds in DATASETS:
        d = method_result_dir(output_root, method_dir, ds)
        if not d.exists():
            continue
        for p in d.glob("result_thr_*.json"):
            m = re.match(r"result_thr_(.+)\.json$", p.name)
            if m:
                tags.add(m.group(1))
    # fallback: if no threshold sweep files exist, use single "default"
    if not tags:
        return ["default"]
    return sorted(tags)


def pick_result_path(output_root: Path, method_dir: str, dataset_name: str, thr_tag: str) -> Path:
    d = method_result_dir(output_root, method_dir, dataset_name)
    if thr_tag != "default":
        p_thr = d / f"result_thr_{thr_tag}.json"
        if p_thr.exists():
            return p_thr
    p_default = d / "result.json"
    if p_default.exists():
        return p_default
    # Last fallback for strict threshold mode
    p_thr = d / f"result_thr_{thr_tag}.json"
    return p_thr


def build_tables(output_root: Path, thr_tag: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        ds_label = short_dataset_label(ds)
        method_data: Dict[str, Dict[str, Any]] = {}
        for method, method_dir in METHOD_TO_DIR.items():
            p = pick_result_path(output_root, method_dir, ds, thr_tag)
            if not p.exists():
                raise FileNotFoundError(f"Missing result file: {p}")
            method_data[method] = load_result(p)

        summary_rows.append({
            "dataset": ds_label,
            "origin_total_avg": float(method_data["origin"]["Total Average"]),
            "none_total_avg": float(method_data["none"]["Total Average"]),
            "mean_total_avg": float(method_data["mean"]["Total Average"]),
            "area_weighted_total_avg": float(method_data["area_weighted"]["Total Average"]),
        })

        # Collect appeared classes from any method (0 int means absent in that eval output)
        appeared_classes = set()
        for method in METHOD_TO_DIR.keys():
            per_class = method_data[method]["Total Avg Per Class"]
            for cls_name, val in per_class.items():
                if is_present_value(val):
                    appeared_classes.add(cls_name)

        for cls_name in sorted(appeared_classes):
            row = {"dataset": ds_label, "class": cls_name}
            for method in METHOD_TO_DIR.keys():
                per_class = method_data[method]["Total Avg Per Class"]
                val = per_class.get(cls_name, 0)
                row[method] = float(val) if isinstance(val, (int, float)) else val
            class_rows.append(row)

    return summary_rows, class_rows


def to_markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def to_csv(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    out = [",".join(headers)]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            vals.append(str(v))
        out.append(",".join(vals))
    return "\n".join(out) + "\n"


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Requested custom highlights in per-class sheet.
# dataset label (short) -> class names to highlight
HIGHLIGHT_TARGETS = {
    "00080": {"liver"},
    "00240": {"grasper", "liver"},
    "15019": {"abdominal wall", "grasper"},
    "15750": {"fat", "l-hook electrocautery"},
    "01803": {"abdominal wall", "grasper"},
}


def col_to_excel(n: int) -> str:
    # 1-based index -> Excel column letters
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def write_sheet_xml(
    rows: List[List[Any]],
    bold_cells: set,
    separator_rows: set,
    highlight_cells: set,
) -> str:
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        out.append(f'<row r="{r_idx}">')
        for c_idx, v in enumerate(row, start=1):
            ref = f"{col_to_excel(c_idx)}{r_idx}"
            is_num = isinstance(v, (int, float))

            # style id:
            # 0: normal
            # 1: bold
            # 2: top-border
            # 3: bold + top-border
            # 4: highlight
            # 5: highlight + top-border
            # 6: highlight + bold
            # 7: highlight + bold + top-border
            style_id = 0
            if r_idx in separator_rows:
                style_id = 2
            if (r_idx, c_idx) in highlight_cells:
                style_id = 5 if style_id == 2 else 4
            if (r_idx, c_idx) in bold_cells:
                if style_id == 2:
                    style_id = 3
                elif style_id == 5:
                    style_id = 7
                elif style_id == 4:
                    style_id = 6
                else:
                    style_id = 1

            if is_num:
                val = f"{float(v):.6f}" if isinstance(v, float) else str(v)
                out.append(f'<c r="{ref}" s="{style_id}"><v>{val}</v></c>')
            else:
                txt = escape(str(v))
                out.append(
                    f'<c r="{ref}" t="inlineStr" s="{style_id}"><is><t>{txt}</t></is></c>'
                )
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write_xlsx_tables(
    xlsx_path: Path,
    summary_rows: List[Dict[str, Any]],
    class_rows: List[Dict[str, Any]],
    summary_headers: List[str],
    class_headers: List[str],
) -> None:
    # Build table rows (header + body)
    summary_table = [summary_headers] + [[r.get(h, "") for h in summary_headers] for r in summary_rows]
    class_table = [class_headers] + [[r.get(h, "") for h in class_headers] for r in class_rows]

    # Bold max values per class row across methods
    # class headers: dataset,class,origin,none,mean,area_weighted
    method_start_col = 3  # 1-based
    method_end_col = 6
    class_bold_cells = set()
    class_separator_rows = set()
    prev_ds = None
    for i, r in enumerate(class_rows, start=2):  # + header row
        ds = r["dataset"]
        if prev_ds is None or ds != prev_ds:
            class_separator_rows.add(i)
        prev_ds = ds

        vals = []
        for c in range(method_start_col, method_end_col + 1):
            v = class_table[i - 1][c - 1]
            vals.append(float(v) if isinstance(v, (int, float)) else float("-inf"))
        mx = max(vals)
        for offset, v in enumerate(vals):
            if v == mx:
                class_bold_cells.add((i, method_start_col + offset))

    # Summary: separator on every data row for readability
    summary_separator_rows = set(range(2, 2 + len(summary_rows)))
    summary_bold_cells = set()
    summary_highlight_cells = set()

    class_highlight_cells = set()
    for i, r in enumerate(class_rows, start=2):
        ds = str(r["dataset"])
        cls = str(r["class"])
        cls_norm = norm_text(cls)
        target_norms = {norm_text(x) for x in HIGHLIGHT_TARGETS.get(ds, set())}
        if cls_norm in target_norms:
            # highlight the full row across visible columns for better readability
            for c in range(1, 7):
                class_highlight_cells.add((i, c))

    sheet1_xml = write_sheet_xml(summary_table, summary_bold_cells, summary_separator_rows, summary_highlight_cells)
    sheet2_xml = write_sheet_xml(class_table, class_bold_cells, class_separator_rows, class_highlight_cells)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    wb = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="total_average" sheetId="1" r:id="rId1"/>
    <sheet name="per_class" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""

    wb_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    # style 0 normal, 1 bold, 2 top border, 3 bold+top border
    # style 4 highlight, 5 highlight+top border, 6 highlight+bold, 7 highlight+bold+top border
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFC000"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left/><right/><top style="thin"/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="1" borderId="0" xfId="0" applyFill="1"/>
    <xf numFmtId="0" fontId="0" fillId="1" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="0" fontId="1" fillId="1" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
  </cellXfs>
</styleSheet>"""

    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""

    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>cholecseg eval summary</dc:title>
</cp:coreProperties>"""

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("docProps/app.xml", app)
        z.writestr("docProps/core.xml", core)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/styles.xml", styles)
        z.writestr("xl/worksheets/sheet1.xml", sheet1_xml)
        z.writestr("xl/worksheets/sheet2.xml", sheet2_xml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--save_dir", type=str, default="output/cholecseg_eval_compare")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    threshold_tags = discover_threshold_tags(output_root)

    summary_headers = [
        "dataset",
        "origin_total_avg",
        "none_total_avg",
        "mean_total_avg",
        "area_weighted_total_avg",
    ]
    class_headers = [
        "dataset",
        "class",
        "origin",
        "none",
        "mean",
        "area_weighted",
    ]

    generated_files = []
    for thr_tag in threshold_tags:
        summary_rows, class_rows = build_tables(output_root, thr_tag)
        total_md = save_dir / f"total_average_comparison_thr_{thr_tag}.md"
        class_md = save_dir / f"per_class_comparison_appeared_only_thr_{thr_tag}.md"
        total_csv = save_dir / f"total_average_comparison_thr_{thr_tag}.csv"
        class_csv = save_dir / f"per_class_comparison_appeared_only_thr_{thr_tag}.csv"

        total_md.write_text(to_markdown_table(summary_rows, summary_headers), encoding="utf-8")
        class_md.write_text(to_markdown_table(class_rows, class_headers), encoding="utf-8")
        total_csv.write_text(to_csv(summary_rows, summary_headers), encoding="utf-8")
        class_csv.write_text(to_csv(class_rows, class_headers), encoding="utf-8")
        xlsx_file = save_dir / f"comparison_thr_{thr_tag}.xlsx"
        write_xlsx_tables(xlsx_file, summary_rows, class_rows, summary_headers, class_headers)
        generated_files.extend([total_md, class_md, total_csv, class_csv, xlsx_file])

    print("[Done]")
    print(f"  save_dir: {save_dir}")
    print(f"  threshold_tags: {threshold_tags}")
    print(f"  files:")
    for p in generated_files:
        print(f"    - {p}")


if __name__ == "__main__":
    main()
