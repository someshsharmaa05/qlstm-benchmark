"""
fill_manuscript.py
==================
Replace placeholder tables and figure placeholders in the docx manuscript
with the real results.

Strategy: open the manuscript with python-docx, find each table by its
caption text immediately above ('Table 2  Mean test root...'), and rewrite
the cell contents from the corresponding CSV in `results/tables/`.
Figure placeholders are likewise replaced by inserting the actual PNGs.

Usage
-----
    python -m src.fill_manuscript --in QLSTM_Multi_Asset_Benchmark_JoSC.docx \
                                  --out results/manuscript_filled.docx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


TABLE_TITLE_TO_CSV = {
    "Table 1": "table1_dataset.csv",
    "Table 2": "table2_test_rmse.csv",
    "Table 3": "table3_directional_accuracy.csv",
    "Table 4": "table4_wilcoxon.csv",
    "Table 5": "table5_walk_forward.csv",
    "Table 6": "table6_trading.csv",
    "Table 7": "table7_noise.csv",
    "Table 8": "table8_vqc_depth.csv",
}

FIGURE_PLACEHOLDER_TO_PNG = {
    "Fig. 1 placeholder": "fig1_pipeline.png",
    "Fig. 2 placeholder": "fig2_qlstm_cell.png",
    "Fig. 3 placeholder": "fig3_rmse_bar.png",
    "Fig. 4 placeholder": "fig4_noise_and_depth.png",
}


def _set_cell(cell, text: str, bold: bool = False):
    cell.text = ""                              # clear
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(str(text))
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.5)
    run.font.bold = bold


def replace_table(doc_table, df: pd.DataFrame):
    """Rewrite an existing docx table with values from df, keeping its grid shape."""
    needed_rows = len(df) + 1
    needed_cols = len(df.columns)
    cur_rows = len(doc_table.rows)
    cur_cols = len(doc_table.columns)

    # Adjust row count to match the data exactly
    while cur_rows < needed_rows:
        doc_table.add_row(); cur_rows += 1
    while cur_rows > needed_rows:
        tbl_el = doc_table._tbl
        tbl_el.remove(doc_table.rows[-1]._tr); cur_rows -= 1

    # Header row
    for j, col in enumerate(df.columns):
        if j < cur_cols:
            _set_cell(doc_table.rows[0].cells[j], col, bold=True)

    # Body rows
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            if j < cur_cols:
                _set_cell(doc_table.rows[i].cells[j], row[col])


def fill(in_docx: Path, out_docx: Path, tables_dir: Path, figures_dir: Path):
    doc = Document(str(in_docx))

    # ---- 1. Replace tables ----
    # Iterate document body, find paragraphs that begin with "Table N", then
    # locate the next table that follows.
    body_elems = list(doc.element.body.iter())
    table_idx = 0
    paragraphs = list(doc.paragraphs)
    doc_tables = list(doc.tables)

    # Map of "Table N" -> index of doc table that comes after that caption
    # Iterate by document order
    seen_titles = []
    elems_in_order = doc.element.body
    table_counter = -1
    pending_title = None
    title_to_table = {}
    for child in elems_in_order.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = "".join(t.text or "" for t in child.iter() if t.tag.endswith("}t"))
            m = re.match(r"\s*Table\s+(\d+)\b", text)
            if m:
                pending_title = f"Table {m.group(1)}"
        elif tag == "tbl":
            table_counter += 1
            if pending_title is not None:
                title_to_table[pending_title] = table_counter
                pending_title = None

    for title, idx in title_to_table.items():
        if title not in TABLE_TITLE_TO_CSV: continue
        csv_path = tables_dir / TABLE_TITLE_TO_CSV[title]
        if not csv_path.exists():
            print(f"  skip {title}: {csv_path.name} not found")
            continue
        df = pd.read_csv(csv_path)
        if idx >= len(doc_tables):
            print(f"  skip {title}: table index {idx} out of range")
            continue
        print(f"  filling {title} from {csv_path.name}  ({len(df)} rows)")
        replace_table(doc_tables[idx], df)

    # ---- 2. Replace figure placeholders ----
    for p in doc.paragraphs:
        for placeholder, png_name in FIGURE_PLACEHOLDER_TO_PNG.items():
            if placeholder in p.text:
                png_path = figures_dir / png_name
                if not png_path.exists():
                    print(f"  skip {placeholder}: {png_name} not found")
                    continue
                print(f"  inserting {png_name} for {placeholder}")
                # Clear and insert image
                for r_el in list(p.runs): r_el.text = ""
                run = p.add_run()
                run.add_picture(str(png_path), width=Inches(6.0))
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Same for placeholders found inside single-cell wrapper tables
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for placeholder, png_name in FIGURE_PLACEHOLDER_TO_PNG.items():
                    if placeholder in cell.text:
                        png_path = figures_dir / png_name
                        if not png_path.exists(): continue
                        print(f"  inserting {png_name} (cell) for {placeholder}")
                        cell.text = ""
                        run = cell.paragraphs[0].add_run()
                        run.add_picture(str(png_path), width=Inches(5.8))
                        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_docx))
    print(f"\nWrote {out_docx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="QLSTM_Multi_Asset_Benchmark_JoSC.docx")
    ap.add_argument("--out", dest="out", default="results/manuscript_filled.docx")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()
    fill(Path(args.inp), Path(args.out), Path(args.tables), Path(args.figures))


if __name__ == "__main__":
    main()
