"""
DocLING layout extraction microservice for PDFTrusted translate pipeline.
POST /extract?max_pages=10&page_offset=0 — PDF → bbox blocks + table cells.

Engine priority (DOCLING_ENGINE=auto):
  1. docling (best layout, needs RapidOCR deps)
  2. pymupdf line spans (digital PDFs — no OCR)
  3. pdfplumber words (last resort)
"""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="PDFTrusted DocLING Extract", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractBlock(BaseModel):
    id: str
    page_index: int
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float = 12.0
    font_name: str | None = None
    rotation: float = 0.0
    block_type: str = "text"
    table_id: str | None = None
    row: int | None = None
    col: int | None = None
    color_r: float | None = None
    color_g: float | None = None
    color_b: float | None = None
    zone: str | None = None


class ExtractResponse(BaseModel):
    blocks: list[ExtractBlock]
    page_count: int
    engine: str = "docling"
    tables_found: int = 0


_converter = None
_last_extract_error: str | None = None


def reset_converter() -> None:
    global _converter
    _converter = None


def docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
    return _converter


def page_height(doc: Any, page_no: int) -> float:
    pages = getattr(doc, "pages", None) or {}
    page = pages.get(page_no) or pages.get(str(page_no))
    if page is not None and hasattr(page, "size"):
        return float(page.size.height)
    if page is not None and isinstance(page, dict):
        size = page.get("size") or {}
        return float(size.get("height") or 792.0)
    return 792.0


def bbox_to_block(
    bbox: Any,
    page_no: int,
    page_h: float,
    text: str,
    block_id: str,
    *,
    block_type: str = "text",
    table_id: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> ExtractBlock:
    l = float(bbox.l)
    t = float(bbox.t)
    r = float(bbox.r)
    b = float(bbox.b)
    w = max(4.0, r - l)
    raw_h = max(4.0, b - t)
    clean = (text or "").strip()
    est_lines = max(1, clean.count("\n") + 1, (len(clean) // max(40, int(w * 0.15))) + 1)
    h = raw_h if raw_h >= 10 else max(12.0, est_lines * 11.0)
    font_size = max(9.0, min(h * 0.88, 14.0))
    return ExtractBlock(
        id=block_id,
        page_index=page_no - 1,
        text=clean,
        x=l,
        y=page_h - b,
        width=w,
        height=h,
        font_size=font_size,
        block_type=block_type,
        table_id=table_id,
        row=row,
        col=col,
    )


def rect_iou(a: ExtractBlock, x: float, y: float, w: float, h: float) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = x + w, y + h
    ox = max(0.0, min(ax2, bx2) - max(a.x, x))
    oy = max(0.0, min(ay2, by2) - max(a.y, y))
    inter = ox * oy
    if inter <= 0:
        return 0.0
    union = a.width * a.height + w * h - inter
    return inter / union if union > 0 else 0.0


def classify_zones(blocks: list[ExtractBlock], page_heights: dict[int, float]) -> None:
    for b in blocks:
        page_h = page_heights.get(b.page_index, 792.0)
        top = b.y / page_h
        bottom = (b.y + b.height) / page_h
        if bottom >= 0.9:
            b.zone = "footer"
        elif top <= 0.1:
            b.zone = "header"
        else:
            b.zone = "body"


def enrich_with_pymupdf(pdf_bytes: bytes, blocks: list[ExtractBlock]) -> list[ExtractBlock]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return blocks

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_heights: dict[int, float] = {i: float(doc[i].rect.height) for i in range(len(doc))}
        classify_zones(blocks, page_heights)

        for page_idx in {b.page_index for b in blocks}:
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            page_h = float(page.rect.height)
            spans: list[dict[str, Any]] = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        bbox = span.get("bbox")
                        if not text or not bbox:
                            continue
                        x0, y0, x1, y1 = bbox
                        spans.append(
                            {
                                "x": float(x0),
                                "y": page_h - float(y1),
                                "width": float(x1 - x0),
                                "height": float(y1 - y0),
                                "size": float(span.get("size", 12)),
                                "font": span.get("font"),
                                "color": int(span.get("color", 0) or 0),
                            }
                        )

            for pb in [b for b in blocks if b.page_index == page_idx]:
                best: dict[str, Any] | None = None
                best_iou = 0.0
                for sp in spans:
                    iou = rect_iou(pb, sp["x"], sp["y"], sp["width"], sp["height"])
                    if iou > best_iou:
                        best_iou = iou
                        best = sp
                if not best or best_iou < 0.1:
                    continue
                pb.font_size = max(pb.font_size, float(best["size"]))
                if best.get("font"):
                    pb.font_name = str(best["font"])
                color = int(best.get("color", 0) or 0)
                if color != 0:
                    pb.color_r = ((color >> 16) & 255) / 255.0
                    pb.color_g = ((color >> 8) & 255) / 255.0
                    pb.color_b = (color & 255) / 255.0
    finally:
        doc.close()

    colored = sum(1 for b in blocks if b.color_r is not None)
    print(f"[pymupdf-enrich] blocks={len(blocks)} colored={colored}")
    return blocks


def iter_docling_blocks(doc: Any, page_offset: int, max_pages: int) -> tuple[list[ExtractBlock], int]:
    blocks: list[ExtractBlock] = []
    seq = 0
    page_end = page_offset + max_pages

    for item in getattr(doc, "texts", None) or []:
        text = getattr(item, "text", None) or ""
        if not str(text).strip():
            continue
        for prov in getattr(item, "prov", None) or []:
            page_no = int(getattr(prov, "page_no", 0) or 0)
            if page_no < page_offset + 1 or page_no > page_end:
                continue
            bbox = getattr(prov, "bbox", None)
            if bbox is None:
                continue
            label = getattr(item, "label", None)
            block_type = str(getattr(label, "value", label) or "text")
            blocks.append(
                bbox_to_block(
                    bbox,
                    page_no,
                    page_height(doc, page_no),
                    str(text),
                    f"t{seq}",
                    block_type=block_type,
                )
            )
            seq += 1

    table_idx = 0
    for table in getattr(doc, "tables", None) or []:
        table_id = f"tbl{table_idx}"
        table_idx += 1
        data = getattr(table, "data", None)
        cells = getattr(data, "table_cells", None) if data else None
        if not cells:
            continue
        for cell in cells:
            text = getattr(cell, "text", None) or ""
            if not str(text).strip():
                continue
            prov_list = getattr(cell, "prov", None) or getattr(table, "prov", None) or []
            for prov in prov_list:
                page_no = int(getattr(prov, "page_no", 0) or 0)
                if page_no < page_offset + 1 or page_no > page_end:
                    continue
                bbox = getattr(prov, "bbox", None) or getattr(cell, "bbox", None)
                if bbox is None:
                    continue
                row = getattr(cell, "row", None)
                col = getattr(cell, "col", None)
                blocks.append(
                    bbox_to_block(
                        bbox,
                        page_no,
                        page_height(doc, page_no),
                        str(text),
                        f"c{seq}",
                        block_type="table_cell",
                        table_id=table_id,
                        row=int(row) if row is not None else None,
                        col=int(col) if col is not None else None,
                    )
                )
                seq += 1

    tables_found = sum(1 for b in blocks if b.block_type == "table_cell")
    return blocks, tables_found


def extract_with_docling(pdf_bytes: bytes, max_pages: int, page_offset: int) -> ExtractResponse:
    suffix = f"-{uuid.uuid4().hex}.pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        result = get_converter().convert(tmp_path)
        doc = result.document
        blocks, tables_found = iter_docling_blocks(doc, page_offset, max_pages)
        blocks = enrich_with_pymupdf(pdf_bytes, blocks)
        page_count = len(getattr(doc, "pages", {}) or {})
        return ExtractResponse(
            blocks=blocks,
            page_count=page_count,
            engine="docling",
            tables_found=tables_found,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_with_pymupdf_lines(pdf_bytes: bytes, max_pages: int, page_offset: int) -> ExtractResponse:
    """Line-level extract via PyMuPDF — stable for digital CVs, no RapidOCR."""
    import fitz

    blocks: list[ExtractBlock] = []
    seq = 0
    page_heights: dict[int, float] = {}
    page_count = 0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = len(doc)
        end = min(page_count, page_offset + max_pages)
        for page_idx in range(page_offset, end):
            page = doc[page_idx]
            page_h = float(page.rect.height)
            page_heights[page_idx] = page_h
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    parts = [str(s.get("text", "")) for s in spans if s.get("text")]
                    text = "".join(parts).strip()
                    if not text:
                        continue
                    x0 = min(float(s["bbox"][0]) for s in spans)
                    y0 = min(float(s["bbox"][1]) for s in spans)
                    x1 = max(float(s["bbox"][2]) for s in spans)
                    y1 = max(float(s["bbox"][3]) for s in spans)
                    max_size = max(float(s.get("size", 10) or 10) for s in spans)
                    font_name = next((str(s.get("font")) for s in spans if s.get("font")), None)
                    color = next((int(s.get("color", 0) or 0) for s in spans if s.get("color")), 0)
                    eb = ExtractBlock(
                        id=f"m{seq}",
                        page_index=page_idx,
                        text=text,
                        x=x0,
                        y=page_h - y1,
                        width=max(1.0, x1 - x0),
                        height=max(1.0, y1 - y0),
                        font_size=max_size,
                        font_name=font_name,
                        block_type="text",
                    )
                    if color != 0:
                        eb.color_r = ((color >> 16) & 255) / 255.0
                        eb.color_g = ((color >> 8) & 255) / 255.0
                        eb.color_b = (color & 255) / 255.0
                    blocks.append(eb)
                    seq += 1
    finally:
        doc.close()

    classify_zones(blocks, page_heights)
    print(f"[pymupdf-extract] blocks={len(blocks)} pages={page_count}")
    return ExtractResponse(blocks=blocks, page_count=page_count, engine="pymupdf", tables_found=0)


def extract_with_pdfplumber(pdf_bytes: bytes, max_pages: int, page_offset: int) -> ExtractResponse:
    import pdfplumber

    blocks: list[ExtractBlock] = []
    key = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        start = page_offset
        end = min(len(pdf.pages), start + max_pages)
        for page_idx in range(start, end):
            page = pdf.pages[page_idx]
            page_h = float(page.height)
            for word in page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False):
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                x0 = float(word["x0"])
                x1 = float(word["x1"])
                bottom = float(word["bottom"])
                top = float(word["top"])
                blocks.append(
                    ExtractBlock(
                        id=f"p{key}",
                        page_index=page_idx,
                        text=text,
                        x=x0,
                        y=page_h - bottom,
                        width=max(1.0, x1 - x0),
                        height=max(1.0, bottom - top),
                        font_size=max(6.0, min(bottom - top, 24.0)),
                        block_type="text",
                    )
                )
                key += 1
        page_count = len(pdf.pages)

    blocks = enrich_with_pymupdf(pdf_bytes, blocks)
    return ExtractResponse(blocks=blocks, page_count=page_count, engine="pdfplumber", tables_found=0)


def run_extract(pdf_bytes: bytes, max_pages: int, page_offset: int) -> ExtractResponse:
    """Try docling first; fall back to pymupdf lines, then pdfplumber."""
    global _last_extract_error

    engine = os.getenv("DOCLING_ENGINE", "auto").lower()
    use_docling = engine == "docling" or (engine == "auto" and docling_available())
    use_pymupdf = engine in ("auto", "pymupdf", "pdfplumber")
    use_pdfplumber = engine in ("auto", "pdfplumber")

    if use_docling and docling_available():
        try:
            result = extract_with_docling(pdf_bytes, max_pages, page_offset)
            if result.blocks:
                _last_extract_error = None
                return result
            print("[extract] docling returned 0 blocks, trying pymupdf")
        except Exception as exc:
            _last_extract_error = str(exc)
            print(f"[extract] docling failed ({exc}), falling back to pymupdf")
            reset_converter()

    if use_pymupdf and engine != "pdfplumber":
        try:
            result = extract_with_pymupdf_lines(pdf_bytes, max_pages, page_offset)
            if result.blocks:
                _last_extract_error = None
                return result
            print("[extract] pymupdf returned 0 blocks, trying pdfplumber")
        except Exception as exc:
            _last_extract_error = str(exc)
            print(f"[extract] pymupdf failed ({exc}), trying pdfplumber")

    if use_pdfplumber:
        result = extract_with_pdfplumber(pdf_bytes, max_pages, page_offset)
        _last_extract_error = None
        return result

    raise RuntimeError(_last_extract_error or "no extraction engine available")


@app.get("/health")
def health() -> dict[str, Any]:
    installed = docling_available()
    engine_env = os.getenv("DOCLING_ENGINE", "auto")
    return {
        "ok": True,
        "docling_installed": installed,
        "engine": engine_env,
        "fallback_chain": ["docling", "pymupdf", "pdfplumber"],
        "last_extract_error": _last_extract_error,
        "service": "docling-extract",
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    max_pages: int = Query(10, ge=1, le=100),
    page_offset: int = Query(0, ge=0),
):
    raw = await file.read()
    if not raw or raw[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="invalid_pdf")

    try:
        return run_extract(raw, max_pages, page_offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
