"""
DocLING layout extraction microservice for PDFTrusted translate pipeline.
POST /extract?max_pages=10&page_offset=0 — PDF → bbox blocks + table cells.
Falls back to pdfplumber when docling package is unavailable.
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


class ExtractResponse(BaseModel):
    blocks: list[ExtractBlock]
    page_count: int
    engine: str = "docling"
    tables_found: int = 0


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

    return ExtractResponse(blocks=blocks, page_count=page_count, engine="pdfplumber", tables_found=0)


@app.get("/health")
def health() -> dict[str, Any]:
    installed = docling_available()
    engine_env = os.getenv("DOCLING_ENGINE", "auto")
    active = "docling" if (engine_env == "docling" or (engine_env == "auto" and installed)) else "pdfplumber"
    return {
        "ok": True,
        "docling_installed": installed,
        "engine": engine_env,
        "active_engine": active,
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

    engine = os.getenv("DOCLING_ENGINE", "auto").lower()
    use_docling = engine == "docling" or (engine == "auto" and docling_available())

    try:
        if use_docling and docling_available():
            return extract_with_docling(raw, max_pages, page_offset)
        return extract_with_pdfplumber(raw, max_pages, page_offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
