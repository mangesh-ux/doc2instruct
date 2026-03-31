from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz


@dataclass
class PageImage:
    book_path: Path
    page_number: int
    image_data_url: str


def _render_page_data_url(doc: fitz.Document, *, page_index: int, dpi: int) -> str:
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def iter_pdf_pages_as_data_urls(
    pdf_path: Path, *, dpi: int, max_pages: int
) -> Iterator[PageImage]:
    doc = fitz.open(pdf_path)
    max_page_count = min(doc.page_count, max_pages)

    for i in range(max_page_count):
        yield PageImage(
            book_path=pdf_path,
            page_number=i + 1,
            image_data_url=_render_page_data_url(doc, page_index=i, dpi=dpi),
        )

    doc.close()


def render_single_page_as_data_url(pdf_path: Path, *, page_number: int, dpi: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"Invalid page number {page_number} for {pdf_path}")
        return _render_page_data_url(doc, page_index=page_index, dpi=dpi)
    finally:
        doc.close()


def extract_page_text(pdf_path: Path, *, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"Invalid page number {page_number} for {pdf_path}")
        page = doc.load_page(page_index)
        return page.get_text("text")
    finally:
        doc.close()


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return int(doc.page_count)
    finally:
        doc.close()
