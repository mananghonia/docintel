"""Image preprocessing + OCR wrapper.

Turns an image (or PDF page) into a Document of Tokens. Preprocessing:
grayscale -> denoise -> binarise -> deskew. OCR via pytesseract if the
tesseract binary is installed; ocr_available() lets callers degrade
gracefully (synthetic documents never need OCR at all).
"""

from __future__ import annotations

import shutil
import uuid

from PIL import Image, ImageOps

from ml.labeling import Document, Token


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _estimate_skew(img: Image.Image, max_angle: float = 4.0, step: float = 0.5) -> float:
    """Brute-force skew estimate: the rotation that maximises the variance of
    row-wise ink density is the one that lines text rows up horizontally."""
    import numpy as np

    small = img.convert("L").resize((min(img.width, 600),
                                     int(img.height * min(img.width, 600) / img.width)))
    arr0 = 255 - np.asarray(small, dtype=np.float32)
    best_angle, best_score = 0.0, -1.0
    angle = -max_angle
    while angle <= max_angle:
        rotated = small.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
        arr = 255 - np.asarray(rotated, dtype=np.float32)
        score = float(np.var(arr.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = angle, score
        angle += step
    # No-op guard: don't rotate if the win over 0° is marginal.
    if abs(best_angle) < step:
        return 0.0
    return best_angle


def preprocess(img: Image.Image, deskew: bool = True) -> Image.Image:
    """Grayscale, autocontrast, deskew. Deliberately NO binarisation or
    median filtering: tesseract binarises internally with adaptive Otsu, and
    a fixed global threshold + median erosion destroys thin/anti-aliased
    strokes (measured: 65 tokens -> 6 on a blurred scan)."""
    g = ImageOps.autocontrast(img.convert("L"))
    if deskew:
        angle = _estimate_skew(g)
        if angle:
            g = g.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
    return g


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(img: Image.Image, doc_id: str | None = None,
              do_preprocess: bool = True, page: int = 0) -> Document:
    """Run tesseract on an image and return a Document of word Tokens."""
    if not ocr_available():
        raise RuntimeError(
            "tesseract binary not found on PATH. Install it "
            "(https://github.com/tesseract-ocr/tesseract) or use synthetic "
            "documents, which carry their own tokens."
        )
    import pytesseract

    work = preprocess(img) if do_preprocess else img
    data = pytesseract.image_to_data(work, output_type=pytesseract.Output.DICT)

    tokens: list[Token] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:  # conf -1 marks structural rows, not words
            continue
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        tokens.append(Token(text=text, x0=x, y0=y, x1=x + w, y1=y + h,
                            page=page, ocr_conf=conf / 100.0))

    return Document(
        doc_id=doc_id or f"ocr-{uuid.uuid4().hex[:12]}",
        tokens=tokens,
        page_width=img.width,
        page_height=img.height,
        meta={"source": "ocr"},
    )


def ocr_pdf(path: str, doc_id: str | None = None, dpi: int = 150) -> Document:
    """OCR every page of a PDF into one Document.

    Rendered with PyMuPDF (pure pip install) rather than pdf2image, which
    needs the poppler system binary — a real deployment obstacle on Windows.
    """
    import fitz  # PyMuPDF

    zoom = dpi / 72.0
    pages = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    if not pages:
        raise ValueError(f"No pages rendered from {path}")
    docs = [ocr_image(img, page=i) for i, img in enumerate(pages)]
    tokens = [t for d in docs for t in d.tokens]
    return Document(
        doc_id=doc_id or f"pdf-{uuid.uuid4().hex[:12]}",
        tokens=tokens,
        page_width=pages[0].width,
        page_height=pages[0].height,
        meta={"source": "ocr_pdf", "n_pages": len(pages)},
    )
