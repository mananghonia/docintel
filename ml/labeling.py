"""Field schema, BIO tag scheme, and bbox->token label assignment.

This module is the ground truth of the whole system: every model, metric and
UI colour ultimately refers to the FIELDS list and the BIO tags defined here.

Trap #2 from the project plan lives here: if assign_labels() is wrong, every
downstream number is garbage. Use visualize_alignment() on at least 5 real
documents before training anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

# ---------------------------------------------------------------------------
# Field schema
# ---------------------------------------------------------------------------

FIELDS: list[str] = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "vendor_gstin",
    "buyer_name",
    "buyer_gstin",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
    "po_number",
]

# BIO scheme: O + B-/I- per field.
O_TAG = "O"
TAGS: list[str] = [O_TAG] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
TAG2ID: dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2TAG: dict[int, str] = {i: t for t, i in TAG2ID.items()}
NUM_TAGS = len(TAGS)

# Fields whose values are amounts (used by postprocess arithmetic check).
AMOUNT_FIELDS = {"subtotal", "tax_amount", "total_amount"}


def tag_field(tag: str) -> str | None:
    """'B-total_amount' -> 'total_amount'; 'O' -> None."""
    return None if tag == O_TAG else tag.split("-", 1)[1]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """One OCR token: text plus its bounding box in pixel coordinates."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0
    ocr_conf: float = 1.0  # OCR engine's own confidence, 0-1
    tag: str = O_TAG       # ground-truth or predicted BIO tag

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Token":
        return cls(**{k: d[k] for k in
                      ("text", "x0", "y0", "x1", "y1", "page", "ocr_conf", "tag")
                      if k in d})


@dataclass
class FieldAnnotation:
    """Ground-truth annotation: a field, its value, and where it is on the page."""

    field: str
    value: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0


@dataclass
class Document:
    """A document = ordered tokens + page geometry + optional annotations."""

    doc_id: str
    tokens: list[Token]
    page_width: float
    page_height: float
    annotations: list[FieldAnnotation] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def tags(self) -> list[str]:
        return [t.tag for t in self.tokens]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "tokens": [t.to_dict() for t in self.tokens],
            "annotations": [asdict(a) for a in self.annotations],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            tokens=[Token.from_dict(t) for t in d["tokens"]],
            page_width=d["page_width"],
            page_height=d["page_height"],
            annotations=[FieldAnnotation(**a) for a in d.get("annotations", [])],
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# bbox -> token label assignment
# ---------------------------------------------------------------------------

def _overlap_ratio(tok: Token, ann: FieldAnnotation) -> float:
    """Fraction of the token's area covered by the annotation box."""
    ix0, iy0 = max(tok.x0, ann.x0), max(tok.y0, ann.y0)
    ix1, iy1 = min(tok.x1, ann.x1), min(tok.y1, ann.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    tok_area = max(tok.width * tok.height, 1e-6)
    return inter / tok_area


def assign_labels(doc: Document, min_overlap: float = 0.5) -> Document:
    """Assign BIO tags to tokens from annotation bounding boxes (in place).

    A token belongs to an annotation when >= min_overlap of the token's own
    area lies inside the annotation box. Within one annotation, tokens are
    ordered by reading order (page, then y, then x) and tagged B-, I-, I-...

    If a token matches several annotations, the one covering it most wins.
    """
    for tok in doc.tokens:
        tok.tag = O_TAG

    for ann in doc.annotations:
        if ann.field not in FIELDS:
            raise ValueError(f"Unknown field in annotation: {ann.field!r}")
        matched = [
            (idx, _overlap_ratio(tok, ann))
            for idx, tok in enumerate(doc.tokens)
            if tok.page == ann.page and _overlap_ratio(tok, ann) >= min_overlap
        ]
        # Reading order inside the annotation.
        matched.sort(key=lambda m: (doc.tokens[m[0]].cy, doc.tokens[m[0]].cx))
        for rank, (idx, ratio) in enumerate(matched):
            tok = doc.tokens[idx]
            # Keep an existing tag only if that annotation covers the token better.
            prev_ratio = doc.meta.get("_ratio", {}).get(idx, 0.0)
            if tok.tag != O_TAG and prev_ratio >= ratio:
                continue
            tok.tag = f"{'B' if rank == 0 else 'I'}-{ann.field}"
            doc.meta.setdefault("_ratio", {})[idx] = ratio

    doc.meta.pop("_ratio", None)
    return doc


def extract_field_values(tokens: Iterable[Token]) -> dict[str, str]:
    """Collapse a tagged token sequence into {field: text} (first span wins)."""
    values: dict[str, str] = {}
    current_field: str | None = None
    parts: list[str] = []
    for tok in tokens:
        f = tag_field(tok.tag)
        starts = tok.tag.startswith("B-")
        if f is None or starts or f != current_field:
            if current_field and current_field not in values:
                values[current_field] = " ".join(parts)
            current_field, parts = (f, [tok.text]) if f else (None, [])
        else:
            parts.append(tok.text)
    if current_field and current_field not in values:
        values[current_field] = " ".join(parts)
    return values


# ---------------------------------------------------------------------------
# Alignment sanity check (trap #2)
# ---------------------------------------------------------------------------

def visualize_alignment(doc: Document, out_path: str) -> str:
    """Render tokens + labels on a blank page so a human can eyeball alignment.

    Green boxes = labeled tokens (with their field name), grey = O tokens,
    red outlines = the raw annotation boxes. If green text doesn't sit inside
    red boxes, assign_labels or the OCR geometry is broken.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (int(doc.page_width), int(doc.page_height)), "white")
    d = ImageDraw.Draw(img)
    for ann in doc.annotations:
        d.rectangle([ann.x0, ann.y0, ann.x1, ann.y1], outline="red", width=2)
    for tok in doc.tokens:
        labeled = tok.tag != O_TAG
        d.rectangle([tok.x0, tok.y0, tok.x1, tok.y1],
                    outline="green" if labeled else "#cccccc")
        d.text((tok.x0, max(tok.y0 - 10, 0)), tok.text,
               fill="black" if labeled else "#999999")
        if labeled:
            d.text((tok.x0, tok.y1 + 1), tok.tag, fill="green")
    img.save(out_path)
    return out_path
