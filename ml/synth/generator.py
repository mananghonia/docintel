"""Synthetic invoice generator.

Generates Document objects (tokens + geometry + annotations) directly, so the
labels are perfect and free — no OCR involved. Optionally renders a document
to a PIL image with scan-style augmentations for the visual tier and for
exercising the OCR pipeline.

Two difficulty modes:

easy (hard=False)   fully-random layouts, clean text. Every model saturates
                    quickly — good for pipeline tests, useless for comparing
                    models or acquisition strategies.

hard (hard=True)    what real scanned invoices actually look like:
                    - 8 vendor template FAMILIES drawn from a skewed (Zipf)
                      distribution: layout signatures cluster, so diversity
                      sampling has real structure to exploit, and rare
                      families are where uncertainty sampling earns its keep
                    - label-above layouts that break same-line keyword anchors
                    - DISTRACTOR fields (quotation no, delivery date, IRN,
                      discounts, bank a/c) that look exactly like targets
                    - OCR-style character corruption (0<->O, 1<->l, 5<->S...),
                      token dropout (sometimes the anchor itself), bbox
                      jitter, and degraded ocr_conf
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass
from datetime import date, timedelta

from ml.labeling import Document, FieldAnnotation, Token
from ml.postprocess import gstin_check_char

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VENDOR_STEMS = [
    "Sharma", "Patel", "Mehta", "Acme", "Zenith", "Orbit", "Kaveri", "Indus",
    "Falcon", "Nimbus", "Vertex", "Stellar", "Global", "Pioneer", "Crescent",
]
VENDOR_SUFFIXES = [
    "Traders", "Industries", "Enterprises", "Pvt Ltd", "LLP", "& Sons",
    "Solutions", "Exports", "Textiles", "Electricals", "Logistics",
]
BUYER_NAMES = [
    "Reliant Retail Ltd", "BlueKart Commerce", "Meridian Hotels",
    "Apex Manufacturing Co", "Trident Pharma", "Everest Foods",
    "Sunrise Automobiles", "Metro Wholesale Mart",
]
ITEM_NAMES = [
    "Copper Wire 2.5mm", "Office Chair", "A4 Paper Ream", "LED Panel 40W",
    "Steel Bracket", "USB-C Cable", "Packing Tape", "Safety Gloves",
    "Cement Bag 50kg", "Printer Toner", "Ethernet Switch", "Wall Clock",
]

# Label wording variations — the model must key on patterns, not one string.
LABELS = {
    "invoice_number": ["Invoice No:", "Invoice #", "Inv No.", "Bill Number:", "Invoice Number:"],
    "invoice_date": ["Invoice Date:", "Date:", "Dated:", "Bill Date:"],
    "due_date": ["Due Date:", "Payment Due:", "Due By:"],
    "po_number": ["PO Number:", "P.O. No:", "Purchase Order:"],
    "vendor_gstin": ["GSTIN:", "GST No:", "GSTIN/UIN:"],
    "buyer_gstin": ["Buyer GSTIN:", "GSTIN:", "GST No:"],
    "subtotal": ["Subtotal:", "Sub Total:", "Taxable Value:"],
    "tax_amount": ["GST (18%):", "Tax:", "IGST:", "CGST+SGST:"],
    "total_amount": ["Total:", "Grand Total:", "Amount Payable:", "TOTAL:"],
}

DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%Y-%m-%d", "%d.%m.%Y"]

PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi
CHAR_W, LINE_H = 10, 26      # crude monospace geometry for token boxes

N_FAMILIES = 8


# ---------------------------------------------------------------------------
# Random value factories
# ---------------------------------------------------------------------------

def _rand_gstin(rng: random.Random) -> str:
    state = f"{rng.randint(1, 37):02d}"
    pan = (
        "".join(rng.choices(string.ascii_uppercase, k=5))
        + "".join(rng.choices(string.digits, k=4))
        + rng.choice(string.ascii_uppercase)
    )
    body = state + pan + str(rng.randint(1, 9)) + "Z"
    return body + gstin_check_char(body)


def _rand_invoice_number(rng: random.Random) -> str:
    styles = [
        lambda: f"INV-{rng.randint(2023, 2026)}-{rng.randint(1, 9999):04d}",
        lambda: f"{rng.choice(['SB', 'TX', 'GT'])}/{rng.randint(100, 999)}/{rng.randint(23, 26)}",
        lambda: f"{rng.randint(100000, 999999)}",
    ]
    return rng.choice(styles)()


def _indian_group(n: float) -> str:
    """1234567.89 -> '12,34,567.89' (lakh/crore grouping)."""
    whole, frac = f"{n:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{frac}"


def _fmt_amount(v: float, rng: random.Random, hard: bool = False) -> str:
    s = _indian_group(v) if (hard and rng.random() < 0.5) else f"{v:,.2f}"
    return rng.choice([s, f"Rs. {s}", f"INR {s}", f"₹{s}"])


# ---------------------------------------------------------------------------
# Token emission
# ---------------------------------------------------------------------------

class _Page:
    """Accumulates tokens; words are split so each Token is one word."""

    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.annotations: list[FieldAnnotation] = []

    def put(self, text: str, x: float, y: float,
            field: str | None = None, value: str | None = None) -> float:
        """Write `text` at (x, y) word by word. Returns the x after the text.

        If `field` is given, an annotation box is recorded around the words.
        """
        start_x = x
        for word in text.split():
            w = len(word) * CHAR_W
            self.tokens.append(Token(word, x, y, x + w, y + LINE_H - 6))
            x += w + CHAR_W  # one space
        if field is not None:
            self.annotations.append(FieldAnnotation(
                field=field, value=value if value is not None else text,
                x0=start_x - 2, y0=y - 2, x1=x - CHAR_W + 2, y1=y + LINE_H - 4,
            ))
        return x

    def kv(self, label: str, value: str, x: float, y: float, field: str,
           labels_above: bool = False) -> float:
        """Label + value inline, or label on one line and value below it
        (label-above layouts break same-line keyword anchors)."""
        if labels_above:
            self.put(label, x, y)
            self.put(value, x, y + LINE_H, field=field)
            return y + LINE_H * 2
        end = self.put(label, x, y)
        self.put(value, end + CHAR_W, y, field=field)
        return y + LINE_H


# ---------------------------------------------------------------------------
# Template families (hard mode)
# ---------------------------------------------------------------------------

@dataclass
class Style:
    family: int
    meta_side: str        # "right" | "left"
    labels_above: bool
    date_fmt: str
    label_idx: dict       # field -> fixed wording index for this family
    totals_x: int
    margin: int
    p_due: float
    p_po: float
    distractors: list[str]
    noise: float          # 0..1 corruption severity


_DISTRACTOR_POOL = [
    "quotation", "delivery_date", "irn", "phone", "discount",
    "round_off", "bank", "advance", "eway",
]


def _family_style(family: int) -> Style:
    """Deterministic style per family; rare families are the weird ones."""
    frng = random.Random(family * 7919)
    return Style(
        family=family,
        meta_side="left" if family in (5, 7) else "right",
        labels_above=family in (3, 6, 7),
        date_fmt=DATE_FORMATS[family % len(DATE_FORMATS)],
        label_idx={f: frng.randrange(len(v)) for f, v in LABELS.items()},
        totals_x=frng.choice([560, 640, 720, 780]),
        margin=frng.choice([50, 70, 90, 110]),
        p_due=frng.choice([0.9, 0.5, 0.2]),
        p_po=frng.choice([0.8, 0.4, 0.1]),
        distractors=frng.sample(_DISTRACTOR_POOL, k=3 + family % 4),
        noise=(0.4 + 0.6 * family / (N_FAMILIES - 1)),  # later families = uglier scans
    )


def _random_style(rng: random.Random) -> Style:
    """Easy mode: everything independent per document (the original behaviour)."""
    return Style(
        family=-1,
        meta_side="right",
        labels_above=False,
        date_fmt=rng.choice(DATE_FORMATS),
        label_idx={f: rng.randrange(len(v)) for f, v in LABELS.items()},
        totals_x=rng.randint(600, 800),
        margin=rng.randint(50, 120),
        p_due=0.8,
        p_po=0.6,
        distractors=[],
        noise=0.0,
    )


def _pick_family(rng: random.Random) -> int:
    """Zipf-ish: family 0 dominates, the tail is rare. Uncertainty sampling
    earns its keep on the tail; random sampling mostly re-labels family 0."""
    weights = [1.0 / (k + 1) for k in range(N_FAMILIES)]
    return rng.choices(range(N_FAMILIES), weights=weights)[0]


# ---------------------------------------------------------------------------
# Distractors: field-lookalikes that punish keyword anchors
# ---------------------------------------------------------------------------

def _emit_distractor(kind: str, page: _Page, x: float, y: float,
                     rng: random.Random, date_fmt: str) -> float:
    d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 500))
    lines = {
        "quotation": f"Quotation No: Q-{rng.randint(1000, 9999)}",
        "delivery_date": f"Delivery Date: {d.strftime(date_fmt)}",
        "irn": f"IRN: {''.join(rng.choices('0123456789abcdef', k=32))}",
        "phone": f"Ph: {rng.randint(6000000000, 9999999999)}",
        "discount": f"Discount: {rng.uniform(50, 2000):,.2f}",
        "round_off": f"Round Off: 0.0{rng.randint(1, 9)}",
        "bank": f"A/c No: {rng.randint(100000000, 999999999999)} IFSC: {''.join(rng.choices(string.ascii_uppercase, k=4))}0{rng.randint(100000, 999999)}",
        "advance": f"Advance Paid: {rng.uniform(100, 9000):,.2f}",
        "eway": f"E-Way Bill: {rng.randint(100000000000, 999999999999)}",
    }
    page.put(lines[kind], x, y)
    return y + LINE_H


# ---------------------------------------------------------------------------
# OCR-style corruption (hard mode)
# ---------------------------------------------------------------------------

_OCR_CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
                   "8": "B", "B": "8", "2": "Z", "Z": "2", "g": "q", "e": "c"}


def _corrupt(doc: Document, rng: random.Random, severity: float) -> None:
    """Char swaps, token dropout (including anchors), bbox jitter, ocr_conf.

    Value tokens are corrupted more gently than boilerplate: heavy corruption
    of values would cap field-F1 so low that model differences drown. The cap
    that remains is realistic — OCR errors on values are unfixable downstream.
    """
    in_annotation = []
    for tok in doc.tokens:
        hit = any(a.x0 <= tok.cx <= a.x1 and a.y0 <= tok.cy <= a.y1
                  for a in doc.annotations)
        in_annotation.append(hit)

    survivors, surv_flags = [], []
    for tok, labeled in zip(doc.tokens, in_annotation):
        # Token dropout: boilerplate (and crucially, anchor labels) vanish
        # the way faint print does. Never drop value tokens.
        if not labeled and rng.random() < 0.03 * severity:
            continue
        p_char = (0.004 if labeled else 0.02) * severity
        if p_char and rng.random() < 0.5:
            chars = list(tok.text)
            for i, c in enumerate(chars):
                if c in _OCR_CONFUSIONS and rng.random() < p_char * 8:
                    chars[i] = _OCR_CONFUSIONS[c]
            tok.text = "".join(chars)
        jitter = 2.5 * severity
        tok.x0 += rng.uniform(-jitter, jitter)
        tok.x1 += rng.uniform(-jitter, jitter)
        tok.y0 += rng.uniform(-jitter, jitter)
        tok.y1 += rng.uniform(-jitter, jitter)
        tok.ocr_conf = max(0.3, min(1.0, rng.gauss(0.95 - 0.25 * severity, 0.08)))
        survivors.append(tok)
        surv_flags.append(labeled)
    doc.tokens = survivors


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_document(seed: int | None = None, hard: bool = False) -> Document:
    rng = random.Random(seed)
    style = _family_style(_pick_family(rng)) if hard else _random_style(rng)
    page = _Page()

    def lab(field: str) -> str:
        return LABELS[field][style.label_idx[field]]

    # Same-family docs share a vendor pool: layouts AND names cluster.
    if hard:
        stem = VENDOR_STEMS[(style.family * 2 + rng.randint(0, 1)) % len(VENDOR_STEMS)]
    else:
        stem = rng.choice(VENDOR_STEMS)
    vendor = f"{stem} {rng.choice(VENDOR_SUFFIXES)}"
    buyer = rng.choice(BUYER_NAMES)
    inv_no = _rand_invoice_number(rng)
    inv_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 500))
    due_date = inv_date + timedelta(days=rng.choice([15, 30, 45, 60]))
    vendor_gstin, buyer_gstin = _rand_gstin(rng), _rand_gstin(rng)
    po_no = f"PO-{rng.randint(1000, 99999)}"
    currency = "INR"

    n_items = rng.randint(1, 6)
    unit_prices = [round(rng.uniform(50, 20000), 2) for _ in range(n_items)]
    qtys = [rng.randint(1, 20) for _ in range(n_items)]
    subtotal = round(sum(p * q for p, q in zip(unit_prices, qtys)), 2)
    tax_rate = rng.choice([0.05, 0.12, 0.18, 0.28])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    margin = style.margin
    y = rng.randint(40, 90)

    # --- header: vendor block --------------------------------------------
    vx = margin if style.meta_side == "right" else 700
    page.put(vendor, vx, y, field="vendor_name")
    y += LINE_H
    page.put(f"{rng.randint(1, 400)} {rng.choice(['MG Road', 'Industrial Area', 'Ring Road', 'Sector 12'])}", vx, y)
    y += LINE_H
    y = page.kv(lab("vendor_gstin"), vendor_gstin, vx, y, "vendor_gstin",
                style.labels_above)

    # --- header: invoice meta ---------------------------------------------
    meta_x = 700 if style.meta_side == "right" else margin
    meta_y = rng.randint(40, 90)
    page.put(rng.choice(["TAX INVOICE", "INVOICE", "GST INVOICE"]), meta_x, meta_y)
    meta_y += LINE_H
    meta_y = page.kv(lab("invoice_number"), inv_no, meta_x, meta_y,
                     "invoice_number", style.labels_above)
    meta_y = page.kv(lab("invoice_date"), inv_date.strftime(style.date_fmt),
                     meta_x, meta_y, "invoice_date", style.labels_above)
    if rng.random() < style.p_due:
        meta_y = page.kv(lab("due_date"), due_date.strftime(style.date_fmt),
                         meta_x, meta_y, "due_date", style.labels_above)
    if rng.random() < style.p_po:
        meta_y = page.kv(lab("po_number"), po_no, meta_x, meta_y,
                         "po_number", style.labels_above)
    # Distractors right in the meta block, where they hurt most.
    for kind in style.distractors[:2]:
        meta_y = _emit_distractor(kind, page, meta_x, meta_y, rng, style.date_fmt)

    # --- buyer block --------------------------------------------------------
    y = max(y, meta_y) + LINE_H * 2
    page.put(rng.choice(["Bill To:", "Buyer:", "Billed To:"]), margin, y)
    y += LINE_H
    page.put(buyer, margin + 20, y, field="buyer_name")
    y += LINE_H
    y = page.kv(lab("buyer_gstin"), buyer_gstin, margin + 20, y,
                "buyer_gstin", style.labels_above)

    # --- line items ---------------------------------------------------------
    y += LINE_H
    page.put("# Description Qty Rate Amount", margin, y)
    y += LINE_H
    for i in range(n_items):
        amount = round(unit_prices[i] * qtys[i], 2)
        page.put(f"{i + 1} {rng.choice(ITEM_NAMES)} {qtys[i]} {unit_prices[i]:,.2f} {amount:,.2f}", margin, y)
        y += LINE_H

    # --- totals block ---------------------------------------------------------
    y += LINE_H
    tx = style.totals_x
    y = page.kv(lab("subtotal"), _fmt_amount(subtotal, rng, hard), tx, y,
                "subtotal", style.labels_above)
    y = page.kv(lab("tax_amount"), _fmt_amount(tax, rng, hard), tx, y,
                "tax_amount", style.labels_above)
    for kind in style.distractors[2:]:
        y = _emit_distractor(kind, page, tx, y, rng, style.date_fmt)
    y = page.kv(lab("total_amount"), _fmt_amount(total, rng, hard), tx, y,
                "total_amount", style.labels_above)
    y = page.kv("Currency:", currency, tx, y, "currency", False)

    # --- footer noise ---------------------------------------------------------
    y += LINE_H
    page.put(rng.choice([
        "Thank you for your business",
        "This is a computer generated invoice",
        "E & O E - Subject to local jurisdiction",
    ]), margin, y)

    doc = Document(
        doc_id=f"synth-{uuid.uuid4().hex[:12]}",
        tokens=page.tokens,
        page_width=PAGE_W,
        page_height=PAGE_H,
        annotations=page.annotations,
        meta={
            "source": "synthetic-hard" if hard else "synthetic",
            "family": style.family,
            "truth": {
                "invoice_number": inv_no,
                "invoice_date": inv_date.isoformat(),
                "due_date": due_date.isoformat(),
                "vendor_name": vendor,
                "vendor_gstin": vendor_gstin,
                "buyer_name": buyer,
                "buyer_gstin": buyer_gstin,
                "subtotal": subtotal,
                "tax_amount": tax,
                "total_amount": total,
                "currency": currency,
                "po_number": po_no,
            },
        },
    )
    if hard:
        _corrupt(doc, rng, style.noise)
    return doc


def generate_dataset(n: int, seed: int = 0, hard: bool = False) -> list[Document]:
    return [generate_document(seed=seed * 1_000_003 + i, hard=hard) for i in range(n)]


# ---------------------------------------------------------------------------
# Rendering + augmentation (for the visual tier / OCR pipeline tests)
# ---------------------------------------------------------------------------

def render(doc: Document, augment: bool = False, seed: int | None = None):
    """Render a Document to a PIL image; optionally add scan-style damage."""
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    img = Image.new("RGB", (int(doc.page_width), int(doc.page_height)), "white")
    d = ImageDraw.Draw(img)
    for tok in doc.tokens:
        d.text((tok.x0, tok.y0), tok.text, fill="black")

    if augment:
        img = img.rotate(rng.uniform(-2.5, 2.5), fillcolor="white",
                         resample=Image.BILINEAR, expand=False)
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        # salt-and-pepper scan noise
        px = img.load()
        w, h = img.size
        for _ in range(int(w * h * 0.001)):
            x, y = rng.randrange(w), rng.randrange(h)
            v = rng.choice([0, 255])
            px[x, y] = (v, v, v)
    return img
