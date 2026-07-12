"""Synthetic invoice generator.

Generates Document objects (tokens + geometry + annotations) directly, so the
labels are perfect and free — no OCR involved. Optionally renders a document
to a PIL image with scan-style augmentations for the visual tier and for
exercising the OCR pipeline.

Layout, vendor, label wording, amounts and dates are all randomised so models
cannot memorise absolute positions.
"""

from __future__ import annotations

import random
import string
import uuid
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


def _fmt_amount(v: float, rng: random.Random) -> str:
    s = f"{v:,.2f}"
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

    def kv(self, label: str, value: str, x: float, y: float, field: str) -> None:
        end = self.put(label, x, y)
        self.put(value, end + CHAR_W, y, field=field)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_document(seed: int | None = None) -> Document:
    rng = random.Random(seed)
    page = _Page()

    vendor = f"{rng.choice(VENDOR_STEMS)} {rng.choice(VENDOR_SUFFIXES)}"
    buyer = rng.choice(BUYER_NAMES)
    inv_no = _rand_invoice_number(rng)
    inv_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 500))
    due_date = inv_date + timedelta(days=rng.choice([15, 30, 45, 60]))
    date_fmt = rng.choice(DATE_FORMATS)
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

    margin = rng.randint(50, 120)
    y = rng.randint(40, 90)

    # --- header: vendor block (left) --------------------------------------
    page.put(vendor, margin, y, field="vendor_name")
    y += LINE_H
    page.put(f"{rng.randint(1, 400)} {rng.choice(['MG Road', 'Industrial Area', 'Ring Road', 'Sector 12'])}", margin, y)
    y += LINE_H
    page.kv(rng.choice(LABELS["vendor_gstin"]), vendor_gstin, margin, y, "vendor_gstin")

    # --- header: invoice meta (right or below, randomised layout) ---------
    right_x = rng.randint(700, 850)
    meta_y = rng.randint(40, 90)
    title = rng.choice(["TAX INVOICE", "INVOICE", "GST INVOICE"])
    page.put(title, right_x, meta_y)
    meta_y += LINE_H
    page.kv(rng.choice(LABELS["invoice_number"]), inv_no, right_x, meta_y, "invoice_number")
    meta_y += LINE_H
    page.kv(rng.choice(LABELS["invoice_date"]), inv_date.strftime(date_fmt), right_x, meta_y, "invoice_date")
    meta_y += LINE_H
    if rng.random() < 0.8:
        page.kv(rng.choice(LABELS["due_date"]), due_date.strftime(date_fmt), right_x, meta_y, "due_date")
        meta_y += LINE_H
    if rng.random() < 0.6:
        page.kv(rng.choice(LABELS["po_number"]), po_no, right_x, meta_y, "po_number")

    # --- buyer block --------------------------------------------------------
    y = max(y, meta_y) + LINE_H * 2
    page.put(rng.choice(["Bill To:", "Buyer:", "Billed To:"]), margin, y)
    y += LINE_H
    page.put(buyer, margin + 20, y, field="buyer_name")
    y += LINE_H
    page.kv(rng.choice(LABELS["buyer_gstin"]), buyer_gstin, margin + 20, y, "buyer_gstin")

    # --- line items ---------------------------------------------------------
    y += LINE_H * 2
    page.put("# Description Qty Rate Amount", margin, y)
    y += LINE_H
    for i in range(n_items):
        amount = round(unit_prices[i] * qtys[i], 2)
        page.put(f"{i + 1} {rng.choice(ITEM_NAMES)} {qtys[i]} {unit_prices[i]:,.2f} {amount:,.2f}", margin, y)
        y += LINE_H

    # --- totals block (right-aligned-ish) ------------------------------------
    y += LINE_H
    totals_x = rng.randint(600, 800)
    page.kv(rng.choice(LABELS["subtotal"]), _fmt_amount(subtotal, rng), totals_x, y, "subtotal")
    y += LINE_H
    page.kv(rng.choice(LABELS["tax_amount"]), _fmt_amount(tax, rng), totals_x, y, "tax_amount")
    y += LINE_H
    page.kv(rng.choice(LABELS["total_amount"]), _fmt_amount(total, rng), totals_x, y, "total_amount")
    y += LINE_H
    page.kv("Currency:", currency, totals_x, y, "currency")

    # --- footer noise ---------------------------------------------------------
    y += LINE_H * 2
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
            "source": "synthetic",
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
    return doc


def generate_dataset(n: int, seed: int = 0) -> list[Document]:
    return [generate_document(seed=seed * 1_000_003 + i) for i in range(n)]


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
