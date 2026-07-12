"""Tier 0: regex + keyword-anchor baseline.

No learning. Groups tokens into lines, looks for label keywords ("Invoice
No", "GSTIN", "Total", ...), and tags the value tokens that follow the
anchor on the same line. Pure-pattern fields (GSTIN, dates) also match
anywhere by regex.

This exists to produce the number every model must beat, and to be honest
about how far rules alone get you (~60% field F1 on messy layouts).
"""

from __future__ import annotations

import re

from ml.labeling import Document, O_TAG, Token
from ml.postprocess import GSTIN_RE, parse_amount, parse_date

# Keyword anchors, matched case-insensitively against the concatenated line.
# Order matters: more specific anchors first (e.g. "due date" before "date").
ANCHORS: list[tuple[str, str]] = [
    ("due_date", r"\b(due date|payment due|due by)\b"),
    ("invoice_date", r"\b(invoice date|bill date|dated|date)\b"),
    ("invoice_number", r"\b(invoice no|invoice #|inv no|bill number|invoice number)\b"),
    ("po_number", r"\b(po number|p\.o\. no|purchase order)\b"),
    ("subtotal", r"\b(subtotal|sub total|taxable value)\b"),
    ("tax_amount", r"\b(gst \(|tax:|igst|cgst)\b"),
    ("total_amount", r"\b(grand total|amount payable|total)\b"),
    ("currency", r"\b(currency)\b"),
]

VALUE_PATTERNS = {
    "invoice_number": re.compile(r"^[A-Z0-9][A-Z0-9/\-]{2,}$", re.I),
    "po_number": re.compile(r"^[A-Z0-9][A-Z0-9/\-]{2,}$", re.I),
    "currency": re.compile(r"^(INR|USD|EUR|GBP|Rs\.?|₹)$", re.I),
}

GSTIN_TOKEN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def _lines(doc: Document) -> list[list[Token]]:
    """Group tokens into lines by vertical proximity, sorted left-to-right."""
    toks = sorted(doc.tokens, key=lambda t: (t.page, t.cy, t.cx))
    lines: list[list[Token]] = []
    for tok in toks:
        if lines and lines[-1][0].page == tok.page and abs(lines[-1][-1].cy - tok.cy) < tok.height * 0.6:
            lines[-1].append(tok)
        else:
            lines.append([tok])
    return lines


def _looks_like_value(field: str, tok: Token) -> bool:
    text = tok.text
    if field.endswith("_date"):
        return parse_date(text) is not None or bool(re.search(r"\d", text))
    if field in ("subtotal", "tax_amount", "total_amount"):
        return parse_amount(text) is not None
    pat = VALUE_PATTERNS.get(field)
    return bool(pat and pat.match(text))


def predict(doc: Document) -> list[str]:
    """Return one predicted BIO tag per token (same order as doc.tokens)."""
    tags = [O_TAG] * len(doc.tokens)
    index = {id(t): i for i, t in enumerate(doc.tokens)}
    seen_gstin = 0

    for line in _lines(doc):
        line_text = " ".join(t.text for t in line).lower()

        # GSTIN by pure pattern: first hit = vendor, second = buyer.
        for tok in line:
            if GSTIN_TOKEN_RE.match(tok.text.strip().upper()):
                field = "vendor_gstin" if seen_gstin == 0 else "buyer_gstin"
                tags[index[id(tok)]] = f"B-{field}"
                seen_gstin += 1

        for field, pattern in ANCHORS:
            m = re.search(pattern, line_text)
            if not m:
                continue
            # Anchor found: tag the value-looking tokens to the right of it.
            anchor_end_chars = m.end()
            consumed, begun = 0, False
            for tok in line:
                consumed += len(tok.text) + 1
                if consumed <= anchor_end_chars:
                    continue  # still inside the anchor phrase
                i = index[id(tok)]
                if tags[i] != O_TAG:
                    continue
                if _looks_like_value(field, tok):
                    tags[i] = f"{'B' if not begun else 'I'}-{field}"
                    begun = True
                elif begun:
                    break  # value span ended
            break  # one anchor per line

    return tags
