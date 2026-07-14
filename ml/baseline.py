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
# Both Indian-GST and US/EU wordings so rules fire on either invoice style.
ANCHORS: list[tuple[str, str]] = [
    ("due_date", r"\b(due date|payment due|due by)\b"),
    ("invoice_date", r"\b(invoice date|bill date|date paid|date of issue|issue date|dated|date)\b"),
    ("invoice_number", r"\b(invoice no|invoice #|inv no|bill number|invoice number)\b"),
    ("po_number", r"\b(po number|p\.o\. no|purchase order|po #)\b"),
    ("subtotal", r"\b(subtotal|sub total|taxable value|total excluding tax)\b"),
    ("tax_amount", r"\b(gst \(|tax:|tax \(|igst|cgst|sgst|vat|sales tax)\b"),
    ("total_amount", r"\b(grand total|amount payable|amount due|balance due|amount paid|total due|total)\b"),
    ("currency", r"\b(currency)\b"),
]

VALUE_PATTERNS = {
    # Invoice/PO numbers contain at least one digit — stops label words like
    # "number" from being mistaken for the value.
    "invoice_number": re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9/\-]{2,}$", re.I),
    "po_number": re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9/\-]{2,}$", re.I),
    "currency": re.compile(r"^(INR|USD|EUR|GBP|Rs\.?|₹|\$|€|£)$", re.I),
}

GSTIN_TOKEN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_RE_MONTH = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.,]?$", re.I)


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
        # A date token is a month word ("June"), or anything with a digit
        # that isn't obviously not-a-date. This lets "June 30, 2026" be
        # captured as a span instead of just the numeric fragments.
        return (parse_date(text) is not None
                or bool(_RE_MONTH.match(text)) or bool(re.search(r"\d", text)))
    if field in ("subtotal", "tax_amount", "total_amount"):
        # A percentage ("18%", "(18%") is a tax RATE, not an amount — skipping
        # it lets the following currency amount ("$0.90") be tagged instead.
        if "%" in text:
            return False
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
            # Anchor found: tag value-looking tokens to the right of it. Skip
            # any token that *starts* inside the matched anchor phrase (tracking
            # each token's start char avoids the off-by-one that used to pull
            # the label word "number" into "Invoice number 2TJFPRKM-0008").
            anchor_end_chars = m.end()
            pos, begun = 0, False
            for tok in line:
                tok_start = pos
                pos += len(tok.text) + 1  # + the joining space
                if tok_start < anchor_end_chars:
                    continue  # this token is part of the anchor phrase
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
