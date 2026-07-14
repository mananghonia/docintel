"""Business-rule post-processing.

Three jobs:
1. Validate what can be validated exactly (GSTIN mod-36 checksum).
2. Normalise free-text values into typed values (dates, amounts).
3. Cross-field arithmetic consistency: if subtotal + tax != total, *reduce
   the confidence* of every amount field involved — a mechanism for catching
   errors the model does not know it made.
"""

from __future__ import annotations

import re
from datetime import date, datetime

GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# Document-title words that must never be accepted as a party name.
_TITLE_WORDS = {"receipt", "invoice", "tax invoice", "bill", "statement",
                "gst invoice", "estimate", "quotation", "proforma"}


# ---------------------------------------------------------------------------
# GSTIN
# ---------------------------------------------------------------------------

def gstin_check_char(body14: str) -> str:
    """Compute the 15th (checksum) character for a 14-char GSTIN body.

    Standard GSTIN algorithm: alternate multipliers 1/2 over base-36 digit
    values, sum quotient+remainder of each product base 36, checksum is
    (36 - sum mod 36) mod 36 mapped back to the alphabet.
    """
    total = 0
    for i, ch in enumerate(body14):
        v = GSTIN_ALPHABET.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return GSTIN_ALPHABET[(36 - total % 36) % 36]


def validate_gstin(gstin: str) -> bool:
    g = gstin.strip().upper().replace(" ", "")
    if not GSTIN_RE.match(g):
        return False
    return gstin_check_char(g[:14]) == g[14]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y", "%d/%m/%y", "%d-%m-%y", "%m/%d/%Y",
]


def parse_date(text: str) -> date | None:
    t = re.sub(r"\s+", " ", text.strip().rstrip(".,;"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


_AMOUNT_RE = re.compile(r"-?[\d,]+(?:\.\d{1,2})?")


def parse_amount(text: str) -> float | None:
    """'Rs. 1,23,456.78' -> 123456.78. Returns None if nothing numeric."""
    m = _AMOUNT_RE.search(text.replace("₹", "").replace("INR", "").replace("Rs.", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Field-level post-processing
# ---------------------------------------------------------------------------

def postprocess_fields(
    values: dict[str, str],
    confidences: dict[str, float],
    arithmetic_tolerance: float = 0.02,
    arithmetic_penalty: float = 0.5,
    checksum_penalty: float = 0.3,
) -> dict[str, dict]:
    """Normalise values and adjust confidences with business rules.

    Returns {field: {"raw", "value", "confidence", "flags"}}.

    Rules:
    - GSTIN failing its checksum: confidence *= checksum_penalty.
    - Unparseable date/amount: flagged, confidence *= 0.7.
    - subtotal + tax != total (beyond tolerance): every amount field's
      confidence *= arithmetic_penalty, even if the model was confident.
    - subtotal + tax == total exactly: mild confidence boost (rules agree).
    """
    out: dict[str, dict] = {}
    for f, raw in values.items():
        conf = confidences.get(f, 0.0)
        flags: list[str] = []
        value: object = raw

        if f.endswith("_name"):
            # A name that is just a document-title word ("Receipt", "Invoice")
            # is a header misfire, not a party name — slash its confidence so
            # it routes to review instead of showing as a confident answer.
            if raw.strip().lower().strip(":") in _TITLE_WORDS:
                flags.append("likely_title_not_name")
                conf *= 0.25
        elif f.endswith("_gstin"):
            g = raw.strip().upper().replace(" ", "")
            value = g
            if not validate_gstin(g):
                flags.append("gstin_checksum_failed")
                conf *= checksum_penalty
        elif f.endswith("_date"):
            parsed = parse_date(raw)
            if parsed is None:
                flags.append("unparseable_date")
                conf *= 0.7
            else:
                value = parsed.isoformat()
        elif f in ("subtotal", "tax_amount", "total_amount"):
            parsed = parse_amount(raw)
            if parsed is None:
                flags.append("unparseable_amount")
                conf *= 0.7
            else:
                value = parsed

        out[f] = {"raw": raw, "value": value, "confidence": conf, "flags": flags}

    # Cross-field arithmetic consistency.
    amounts = {f: out[f]["value"] for f in ("subtotal", "tax_amount", "total_amount")
               if f in out and isinstance(out[f]["value"], float)}
    if len(amounts) == 3:
        gap = abs(amounts["subtotal"] + amounts["tax_amount"] - amounts["total_amount"])
        rel = gap / max(amounts["total_amount"], 1.0)
        if rel > arithmetic_tolerance:
            for f in amounts:
                out[f]["confidence"] *= arithmetic_penalty
                out[f]["flags"].append("arithmetic_inconsistent")
        else:
            for f in amounts:
                out[f]["confidence"] = min(1.0, out[f]["confidence"] * 1.05)
                out[f]["flags"].append("arithmetic_ok")

    return out
