"""Per-token feature engineering (Tier 1 input).

Each token gets ~120 features across four families:

  text shape   what the token looks like (digits, case, punctuation, regex hits)
  geometry     where it sits on the page (normalised coords, line position)
  context      the same shape features for neighbours in a +/-2 window,
               because "the token right after 'Total:'" is the whole game
  anchors      distance-decayed presence of label keywords to the left/above

Everything is deterministic and model-agnostic: the same matrix feeds
LogReg, RF, XGBoost, and (concatenated with embeddings) the BiLSTM.
"""

from __future__ import annotations

import re

import numpy as np

from ml.labeling import Document, Token
from ml.postprocess import parse_amount, parse_date

# Keywords that anchor fields. Lowercase, matched on cleaned token text.
KEYWORDS = [
    "invoice", "inv", "bill", "date", "dated", "due", "gstin", "gst",
    "total", "subtotal", "sub", "tax", "igst", "cgst", "sgst", "amount",
    "payable", "po", "purchase", "order", "no", "number", "currency",
    "buyer", "vendor", "to",
]
_KW_INDEX = {k: i for i, k in enumerate(KEYWORDS)}

_RE_DATEISH = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}")
_RE_AMOUNTISH = re.compile(r"^-?[\d,]+\.\d{1,2}$")
_RE_GSTIN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_RE_INVNUM = re.compile(r"^[A-Z]{2,4}[-/]\w", re.I)
_RE_MONTH = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*$", re.I)


def _clean(text: str) -> str:
    return text.strip().strip(".,:;#").lower()


def _shape_features(tok: Token) -> list[float]:
    """17 features describing the token string itself."""
    t = tok.text
    n = max(len(t), 1)
    digits = sum(c.isdigit() for c in t)
    alphas = sum(c.isalpha() for c in t)
    uppers = sum(c.isupper() for c in t)
    return [
        min(len(t), 30) / 30.0,
        digits / n,
        alphas / n,
        uppers / max(alphas, 1),
        float(t.isdigit()),
        float(t.isalpha()),
        float(t.istitle()),
        float(t.isupper() and alphas > 1),
        float("," in t),
        float("." in t),
        float("/" in t or "-" in t),
        float(any(c in t for c in "₹$€£") or "rs" == _clean(t) or "inr" == _clean(t)),
        float(bool(_RE_DATEISH.search(t)) or bool(_RE_MONTH.match(t))),
        float(bool(_RE_AMOUNTISH.match(t.replace("₹", "")))),
        float(bool(_RE_GSTIN.match(t.upper()))),
        float(bool(_RE_INVNUM.match(t))),
        float(t.endswith(":")),
    ]


_N_SHAPE = 17

SHAPE_NAMES = [
    "len", "digit_ratio", "alpha_ratio", "upper_ratio", "is_digit", "is_alpha",
    "is_title", "is_allcaps", "has_comma", "has_dot", "has_slash_dash",
    "is_currencyish", "dateish", "amountish", "gstinish", "invnumish",
    "ends_colon",
]


def _line_ids(doc: Document) -> np.ndarray:
    """Assign a line id to each token by vertical proximity (reading order)."""
    order = sorted(range(len(doc.tokens)),
                   key=lambda i: (doc.tokens[i].page, doc.tokens[i].cy, doc.tokens[i].cx))
    line_ids = np.zeros(len(doc.tokens), dtype=int)
    line = 0
    prev: Token | None = None
    for i in order:
        tok = doc.tokens[i]
        if prev is not None and (tok.page != prev.page or abs(tok.cy - prev.cy) >= tok.height * 0.6):
            line += 1
        line_ids[i] = line
        prev = tok
    return line_ids


def featurize_document(doc: Document, context: bool = True) -> np.ndarray:
    """Return (n_tokens, n_features) float32 matrix, order = doc.tokens.

    context=False drops the neighbour window and keyword-anchor families,
    leaving only per-token shape + geometry (31 features). Sequence models
    (Tier 2/3) use this: hand-coding context into their input would let a
    memoryless model fake sequence understanding, and the RNN/LSTM/BiLSTM
    comparison would measure nothing.
    """
    n = len(doc.tokens)
    if n == 0:
        width = len(FEATURE_NAMES) if context else _N_SHAPE + len(GEOM_NAMES)
        return np.zeros((0, width), dtype=np.float32)

    pw, ph = max(doc.page_width, 1), max(doc.page_height, 1)
    line_ids = _line_ids(doc)
    shape = np.array([_shape_features(t) for t in doc.tokens], dtype=np.float32)

    # Date-order signal: rank each date-like token among all date-like tokens in
    # reading order. Invoice date is usually the FIRST date, due date a later
    # one — this disambiguates the two fields the model most often confuses.
    date_like = [i for i, t in enumerate(doc.tokens)
                 if _RE_DATEISH.search(t.text) or _RE_MONTH.match(t.text.strip())]
    ro = sorted(date_like, key=lambda i: (doc.tokens[i].page, doc.tokens[i].cy, doc.tokens[i].cx))
    date_rank = {i: r / max(len(ro) - 1, 1) for r, i in enumerate(ro)}
    first_date, last_date = (ro[0], ro[-1]) if ro else (-1, -1)

    # Line-level positions.
    line_members: dict[int, list[int]] = {}
    for i, lid in enumerate(line_ids):
        line_members.setdefault(int(lid), []).append(i)
    for members in line_members.values():
        members.sort(key=lambda i: doc.tokens[i].cx)

    rows = []
    for i, tok in enumerate(doc.tokens):
        members = line_members[int(line_ids[i])]
        pos_in_line = members.index(i)

        geom = [
            tok.x0 / pw, tok.cy / ph, tok.cx / pw, tok.y0 / ph,
            tok.width / pw, tok.height / ph,
            float(tok.cx > pw * 0.5),                # right half
            float(tok.cy < ph * 0.25),               # top quarter
            float(tok.cy > ph * 0.75),               # bottom quarter
            pos_in_line / max(len(members) - 1, 1),
            min(len(members), 20) / 20.0,
            float(pos_in_line == 0),
            float(pos_in_line == len(members) - 1),
            tok.ocr_conf,
            date_rank.get(i, 0.0),                   # 0 = first date, 1 = last
            float(i == first_date),
            float(i == last_date),
        ]

        if not context:
            rows.append(np.concatenate([shape[i], geom]))
            continue

        # Context: shape features of the 2 tokens left and right on the line,
        # zeros past the boundary.
        ctx = []
        for off in (-2, -1, 1, 2):
            j = pos_in_line + off
            if 0 <= j < len(members):
                ctx.extend(_shape_features(doc.tokens[members[j]]))
            else:
                ctx.extend([0.0] * _N_SHAPE)

        # Anchor keywords: nearest occurrence to the LEFT on this line,
        # distance-decayed, one slot per keyword.
        kw = [0.0] * len(KEYWORDS)
        for j in range(pos_in_line - 1, max(pos_in_line - 5, -1), -1):
            word = _clean(doc.tokens[members[j]].text)
            if word in _KW_INDEX:
                dist = pos_in_line - j
                kw[_KW_INDEX[word]] = max(kw[_KW_INDEX[word]], 1.0 / dist)
        # ...and on the previous line at a similar x (label-above layouts).
        prev_line = int(line_ids[i]) - 1
        if prev_line in line_members:
            for j in line_members[prev_line]:
                other = doc.tokens[j]
                if abs(other.cx - tok.cx) < pw * 0.08:
                    word = _clean(other.text)
                    if word in _KW_INDEX:
                        kw[_KW_INDEX[word]] = max(kw[_KW_INDEX[word]], 0.5)

        rows.append(np.concatenate([shape[i], geom, ctx, kw]))

    return np.asarray(rows, dtype=np.float32)


GEOM_NAMES = [
    "x0_norm", "cy_norm", "cx_norm", "y0_norm", "w_norm", "h_norm",
    "right_half", "top_quarter", "bottom_quarter", "pos_in_line",
    "line_len", "line_first", "line_last", "ocr_conf",
    "date_rank", "date_first", "date_last",
]

FEATURE_NAMES: list[str] = (
    SHAPE_NAMES
    + GEOM_NAMES
    + [f"{name}@{off}" for off in (-2, -1, 1, 2) for name in SHAPE_NAMES]
    + [f"kw_{k}" for k in KEYWORDS]
)


def featurize_dataset(docs: list[Document]):
    """Stack features/labels/groups for classical models.

    Returns (X, y, groups): X (N, F) float32; y (N,) int tag ids;
    groups (N,) doc index for GroupKFold — trap #1: always split by document.
    """
    from ml.labeling import TAG2ID

    Xs, ys, gs = [], [], []
    for gi, doc in enumerate(docs):
        Xs.append(featurize_document(doc))
        ys.append(np.array([TAG2ID[t.tag] for t in doc.tokens], dtype=np.int64))
        gs.append(np.full(len(doc.tokens), gi, dtype=np.int64))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
