"""Adversarial edge-case / robustness test. Throws degenerate and malicious
inputs at every ML + inference module; any uncaught crash is a bug.

    python scripts/edge_case_test.py
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np

fails = []


def case(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH {name}: {type(e).__name__}: {e}")
        fails.append((name, traceback.format_exc()))


def main():
    from ml.baseline import predict as baseline_predict
    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_document
    from ml.labeling import (Document, FieldAnnotation, Token, assign_labels,
                             extract_field_values)
    from ml.postprocess import (parse_amount, parse_date, postprocess_fields,
                                validate_gstin)

    def doc(tokens, anns=None, w=600, h=800):
        return Document("t", tokens, w, h, annotations=anns or [])

    print("== labeling / geometry ==")
    case("empty document", lambda: assign_labels(doc([])))
    case("single token", lambda: assign_labels(doc([Token("x", 0, 0, 1, 1)])))
    case("zero-area token", lambda: assign_labels(doc([Token("x", 5, 5, 5, 5)])))
    case("zero-size page", lambda: featurize_document(doc([Token("x", 0, 0, 1, 1)], w=0, h=0)))
    case("negative coords", lambda: featurize_document(doc([Token("x", -10, -10, -5, -5)])))
    case("annotation off-page", lambda: assign_labels(doc(
        [Token("x", 0, 0, 10, 10)], [FieldAnnotation("total_amount", "x", 999, 999, 1099, 1099)])))
    case("overlapping annotations", lambda: assign_labels(doc(
        [Token("5.00", 10, 10, 60, 30)],
        [FieldAnnotation("subtotal", "5", 8, 8, 62, 32),
         FieldAnnotation("total_amount", "5", 8, 8, 62, 32)])))

    print("\n== features on degenerate docs ==")
    case("features empty", lambda: featurize_document(doc([])))
    case("features empty context=False", lambda: featurize_document(doc([]), context=False))
    case("unicode/emoji tokens", lambda: featurize_document(doc(
        [Token("₹💰", 0, 0, 20, 10), Token("naïve", 20, 0, 40, 10), Token("日本語", 40, 0, 60, 10)])))
    case("very long token", lambda: featurize_document(doc([Token("A" * 5000, 0, 0, 20, 10)])))
    case("whitespace/empty text token", lambda: featurize_document(doc(
        [Token("   ", 0, 0, 10, 10), Token("", 10, 0, 20, 10)])))
    case("thousands of tokens", lambda: featurize_document(doc(
        [Token(str(i), (i % 50) * 12, (i // 50) * 12, (i % 50) * 12 + 10, (i // 50) * 12 + 10)
         for i in range(3000)], w=700, h=800)))

    print("\n== baseline on degenerate docs ==")
    case("baseline empty", lambda: baseline_predict(doc([])))
    case("baseline all-numbers", lambda: baseline_predict(doc(
        [Token(str(i), i * 10, 0, i * 10 + 8, 10) for i in range(20)])))
    case("baseline single line huge", lambda: baseline_predict(doc(
        [Token("Total:", 0, 0, 40, 10)] + [Token("x", 40 + i, 0, 48 + i, 10) for i in range(200)])))

    print("\n== postprocess edge cases ==")
    case("parse_amount garbage", lambda: [parse_amount(x) for x in
         ["", "   ", "abc", "$", "1.2.3.4", "-", "1,,2,,3", "₹₹₹", "1e10", "NaN", "999" * 100]])
    case("parse_date garbage", lambda: [parse_date(x) for x in
         ["", "  ", "32/13/9999", "not a date", "0/0/0", "2026", "Feb 30 2026"]])
    case("validate_gstin garbage", lambda: [validate_gstin(x) for x in
         ["", "x", "1" * 15, "27AAPFU0939F1Z" + "?", "😀" * 15]])
    case("postprocess empty", lambda: postprocess_fields({}, {}))
    case("postprocess None-ish values", lambda: postprocess_fields(
        {"subtotal": "", "tax_amount": "  ", "total_amount": "abc", "vendor_gstin": ""},
        {"subtotal": 0.9, "tax_amount": 0.9, "total_amount": 0.9, "vendor_gstin": 0.9}))
    case("postprocess missing confidences", lambda: postprocess_fields(
        {"total_amount": "100.00"}, {}))
    case("postprocess huge amounts", lambda: postprocess_fields(
        {"subtotal": "9" * 40, "tax_amount": "1", "total_amount": "2"},
        {"subtotal": 1.0, "tax_amount": 1.0, "total_amount": 1.0}))

    print("\n== extract / evaluate edge cases ==")
    case("extract_field_values empty", lambda: extract_field_values([]))
    case("extract only I- tags (no B-)", lambda: extract_field_values(
        [Token("x", 0, 0, 1, 1, tag="I-total_amount")]))
    case("evaluate empty lists", lambda: evaluate_field_extraction([], []))
    case("evaluate doc with no truth", lambda: evaluate_field_extraction(
        [doc([Token("x", 0, 0, 1, 1)])], [["O"]]))

    print("\n== inference (Django) edge cases ==")
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()
    from documents.inference import _merge_rules, build_fields

    case("build_fields empty", lambda: build_fields(doc([]), [], np.zeros((0, 25))))
    case("build_fields mismatched (defensive)", lambda: build_fields(
        doc([Token("x", 0, 0, 10, 10)]), ["B-total_amount"],
        _one_hot(["B-total_amount"])))
    case("merge_rules empty", lambda: _merge_rules(doc([]), [], np.zeros((0, 25)), "champion:1"))
    case("merge_rules all-O", lambda: _merge_rules(
        doc([Token("hello", 0, 0, 40, 10)]), ["O"], _one_hot(["O"]), "champion:1"))

    print(f"\n{'='*50}")
    if fails:
        print(f"EDGE CASES: {len(fails)} CRASHES")
        for name, tb in fails:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("ALL EDGE CASES SURVIVED (no crashes)")


def _one_hot(tags):
    from ml.labeling import NUM_TAGS, TAG2ID
    p = np.zeros((len(tags), NUM_TAGS))
    for i, t in enumerate(tags):
        p[i, TAG2ID[t]] = 0.9
    return p


if __name__ == "__main__":
    main()
