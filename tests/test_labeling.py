"""bbox->token alignment is trap #2: if this is wrong, everything is wrong."""

from ml.labeling import (Document, FieldAnnotation, Token, assign_labels,
                         extract_field_values)


def _doc():
    # One line: "Total: 1,234.00" — label token + two value tokens.
    tokens = [
        Token("Total:", 10, 10, 70, 30),
        Token("Rs.", 80, 10, 110, 30),
        Token("1,234.00", 120, 10, 200, 30),
        Token("footer", 10, 100, 70, 120),
    ]
    ann = [FieldAnnotation("total_amount", "Rs. 1,234.00", 78, 8, 202, 32)]
    return Document("t1", tokens, 600, 400, annotations=ann)


def test_assign_labels_bio():
    doc = assign_labels(_doc())
    assert doc.tokens[0].tag == "O"          # label word not inside the box
    assert doc.tokens[1].tag == "B-total_amount"
    assert doc.tokens[2].tag == "I-total_amount"
    assert doc.tokens[3].tag == "O"


def test_partial_overlap_below_threshold_stays_O():
    doc = _doc()
    # Shrink the annotation so it covers <50% of the second value token.
    doc.annotations[0].x1 = 130
    assign_labels(doc)
    assert doc.tokens[2].tag == "O"


def test_extract_field_values_joins_span():
    doc = assign_labels(_doc())
    values = extract_field_values(doc.tokens)
    assert values == {"total_amount": "Rs. 1,234.00"}


def test_document_roundtrip():
    doc = assign_labels(_doc())
    clone = Document.from_dict(doc.to_dict())
    assert [t.tag for t in clone.tokens] == [t.tag for t in doc.tokens]
    assert clone.page_width == doc.page_width
