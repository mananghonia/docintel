from ml.postprocess import (gstin_check_char, parse_amount, parse_date,
                            postprocess_fields, validate_gstin)


def test_gstin_checksum_roundtrip():
    body = "27AAPFU0939F1Z"
    assert validate_gstin(body + gstin_check_char(body))


def test_gstin_rejects_wrong_check_char():
    body = "27AAPFU0939F1Z"
    good = gstin_check_char(body)
    bad = "A" if good != "A" else "B"
    assert not validate_gstin(body + bad)


def test_parse_amount_variants():
    assert parse_amount("Rs. 1,23,456.78") == 123456.78
    assert parse_amount("₹500") == 500.0
    assert parse_amount("no numbers here") is None


def test_parse_date_variants():
    assert parse_date("25/03/2025").isoformat() == "2025-03-25"
    assert parse_date("25 Mar 2025").isoformat() == "2025-03-25"
    assert parse_date("garbage") is None


def test_arithmetic_inconsistency_cuts_confidence():
    values = {"subtotal": "100.00", "tax_amount": "18.00", "total_amount": "200.00"}
    out = postprocess_fields(values, {f: 0.95 for f in values})
    assert all("arithmetic_inconsistent" in out[f]["flags"] for f in values)
    assert out["total_amount"]["confidence"] < 0.95 / 1.5


def test_arithmetic_consistency_flags_ok():
    values = {"subtotal": "100.00", "tax_amount": "18.00", "total_amount": "118.00"}
    out = postprocess_fields(values, {f: 0.9 for f in values})
    assert all("arithmetic_ok" in out[f]["flags"] for f in values)


def test_bad_gstin_penalised():
    out = postprocess_fields({"vendor_gstin": "NOTAGSTIN"}, {"vendor_gstin": 0.9})
    assert "gstin_checksum_failed" in out["vendor_gstin"]["flags"]
    assert out["vendor_gstin"]["confidence"] < 0.9
