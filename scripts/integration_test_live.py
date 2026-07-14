"""Integration test against a running DocIntel deployment (default: the live
Render URL). Exercises every endpoint and the full product loop over HTTP.

    python scripts/integration_test_live.py [BASE_URL]

Tolerant of free-tier cold starts (long first timeout + retries).
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://docintel-dvze.onrender.com").rstrip("/")

results = []


def _req(method, path, body=None, timeout=90, ctype="application/json"):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            dt = time.time() - t0
            try:
                return r.status, json.loads(raw), dt
            except json.JSONDecodeError:
                return r.status, raw.decode("utf-8", "replace"), dt
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, str(e), time.time() - t0


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def wake():
    print(f"== waking {BASE} (free-tier cold start can take ~60s) ==")
    for attempt in range(6):
        status, _, dt = _req("GET", "/api/monitoring/metrics/", timeout=120)
        if status == 200:
            print(f"   awake after {dt:.0f}s")
            return True
        print(f"   attempt {attempt + 1}: status={status}, retrying...")
        time.sleep(5)
    return False


def main():
    if not wake():
        print("SERVICE DID NOT WAKE — aborting")
        sys.exit(1)

    print("\n== SPA routes serve the app shell ==")
    for path in ["/", "/queue", "/dashboard", "/review/does-not-exist"]:
        status, body, _ = _req("GET", path, timeout=60)
        is_html = isinstance(body, str) and "<div id=\"root\">" in body
        check(f"GET {path} -> 200 html", status == 200 and is_html, f"status={status}")

    print("\n== read endpoints ==")
    for path in ["/api/monitoring/metrics/", "/api/monitoring/drift/",
                 "/api/documents/", "/api/documents/review_queue/",
                 "/api/training/models/", "/api/training/runs/"]:
        status, body, _ = _req("GET", path)
        check(f"GET {path} -> 200 json", status == 200 and not isinstance(body, str),
              f"status={status}")

    print("\n== dashboard shows a champion model ==")
    status, body, _ = _req("GET", "/api/training/models/")
    rows = (body.get("results", body) if isinstance(body, dict) else body) or []
    champ = [m for m in rows if m.get("is_champion")]
    check("a champion ModelVersion exists", len(champ) >= 1,
          f"{len(rows)} versions" if isinstance(rows, list) else str(rows)[:80])

    print("\n== ingest -> extract -> export -> review flow ==")
    status, body, _ = _req("POST", "/api/documents/ingest_synthetic/",
                           {"n": 4, "holdout_every": 0}, timeout=120)
    ids = body.get("created", []) if isinstance(body, dict) else []
    check("ingest_synthetic creates docs", status == 201 and len(ids) == 4,
          f"status={status}, created={len(ids)}")

    if ids:
        did = ids[0]
        # give eager processing a moment (it's inline, but be safe)
        time.sleep(2)
        status, doc, _ = _req("GET", f"/api/documents/{did}/")
        ext = doc.get("latest_extraction") if isinstance(doc, dict) else None
        check("document has an extraction", bool(ext) and bool(ext.get("fields")),
              f"fields={len(ext['fields']) if ext else 0}")

        status, exp, _ = _req("GET", f"/api/documents/{did}/export/")
        ok = status == 200 and isinstance(exp, dict) and "fields" in exp
        check("export returns structured fields", ok, f"status={status}")

        # review with the synthetic ground truth
        truth = doc["doc_json"]["meta"]["truth"]
        anns = {a["field"]: a for a in doc["doc_json"]["annotations"]}
        corrections = [{
            "field": f, "corrected_value": str(v), "accepted_as_is": False,
            "bbox": {k: anns[f][k] for k in ("x0", "y0", "x1", "y1", "page")}
                    if f in anns else None,
        } for f, v in truth.items()]
        status, rr, _ = _req("POST", f"/api/documents/{did}/review/",
                             {"corrections": corrections, "review_seconds": 30})
        check("review -> verified", status == 200, f"status={status}, {str(rr)[:80]}")
        status, doc2, _ = _req("GET", f"/api/documents/{did}/")
        check("document status is verified",
              isinstance(doc2, dict) and doc2.get("status") == "verified",
              doc2.get("status") if isinstance(doc2, dict) else "")

    print("\n== upload validation ==")
    # multipart with a bad extension via raw body
    boundary = "----docintelITEST"
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"x.exe\"\r\nContent-Type: application/octet-stream\r\n\r\n"
               f"junk\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(BASE + "/api/documents/upload/", data=payload,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=60); code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    check("upload rejects .exe (400)", code == 400, f"status={code}")

    print("\n== error handling ==")
    status, _, _ = _req("GET", "/api/documents/00000000-0000-0000-0000-000000000000/")
    check("unknown document -> 404", status == 404, f"status={status}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*50}\nLIVE INTEGRATION: {n_pass}/{len(results)} passed")
    if n_pass != len(results):
        print("FAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    print("ALL LIVE INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
