import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, confColor } from "../api.js";
import DocCanvas from "../components/DocCanvas.jsx";

const ALL_FIELDS = [
  "invoice_number", "invoice_date", "due_date", "vendor_name", "vendor_gstin",
  "buyer_name", "buyer_gstin", "subtotal", "tax_amount", "total_amount",
  "currency", "po_number",
];

export default function Review() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [doc, setDoc] = useState(null);
  const [edits, setEdits] = useState({});      // field -> corrected value
  const [drawn, setDrawn] = useState({});      // field -> bbox (missed fields)
  const [hovered, setHovered] = useState(null);
  const [drawTarget, setDrawTarget] = useState(null);
  const started = useRef(Date.now());          // time-to-review metric

  useEffect(() => { api.document(id).then(setDoc); }, [id]);

  const extraction = doc?.latest_extraction;
  const fields = extraction?.fields ?? [];
  const predicted = useMemo(() => Object.fromEntries(fields.map((f) => [f.field, f])), [fields]);
  // Low-confidence fields first: the reviewer only touches what needs attention.
  const sorted = [...fields].sort((a, b) => a.confidence - b.confidence);
  const missing = ALL_FIELDS.filter((f) => !predicted[f]);

  const submit = async () => {
    const corrections = [];
    for (const f of fields) {
      const edited = edits[f.field] !== undefined && edits[f.field] !== String(f.raw);
      corrections.push({
        field: f.field,
        corrected_value: edited ? edits[f.field] : String(f.raw),
        accepted_as_is: !edited,
      });
    }
    for (const [field, bbox] of Object.entries(drawn)) {
      if (edits[field]) corrections.push({ field, corrected_value: edits[field], accepted_as_is: false, bbox });
    }
    await api.submitReview(id, corrections, (Date.now() - started.current) / 1000);
    navigate("/queue");
  };

  const downloadJson = () => {
    // Export the current structured extraction (with any edits) as a file.
    const out = {
      document_id: doc.id,
      status: doc.status,
      fields: fields.map((f) => ({
        field: f.field,
        value: edits[f.field] !== undefined ? edits[f.field] : f.value,
        confidence: f.confidence,
        flags: f.flags,
        bbox: f.bbox,
      })),
    };
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `docintel-${doc.id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!doc) return <p>Loading…</p>;

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".6rem" }}>
        <h2 style={{ margin: 0 }}>
          Review <code>{doc.id.slice(0, 8)}</code>{" "}
          <small style={{ color: "#64748b" }}>
            min conf {extraction ? extraction.min_confidence.toFixed(2) : "—"} ·
            avg {extraction ? extraction.avg_confidence.toFixed(2) : "—"}
          </small>
        </h2>
        <div style={{ display: "flex", gap: ".5rem" }}>
          <button className="secondary" onClick={downloadJson}>Download JSON</button>
          <button onClick={submit}>Submit review</button>
        </div>
      </div>

      <div className="review-layout">
        <div className="canvas-wrap">
          <DocCanvas
            docJson={doc.doc_json}
            fields={fields}
            hovered={hovered}
            onHover={setHovered}
            drawTarget={drawTarget}
            onDrawn={(bbox, text) => {
              setDrawn((d) => ({ ...d, [drawTarget]: bbox }));
              setEdits((e) => ({ ...e, [drawTarget]: text }));
              setDrawTarget(null);
            }}
          />
        </div>

        <div className="field-panel">
          {sorted.map((f) => (
            <div
              key={f.field}
              className={`field-card${hovered === f.field ? " hover" : ""}`}
              style={{ borderLeftColor: confColor(f.confidence) }}
              onMouseEnter={() => setHovered(f.field)}
              onMouseLeave={() => setHovered(null)}
            >
              <div className="field-name">
                {f.field}{" "}
                <span style={{ color: confColor(f.confidence) }}>
                  {(f.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <input
                defaultValue={String(f.raw)}
                autoFocus={f === sorted[0] && f.confidence < 0.9}
                onChange={(e) => setEdits((ed) => ({ ...ed, [f.field]: e.target.value }))}
              />
              {f.flags?.length > 0 && <div className="flags">{f.flags.join(", ")}</div>}
            </div>
          ))}

          {missing.length > 0 && (
            <div className="field-card" style={{ borderLeftColor: "#64748b" }}>
              <div className="field-name">Model missed a field?</div>
              {missing.map((f) => (
                <div key={f} style={{ display: "flex", gap: ".4rem", alignItems: "center", marginTop: ".3rem" }}>
                  <button
                    className="secondary"
                    style={{ padding: ".15rem .5rem", fontSize: ".72rem" }}
                    onClick={() => setDrawTarget(drawTarget === f ? null : f)}
                  >
                    {drawTarget === f ? "drawing… (drag on doc)" : "draw box"}
                  </button>
                  <span style={{ fontSize: ".78rem" }}>{f}</span>
                  {drawn[f] && (
                    <input
                      style={{ margin: 0 }}
                      value={edits[f] ?? ""}
                      onChange={(e) => setEdits((ed) => ({ ...ed, [f]: e.target.value }))}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
