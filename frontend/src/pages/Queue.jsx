import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

const STATUS_COLORS = {
  needs_review: "#b45309",
  extracted: "#15803d",
  verified: "#0369a1",
  failed: "#b91c1c",
  processing: "#64748b",
  uploaded: "#64748b",
};

export default function Queue() {
  const [docs, setDocs] = useState([]);
  const [busy, setBusy] = useState(false);

  const load = () => api.documents().then((d) => setDocs(d.results ?? d));
  useEffect(() => { load(); }, []);

  const ingest = async () => {
    setBusy(true);
    try { await api.ingestSynthetic(10); await load(); }
    finally { setBusy(false); }
  };

  const upload = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true);
    try { await api.upload(f); await load(); }
    finally { setBusy(false); e.target.value = ""; }
  };

  return (
    <>
      <div style={{ display: "flex", gap: ".6rem", marginBottom: "1rem", alignItems: "center" }}>
        <button onClick={ingest} disabled={busy}>Ingest 10 synthetic</button>
        <label>
          <button className="secondary" onClick={(e) => e.currentTarget.parentElement.querySelector("input").click()} disabled={busy}>
            Upload PDF/image
          </button>
          <input type="file" accept=".pdf,image/*" style={{ display: "none" }} onChange={upload} />
        </label>
        <button className="secondary" onClick={load}>Refresh</button>
      </div>
      <table>
        <thead>
          <tr><th>Document</th><th>Source</th><th>Status</th><th>Holdout</th><th>Created</th><th /></tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.id}>
              <td style={{ fontFamily: "monospace" }}>{d.id.slice(0, 8)}</td>
              <td>{d.source}</td>
              <td>
                <span className="badge" style={{ background: (STATUS_COLORS[d.status] ?? "#64748b") + "22", color: STATUS_COLORS[d.status] ?? "#64748b" }}>
                  {d.status}
                </span>
              </td>
              <td>{d.is_holdout ? "frozen" : ""}</td>
              <td>{new Date(d.created_at).toLocaleString()}</td>
              <td>
                {(d.status === "needs_review" || d.status === "extracted") && (
                  <Link to={`/review/${d.id}`}>review →</Link>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
