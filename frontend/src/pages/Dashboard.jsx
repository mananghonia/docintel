import { useEffect, useState } from "react";
import { api } from "../api.js";

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const num = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [drift, setDrift] = useState(null);
  const [models, setModels] = useState([]);
  const [runs, setRuns] = useState([]);
  const [busy, setBusy] = useState(false);

  const load = () => {
    api.metrics().then(setMetrics);
    api.drift().then(setDrift);
    api.modelVersions().then((d) => setModels(d.results ?? d));
    api.trainingRuns().then((d) => setRuns(d.results ?? d));
  };
  useEffect(load, []);

  const retrain = async () => {
    setBusy(true);
    try { await api.triggerRetrain(); load(); } finally { setBusy(false); }
  };

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2>Dashboard</h2>
        <div style={{ display: "flex", gap: ".5rem" }}>
          <button onClick={retrain} disabled={busy}>Trigger retrain</button>
          <button className="secondary" onClick={load}>Refresh</button>
        </div>
      </div>

      <div className="cards">
        <div className="stat"><b>{pct(metrics?.review_rate)}</b><span>docs routed to review</span></div>
        <div className="stat"><b>{num(metrics?.avg_confidence)}</b><span>avg field confidence</span></div>
        <div className="stat"><b>{num(metrics?.avg_review_seconds, 0)}s</b><span>avg time-to-review</span></div>
        <div className="stat"><b>{pct(metrics?.field_acceptance_rate)}</b><span>fields accepted as-is</span></div>
        <div className="stat"><b>{num(metrics?.avg_latency_ms, 0)}ms</b><span>avg extraction latency</span></div>
        <div className="stat">
          <b style={{ color: drift?.status === "alert" ? "#b91c1c" : drift?.status === "drifting" ? "#b45309" : "#15803d" }}>
            {drift?.psi ?? "—"}
          </b>
          <span>confidence PSI ({drift?.status ?? "insufficient data"})</span>
        </div>
      </div>

      <h3>Model versions</h3>
      <table>
        <thead><tr><th>Version</th><th>Champion</th><th>Holdout field-F1</th><th>Temp</th><th>Train docs</th><th>Created</th></tr></thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.id}>
              <td>v{m.version}</td>
              <td>{m.is_champion ? "★" : ""}</td>
              <td>{num(m.metrics?.field_macro_f1, 4)}</td>
              <td>{num(m.metrics?.temperature)}</td>
              <td>{m.n_training_docs}</td>
              <td>{new Date(m.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Training runs (rejections are logged, not hidden)</h3>
      <table>
        <thead><tr><th>When</th><th>Trigger</th><th>Outcome</th><th>Challenger F1</th><th>Champion F1</th><th>Detail</th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>{new Date(r.created_at).toLocaleString()}</td>
              <td>{r.triggered_by}</td>
              <td>{r.outcome}</td>
              <td>{num(r.challenger_f1, 4)}</td>
              <td>{num(r.champion_f1, 4)}</td>
              <td style={{ fontSize: ".8rem", color: "#64748b" }}>{r.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
