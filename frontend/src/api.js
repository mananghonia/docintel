const json = (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

export const api = {
  documents: (status) =>
    fetch(`/api/documents/${status ? `?status=${status}` : ""}`).then(json),
  document: (id) => fetch(`/api/documents/${id}/`).then(json),
  reviewQueue: () => fetch("/api/documents/review_queue/").then(json),
  ingestSynthetic: (n) =>
    fetch("/api/documents/ingest_synthetic/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n }),
    }).then(json),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/documents/upload/", { method: "POST", body: form }).then(json);
  },
  exportDocument: (id) => fetch(`/api/documents/${id}/export/`).then(json),
  submitReview: (id, corrections, reviewSeconds) =>
    fetch(`/api/documents/${id}/review/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corrections, review_seconds: reviewSeconds }),
    }).then(json),
  metrics: () => fetch("/api/monitoring/metrics/").then(json),
  drift: () => fetch("/api/monitoring/drift/").then(json),
  modelVersions: () => fetch("/api/training/models/").then(json),
  trainingRuns: () => fetch("/api/training/runs/").then(json),
  triggerRetrain: () =>
    fetch("/api/training/runs/trigger/", { method: "POST" }).then(json),
};

export const confColor = (c) =>
  c >= 0.9 ? "#15803d" : c >= 0.7 ? "#b45309" : "#b91c1c";
export const confBg = (c) =>
  c >= 0.9 ? "rgba(34,197,94,.18)" : c >= 0.7 ? "rgba(245,158,11,.20)" : "rgba(239,68,68,.20)";
