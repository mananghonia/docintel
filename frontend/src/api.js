const json = (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

// Attach a token when one is stored (a REQUIRE_AUTH=1 deployment); in local
// dev nothing is stored, no header is sent, and the open API works unchanged.
// Set one from the console: localStorage.setItem("docintel_token", "<token>").
const af = (url, opts = {}) => {
  const token = localStorage.getItem("docintel_token");
  const headers = { ...(opts.headers || {}) };
  if (token) headers.Authorization = `Token ${token}`;
  return fetch(url, { ...opts, headers });
};

export const api = {
  login: (username, password) =>
    af("/api/auth/token/", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }).then(json).then((d) => {
      localStorage.setItem("docintel_token", d.token);
      return d;
    }),
  documents: (status) =>
    af(`/api/documents/${status ? `?status=${status}` : ""}`).then(json),
  document: (id) => af(`/api/documents/${id}/`).then(json),
  reviewQueue: () => af("/api/documents/review_queue/").then(json),
  ingestSynthetic: (n) =>
    af("/api/documents/ingest_synthetic/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n }),
    }).then(json),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return af("/api/documents/upload/", { method: "POST", body: form }).then(json);
  },
  exportDocument: (id) => af(`/api/documents/${id}/export/`).then(json),
  submitReview: (id, corrections, reviewSeconds) =>
    af(`/api/documents/${id}/review/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corrections, review_seconds: reviewSeconds }),
    }).then(json),
  metrics: () => af("/api/monitoring/metrics/").then(json),
  drift: () => af("/api/monitoring/drift/").then(json),
  modelVersions: () => af("/api/training/models/").then(json),
  trainingRuns: () => af("/api/training/runs/").then(json),
  triggerRetrain: () =>
    af("/api/training/runs/trigger/", { method: "POST" }).then(json),
};

export const confColor = (c) =>
  c >= 0.9 ? "#15803d" : c >= 0.7 ? "#b45309" : "#b91c1c";
export const confBg = (c) =>
  c >= 0.9 ? "rgba(34,197,94,.18)" : c >= 0.7 ? "rgba(245,158,11,.20)" : "rgba(239,68,68,.20)";
