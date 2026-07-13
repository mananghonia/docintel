"""Tier 2: sequence tagger — vanilla RNN vs LSTM vs BiLSTM (the ablation).

Input per token: the engineered feature vector (ml.features) concatenated
with a learned embedding of the hashed lowercased token text. The sequence
is the document in reading order, so recurrence can carry "we are inside the
totals block" across tokens — the thing per-token classifiers can't do.

The ablation is a deliverable: vanilla RNN degrading on long invoices is the
vanishing-gradient lesson made concrete; the BiLSTM winning shows why seeing
the RIGHT context ("the token before 'Total:'" vs after) matters.

torch is imported lazily: everything else in ml/ works without it.
"""

from __future__ import annotations

import numpy as np

from ml.labeling import Document, ID2TAG, NUM_TAGS, TAG2ID

VOCAB_HASH_SIZE = 4096
EMBED_DIM = 32


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:
        raise ImportError(
            "Tier 2 needs PyTorch: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from e


def _hash_token(text: str) -> int:
    import zlib
    return zlib.crc32(text.strip().lower().encode("utf-8")) % VOCAB_HASH_SIZE


def _doc_tensors(doc: Document, torch):
    from ml.features import featurize_document

    # context=False: shape+geometry only. Context must come from recurrence —
    # that's what the RNN/LSTM/BiLSTM comparison is measuring.
    feats = torch.tensor(featurize_document(doc, context=False))
    ids = torch.tensor([_hash_token(t.text) for t in doc.tokens], dtype=torch.long)
    tags = torch.tensor([TAG2ID[t.tag] for t in doc.tokens], dtype=torch.long)
    return feats, ids, tags


class SequenceTagger:
    """rnn_type in {'rnn', 'lstm', 'bilstm'} — same everything else, so the
    comparison isolates the recurrence architecture."""

    def __init__(self, rnn_type: str = "bilstm", hidden: int = 128,
                 lr: float = 1e-3, epochs: int = 12, device: str | None = None,
                 seed: int | None = None):
        assert rnn_type in ("rnn", "lstm", "bilstm")
        self.rnn_type = rnn_type
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self._net = None

    def _build(self, n_features: int, torch):
        import torch.nn as nn

        bidirectional = self.rnn_type == "bilstm"
        rnn_cls = nn.RNN if self.rnn_type == "rnn" else nn.LSTM
        out_dim = self.hidden * (2 if bidirectional else 1)

        class Net(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.embed = nn.Embedding(VOCAB_HASH_SIZE, EMBED_DIM)
                self.rnn = rnn_cls(n_features + EMBED_DIM, hidden,
                                   batch_first=True, bidirectional=bidirectional)
                self.head = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(out_dim, NUM_TAGS))

            def forward(self, feats, ids):
                x = torch.cat([feats, self.embed(ids)], dim=-1)
                out, _ = self.rnn(x)
                return self.head(out)

        return Net(self.hidden)

    def fit(self, docs: list[Document], val_docs: list[Document] | None = None,
            verbose: bool = True) -> "SequenceTagger":
        torch = _require_torch()
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        data = [_doc_tensors(d, torch) for d in docs if d.tokens]
        n_features = data[0][0].shape[1]
        self._net = self._build(n_features, torch).to(device)

        # Class weights: O dominates ~85/15; without weighting the net just
        # predicts O everywhere and looks great on accuracy (trap #3).
        counts = np.bincount(
            np.concatenate([t[2].numpy() for t in data]), minlength=NUM_TAGS)
        weights = torch.tensor(
            (1.0 / np.sqrt(np.maximum(counts, 1))), dtype=torch.float32).to(device)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):
            self._net.train()
            total = 0.0
            perm = np.random.permutation(len(data))
            for i in perm:  # batch = one document (variable length)
                feats, ids, tags = (t.to(device) for t in data[i])
                opt.zero_grad()
                logits = self._net(feats.unsqueeze(0), ids.unsqueeze(0))[0]
                loss = loss_fn(logits, tags)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                opt.step()
                total += loss.item()
            if verbose:
                msg = f"[{self.rnn_type}] epoch {epoch + 1}/{self.epochs} loss {total / len(data):.4f}"
                if val_docs:
                    from ml.evaluate import evaluate_field_extraction
                    tag_lists = [self.predict_tags(d) for d in val_docs]
                    f1 = evaluate_field_extraction(val_docs, tag_lists).macro_f1()
                    msg += f"  val field-F1 {f1:.3f}"
                print(msg)
        return self

    def predict_logits(self, doc: Document) -> np.ndarray:
        torch = _require_torch()
        device = next(self._net.parameters()).device
        self._net.eval()
        feats, ids, _ = _doc_tensors(doc, torch)
        with torch.no_grad():
            logits = self._net(feats.unsqueeze(0).to(device),
                               ids.unsqueeze(0).to(device))[0]
        return logits.cpu().numpy()

    def predict_proba(self, doc: Document) -> np.ndarray:
        z = self.predict_logits(doc)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_tags(self, doc: Document) -> list[str]:
        return [ID2TAG[int(i)] for i in self.predict_logits(doc).argmax(axis=1)]

    def save(self, path: str) -> None:
        torch = _require_torch()
        torch.save({"state": self._net.state_dict(), "rnn_type": self.rnn_type,
                    "hidden": self.hidden}, path)


def run_ablation(train_docs: list[Document], test_docs: list[Document],
                 epochs: int = 10, seeds: tuple = (0, 1, 2)) -> dict[str, dict]:
    """RNN vs LSTM vs BiLSTM, identical data/budget, mean±std over seeds.

    Multi-seed is not optional: on a few-hundred-doc corpus, single-seed
    differences of ±0.02 field-F1 are init noise, and a single run will
    happily 'show' whichever architecture got the luckiest seed.
    """
    import numpy as np

    from ml.evaluate import evaluate_field_extraction

    results: dict[str, dict] = {}
    for rnn_type in ("rnn", "lstm", "bilstm"):
        f1s = []
        for seed in seeds:
            tagger = SequenceTagger(rnn_type=rnn_type, epochs=epochs, seed=seed)
            tagger.fit(train_docs, verbose=False)
            tag_lists = [tagger.predict_tags(d) for d in test_docs]
            f1s.append(evaluate_field_extraction(test_docs, tag_lists).macro_f1())
        results[rnn_type] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)),
                             "runs": f1s}
        print(f"{rnn_type:>8}: field-F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"({' '.join(f'{f:.3f}' for f in f1s)})")
    return results
