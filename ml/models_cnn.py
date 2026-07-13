"""Tier 3: BiLSTM + CNN visual stream on token crops.

Each token's image crop (from the rendered page, or a real scan) is encoded
by a small CNN into a 32-d visual embedding, concatenated with the engineered
features and the hash embedding, and fed to the same BiLSTM as Tier 2 —
trained end to end.

What the visual stream can see that features can't: glyph shapes OCR got
wrong (a mangled GSTIN still LOOKS like a GSTIN), boldness/size of headers,
stamps and rules on real scans. On clean synthetic renders the gain is
small — measuring that honestly is the point of run_visual_ablation().
"""

from __future__ import annotations

import numpy as np

from ml.labeling import Document, ID2TAG, NUM_TAGS, TAG2ID
from ml.models_bilstm import (EMBED_DIM, VOCAB_HASH_SIZE, _hash_token,
                              _require_torch)

CROP_H, CROP_W = 16, 48
VISUAL_DIM = 32


def token_crops(doc: Document, image=None) -> np.ndarray:
    """(n_tokens, 1, CROP_H, CROP_W) float32 in [0,1], dark ink = high value.

    `image`: PIL image of the page; rendered from tokens if not given
    (synthetic docs have no scan).
    """
    from PIL import Image

    if image is None:
        from ml.synth.generator import render
        image = render(doc)
    gray = image.convert("L")
    W, H = gray.size
    sx, sy = W / max(doc.page_width, 1), H / max(doc.page_height, 1)

    crops = np.zeros((len(doc.tokens), 1, CROP_H, CROP_W), dtype=np.float32)
    for i, tok in enumerate(doc.tokens):
        box = (max(int(tok.x0 * sx) - 2, 0), max(int(tok.y0 * sy) - 2, 0),
               min(int(tok.x1 * sx) + 2, W), min(int(tok.y1 * sy) + 2, H))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = gray.crop(box).resize((CROP_W, CROP_H), Image.BILINEAR)
        crops[i, 0] = 1.0 - np.asarray(crop, dtype=np.float32) / 255.0
    return crops


class VisualSequenceTagger:
    """BiLSTM tagger with an optional CNN stream; use_visual=False makes it
    exactly comparable to Tier 2 under the same training loop."""

    def __init__(self, use_visual: bool = True, hidden: int = 128,
                 lr: float = 1e-3, epochs: int = 12, device: str | None = None):
        self.use_visual = use_visual
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self._net = None

    def _build(self, n_features: int, torch):
        import torch.nn as nn

        use_visual = self.use_visual
        in_dim = n_features + EMBED_DIM + (VISUAL_DIM if use_visual else 0)

        class Net(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.embed = nn.Embedding(VOCAB_HASH_SIZE, EMBED_DIM)
                if use_visual:
                    self.cnn = nn.Sequential(
                        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d((2, 6)), nn.Flatten(),
                        nn.Linear(16 * 2 * 6, VISUAL_DIM), nn.ReLU(),
                    )
                self.rnn = nn.LSTM(in_dim, hidden, batch_first=True,
                                   bidirectional=True)
                self.head = nn.Sequential(nn.Dropout(0.3),
                                          nn.Linear(hidden * 2, NUM_TAGS))

            def forward(self, feats, ids, crops=None):
                parts = [feats, self.embed(ids)]
                if use_visual:
                    v = self.cnn(crops.squeeze(0)).unsqueeze(0)  # (1, T, VISUAL_DIM)
                    parts.append(v)
                out, _ = self.rnn(torch.cat(parts, dim=-1))
                return self.head(out)

        return Net(self.hidden)

    def _tensors(self, doc: Document, torch):
        from ml.features import featurize_document

        feats = torch.tensor(featurize_document(doc, context=False))
        ids = torch.tensor([_hash_token(t.text) for t in doc.tokens],
                           dtype=torch.long)
        tags = torch.tensor([TAG2ID[t.tag] for t in doc.tokens], dtype=torch.long)
        crops = (torch.tensor(token_crops(doc)) if self.use_visual else None)
        return feats, ids, tags, crops

    def fit(self, docs: list[Document], verbose: bool = True) -> "VisualSequenceTagger":
        torch = _require_torch()
        import torch.nn as nn

        device = self.device or "cpu"
        data = [self._tensors(d, torch) for d in docs if d.tokens]
        n_features = data[0][0].shape[1]
        self._net = self._build(n_features, torch).to(device)

        counts = np.bincount(
            np.concatenate([t[2].numpy() for t in data]), minlength=NUM_TAGS)
        weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)),
                               dtype=torch.float32).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):
            self._net.train()
            total = 0.0
            for i in np.random.permutation(len(data)):
                feats, ids, tags, crops = data[i]
                opt.zero_grad()
                logits = self._net(
                    feats.unsqueeze(0).to(device), ids.unsqueeze(0).to(device),
                    crops.unsqueeze(0).to(device) if crops is not None else None)[0]
                loss = loss_fn(logits, tags.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                opt.step()
                total += loss.item()
            if verbose:
                print(f"[{'bilstm+cnn' if self.use_visual else 'bilstm'}] "
                      f"epoch {epoch + 1}/{self.epochs} loss {total / len(data):.4f}")
        return self

    def predict_tags(self, doc: Document) -> list[str]:
        torch = _require_torch()
        self._net.eval()
        feats, ids, _, crops = self._tensors(doc, torch)
        with torch.no_grad():
            logits = self._net(
                feats.unsqueeze(0), ids.unsqueeze(0),
                crops.unsqueeze(0) if crops is not None else None)[0]
        return [ID2TAG[int(i)] for i in logits.argmax(dim=1)]


def run_visual_ablation(train_docs: list[Document], test_docs: list[Document],
                        epochs: int = 8) -> dict[str, float]:
    """Tier 2 (BiLSTM) vs Tier 3 (BiLSTM+CNN), same budget. Field-F1 each."""
    from ml.evaluate import evaluate_field_extraction

    results: dict[str, float] = {}
    for use_visual in (False, True):
        name = "bilstm+cnn" if use_visual else "bilstm"
        tagger = VisualSequenceTagger(use_visual=use_visual, epochs=epochs)
        tagger.fit(train_docs, verbose=False)
        tag_lists = [tagger.predict_tags(d) for d in test_docs]
        results[name] = evaluate_field_extraction(test_docs, tag_lists).macro_f1()
        print(f"{name:>12}: field-F1 {results[name]:.3f}")
    return results
