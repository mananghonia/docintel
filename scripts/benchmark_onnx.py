"""ONNX export + INT8 quantization latency benchmark (week 8 deliverable).

Trains a small BiLSTM tagger, then measures per-document latency and
field-F1 across three serving paths:

    torch eager  ->  ONNX fp32  ->  ONNX INT8 (dynamic quantization)

Reports p50/p95/p99 and the accuracy cost of quantization. Run:

    python scripts/benchmark_onnx.py [--docs 120] [--epochs 8]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def percentiles(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def main(n_docs: int, epochs: int) -> None:
    import torch

    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_document
    from ml.labeling import ID2TAG, assign_labels
    from ml.models_bilstm import SequenceTagger, _hash_token
    from ml.synth.generator import generate_dataset

    out_dir = ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== train BiLSTM on {n_docs} hard docs ==")
    docs = generate_dataset(n_docs, seed=7, hard=True)
    for d in docs:
        assign_labels(d)
    n_test = n_docs // 5
    test, train = docs[:n_test], docs[n_test:]
    tagger = SequenceTagger(rnn_type="bilstm", epochs=epochs)
    tagger.fit(train, verbose=False)
    net = tagger._net.eval()

    # Pre-featurize test docs so the benchmark isolates MODEL latency.
    inputs = []
    for d in test:
        feats = torch.tensor(featurize_document(d, context=False)).unsqueeze(0)
        ids = torch.tensor([_hash_token(t.text) for t in d.tokens]).unsqueeze(0)
        inputs.append((feats, ids))

    # --- torch eager ---------------------------------------------------------
    def run_torch(feats, ids):
        with torch.no_grad():
            return net(feats, ids)[0].numpy()

    # --- ONNX export -----------------------------------------------------------
    fp32_path = str(out_dir / "bilstm_fp32.onnx")
    int8_path = str(out_dir / "bilstm_int8.onnx")
    # dynamo=False: the legacy exporter honours dynamic_axes for LSTMs; the
    # dynamo path bakes the example sequence length into a Reshape node.
    torch.onnx.export(
        net, (inputs[0][0], inputs[0][1]), fp32_path,
        input_names=["feats", "ids"], output_names=["logits"],
        dynamic_axes={"feats": {1: "seq"}, "ids": {1: "seq"},
                      "logits": {1: "seq"}},
        opset_version=17, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    size_fp32 = Path(fp32_path).stat().st_size / 1e6
    size_int8 = Path(int8_path).stat().st_size / 1e6
    print(f"   exported: fp32 {size_fp32:.2f} MB -> int8 {size_int8:.2f} MB")

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-thread: honest per-request numbers
    sess_fp32 = ort.InferenceSession(fp32_path, opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, opts, providers=["CPUExecutionProvider"])

    def run_onnx(sess, feats, ids):
        return sess.run(["logits"], {"feats": feats.numpy(),
                                     "ids": ids.numpy().astype(np.int64)})[0][0]

    engines = {
        "torch_eager": lambda f, i: run_torch(f, i),
        "onnx_fp32": lambda f, i: run_onnx(sess_fp32, f, i),
        "onnx_int8": lambda f, i: run_onnx(sess_int8, f, i),
    }

    print(f"\n== latency ({len(test)} docs x 5 repeats, 1 thread) and field-F1 ==")
    results = {}
    for name, fn in engines.items():
        for feats, ids in inputs[:3]:  # warmup
            fn(feats, ids)
        times, tag_lists = [], []
        for feats, ids in inputs:
            best = np.inf
            for _ in range(5):
                t0 = time.perf_counter()
                logits = fn(feats, ids)
                best = min(best, (time.perf_counter() - t0) * 1000)
            times.append(best)
            tag_lists.append([ID2TAG[int(i)] for i in logits.argmax(axis=1)])
        f1 = evaluate_field_extraction(test, tag_lists).macro_f1()
        p = percentiles(times)
        results[name] = {**p, "field_f1": f1}
        print(f"   {name:>12}  p50 {p['p50']:6.2f}ms  p95 {p['p95']:6.2f}ms  "
              f"p99 {p['p99']:6.2f}ms  field-F1 {f1:.4f}")

    delta = results["onnx_fp32"]["field_f1"] - results["onnx_int8"]["field_f1"]
    speedup = results["torch_eager"]["p95"] / results["onnx_int8"]["p95"]
    print(f"\n   int8 vs torch: {speedup:.1f}x faster at p95, "
          f"F1 cost {delta:+.4f}, {size_fp32 / size_int8:.1f}x smaller")

    results["model_size_mb"] = {"fp32": size_fp32, "int8": size_int8}
    (out_dir / "onnx_benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"   written to {out_dir / 'onnx_benchmark.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    main(args.docs, args.epochs)
