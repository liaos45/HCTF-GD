"""Read-only CUDA repeatability diagnosis for the first locked Validation graph."""

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

import evaluate_factorial_validation as ev


def main() -> None:
    if os.environ.get("HCTF_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    device = "cuda"
    domain, seed = "enzymes", 17
    checkpoint = ev.ROOT / "checkpoints/hctf_strict_size_ood/strictsize_enzymes_adaptive_supervised_full_s17_best.pt"
    args, model, expected = ev.load_model(checkpoint, device)
    ev.train.set_random_seed(seed)
    _, validation, reserved = ev.train.load_graph_dataset(args)
    del reserved
    batch = next(iter(ev.train.make_loader(validation, 1, shuffle=False))).to(device)
    metric = ev.train.Stress()
    raw1, gates1, _ = ev.raw_forward(batch, args, model, metric, seed, 0, device, None)
    raw2, gates2, _ = ev.raw_forward(batch, args, model, metric, seed, 0, device, None)
    replay, gates_replay, _ = ev.raw_forward(batch, args, model, metric, seed, 0, device, "111", gates1)
    report = {
        "graph_id": str(validation[0].name), "expected_insertions": expected,
        "raw_vs_raw_max_abs": float((raw1 - raw2).abs().max()),
        "raw_vs_replay_max_abs": float((raw1 - replay).abs().max()),
        "raw_vs_raw_close_1e6": bool(torch.allclose(raw1, raw2, atol=1e-6, rtol=1e-6)),
        "raw_vs_replay_close_1e6": bool(torch.allclose(raw1, replay, atol=1e-6, rtol=1e-6)),
        "raw_vs_replay_hidden_max_abs": max(float((a-b).abs().max()) for a,b in zip(gates1["hctf"], gates_replay["hctf"])),
        "raw_vs_raw_hidden_max_abs": max(float((a-b).abs().max()) for a,b in zip(gates1["hctf"], gates2["hctf"])),
    }
    mode = "deterministic" if os.environ.get("HCTF_DETERMINISTIC") == "1" else "default"
    report["cuda_mode"] = mode
    output = Path(__file__).resolve().parent / f"logs/cuda_equivalence_{mode}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
