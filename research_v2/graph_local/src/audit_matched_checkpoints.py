"""Read-only fairness audit of completed four-way Validation checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"
DOMAINS = ("enzymes", "fb15k237", "sbm")
SEEDS = (17, 42, 73)
VARIANTS = ("A_both", "B_caid_only", "C_macpsg_only", "D_neither")


def main() -> None:
    results = []
    for domain in DOMAINS:
        for seed in SEEDS:
            names = [f"HCTF-GD-GraphLocal-v1_{domain}_s{seed}_{v}" for v in VARIANTS]
            paths = [AREA / "checkpoints" / f"{name}_best.pt" for name in names]
            if not all(path.exists() for path in paths):
                results.append({"domain": domain, "seed": seed,
                                "status": "INCOMPLETE", "present": sum(path.exists() for path in paths)})
                continue
            checkpoints = [torch.load(path, weights_only=False, map_location="cpu") for path in paths]
            states = [ckpt["deepgd_state_dict"] for ckpt in checkpoints]
            keyset = set(states[0])
            if any(set(state) != keyset for state in states):
                raise RuntimeError(f"model tensor keys differ: {domain} seed {seed}")
            non_hctf = [key for key in keyset if not key.startswith("hctf_block.")]
            changed = {}
            for variant, state in zip(VARIANTS[1:], states[1:]):
                changed[variant] = sorted(key for key in keyset if not torch.equal(states[0][key], state[key]))
                invalid = [key for key in changed[variant] if key in non_hctf]
                if invalid:
                    raise RuntimeError(f"frozen backbone changed in {domain} seed {seed}: {invalid[:5]}")
            for ckpt in checkpoints:
                args = ckpt["args"]
                if not (args["graph_local_community_objectives"] and args["hctf_num_trees"] == 3
                        and args["hctf_adaptive_structure_gate"] and args["freeze_backbone_for_hctf"]):
                    raise RuntimeError(f"GraphLocal/gated/frozen invariant failed: {domain} seed {seed}")
            leaked_test = any((AREA / "metrics" / f"{name}_TEST_RESERVED.csv").exists() for name in names)
            if leaked_test:
                raise RuntimeError(f"unexpected reserved Test CSV: {domain} seed {seed}")
            results.append({"domain": domain, "seed": seed, "status": "PASS",
                            "model_tensor_count": len(keyset),
                            "frozen_tensor_count": len(non_hctf),
                            "changed_hctf_tensors_by_variant": {k: len(v) for k, v in changed.items()},
                            "selected_epochs": {v: int(c["epoch"]) for v, c in zip(VARIANTS, checkpoints)},
                            "test_csv_exists": leaked_test})
    out = AREA / "manifests/matched_checkpoint_audit.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{sum(x['status']=='PASS' for x in results)}/{len(results)} matched four-way groups PASS; {out}")


if __name__ == "__main__":
    main()
