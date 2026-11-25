# HCTF-GD paper artifact

**HCTF-GD: Hybrid Community-Conditioned Tree Flows with Auditable Safe Selection for Neural Graph Drawing**

Version 0.1.0 bundles the paper implementation, reported experiment records, configuration snapshots, figure sources, and reproduction scripts. The paper's figures and tables are indexed in [PAPER_MAP.md](docs/PAPER_MAP.md). The archive covers the shared warm-start study, the separate strict-size archives, SafeBlend 10+7, the exploratory All-17 studies, and the GraphLocal Validation controls.

## Download the inference weights

When the weights are published, download `HCTF-GD_inference_weights_v0.1.0.zip` from the repository's `v0.1.0` Release. Extract the ZIP separately and merge its `checkpoints/` and `research_v2/` folders into the corresponding paths in a local copy of this repository. The archive includes a model file index. The companion SHA-256 file lets you verify the downloaded ZIP against the checksum posted with the release.

## Verify and reproduce saved results

Use Python 3.10–3.12. The pinned analysis dependencies were tested with Python 3.10.

```bash
python -m pip install -r requirements-analysis.txt
python reproduction/verify.py
python reproduction/raw_stress_tables.py --out regenerated_tables
python reproduction/02_BOOTSTRAP_AUDIT.py
python reproduction/audit_objective_statistics.py
python reproduction/audit_mechanism_all_metrics.py
python reproduction/audit_dataset_records.py
python reproduction/audit_protocol_edges.py
python reproduction/redraw.py --out regenerated_figures
```

Run the Validation scripts below in a separate extracted copy. They regenerate files under `research_v2/*/metrics/` in that copy. Run verification before regenerating archived files.

```bash
python research_v2/graph_local/src/summarize_validation.py
python research_v2/graph_local/src/analyze_graphwise_validation.py
python research_v2/factorial_ablation/analyze_factorial_validation.py
```

The bootstrap script writes new files under `reproduction/` and may take several minutes. A fresh extraction reruns all comparisons. `reference/` contains the comparison targets. Figure PDFs in `paper_figures/` are the archived manuscript assets. `redraw.py` uses the recorded numerical and diagram inputs and writes to a new directory. Font versions and PDF metadata affect the regenerated file bytes.

## Model code and inference weights

- `deepgd/` contains the backbone, HCTF, dataset loaders and shared metric code. `train_cser_spectral_mog_end2end.py` is the training entry point. Training commands for the paper are archived under the matching `results/` directories.
- `scripts/evaluate_hctf_paired_holdout.py` implements the paired SafeBlend 10+7 evaluation. `scripts/evaluate_hctf_safe_blend.py` includes the separate All-17 path. The strict-size run snapshot lives under `results/hctf_strict_size_ood/source_snapshot_strict/`. GraphLocal source is under `research_v2/graph_local/source_snapshot/`.
- The companion inference weight archive follows the same relative paths. Its `*_best.pt` files contain the verified model tensors and portable configuration metadata. The index in `reference/model_weight_index.csv` maps each repackaged file to the SHA-256 of its source. The weight archive supports inference and checkpoint loading. Resuming the historical optimizer state requires original training state.

Install PyTorch and the corresponding PyG build for your platform before running model code. The local CPU fixture and loader checks ran in the existing Python 3.10 environment (PyTorch 2.0.0+cu118, PyG 2.3.1), while weights were extracted and checked with the restricted `weights_only` loader in PyTorch 2.6.0. For new environments, install a maintained PyTorch release with your CPU or CUDA build, then install PyG and the matching scatter/sparse extensions if needed. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
python -m pip install -r requirements-model-extras.txt
python train_cser_spectral_mog_end2end.py --help
python scripts/evaluate_hctf_paired_holdout.py --help
python reproduction/audit_method_fixtures.py
python reproduction/audit_safeblend_records.py
```

Replaying predictions requires the upstream corpora, dataset reconstruction and matching snapshot as described in [DATA.md](docs/DATA.md). The included saved-record checks require no downloads, training, or GPU. The source includes paths for active research configurations with distinct prerequisites. Use the archived command and protocol for the result being reproduced.

## Provenance and scope

The [file manifest](FILE_MANIFEST.csv) records the SHA-256 of staged files, the matching historical digest when available, and any normalization of local path metadata. Upstream DeepGD attribution and its license appear in [LICENSES](LICENSES/README.md). The evidence limits and the release tests are recorded in [REPRODUCTION.md](docs/REPRODUCTION.md).
