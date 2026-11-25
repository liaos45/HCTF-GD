# 2026 baseline source manifest

Downloaded on 2026-09-13 for the HCTF-GD comparison experiments.

## AutoFDP

- Official repository: https://github.com/IDEASLab-AutoFDP/AutoFDP
- Local path: `AutoFDP/`
- Commit: `82d15e2e6d7a183a320d91b6db274eff5d5749b4`
- Upstream environment: Ubuntu 20.04, g++ 9.4.0, GNU Make 4.2.1.
- Windows port: the force and optimization formulas are preserved; the Unix-only
  path/timing code, variable-length arrays, Matrix Market reader, and fixed demo
  entry point were adapted for portable C++17/MSVC execution.
- Windows build: `AutoFDP/build-vs/Release/autofdp.exe`.
- Reproducible runner: `scripts/run_autofdp_baselines.py` exports the existing
  DeepGD/HCTF test splits to one-based symmetric Matrix Market files, executes
  AutoFDP, validates VNA node coordinates, and calls the same metric evaluator
  used for the classical baselines.
- Full comparison (2026-09-14): seed 42, 50 AutoFDP iterations, eight datasets,
  and four predeclared variants (`SE`, `SE+IL`, `SE+CL`, and `SE+NR`). All 32
  dataset/variant tasks succeeded, producing 7,780 graph-variant records with
  zero failed graphs, missing values, infinite values, or duplicate graph keys.
- Result directory: `results/autofdp_2026/`.

## NNP-NET

- Official repository: https://github.com/IlanHartskeerl/NNP-NET
- Local path: `NNP-NET-official/`
- Commit: `ea34cf9275b6c2d969ada2976f2dd0d3e1bd926c`
- Version scope: the repository README links the GD 2025 paper, the 2026 TVCG static/dynamic extension, and the 2026 multivariate extension.
- Windows build: `NNP-NET-official/build-vs/NNP-NET/Release/NNPNET.exe`
- Python 3.13 module: `NNP-NET-official/build-vs/NNP-NET/Release/NNP_NET.cp313-win_amd64.pyd`
- Verification: the Python module imports successfully; the bundled `3elt.mtx` PMDS smoke test completed successfully and produced `NNP-NET-official/build-vs/pmds_smoke.vna`.
- Runtime dependency: the official Python 3.13 module was verified with an isolated
  TensorFlow 2.21 installation under `.deps/nnpnet_py313`. The full DeepGD run
  uses the pre-existing Python 3.10/TensorFlow 2.15 compatibility build so its
  small-graph behavior matches the earlier NNP-NET experiments.
- Method dating: the static NNP-NET core is the GD 2025 method. The repository's
  2026 extensions concern time-series and multivariate graphs; ordinary static
  DeepGD datasets must not be described as a new 2026 static architecture.
- Full static comparison (2026-09-14): 40 epochs, split seed 42, deterministic
  per-graph TensorFlow seeds, and all eight datasets. All 1,945 test graphs
  succeeded with zero failed graphs, missing values, infinite values, or
  duplicate graph keys. Results are in `results/nnpnet_2026_static/`.
- Reproducible runners: `scripts/run_nnpnet_module_batch.py` and
  `scripts/run_nnpnet_2026_baselines.py`.

## Existing NNP-NET experiment snapshot

The pre-existing `NNP-NET-master/` directory was preserved because it contains project-specific graph exporters, evaluation scripts, generated layouts, and metric CSV files. It is not a Git checkout and its `NNPNET.cpp` differs from the current official source. Do not overwrite it when updating the official baseline.
