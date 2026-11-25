# Paper item to source map

Numbering follows the user-provided CVM manuscript PDF revision (14). The accompanying PDF names match the printed figure numbers in this artifact. Some source scripts retain the internal names from earlier paper revisions.

| Paper item | Data and implementation | Reproduction scope |
| --- | --- | --- |
| Algorithm 1, Figures 1 and 3 | `deepgd/model/generator/hctf.py`, strict snapshot, `figure_sources/fig01/`, `figure_sources/fig03/` | Constructor fixture and vector redraw. Figure 1 shows stage layouts, while Figure 3 uses a fixed small graph. |
| Figure 2, CAID and MA-CPSG | Training entry point and `figure_sources/fig02/` | Schematic redraw and synthetic indexing fixture. |
| Algorithm 2 and Figure 4 | `scripts/evaluate_hctf_paired_holdout.py`, `scripts/evaluate_hctf_safe_blend.py`, `figure_sources/fig04/` | Archived layout redraw and selection-loop boundary fixtures. |
| Table 1 | `assets/*_index.txt`, dataset loaders, `results/hctf_graph_metadata.csv`, `reference/05_DATASET_PROVENANCE.csv` | Archived graph ranges and split membership. Original corpus snapshots come from their providers. |
| Table 2 and Figure 7(a) | `results/hctf_submission_main/`, `results/hctf_main_clustered_analysis/crossed_bootstrap_ci.csv`, `figure_sources/fig07/` | Saved-record bootstrap and figure redraw. |
| Table 3 | `results/hctf_submission_main/`, `results/original_deepgd_baselines/`, `results/hctf_classical_baselines/` | Point summaries from archived rows. Original DeepGD is distinct from the matching backbone. |
| Table 4 | `results/autofdp_2026/`, `results/nnpnet_2026/`, `docs/BASELINE_SOURCES.md` | Seed-42 recorded comparison. External implementations and weights come from their authors. |
| Table 5 and Figure 5(a) | `results/hctf_mechanistic_counterfactuals/`, `reference/evidence/mechanism_all_metric_reproduction.csv` | Same-checkpoint intervention statistics from stored records. |
| Table 6 and Figure 5(b) | `results/hctf_objective_ablations/`, `results/hctf_submission_extensions/ablation/`, `reproduction/audit_objective_statistics.py` | 120 archived statistic rows rerun. Identity-aware intervals are separately identified. |
| Table 7 | `research_v2/graph_local/configs/`, `logs/*history.csv`, `metrics/validation_summary.csv` within the GraphLocal directory | 36 checkpoint-selection histories and summary regeneration. |
| Table 8 and Figure 6 | `research_v2/factorial_ablation/validation_graphwise_deterministic_v2/`, `metrics/`, `figure_sources/fig06/` | Validation contrasts, architecture controls, and redraw. |
| Table 9 | `results/hctf_strict_size_ood/`, `results/hctf_strict_size_ood_analysis/crossed_bootstrap_ci.csv` | Original strict archive statistics. |
| Figure 7(b) | `results/hctf_q1_controls/`, `results/hctf_q1_controls_analysis_strict/crossed_bootstrap_ci.csv` | Q1 reevaluation statistics. |
| Figure 8 | `results/hctf_safe_blend_paired_holdout/`, `figure_sources/statistical/source_data/` | Accepted counts, fallback metrics, seven post-selection endpoint intervals and redraw. |
| Table 10 | `results/hctf_safe_blend_all17_extrap6_prefix2_test/` and `results/hctf_safe_blend_all17_proteins_portfolio4_extrap6_prefix2_test/` | Two exploratory scopes with different candidate portfolios. |
| Figure 9 | `results/hctf_end_to_end/`, `results/hctf_submission_main/`, `figure_sources/statistical/` | End-to-end and stage records plus deduplicated batched timing medians. |
| Figure 10 | `figure_sources/fig10/data/` and its redraw script | Saved SBM examples, seed 73 graph 83 and seed 17 graph 13. |
