# Reproduction results and limits

## Verified with the staged artifact

| Check | Result |
| --- | --- |
| Archived inputs | 607 allowlisted input files matched the SHA-256 recorded before packaging. The file manifest also records the digest of each staged copy. |
| Main and strict saved-record bootstrap | 14 comparisons produced 189 matched domain/metric summaries, each with 4,000 replicates. |
| Additional archived statistics | 120 objective-ablation summaries and 160 mechanism summaries matched the original values. |
| Raw stress references | Table 3 seed means and Table 4 seed-42 external comparison means matched the reference files. |
| GraphLocal-v1 | 36 completed Validation histories summarized. Both graphwise Validation analyses executed with 10,000 bootstrap replicates. |
| SafeBlend 10+7 | 19,450 selection records. Full 970/9,725 and Node-only 870/9,725 accepted. The 17 stored endpoints match baseline values exactly on the corresponding fallback records. The code fixture checks exact fallback behavior. |
| Exploratory All-17 | ENZYMES 12/58 with six portfolio sources and PROTEINS 15/108 with four. These are portfolio source counts. |
| Runtime | Archived end-to-end observations were checked. Across 2,745 distinct run/batch observations, the batched medians round to 23.0, 27.5 and 29.8 ms/graph. |
| Figures | All ten figures were generated with their Python plotting sources. The archived PDFs are packaged alongside the redraw scripts. |
| Weights | 255 repackaged checkpoint files passed tensor-by-tensor comparison against the local model states. The current evaluator loaded three SBM variants on CPU. |

## Comparisons that use different protocols

The eight-domain shared warm-start Full-versus-Node stress effect is +0.735343%, with its archived interval [−0.474641%, +1.748100%]. The original strict-size result is +0.232448% [0.090631%, 0.377881%]. The separate strict Q1 reevaluation gives +0.232378% [0.094598%, 0.373853%]. These are different archival comparisons. All-17 uses different candidate portfolios from the 10+7 protocol. The GraphLocal-v1 training and fixed-checkpoint factorial results in this artifact use Validation records.

## Historical limits

- The archived paired selection CSVs retain final selections and endpoint values. Complete historical six-trial records and coordinate arrays are available for selected cases only. Matching fallback metrics across the archive do not establish byte-level equality of every historical coordinate array.
- Some legacy evaluation records predate recording source and checkpoint hashes. The package records the historical links where available and retains the narrower meaning of a saved-record reproduction.
- The optimizer states, complete RNG states, and exact historical data downloads are unavailable for exact training resumption. New model training and complete end-to-end Test reruns were outside this artifact check.
- Raw third-party corpora, external baseline repositories and weights, intermediate logs, caches, non-paper runs, and independent sealed Test records are outside the archive. Dataset sources and needed split context appear in [DATA.md](DATA.md).
- Original source archives remain byte-for-byte. Where a JSON, CSV, or text file used local paths, the staged copy uses portable paths. Both hashes appear in `FILE_MANIFEST.csv`. The inference weight archive omits optimizer state and uses new serialization with tensor equality verified.
