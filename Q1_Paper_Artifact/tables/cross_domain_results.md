# Cross-Domain (Transfer Matrix) Results — full_fusion

**Source**: `results/P0_device_gap` (R_R/S_S/R_S/S_R) + RS_RS pooled-protocol results. Underlies Figure 4.

| Train ↓ / Test → | Recorder (R) | Smartphone (S) | Pooled (R+S) |
|---|---|---|---|
| **Recorder (R)** | 0.5843 (matched) | 0.5311 (cross) | N/A — not evaluated |
| **Smartphone (S)** | 0.5311 (cross)* | 0.5843 (matched)* | N/A — not evaluated |
| **Pooled (R+S)** | N/A — not evaluated | N/A — not evaluated | 0.593 (pooled) |

*Note: R→S and S→R cross-device balanced accuracy differ slightly by direction in the raw data (R→S: 0.5289 avg; S→R: 0.5333 avg per representation-level breakdowns in Table 5 of `manuscript.md`); the value shown here (0.5311) is the direction-averaged figure used in Figure 4 for visual symmetry — see `MASTER_RESULTS.csv` for the exact per-direction values.

**N/A cells are explicit and intentional.** The pooled-training protocol (RS→RS) trains and evaluates on both devices combined; it does not produce a pooled-train/single-device-test cell, and none of the evaluated protocols produce a single-device-train/pooled-test cell. Per this artifact's data-integrity rule, these cells are marked N/A rather than estimated or interpolated.

## Pooled-training comparison, 5 representations (manuscript Table 5)

| Representation | Cross BA | Matched BA | Pooled BA | Gap closed |
|---|---|---|---|---|
| wavlm_large | 0.541 | 0.611 | 0.613 | 101% |
| full_fusion_plus_hear | 0.529 | 0.586 | 0.594 | 115% |
| full_fusion | 0.531 | 0.584 | 0.593 | 118% |
| data2vec_fusion | 0.535 | 0.583 | 0.588 | 111% |
| classical | 0.499 | 0.551 | 0.563 | 123% |

*Source: `results/P0_device_gap` (cross/matched), RS_RS pooled-protocol results (100 combinations: 5 representations × 4 classifiers × 5 folds).*
