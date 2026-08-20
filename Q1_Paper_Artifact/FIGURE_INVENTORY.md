# Figure Inventory

Per-figure accounting: manuscript location, RQ addressed, data source, computation, interpretation, and measured-vs-conceptual status. Captions below are the exact captions used (also stored machine-readably in `figures/captions.json`).

---

### Figure 1 — `figures/fig01_problem_setting.png`
- **Status: CONCEPTUAL / ILLUSTRATIVE.** A 4-box causal-flow diagram, not a data plot.
- **Manuscript location:** Section 1 (Introduction), motivating figure.
- **RQ addressed:** frames RQ1 before it is answered quantitatively.
- **Data source:** none — schematic only, built with matplotlib `FancyBboxPatch`/`annotate`.
- **Computation:** none.
- **Interpretation:** states the study's causal hypothesis chain (concurrent recording → distinct acoustic domains → transfer degradation), each downstream box referencing the real figure that tests it.
- **Caption:** "The study's motivating causal chain: identical physiological events recorded concurrently by two devices in the same acoustic environment nonetheless occupy measurably distinct acoustic domains (Figure 3), and a classifier trained on one device's recordings degrades when evaluated on the other (Figure 4, Table 2)."

### Figure 2 — `figures/fig02_architecture.png`
- **Status: AS-IMPLEMENTED**, reconstructed from code, not aspirational.
- **Manuscript location:** Section 4 (Proposed Method / Evaluation Framework).
- **RQ addressed:** none directly — documents the pipeline RQ2–RQ4 are evaluated on.
- **Data source:** `src/sleep_quadnet/`, `scripts/extract_features.py`, `scripts/train_classifier.py` (reverse-engineered, not run for this figure).
- **Computation:** none — schematic.
- **Interpretation:** 7-stage flow (raw audio → resample/preprocess → 14-way representation branch → 4-way classifier branch → 5 protocols → metrics/significance). PCA/CORAL and the additional-signal branches (ODI/HB, corroboration filter, sliding-window severity) are omitted from the diagram for legibility and their exact insertion points are instead described in Section 3/4 text.
- **Caption:** "The evaluation pipeline's real component stages, reconstructed directly from the codebase (`src/sleep_quadnet/`, `scripts/`), not an idealized or aspirational architecture. PCA/CORAL domain-adaptation and the additional-signal branches attach after feature extraction and are omitted here for legibility."

### Figure 3 — `figures/fig03_domain_shift.png`
- **Status: MEASURED.**
- **Manuscript location:** Section 6.1 (RQ1).
- **RQ addressed:** RQ1 — severity of domain shift, signal level.
- **Data source:** raw WAV files, stratified n=600/device (`tables/domain_shift_audio_stats.csv`).
- **Computation:** `scripts/analyze_domain_shift.py` — librosa RMS/spectral centroid/bandwidth/rolloff/ZCR/flatness; Mann-Whitney U + Cliff's delta per metric.
- **Interpretation:** near-complete distributional separation (|δ|>0.95 for 5/6 metrics) between Recorder and Smartphone.
- **Caveat explicitly stated in caption:** the underlying test is window-level, not subject-level (Section 5.3 caveat applies).
- **Caption:** see `figures/captions.json["fig03_domain_shift"]` (full text reused verbatim in manuscript).

### Figure 4 — `figures/fig04_transfer_matrix.png`
- **Status: MEASURED, with explicit N/A cells.**
- **Manuscript location:** Section 6.3 (RQ3) and referenced from 6.1.
- **RQ addressed:** RQ1/RQ3 — transfer severity and the pooled-training mitigation.
- **Data source:** `results/P0_device_gap` (R_R/S_S/R_S/S_R) + RS_RS pooled-protocol results, representation=`full_fusion`.
- **Computation:** mean balanced accuracy per protocol cell, aggregated over classifiers/folds.
- **Interpretation:** diagonal = matched, off-diagonal = cross-device, one pooled cell (RS→RS). Pooled-train/single-device-test cells are explicitly marked N/A because that specific combination was never evaluated — not estimated or interpolated.
- **Caption:** see `figures/captions.json["fig04_transfer_matrix"]`.

### Figure 5 — `figures/fig05_embedding_space.png`
- **Status: MEASURED.**
- **Manuscript location:** Section 6.1 (RQ1).
- **RQ addressed:** RQ1 — mechanistic, representation-level explanation for the transfer gap.
- **Data source:** cached HuBERT embeddings, stratified n=2,400 (`tables/embedding_pca_coords.csv`).
- **Computation:** `scripts/analyze_embedding_space.py` — 10-component PCA, silhouette score by device grouping and by class grouping (computed in full 10-D space; only PC1/PC2 plotted).
- **Interpretation:** device separates ~20x more cleanly (0.145 vs 0.007 silhouette) than the clinical label in this frozen representation.
- **Caption:** see `figures/captions.json["fig05_embedding_space"]`.

### Figure 6 — `figures/fig06_generalization_gap.png`
- **Status: MEASURED.**
- **Manuscript location:** Section 6.2 (RQ2).
- **RQ addressed:** RQ2 — transfer quality across all representations.
- **Data source:** `results/P0_device_gap`, all 14 representations (odi_hb excluded — degenerate, gap=0 by construction).
- **Computation:** mean matched BA − mean cross BA, per representation, averaged over 4 classifiers and both cross/matched protocol pairs.
- **Interpretation:** every real representation shows a positive gap; statistically confirmed significant specifically for `full_fusion` (paired bootstrap, cited from `results/P0_statistics_v2`) — the figure itself shows point estimates for all 13, significance is not re-tested per-bar in this figure and the caption says so.
- **Caption:** see `figures/captions.json["fig06_generalization_gap"]`.

### Figure 7 — `figures/fig07_representation_comparison.png`
- **Status: MEASURED**, paired explicitly with a significance caveat.
- **Manuscript location:** Section 6.2 (RQ2).
- **RQ addressed:** RQ2 — "which representation is best" leaderboard, and its fragility.
- **Data source:** `results/P0_device_gap`.
- **Computation:** mean cross-device BA (R→S, S→R averaged) per representation.
- **Interpretation:** visible point-estimate spread, but caption explicitly states no representation is confirmed significantly better under the validation-selected significance protocol (`results/P0_statistics_v2`) — this figure is the one the master prompt most explicitly warns against over-reading, and the caption is written defensively for that reason.
- **Caption:** see `figures/captions.json["fig07_representation_comparison"]`.

### Figure 8 — `figures/fig08_ablation.png`
- **Status: MEASURED.**
- **Manuscript location:** Section 6.4 (RQ4).
- **RQ addressed:** RQ4 — component contribution.
- **Data source:** `results/P0_ablation`, `results/P0_ablation_statistics/ablation_vs_full_fusion.csv`.
- **Computation:** leave-one-encoder-out point estimates vs. `full_fusion` reference; orange bars = individually significant vs. reference in ≥1 tested combination (7/89 total).
- **Interpretation:** most point estimates numerically exceed the reference, but the large majority of combinations (82/89) are not significant — no individual encoder is confirmed necessary.
- **Caption:** see `figures/captions.json["fig08_ablation"]`.

---

## Figures considered and NOT generated (with reason)

Per the master prompt's explicit instruction not to generate every figure on the menu automatically:

- **Per-domain confusion matrices** — raw confusion counts (`tn/fp/fn/tp`) exist per fold-run in `MASTER_RESULTS.csv`, but a publication-quality multi-panel confusion-matrix figure was not built this pass; listed in `MISSING_EXPERIMENTS.md` as a low-cost Tier 2 item (data already exists, only the figure is missing).
- **Performance-vs-domain-distance (exploratory) scatter** — would require a per-subject or per-window domain-distance metric joined against per-subject accuracy; the domain-shift statistics computed here (Section 6.1) are aggregate device-level, not per-subject, so this figure is not computable from current outputs without new analysis (`MISSING_EXPERIMENTS.md`, Tier 2).
- **UMAP embedding view** — PCA was used instead (Figure 5); UMAP was not run. Not additive over the PCA silhouette finding for this artifact's purposes, and adding a second embedding method without a clear incremental question risks the "figure menu inflation" the master prompt explicitly warns against.
- **Full 5×5 transfer matrix with every representation** — Figure 4 shows only `full_fusion`; a full per-representation transfer-matrix grid was judged to add visual clutter without a new finding beyond Figures 6/7, which already summarize the same underlying data per-representation.
