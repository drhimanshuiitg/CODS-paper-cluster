# Experiment Plan

Reframes `CLAUDE_CODE_MASTER_PROMPT.md` Sections B–N against the real Stage-0 audit findings (`AUDIT_REPORT.md`, `DATA_INTEGRITY_REPORT.md`). This is the plan to be executed in Stages 2–13; nothing below has been run yet except where explicitly marked.

## Scientific goal (unchanged from the master prompt)

Test whether simultaneous Recorder/Smartphone recordings of the same physiological event can train a representation that preserves respiratory physiology while suppressing acquisition-device shortcuts, and whether that improves bidirectional cross-device generalization at the event level and, if the cohort supports it, at the night level. Not a headline-number exercise; negative results are preserved (Section E's "Required negative-result behavior").

## Task 1 — PSG-annotated sleep respiratory event classification (corrected)

- **Primary target:** respiratory event vs. non-event (binary), event-window-centered.
- **Secondary subtype target:** `{osa, hypo}` **only** — corrected from the master prompt's literal OA/CA/MA/Hypopnea framing per Data Integrity Finding 1. Stated explicitly in every subtype table/figure: *"Apnea subtype (obstructive/central/mixed) is not annotated in this dataset; only the osa/hypo distinction provided by the source PSG scoring is evaluated."*
- **Data:** `metadata/dataset_manifest_aligned.csv` (39,596 windows, 41 subjects), reused as-is — already leakage-asserted, already carries the `paired_positive_id` R/S pairing structure needed for Section C6.

## Task 2 — Night-level screening (corrected thresholds)

- **Primary:** `AHI ≥ 30` (severe OSA) — promoted from secondary per Data Integrity Finding 2 (25 positive / 16 negative of 41 usable subjects — the smallest class-imbalance option this cohort actually supports).
- **Secondary, explicitly caveated:** `AHI ≥ 15` (moderate-to-severe) — 34/7 split; every reported result for this target carries the caveat "only 7 negative-class subjects in the full usable cohort; fold-level negative counts may be 0–2."
- **Also retained:** continuous AHI regression (threshold-independent, sidesteps the class-count problem entirely); ODI and `oximetry_desaturation_burden` regression (renamed per Data Integrity Finding re: Section D3 terminology audit — see below).
- **Input:** full-night Recorder or Smartphone audio, fixed windows, no PSG event timestamps at inference. Window length (30 s vs. 60 s) chosen on validation subjects only, per Section B.

## Section D3 — Hypoxic-burden terminology audit (to be completed in Stage 2/3)

`metadata/odi_hypoxic_burden.csv`'s `hypoxic_burden` column formula has not yet been re-derived and checked against a recognized event-associated whole-night hypoxic-burden definition in this session. **Action required before this value is used in any new PairPhysNet target**: inspect the exact computation in `scripts/compute_odi_hypoxic_burden.py`; if it does not match a recognized formulation, rename it in all new code/config/results to `oximetry_desaturation_burden` (the existing archived column name in `metadata/odi_hypoxic_burden.csv` is left untouched — this is about new usage going forward, not rewriting existing trusted outputs). **NOT YET DONE — tracked as the first item of Stage 2.**

## Architecture — PairPhysNet (Section C)

Implements exactly the master prompt's C1–C6 specification:
- Shared frozen/lightly-fine-tuned SSL encoder, backbone selected from validated baseline evidence — **WavLM-large is the validation-selected best single encoder from the prior benchmark** (`ara/logic/claims.md`, review finding F05), and is the default backbone here unless Stage 5 unit tests find a reason to prefer HuBERT.
- Configurable pooling: mean / mean+std / attentive statistical — implemented as a swappable module, not hardcoded.
- Physiology-content branch `c` (256-D, L2-normalized) and device-style branch `d`, each a small projection head on top of the frozen/shared encoder output.
- Device-adversarial head via gradient reversal on `c`.
- Disentanglement loss: normalized cross-covariance (not raw dot product, per Section C5's explicit numerical-stability instruction).
- Paired NT-Xent/InfoNCE contrastive objective on `(c_R, c_S)` using the confirmed real `paired_positive_id` structure (Audit Finding, `AUDIT_REPORT.md` §4) — positives are exact same-event R/S pairs, already present in the data, not something to construct.

Sleep-QuadNet / `full_fusion` is retained **only as baseline (A0)** — not modified, not deleted, not re-used as the proposed method's own architecture, per Section C's explicit instruction.

## Ablation ladder (Section E) — A0 through A5

| Variant | Loss | Status |
|---|---|---|
| A0 | Frozen baseline (existing benchmark) | **Already exists** — reuse `results/P0_device_gap`, no re-run needed |
| A1 | CE only | To build |
| A2 | CE + paired contrastive | To build |
| A3 | CE + device adversarial | To build |
| A4 | CE + pair + adversarial | To build |
| A5 | Full PairPhysNet (+ disentanglement + SpO2 auxiliary) | To build |

All lambda weights tuned on validation subjects only, recorded in per-run config files, never on test performance.

## Pooled-device controls (Section F) — mandatory, corrected framing vs. prior work

The prior benchmark's `(R+S)→(R+S)` result (Q1_Paper_Artifact `manuscript.md` Table 5) is explicitly flagged by this master prompt as confounded by increased training-set size. The corrected 9-condition matrix (Section F) will be run for both the A0 baseline and the best-performing PairPhysNet variant:
`R→R, S→S, R→S, S→R, (R+S balanced N)→R, (R+S balanced N)→S, (R+S full)→R, (R+S full)→S, (R+S full)→(R+S) [secondary diagnostic only]`.
Device-diversity benefit is only claimed if the **equal-data** mixed-device condition beats single-device training — the full-data pooled condition is diagnostic only, not evidence of a diversity effect by itself.

## Device probe (Section H) — mandatory

Subject-disjoint GPU device classifier (Recorder vs. Smartphone) trained on each of: frozen HuBERT, WavLM, WavLM-large, Wav2Vec2, Data2Vec, HeAR (if the isolated venv still resolves — confirmed present from prior session: `/scratch/pkdas/IEEE_healthcomm_workshop/hear_extractor/`), CE-only PairPhysNet encoder, full PairPhysNet physiology-content branch `c`. This is the direct, mechanistic follow-up to the prior work's embedding-silhouette finding (`Q1_Paper_Artifact/figures/fig05_embedding_space.png`: device ~20x more separable than label in frozen HuBERT space) — now tested as an actual trained classifier with balanced accuracy/AUROC/CI, and specifically tested on whether PairPhysNet's `c` branch reduces this relative to `d` and relative to the frozen baselines.

## Event-type/error phenotyping and case atlas (Sections I, J)

Subtype heatmap: protocol × `{osa, hypo}` only (Data Integrity Finding 1). Error phenotypes 1–6 exactly as specified; phenotype 7 (false positive during snoring) built **only if** a snoring-specific annotation or reliably identifiable acoustic marker exists in this dataset — **not yet audited**, tracked as a Stage-8 open item, not assumed.

Case-atlas selection rule (median-confidence concordant / disagreement / both-wrong) will be predefined and hashed into a JSON manifest **before** any plot is generated, per Section J's explicit anti-cherry-picking instruction.

## Statistics (Section M)

H1–H6 as specified. Paired subject-level bootstrap (reusing this project's existing, validated implementation — `scripts/run_statistics.py` conventions) for all primary hypotheses. Multiple-comparison control (Holm or Benjamini-Hochberg) applied to secondary/exploratory families — this directly closes the uncorrected-multiplicity gap flagged as the top Tier-1 item in the prior `Q1_Paper_Artifact/REVIEWER_AUDIT.md`.

## Stage progress ledger (updated live as stages complete)

| Stage | Status | Key output |
|---|---|---|
| 0 — Audit | **Complete** | `AUDIT_REPORT.md`, `DATA_INTEGRITY_REPORT.md`, `audit/raw_audio_inventory.csv` (145 files), `audit/pair_inventory.csv` (10,325 pairs) |
| 1 — Reproduce trusted baselines | **Not needed** | A0 already exists (`results/P0_device_gap`), reused as-is |
| 2 — Paired dataloader + alignment validation | **Complete** | `scripts/validate_pairing_and_folds.py` (all 5 checks pass — see `audit/pairing_fold_validation_report.json`), `scripts/paired_dataset.py` (real end-to-end smoke test passed, real audio loaded for 3 sample pairs) |
| 3 — Paired device-shift analysis (Section G) | **Complete** | `results/physiology/paired_device_shift_measures.csv` (450 real exact-same-event R/S pairs), `results/physiology/paired_device_shift_summary.json` — **subject-level Wilcoxon signed-rank test (n=40 subjects)** significant for all 6 acoustic statistics (p≤3.9e-7), a stronger, non-pseudo-replicated version of the earlier window-level-only domain-shift finding |
| D2 prerequisite — SpO2 event-timing audit | **Complete** | `results/physiology/spo2_event_timing_audit.json`: median event-to-nadir delay 41s (hypo) / 53s (osa), 150s search window covers >99% of events (justifies the auxiliary-target window empirically, not arbitrarily) |
| 4 — Device probe (Section H) | **Complete for the 7 frozen encoders** | `results/device_probe/` (35/35 jobs complete, 0 errors). **Every representation's device identity is recoverable with balanced accuracy 0.93–0.995 and AUROC 0.98–0.9996** (`results/device_probe/device_probe_summary.csv`) — including HeAR (0.983 BA), the health-acoustic-domain-specific model, which is exactly as device-leaky as general speech encoders. This is the strongest, most decisive evidence yet for the master prompt's core premise. The two remaining device-probe rows (CE-only PairPhysNet encoder, full PairPhysNet `c` branch) require Stage 6 models to exist first — not yet run. |
| 5 — Implement PairPhysNet + unit tests | **Complete** | `models/pairphysnet.py` (SharedEncoder, configurable Pooling, ProjectionHead, GRL, disentanglement/NT-Xent losses); `scripts/test_pairphysnet_forward.py` — real forward+backward on real paired audio, all outputs finite, gradients confirmed reaching every intended component (job 1654, PASSED) |
| 6 — Run A1–A5 model variants | **Script written, one bounded training smoke test in progress** | `scripts/run_pairphysnet_training.py`. First smoke test (job 1655) hit a real CUDA OOM at batch_size=8 on a 24GB MIG slice — fixed via gradient checkpointing + reduced batch/grad-accumulation + mixed precision (see `GPU_EXECUTION_PLAN.md`); re-submitted as job 1656, outcome pending. **The full 25-job (5 variants × 5 folds) matrix is prepared (`scripts/run_pairphysnet_training_single.sbatch`) but NOT submitted** — each real run is ~15 epochs over the full fold, genuinely multi-hour, and is held for explicit confirmation once the smoke test confirms the OOM fix works. |
| 7 — Equal-data pooled controls | Not started | Depends on Stage 6 producing a trained model worth comparing |
| 8 — Subtype/error/case-atlas | Not started | |
| 9 — Night-level screening | Not started | |
| 10 — Statistics + multiple-comparison correction | Not started | |
| 11–13 — Results audit, figures, journal paper | Not started | Explicitly deferred — user instructed "Do not generate the manuscript yet" |

## What is explicitly NOT run this turn

Per the master prompt's own staged model (Section R) and the cluster policy's "do not auto-submit expensive jobs," **no PairPhysNet training, no device-probe training, no feature re-extraction, and no pooled-control runs have been executed yet.** Stage 0 (audit) is complete; Stage 1 (reproduce trusted baselines only where needed — not required, A0 already exists) and Stage 2 (paired-event dataloader + alignment validation, which is mostly already-confirmed-real per this audit) are the next actionable items, followed by Stage 3 (paired device-shift analysis — CPU/signal-level, no GPU training required) as the next concrete, low-cost step before any model training is submitted.
