---
title: "Sleep-QuadNet: Cross-Device Robustness Benchmark for Self-Supervised Audio Representations in Sleep Apnea Screening"
authors: ["pkdas (project owner)"]
year: 2026
venue: "IEEE HealthCom 2026 (resubmission target)"
doi: "not assigned"
ara_version: "1.0"
domain: "clinical audio ML / cross-device domain robustness / sleep medicine"
keywords: ["sleep apnea", "obstructive sleep apnea", "self-supervised audio representations", "cross-device generalization", "domain adaptation", "PCA", "CORAL", "HuBERT", "HeAR", "subject-disjoint evaluation"]
claims_summary:
  - "No representation significantly beats any other among the top single encoders and full multi-encoder fusion (revised post-review, F05): a rigorous, validation-selected re-analysis found full_fusion statistically indistinguishable from the best single encoder, which is itself wavlm_large, not HuBERT as an earlier, less careful point-estimate-only comparison had suggested."
  - "PCA dimensionality reduction's apparent domain-adaptation failure was a correctable implementation-scope bug (source-device-only fit); scoping it like CORAL fixes the collapse and yields a significant cross-device improvement."
  - "CORAL feature-space alignment, SpO2-corroboration training-label filtering, and a health-acoustic foundation model (HeAR) were all tested as mitigations for the cross-device gap and did not help — reported as honest negative results, not omitted. (HeAR's fusion-level effect specifically was not significant once tested directly, post-review — closer to 'no confirmed benefit' than 'confirmed harm.')"
abstract: "This project benchmarks self-supervised audio representations for bidirectional cross-device (clinical recorder <-> smartphone) sleep apnea screening on N=41-50 dual/single-device subjects, under strict subject-disjoint 5-fold cross-validation. The codebase evolved from a rejected paper submission whose headline numbers came from a leaky, non-reproducible script; this ARA covers only the current, re-verified pipeline. It reconstructs 10 experiment families run against a live, still-partially-running results tree: the main 14-representation x 4-classifier x 4-protocol x 5-fold benchmark, a leave-one-encoder-out ablation, PCA and CORAL domain-adaptation attempts (including a root-caused PCA bug and its fix), an SpO2-corroboration training-label-quality ablation, a HeAR foundation-model integration, an ODI/Hypoxic-Burden clinical-feature test, a first-cut sliding-window whole-night severity classifier (in progress), and a partial deployment/efficiency benchmark. Six real, verified bugs and one real incomplete-coverage gap are captured in the exploration trace alongside the clean numbers, since they materially affect which findings are trustworthy."
---

# Sleep-QuadNet: Cross-Device Robustness Benchmark

## Overview

This is a code-and-results-derived ARA (no paper PDF exists yet in final form; `paper/conference_101719.tex` is the in-progress manuscript, itself derived from this same results tree and predates this ARA's Level 2 review revisions -- **the paper has not yet been updated to match the revised C01/C02/C08 findings below and needs a follow-up pass**). The central empirical finding, reconstructed directly from `results/P0_device_gap` (1,120 completed fold-runs) and refreshed significance tests (`results/P0_statistics_v2`, added in response to Level 2 review finding F05), is more nuanced than this ARA's first draft claimed: naive multi-encoder concatenation does not *significantly* beat the best single encoder, but nor is it *confirmed* worse -- the two are a statistical wash under the project's own rigorous, validation-selected methodology, which additionally revealed that HuBERT was never actually the validation-selected best single encoder once wavlm_large was included in the candidate pool (wavlm_large is). Of two tested post-hoc domain-adaptation corrections, one (PCA) was salvageable via a scope fix and one (CORAL) was not, though CORAL's negative finding rested on much narrower evidentiary coverage (3 representations, 1 classifier) than PCA's fix (5 representations, 4 classifiers) until a post-review extension (F04, `results/P1_domain_adaptation`, jobs 1601/1602) broadened it to match. Two additional candidate improvements (SpO2-corroboration label filtering, the HeAR foundation model) both failed to help and are reported as negative results rather than omitted -- HeAR's fusion-level effect specifically was found not significant once directly tested (post-review), a more precise finding than the original point-estimate-only framing. The project also has real in-progress and not-yet-attempted work (sliding-window severity classification, a contrastive device-invariance head, an `RS_RS` pooled-device protocol) which this ARA marks explicitly as unfinished so it is never conflated with the settled findings.

## Layer Index

### Cognitive Layer (`/logic`)
| File | Description |
|------|--------------|
| [problem.md](logic/problem.md) | Observations (O1-O9) -> gaps (G1-G5) -> key insight -> assumptions |
| [claims.md](logic/claims.md) | 11 falsifiable claims (C01-C11), 9 supported, 1 refuted-as-hypothesized, 1 open |
| [concepts.md](logic/concepts.md) | 8 formal concept definitions |
| [experiments.md](logic/experiments.md) | 10 experiment plans (E01-E10) |
| [solution/architecture.md](logic/solution/architecture.md) | Pipeline component graph |
| [solution/algorithm.md](logic/solution/algorithm.md) | Fold-splitting, PCA-refit-fix, CORAL, significance-testing math |
| [solution/constraints.md](logic/solution/constraints.md) | Boundary conditions and known limitations |
| [solution/heuristics.md](logic/solution/heuristics.md) | 8 implementation heuristics (H01-H08) |
| [related_work.md](logic/related_work.md) | Cited baselines and their relationship to this work |

### Physical Layer (`/src`)
| File | Description | Claims |
|------|--------------|--------|
| [configs/training.md](src/configs/training.md) | Classifier hyperparameters, seed, split policy | C01-C11 |
| [configs/model.md](src/configs/model.md) | Encoder specs, feature dimensions, fusion definitions | C01, C02, C08 |
| [execution/pca_target_aware_refit.py](src/execution/pca_target_aware_refit.py) | The novel PCA-fix contribution, as a code stub | C05 |
| [execution/paired_bootstrap_significance.py](src/execution/paired_bootstrap_significance.py) | The subject-level paired bootstrap used for every significance claim | C01-C10 |
| [environment.md](src/environment.md) | Python/PyTorch/cuML/TensorFlow versions, hardware, seeds | all |

### Exploration Graph (`/trace`)
| File | Description |
|------|--------------|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | 33-node research DAG incl. 8 dead_end and 5 decision nodes |

### Evidence (`/evidence`)
| File | Description |
|------|--------------|
| [README.md](evidence/README.md) | Index of 9 tables + 0 figures (quantitative figures not yet generated for this results snapshot) |
