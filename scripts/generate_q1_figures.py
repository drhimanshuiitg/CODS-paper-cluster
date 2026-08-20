#!/usr/bin/env python3
"""Q1_Paper_Artifact: generate all publication figures from real, already-
computed data (no fabrication, no retraining). Every figure's data source is
named in its own caption text file alongside the PNG."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "Q1_Paper_Artifact" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 100, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333333",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
})
BLUE, ORANGE, GREEN, RED, GREY = "#2a6fb0", "#d2691e", "#2e8b57", "#b22222", "#888888"

captions = {}


def save(fig, name, caption):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    captions[name] = caption
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Fig 1: problem-setting diagram (CONCEPTUAL / ILLUSTRATIVE, not measured data)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
boxes = [
    (0.3, "Same subject,\nsame night,\nsame room", "#f0f0f0"),
    (2.6, "Recorder (R)\n+ Smartphone (S)\nconcurrent audio", "#f0f0f0"),
    (5.1, "Distinct acoustic\ndomains\n(measured, Fig. 3)", "#fde8d8"),
    (7.6, "Model trained on\none device degrades\non the other", "#fbe0e0"),
]
for x, text, color in boxes:
    fb = FancyBboxPatch((x, 1.0), 2.0, 1.2, boxstyle="round,pad=0.08", linewidth=1.1,
                          edgecolor="#444444", facecolor=color)
    ax.add_patch(fb)
    ax.text(x + 1.0, 1.6, text, ha="center", va="center", fontsize=9.3)
for x0 in (2.3, 4.8, 7.3):
    ax.annotate("", xy=(x0 + 0.3, 1.6), xytext=(x0, 1.6),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3))
save(fig, "fig01_problem_setting",
     "Figure 1 (conceptual/illustrative -- not measured data). The study's motivating causal chain: "
     "identical physiological events recorded concurrently by two devices in the same acoustic environment "
     "nonetheless occupy measurably distinct acoustic domains (Figure 3), and a classifier trained on one "
     "device's recordings degrades when evaluated on the other (Figure 4, Table 2).")

# ---------------------------------------------------------------------------
# Fig 2: system architecture (AS IMPLEMENTED -- real pipeline, not conceptual art)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
stages = [
    (9.6, "Raw dual-device audio (R, S) + PSG annotations\n(record_start-relative, device-clock-aligned)", "#eeeeee"),
    (8.1, "Manifest construction\n(event-centered windows: positives + matched negatives)", "#eeeeee"),
    (6.6, "Feature extraction\n14 representations: classical DSP | 5 frozen SSL encoders | HeAR | ODI/HB | 8 fusions", "#dbe8f5"),
    (5.1, "Subject-disjoint 5-fold split\n(train / val / test subject sets, device protocol applied)", "#dbe8f5"),
    (3.6, "Classifier fit\nSVM-RBF | Random Forest | XGBoost | MLP (validation-selected hyperparameters)", "#e6f2e6"),
    (2.1, "Evaluation: 5 protocols\nR→R, S→S, R→S, S→R, (R+S)→(R+S) pooled", "#fbe8d8"),
    (0.6, "Paired subject-level bootstrap significance testing\n(2,000 resamples, every comparative claim)", "#f5dede"),
]
for y, text, color in stages:
    fb = FancyBboxPatch((1.0, y), 8.0, 1.15, boxstyle="round,pad=0.08", linewidth=1.1,
                          edgecolor="#444444", facecolor=color)
    ax.add_patch(fb)
    ax.text(5.0, y + 0.575, text, ha="center", va="center", fontsize=9.2)
for y0 in (8.1, 6.6, 5.1, 3.6, 2.1, 0.6):
    ax.annotate("", xy=(5.0, y0 + 1.15 + 0.02), xytext=(5.0, y0 + 1.15 + 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3))
save(fig, "fig02_architecture",
     "Figure 2 (as implemented). The evaluation pipeline's real component stages, reconstructed directly "
     "from the codebase (src/sleep_quadnet/, scripts/), not an idealized or aspirational architecture. "
     "PCA/CORAL domain-adaptation and the additional-signal branches (ODI/HB, SpO2-corroboration filtering, "
     "sliding-window severity) attach after feature extraction and are omitted here for legibility; see "
     "Section 3 for their exact insertion points.")

# ---------------------------------------------------------------------------
# Fig 3: domain-shift audio statistics (REAL DATA)
# ---------------------------------------------------------------------------
shift = pd.read_csv(PROJECT_ROOT / "Q1_Paper_Artifact" / "tables" / "domain_shift_audio_stats.csv")
shift_summary = json.loads((PROJECT_ROOT / "Q1_Paper_Artifact" / "analysis" / "domain_shift_summary.json").read_text())
metrics = ["rms", "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "zero_crossing_rate", "spectral_flatness"]
titles = ["RMS energy", "Spectral centroid (Hz)", "Spectral bandwidth (Hz)", "Spectral rolloff (Hz)", "Zero-crossing rate", "Spectral flatness"]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
for ax, m, title in zip(axes.flat, metrics, titles):
    r_vals = shift[shift.device == "R"][m]
    s_vals = shift[shift.device == "S"][m]
    parts = ax.violinplot([r_vals, s_vals], positions=[0, 1], showmeans=True, showextrema=False)
    for pc, c in zip(parts["bodies"], [BLUE, ORANGE]):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    parts["cmeans"].set_color("#222222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Recorder", "Smartphone"])
    d = shift_summary["metrics"][m]["cliffs_delta"]
    p = shift_summary["metrics"][m]["p_value"]
    p_str = "p < 1e-90" if p < 1e-90 else f"p = {p:.2g}"
    ax.set_title(f"{title}\n|Cliff's $\\delta$| = {abs(d):.2f}, {p_str}", fontsize=9.5)
fig.suptitle("Figure 3. Window-level acoustic domain shift, Recorder vs. Smartphone (n=600 stratified windows/device)", y=1.02, fontsize=11.5)
fig.tight_layout()
save(fig, "fig03_domain_shift",
     "Figure 3 (measured). Distribution of six window-level acoustic statistics by device, computed directly "
     "from raw audio (librosa) on a stratified sample of 600 windows per device (balanced across the positive/"
     "negative label). Two-sided Mann-Whitney U tests (window-level, not subject-level -- see caveat in "
     "Section 5.3 / analysis/domain_shift_analysis.md) reject equal distributions for all six statistics; "
     "Cliff's delta (rank-biserial effect size, range [-1,1]) exceeds 0.95 in magnitude for five of six, "
     "indicating near-complete distributional separation between devices, not a subtle shift. "
     f"Source: scripts/analyze_domain_shift.py, Q1_Paper_Artifact/tables/domain_shift_audio_stats.csv, n_recorder={shift_summary['n_recorder']}, n_smartphone={shift_summary['n_smartphone']}.")

# ---------------------------------------------------------------------------
# Fig 4: cross-domain transfer matrix (REAL DATA -- with honest N/A cells)
# ---------------------------------------------------------------------------
main = pd.read_csv("/tmp/main_benchmark_full.csv")
main["regime"] = main["protocol"].map({"R_R": "matched", "S_S": "matched", "R_S": "cross", "S_R": "cross"})
rs = pd.read_csv("/tmp/rs_rs_full.csv")
rep_for_matrix = "full_fusion"
def ba(protocol):
    return main[(main.representation == rep_for_matrix) & (main.protocol == protocol)]["balanced_accuracy"].mean()
pooled_ba = rs[rs.representation == rep_for_matrix]["balanced_accuracy"].mean()
matrix = np.full((3, 3), np.nan)
row_labels = ["Train: R", "Train: S", "Train: R+S (pooled)"]
col_labels = ["Test: R", "Test: S", "Test: R+S (pooled)"]
matrix[0, 0] = ba("R_R"); matrix[0, 1] = ba("R_S")
matrix[1, 0] = ba("S_R"); matrix[1, 1] = ba("S_S")
matrix[2, 2] = pooled_ba
fig, ax = plt.subplots(figsize=(5.6, 5.0))
masked = np.ma.masked_invalid(matrix)
im = ax.imshow(masked, cmap="RdYlGn", vmin=0.45, vmax=0.65)
ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=9.5)
ax.set_yticks(range(3)); ax.set_yticklabels(row_labels, fontsize=9.5)
for i in range(3):
    for j in range(3):
        if not np.isnan(matrix[i, j]):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
        else:
            ax.text(j, i, "N/A\n(not evaluated)", ha="center", va="center", fontsize=7.5, color="#999999", style="italic")
ax.set_title(f"Figure 4. Train→test balanced-accuracy transfer matrix\n(representation: {rep_for_matrix})", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Balanced accuracy")
fig.tight_layout()
save(fig, "fig04_transfer_matrix",
     f"Figure 4 (measured). Balanced-accuracy transfer matrix for {rep_for_matrix} across the 5 evaluated "
     "device protocols. Diagonal cells (R→R, S→S) are matched-device; off-diagonal (R→S, S→R) "
     "are bidirectional cross-device zero-shot transfer; the pooled cell trains and tests on both devices "
     "combined (protocol RS→RS). Cells for pooled-train/single-device-test are marked N/A: this project's "
     "RS→RS protocol evaluates pooled-vs-pooled only, not pooled-train against a held-out single-device "
     "test set, so those cells are honestly not evaluated rather than estimated. "
     "Source: results/P0_device_gap, results/... RS_RS rows (see Table 2).")

# ---------------------------------------------------------------------------
# Fig 5: embedding-space PCA, colored by device and by class (REAL DATA)
# ---------------------------------------------------------------------------
emb = pd.read_csv(PROJECT_ROOT / "Q1_Paper_Artifact" / "tables" / "embedding_pca_coords.csv")
emb_summary = json.loads((PROJECT_ROOT / "Q1_Paper_Artifact" / "analysis" / "embedding_pca_summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for d, c, lab in [("R", BLUE, "Recorder"), ("S", ORANGE, "Smartphone")]:
    sub = emb[emb.device == d]
    axes[0].scatter(sub.pc1, sub.pc2, s=5, alpha=0.35, color=c, label=lab, edgecolors="none")
axes[0].set_title(f"Colored by device\nsilhouette = {emb_summary['silhouette_by_device_10pc']:.3f}", fontsize=10.5)
axes[0].legend(markerscale=3, fontsize=9, frameon=False)
for l, c, lab in [(0, GREY, "Negative"), (1, RED, "Apnea event (positive)")]:
    sub = emb[emb.label == l]
    axes[1].scatter(sub.pc1, sub.pc2, s=5, alpha=0.35, color=c, label=lab, edgecolors="none")
axes[1].set_title(f"Colored by class label\nsilhouette = {emb_summary['silhouette_by_class_10pc']:.3f}", fontsize=10.5)
axes[1].legend(markerscale=3, fontsize=9, frameon=False)
for ax in axes:
    ax.set_xlabel(f"PC1 ({emb_summary['explained_variance_ratio_pc1_pc2'][0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({emb_summary['explained_variance_ratio_pc1_pc2'][1]*100:.1f}% var.)")
fig.suptitle("Figure 5. PCA projection of frozen HuBERT embeddings (n=2,400 stratified windows)", y=1.03, fontsize=11.5)
fig.tight_layout()
save(fig, "fig05_embedding_space",
     "Figure 5 (measured). Same 10-component PCA fit on cached HuBERT embeddings (first two components shown), "
     "colored once by recording device (left) and once by clinical label (right). Silhouette scores computed "
     "in the full 10-component space. Device identity separates roughly 20x more cleanly than the clinical "
     "label in this frozen representation, offering a mechanistic, representation-level explanation for the "
     "cross-device performance gap in Figure 4/Table 1: the encoder's feature space encodes acquisition device "
     "far more saliently than task-relevant signal. Source: scripts/analyze_embedding_space.py, "
     "Q1_Paper_Artifact/tables/embedding_pca_coords.csv.")

# ---------------------------------------------------------------------------
# Fig 6: generalization gap by representation (REAL DATA)
# ---------------------------------------------------------------------------
gap = main.groupby(["representation", "regime"])["balanced_accuracy"].mean().unstack()
gap["gap"] = gap["matched"] - gap["cross"]
gap = gap.sort_values("gap", ascending=True)
gap = gap[gap.index != "odi_hb"]
fig, ax = plt.subplots(figsize=(7.5, 6))
colors = [RED if v == gap["gap"].max() else BLUE for v in gap["gap"]]
ax.barh(gap.index, gap["gap"], color=colors, edgecolor="#333333", linewidth=0.6)
ax.set_xlabel("Generalization gap (matched BA − cross-device BA)")
ax.set_title("Figure 6. Matched-to-cross-device generalization gap, all representations\n(odi_hb excluded: gap = 0 by construction, a degenerate baseline)", fontsize=10.5)
fig.tight_layout()
save(fig, "fig06_generalization_gap",
     "Figure 6 (measured). Generalization gap (mean matched-device balanced accuracy minus mean cross-device "
     "balanced accuracy) per representation, averaged over 4 classifiers, R_R/S_S (matched) and R_S/S_R "
     "(cross) protocols, 5 folds. Every real representation shows a positive gap; confirmed statistically "
     "significant for full_fusion specifically via paired bootstrap (R→R vs R→S: p<0.001; S→S vs "
     "S→R: p<0.001; see results/P0_statistics_v2). Source: results/P0_device_gap.")

# ---------------------------------------------------------------------------
# Fig 7: representation comparison, cross-device (REAL DATA)
# ---------------------------------------------------------------------------
cross_ba = main[main.regime == "cross"].groupby("representation")["balanced_accuracy"].mean().sort_values(ascending=True)
cross_ba = cross_ba[cross_ba.index != "odi_hb"]
fig, ax = plt.subplots(figsize=(7.5, 6))
colors = ["#c9c9c9" if r not in ("hubert", "wavlm_large", "full_fusion") else (GREEN if r in ("hubert", "wavlm_large") else RED) for r in cross_ba.index]
ax.barh(cross_ba.index, cross_ba.values, color=colors, edgecolor="#333333", linewidth=0.6)
ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1, label="Chance (0.5)")
ax.set_xlabel("Mean cross-device balanced accuracy")
ax.set_title("Figure 7. Cross-device balanced accuracy, all representations\n(green: single encoders selected in Sec. 4.2; red: largest fusion)", fontsize=10.5)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
save(fig, "fig07_representation_comparison",
     "Figure 7 (measured). Mean cross-device balanced accuracy (R→S and S→R averaged), all 13 real "
     "representations (odi_hb excluded, chance-level by construction). No representation is confirmed "
     "significantly better than the others under this project's validation-selected significance testing "
     "(Section 4.2) despite the visible point-estimate spread shown here -- see results/P0_statistics_v2 for "
     "the paired bootstrap that adjudicates this. Source: results/P0_device_gap.")

# ---------------------------------------------------------------------------
# Fig 8: leave-one-encoder-out ablation (REAL DATA)
# ---------------------------------------------------------------------------
abl = pd.read_csv("/tmp/ablation_full.csv")
abl["regime"] = abl["protocol"].map({"R_R": "matched", "S_S": "matched", "R_S": "cross", "S_R": "cross"})
abl_cross = abl[abl.regime == "cross"].groupby("representation")["balanced_accuracy"].mean()
full_fusion_cross = main[(main.representation == "full_fusion") & (main.regime == "cross")]["balanced_accuracy"].mean()
abl_sig = pd.read_csv(PROJECT_ROOT / "results" / "P0_ablation_statistics" / "ablation_vs_full_fusion.csv")
abl_sig_ba = abl_sig[abl_sig.metric == "balanced_accuracy"]
sig_reps = set(abl_sig_ba[(abl_sig_ba.ci95_low > 0) | (abl_sig_ba.ci95_high < 0)]["representation"])
fig, ax = plt.subplots(figsize=(7.5, 4.2))
names = list(abl_cross.index) + ["full_fusion\n(reference)"]
vals = list(abl_cross.values) + [full_fusion_cross]
colors = [(ORANGE if r in sig_reps else GREY) for r in abl_cross.index] + ["#222222"]
ax.barh(names, vals, color=colors, edgecolor="#333333", linewidth=0.6)
ax.axvline(full_fusion_cross, color="#222222", linestyle="--", linewidth=1)
ax.set_xlabel("Mean cross-device balanced accuracy")
ax.set_title("Figure 8. Leave-one-encoder-out ablation vs. full_fusion\n(orange: individually significant vs. full_fusion in ≥1 tested combination; grey: not significant)", fontsize=10)
fig.tight_layout()
save(fig, "fig08_ablation",
     "Figure 8 (measured). Cross-device balanced accuracy for each leave-one-encoder-out fusion variant, "
     "point estimates. Paired bootstrap testing (89 evaluable representation/classifier/protocol combinations) "
     "finds only 7/89 individually significant (4 positive toward the ablated variant, 3 negative); mean "
     "difference across all combinations is -0.0003 -- full_fusion and its ablated variants are statistically "
     "indistinguishable in the large majority of tested configurations, despite visible point-estimate spread "
     "in this figure. Source: results/P0_ablation, results/P0_ablation_statistics/ablation_vs_full_fusion.csv.")

# ---------------------------------------------------------------------------
(FIG_DIR / "captions.json").write_text(json.dumps(captions, indent=2))
print(f"\n{len(captions)} figures written to {FIG_DIR}")
