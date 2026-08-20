# Domain-Shift Analysis — Full Write-Up

Answers the master prompt's central Section-3 question directly: *"Are these datasets sufficiently related to represent the same task while being sufficiently different to constitute meaningful domain/device shift?"*

## Are they the same task?

Yes, unambiguously. Both devices record the same subject, same night, same room, and are aligned to the identical PSG-scored event timeline via a fitted clock-drift correction (Section 3.2 of `manuscript.md`). This is not a different population, different clinical protocol, or different annotation source — it is the same physiological events, captured twice, by different hardware. This is the structural precondition that makes device-shift (rather than confounded population/session shift) the correct causal interpretation of any observed accuracy gap.

## Are they meaningfully different acoustic domains?

Yes, decisively, at two independent levels of analysis.

### Signal level (Table 2, Figure 3)
Six standard acoustic statistics (RMS energy, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, spectral flatness), measured directly from raw WAV files on a stratified sample of 600 windows per device (balanced across the positive/negative clinical label so the comparison is not confounded by class distribution). Every statistic differs with p < 1e-30 (two-sided Mann-Whitney U); five of six show |Cliff's delta| > 0.95, out of a maximum possible magnitude of 1.0 — near-complete distributional separation. Spectral flatness shows |δ| = 1.00 exactly (complete separation in the sampled data). Only RMS energy shows a comparatively "smaller" effect (|δ| = 0.54) — still large by conventional effect-size standards (>0.474 is "large" under Romano et al.'s Cliff's delta guidelines), but the clear outlier among the six, plausibly reflecting that both devices still capture the same room's overall loudness envelope even as their spectral characteristics diverge sharply.

**Caveat**: this is a window-level test (n=600/device), not subject-level. It characterizes the aggregate raw-signal distribution, not a subject-level effect with the same independence guarantee as the classifier-performance significance tests elsewhere in this artifact (Section 5.3 of `manuscript.md`). Given the near-total separation observed (p<1e-90 for 4/6 metrics), this caveat is unlikely to change the qualitative conclusion, but it is disclosed rather than glossed over.

### Representation level (Figure 5)
A 10-component PCA on cached HuBERT embeddings (n=2,400 stratified windows) shows device identity is far more separable than the clinical label in this frozen encoder's own feature space: silhouette score 0.145 by device grouping vs. 0.007 by class-label grouping — roughly a 20x difference. This is a different, complementary line of evidence to the signal-level statistics: it shows the domain shift is not merely present in the raw audio (which a sufficiently invariant encoder could in principle discard) but is actively preserved, and in fact dominant, in the representation actually used for downstream classification. This is the paper's most direct mechanistic evidence for *why* cross-device transfer fails (Section 6.1 of `manuscript.md`), and — to the authors' knowledge — has not been shown for this specific task in prior work.

## Conclusion

The two device recordings are the same task (same events, same subjects, same session) but constitute a genuine, large, and representation-preserved domain shift. This satisfies both halves of the master prompt's framing question, and licenses this paper's core empirical claims (Sections 6.1–6.3 of `manuscript.md`) as claims about device-level domain shift specifically — not population shift, not annotation-protocol shift, not environment shift.

## What this analysis does NOT establish

- It does not identify *which specific acoustic property* (frequency response vs. gain vs. self-noise vs. placement) is most responsible for the downstream accuracy gap — the six statistics are correlated with each other and with device identity as a bundle; no causal decomposition was attempted. **NOT EVALUATED.**
- It does not establish that the same shift magnitude would be observed with a different device pair — this is a property of this specific recorder/smartphone pair, not a general "device shift is always this large" claim (Limitations, Section 8).
