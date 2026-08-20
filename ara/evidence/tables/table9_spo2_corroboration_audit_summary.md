# Table 9: SpO2-corroboration audit summary, by event type

**Source**: results/audit/spo2_corroboration_per_event.csv (13,455 total annotated events), results/audit/spo2_corroboration_per_subject.csv

**Caption**: Fraction of PSG-annotated osa/hypo events with a corroborating SpO2 desaturation within 45s lag tolerance. Per-subject corroboration rate ranges 8.4%-99.3% (results/audit/spo2_corroboration_per_subject.csv). 1,317 uncorroborated-desaturation candidate rows also exist in results/audit/uncorroborated_desat_candidates.csv (desaturations with no matching annotation -- the mirror-image question, not part of this table).

**Extraction type**: raw_table

| event_type | n_events | corroboration_rate |
|---|---|---|
| hypo | 8916 | 0.7446 |
| osa | 4539 | 0.9696 |
| ALL | 13455 | 0.8205 |

Per-subject range: min pct_corroborated = 8.4, max pct_corroborated = 8.4 to 99.3 across 41 subjects with SpO2+annotation coverage.
