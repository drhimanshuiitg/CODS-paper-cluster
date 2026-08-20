# Algorithm

## 1. Subject-disjoint fold splitting

For fold `f in {0..4}`, protocol `p` with `(train_devices, test_devices) = protocol_devices(p)`:

```
test_subjects  = { s : folds[s] = f }
val_subjects   = { s : folds[s] = (f+1) mod 5 }
train_subjects = all_subjects - test_subjects - val_subjects

train = { window w : subject(w) in train_subjects, device(w) in train_devices }
val   = { window w : subject(w) in val_subjects,   device(w) in train_devices }
test  = { window w : subject(w) in test_subjects,  device(w) in test_devices }

assert train_subjects ∩ val_subjects = ∅
assert train_subjects ∩ test_subjects = ∅
assert val_subjects ∩ test_subjects = ∅
assert |train|, |val|, |test| > 0 and each spans both classes
```

Complexity: O(|windows|) per fold-run (single pass with set-membership checks).

## 2. Target-aware PCA refit (the novel fix, C05)

Given source-device train features `X_tr`, source-device val features `X_val`, and (for cross-device protocols only) unlabeled target-device val features `X_tgt`:

```
# tuning stage (unchanged by the fix -- no leakage risk, only picks hyperparameters)
W_tune = PCA_fit(X_tr)
z_tr, z_val = W_tune(X_tr), W_tune(X_val)
best_hparams = select_by_validation_score(z_tr, y_tr, z_val, y_val)

# refit stage (the fix: target_val_idx now included)
target_devices = test_devices - train_devices          # empty for matched-device protocols
X_fit = X_tr ∪ X_val ∪ X_tgt   if target_devices ≠ ∅ else   X_tr ∪ X_val
W_refit = PCA_fit(X_fit)                                 # <-- previously X_tr ∪ X_val only
z_fit, z_test = W_refit(X_fit), W_refit(X_test)
classifier = fit(z_fit[:, :dim], y_fit)
predictions = classifier.predict(z_test[:, :dim])
```

Complexity: dominated by the PCA fit's SVD, `O(min(n,d)^2 * max(n,d))` for `X_fit` of shape `(n,d)`; the fix adds `|X_tgt|` rows to `n` but does not change the asymptotic form.

Mirrors CORAL's own scope exactly:

```
# CORAL (unchanged, already correctly scoped before this session's PCA fix)
transform = CORAL_fit(X_tr ∪ X_val, X_tgt)   # source+val vs. unlabeled target-val
X_fit_aligned = transform(X_tr ∪ X_val)
X_test_aligned = transform(X_test)            # test only ever *transformed*, never *fit*
```

## 3. Paired subject-level bootstrap significance

Given two prediction arms A, B aligned on `(subject_id, logical_window_id, label)`:

```
for subject in test_subjects:
    s_A[subject] = metric(labels[subject], probs_A[subject])   # e.g. balanced_accuracy
    s_B[subject] = metric(labels[subject], probs_B[subject])
diff[subject] = s_A[subject] - s_B[subject]

for i in 1..2000:
    resample = draw_with_replacement(test_subjects)
    bootstrap_mean[i] = mean(diff[s] for s in resample)

ci_low, ci_high = percentile(bootstrap_mean, 2.5), percentile(bootstrap_mean, 97.5)
p = 2 * min( P(bootstrap_mean <= 0), P(bootstrap_mean >= 0) )
significant = (ci_low > 0) or (ci_high < 0)
```

Complexity: `O(iterations * |test_subjects|)` per metric per comparison, dominated by the resampling loop; negligible relative to any classifier fit.

## 4. SpO2 desaturation detection (used by ODI/Hypoxic-Burden, E08, and the sliding-window ground truth, E09)

```
baseline[i] = max(value[j] for j in window of last 100s ending at time[i])
below_threshold[i] = (baseline[i] - value[i] >= 3.0) and not awake[i]
for each maximal run of consecutive below_threshold=True samples with duration >= 8s:
    area = trapezoid_integral(baseline - value, over the run)   # used by hypoxic burden
    event = (start_time, end_time, area)                        # used by ODI (count) and sliding-window binning
```

Complexity: O(n) with a sliding deque for the rolling-max baseline, single pass.

## 5. Sliding-window epoch binning (E09 ground truth)

```
epoch_length = 300s (5 minutes), fixed clock grid from 0 to night_end
for each desaturation event (start, end, area):
    epoch_index = min(floor(start / epoch_length), n_epochs - 1)
    epoch_desat_count[epoch_index] += 1
    epoch_area[epoch_index] += area

for each epoch:
    hourly_scale = 3600 / epoch_duration
    ahi_proxy = epoch_desat_count * hourly_scale
    severity_bin = normal if ahi_proxy<5, mild if <15, moderate if <30, else severe
```

Complexity: O(n_events + n_epochs).
