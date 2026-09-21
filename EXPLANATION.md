# Fetal ECG Extraction Under Maternal Overlap — Project Explainer

A guide to what this project is trying to do, why it is built the way it is, and
what the results so far actually show. Written for someone new to the codebase who
may also be new to fetal ECG.

---

## 1. What this project is

It extracts the **fetal heartbeat** from ECG electrodes placed on a pregnant
mother's abdomen, and it focuses on the single hardest case: the moments when a
fetal beat is hidden underneath a maternal beat.

The repository implements five methods for this, evaluates them all through one
shared harness, and reports metrics that are deliberately split so the hard case
cannot hide inside the average.

---

## 2. The clinical problem

Fetal heart rate (FHR) is the main signal clinicians use to detect fetal distress.
The routine tool is Doppler ultrasound CTG, which reports an **averaged** rate. That
averaging destroys *beat-to-beat variability* — which is precisely the component
that carries distress information.

True beat-to-beat timing today requires a **fetal scalp electrode**, which needs
ruptured membranes and an accessible fetus. In practice: late labour only.

Non-invasive fetal ECG from abdominal electrodes would give real beat-to-beat
timing, throughout pregnancy, with nothing invasive. That is the motivation.

---

## 3. Why it is hard

An abdominal electrode records the mother's heart and the fetus's heart summed
together. The maternal QRS complex is roughly **10–30× larger** than the fetal QRS.

The conventional strategy is **maternal suppression**:

1. Detect the maternal R-peaks.
2. Estimate the maternal contribution to the signal.
3. Subtract it.
4. Detect fetal peaks in whatever is left (the "residual").

This works tolerably — except in one specific situation.

---

## 4. The core research question: coincidence

The mother beats at roughly 75 bpm. The fetus beats at roughly 140 bpm. **These two
rhythms are independent**, so they drift in and out of phase, and periodically a
fetal beat lands directly underneath a maternal QRS.

This project calls those **coincident beats**, and they are hard for a structural
reason, not a tuning reason:

> Every suppression method estimates the maternal component *from the signal
> itself*. At a coincident beat, the maternal estimate has the fetal beat baked
> into it. Subtracting that estimate removes the fetal beat along with the mother's.

So the methods fail exactly where the information is hidden. Getting a good
aggregate score while silently dropping every coincident beat is easy — and
useless, because a systematically missed subset of beats corrupts variability
analysis far more than randomly scattered misses would.

---

## 5. The key evaluation decision

Because of the above, **metrics in this project are stratified by overlap**
([`fecg/evaluate.py`](fecg/evaluate.py)).

Every fetal beat in the ground truth is labelled:

- **coincident** — within ±50 ms of a maternal R-peak
- **non-coincident** — everything else

Sensitivity is then reported *separately* for each group. The gap between
`non_coincident_Se` and `coincident_Se` is the actual measurement of what overlap
costs a given method. An aggregate F1 alone would average the hard beats into the
easy ones and hide the failure entirely.

This single decision is why the codebase looks the way it does.

---

## 6. Repository map

```
fecg/
  config.py          PipelineConfig — every tunable knob in one dataclass
  io.py              Record container, PhysioNet loaders, synthetic generator
  preprocess.py      bandpass + powerline notch + Pan-Tompkins maternal QRS
  fetal.py           fetal QRS detection, FHR series, HRV metrics
  evaluate.py        peak matching + OVERLAP-STRATIFIED metrics + FHR error
  pipeline.py        the orchestrator + leave-one-record-out harness
  suppression/
    base.py          Suppressor interface — the shared contract
    template.py      adaptive template subtraction
    adaptive.py      LMS / RLS noise cancellation
    bss.py           FastICA blind source separation
    kalman.py        beat-wise Kalman template tracker
  dl/
    model.py         1D U-Net
    dataset.py       windowing + Gaussian R-peak targets
    train.py         training loop + DLDetector inference wrapper
  recovery/          the novel occlusion-aware arm (see §9)

run_experiment.py        classical method comparison table
run_recovery_experiment.py  benchmarks the recovery arm vs baselines
run_full_comparison.py   all 5 methods on one held-out real test split
plot_comparison.py       aggregate result charts
plot_diagnostics.py      signal-level and per-record diagnostic charts

testbench.py             dataset registry, model loading, scoring helpers (no UI)
app.py                   Streamlit test bench built on testbench.py (see §16)
```

---

## 7. Pipeline architecture

One orchestrator, used identically by every method
([`fecg/pipeline.py`](fecg/pipeline.py)):

```
preprocess → detect maternal QRS → SUPPRESS → detect fetal QRS → FHR / HRV → evaluate
                                      ▲
                          the only stage that varies
```

The `Suppressor` interface is the seam. Each classical method is a drop-in at that
one slot, so every arm shares the same preprocessing, the same fetal detector, and
the same evaluator. Differences in the results come from the method, not from
incidental differences in the surrounding machinery.

The deep-learning arm is the exception: it replaces the *detection* stage rather
than the suppression stage, consuming the multichannel signal directly and emitting
a fetal R-peak probability heatmap. It still flows into the identical FHR and
evaluation code.

---

## 8. The five methods and what each one bets on

| Method | Mechanism | The bet |
|---|---|---|
| **template** | Median maternal beat template, with a **per-beat least-squares gain** fitted before subtraction | A fixed template leaves a residual proportional to maternal amplitude drift; fitting gain per beat shrinks that residual |
| **adaptive** | Widrow adaptive noise cancellation (LMS / RLS) | Given a maternal **reference channel** uncorrelated with the fetus, cancellation does not depend on temporal separation at all — so coincident beats survive |
| **bss_ica** | FastICA; identifies the maternal component by lowest dominant rate, zeroes it, re-mixes | Sources can be separated on the **spatial** axis, which works even at exact temporal coincidence |
| **kalman** | Maternal template treated as a slowly time-varying state, tracked beat-to-beat with a random-walk model | **Adapting to morphology drift** predicts the maternal contribution better at the coincident beat than a fixed median does |
| **dl_unet** | 1D U-Net over all channels → per-sample fetal R-peak probability | A network can learn the spatial + temporal discriminant **directly**, with no explicit subtraction step to corrupt |

---

## 9. The novel arm: occlusion-aware recovery

`fecg/recovery/` reframes the problem rather than improving the subtraction.

The insight: a maternal QRS **occludes** the fetal beat at *known, predictable*
locations. So instead of trusting every sample equally:

1. [`occlusion.py`](fecg/recovery/occlusion.py) marks samples under each maternal
   QRS as unreliable, producing a soft per-sample reliability weight.
2. [`rhythm_prior.py`](fecg/recovery/rhythm_prior.py) exploits the fact that the
   fetal heart is a quasi-periodic oscillator *independent of the mother's*. The
   visible beats around an occlusion tightly constrain where the hidden one must
   have fallen.
3. A predicted beat is accepted **only when an occlusion actually explains the
   miss**. That discipline is what stops the method inventing beats and destroying
   precision.

Status: on synthetic data it recovered **zero** beats, because the baselines
already score ~1.00 there and leave no headroom. It remains genuinely untested.

---

## 10. Data

All three PhysioNet databases are present and verified in `data/`:

| Dataset | Records | Format | Why it matters |
|---|---|---|---|
| **ADFECGDB** | 5 | EDF | Includes a **direct fetal scalp electrode** — the only true ground truth at coincident beats |
| **CinC2013** | 75 | WFDB | Challenge set with expert fetal annotations; the main benchmark used here |
| **NIFECGDB** | 55 | EDF | Larger non-invasive cohort |

A **synthetic generator** ([`fecg/io.py`](fecg/io.py)) complements these. It forces
a controllable `overlap_frac`, which is the only way to dial overlap directly and
study it in isolation. Its limitation: with only two sources it is too easy, so all
methods score near 1.00 and the comparison carries no information. Synthetic runs
are a **functional smoke test, not a finding**.

---

## 11. Metrics reference

| Metric | Definition | Why it matters clinically |
|---|---|---|
| **TP / FP / FN** | Detections matched to truth within ±50 ms | The raw counts behind everything below |
| **Se** (sensitivity) | `TP / (TP + FN)` — fraction of real beats found | Missed beats fabricate apparent bradycardia — distress that isn't there |
| **PPV** (precision) | `TP / (TP + FP)` — fraction of detections that are real | Invented beats create false reassurance. This is the **dangerous** failure mode, because it can mask genuine distress |
| **F1** | Harmonic mean of Se and PPV | One balanced number; punishes trading one against the other |
| **coincident_Se** | Se restricted to beats overlapping a maternal QRS | **The project's whole point** — the beats suppression destroys |
| **non_coincident_Se** | Se on the easy, non-overlapping beats | The control group; the gap vs `coincident_Se` isolates the cost of overlap |
| **FHR MAE** | Mean absolute error of beat-to-beat heart rate, in bpm | The number actually shown to a clinician. ~2 bpm is usable; ~18 bpm is not |

---

## 12. Results

All five methods evaluated on the **same 15 held-out real CinC2013 records**. The
60 remaining records were used for DL training only, so the comparison is
like-for-like.

| method | F1 | Se | PPV | non-coin Se | coin Se | FHR MAE |
|---|---|---|---|---|---|---|
| **dl_unet** | **0.884** | 0.846 | **0.960** | **0.867** | **0.727** | **4.31 bpm** |
| template | 0.623 | 0.615 | 0.633 | 0.606 | 0.656 | 17.13 |
| kalman | 0.603 | 0.595 | 0.614 | 0.585 | 0.639 | 18.77 |
| bss_ica | 0.566 | 0.558 | 0.575 | 0.550 | 0.596 | 19.13 |
| adaptive | 0.075 | 0.046 | 0.273 | 0.035 | 0.105 | 36.98 |

Error composition over the test set (2167 true beats):

| method | TP | FP (invented) | FN (missed) |
|---|---|---|---|
| dl_unet | 1827 | **31** | 340 |
| template | 1325 | 742 | 842 |
| kalman | 1279 | 772 | 888 |
| bss_ica | 1202 | 872 | 965 |
| adaptive | 100 | 350 | 2067 |

**Production recommendation: `dl_unet`.** It leads on every axis, but the decisive
margins are PPV (0.960 vs 0.633) and FHR error (4.3 vs 17.1 bpm). It produces **31
false beats where template produces 742** — a ~24× reduction in the clinically
dangerous failure mode. Its per-record F1 is also far more stable (median 0.959)
than any classical arm.

**Threshold tuning is free performance.** The detector's 0.3 probability cutoff was
never tuned. Sweeping it (`fig_dl_threshold_sweep.png`) puts the F1 optimum at
**0.06**: F1 0.908 → **0.922**, with sensitivity rising 0.846 → 0.889 while
precision barely moves (0.960 → 0.957). The signal-level plot shows why — missed
coincident beats appear as sub-threshold bumps around P≈0.15, well below 0.3 but
comfortably above 0.06.

**Caveat on run-to-run variance.** The four classical arms are deterministic and
reproduce bit-identically. `dl_unet` is not: two training runs with identical
settings gave F1 0.852 vs 0.884 and FHR MAE 2.48 vs 4.31 bpm. The ranking is stable,
but single-decimal DL figures should not be over-read.

---

### Figure guide (`results/`)

| Figure | What it shows |
|---|---|
| `fig_signal_anatomy.png` | **Start here.** Raw abdominal ECG → template residual → U-Net probability, on one 6 s window, with overlap zones shaded. The whole argument in one image |
| `fig_signal_anatomy_worstcase.png` | The same view on the record where the U-Net does *worst* — the honest counterweight |
| `fig_fhr_trace.png` | The clinician-facing output: FHR over time vs truth |
| `fig_bland_altman.png` | FHR agreement with 95% limits (clinical standard) |
| `fig_dl_probability.png` | U-Net confidence trace over a longer window |
| `fig_dl_threshold_sweep.png` | Se/PPV/F1 vs detection threshold — the deployment tuning knob |
| `fig_headline_metrics.png` | F1 / Se / PPV per method |
| `fig_coincident_gap.png` | Coincident vs non-coincident Se — the headline comparison |
| `fig_error_breakdown.png` | TP / FP / FN composition; makes the FP gap unmissable |
| `fig_f1_distribution.png` | Per-record F1 spread — robustness, not just means |
| `fig_paired_records.png` | Per-record win/loss of `dl_unet` vs `template` |
| `fig_se_ppv_scatter.png` | Per-record operating points |
| `fig_overlap_vs_f1.png` | Does overlap fraction predict record difficulty? |
| `fig_fhr_error.png` | FHR MAE per method |
| `fig_training_loss.png` | U-Net training curve |

---

## 13. Four findings that matter more than the leaderboard

**`adaptive` collapsed for an architectural reason, not a tuning one.** It requires
a thoracic reference lead. CinC2013 provides only abdominal leads, so the code falls
back to synthesising a reference from the template reconstruction — meaning it
cancels using an estimate derived from the very signal it is cleaning. The bet the
method makes is simply unavailable on this dataset. It is not a fair test of RLS
cancellation; it is a demonstration that the method needs hardware this dataset
lacks.

**The classical arms' apparent "overlap advantage" is probably an artifact.** All
four classical methods score *higher* on coincident than non-coincident beats, which
is backwards — overlap should hurt. The likely explanation: they false-fire on
maternal QRS residual, and coincident fetal beats sit adjacent to a maternal QRS by
definition, so those false fires land on the right answer by coincidence. Their
enormous FP counts support this reading, and `fig_signal_anatomy.png` shows it
directly: the template residual is essentially noise, and its detections fall on a
near-regular grid unrelated to the true beats. `dl_unet` is the only arm showing the
expected direction (0.867 non-coincident vs 0.727 coincident) and therefore the only
one whose overlap number reflects genuine recovery.

**The two approaches fail in opposite ways, and one failure is much safer.** On the
hardest record (a71, `fig_signal_anatomy_worstcase.png`) both methods fail — but
template keeps emitting a near-regular grid of confident false beats, while the
U-Net's probability collapses below threshold almost everywhere and it simply
returns nothing. Clinically these are not equivalent. A monitor that outputs
*nothing* can raise a signal-loss alarm; a monitor that outputs a plausible but
fabricated FHR trace cannot be detected as wrong. The U-Net's tendency to abstain
rather than guess is a deployment asset, not just a metric artifact.

**A latent data bug was corrupting every method.** Several CinC2013 records contain
NaN sample dropouts. `scipy.signal.filtfilt` propagates a single NaN across an
entire filtered channel, silently reducing those records to all-NaN for *every*
arm — and driving DL training to `loss=nan`. Fixed at the loader level in
[`fecg/io.py`](fecg/io.py) with per-channel linear interpolation applied before any
filtering.

---

## 14. Honest limitations

- **Single split, single seed.** 15 test records, no k-fold cross-validation. The
  ranking is not yet statistically robust.
- **`dl_unet` carries deployment cost.** It is fitted to this electrode montage and
  this population, and needs annotated training data plus a torch runtime. The
  classical arms need no training and transfer freely. That trade is real.
- **ADFECGDB is still unused.** It is the only dataset with a direct fetal scalp
  electrode — the only true ground truth at coincident beats — but it is EDF-format
  and the loader path expects WFDB. This is the most valuable untapped validation
  available to the project.
- **The recovery arm is untested.** Synthetic data is too easy to exercise it.
- **The detection threshold default (0.3) is untuned.** The sweep shows 0.06 is
  better on this test set, but that figure is itself fitted to the test set — it
  needs a separate validation split before being adopted as the production default.
- **`dl_unet` is not deterministic.** Retraining moves F1 by ~0.03 and FHR MAE by
  ~2 bpm (see §12).

---

## 15. Reproducing the results

```bash
pip install -r requirements.txt

# synthetic smoke test (no data download needed)
python run_experiment.py
python tests/test_smoke.py

# classical arms on real CinC2013 data
python run_experiment.py --dataset cinc2013

# full comparison: all 5 methods, held-out real test split (~20 min, trains the U-Net)
python run_full_comparison.py --n-test 15 --epochs 20

# charts
python plot_comparison.py     # aggregate comparison
python plot_diagnostics.py    # signal-level + per-record diagnostics
```

Outputs land in `results/` — `comparison_metrics.json` plus the figures.

---

## 16. The interactive test bench (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py          # opens in the browser at localhost:8501
```

### Why it exists

The scripts in §15 answer "what is the average F1 over the test split?". They do not
answer the questions you actually ask while debugging a detector: *which* beats did it
miss, did it miss them because the model is blind to them or because the threshold is
wrong, and does this record's flattering score come from the model having memorised it
during training. Each of those needs to be looked at, one record at a time, with the
knobs live. That is what the test bench is for — it runs the *same* pipeline code as
the scripts, but lets you move the threshold and watch the metrics and the waveform
respond.

### Architecture: `testbench.py` vs `app.py`

The split is deliberate:

- **`testbench.py`** holds everything that computes a number — the dataset registry
  (`DATASETS`), record loading and channel/duration subsetting, checkpoint loading
  (input-channel count is inferred from the weights), inference, peak-picking, and
  scoring via the same `evaluate()` used everywhere else. It imports no Streamlit.
- **`app.py`** holds only the UI — widgets, plots, layout, warnings.

So every number the app shows is reproducible from a plain Python session, with no
web server involved:

```python
from fecg import PipelineConfig
from testbench import (load_record, subset_record, load_model, dl_probability,
                       peaks_from_probability, score)
model, in_ch = load_model("results/dl_unet.pth")
rec = subset_record(load_record("CinC2013 set-a", "a06"),   # a06 is held-out;
                    ["AECG1", "AECG2", "AECG3", "AECG4"], 60)  # check split_label()
cfg = PipelineConfig(fs=rec.fs)
prob = dl_probability(model, rec, cfg)
res  = score(rec, peaks_from_probability(prob, cfg, 0.30), cfg)
print(res["eval"])            # overall + coincident + non_coincident
```

A UI that owns its own maths grows a second, subtly different implementation; this
layout makes that impossible.

### Shared sidebar controls

Dataset · model checkpoint (any `results/*.pth`) · **detection threshold** (peak
height on the probability trace, default 0.30) · analysis window (first N seconds,
default 60) · which classical arms to run alongside the U-Net. Results are memoised
with `st.cache_data` / `st.cache_resource` keyed on
`(dataset, record, channels, seconds, checkpoint mtime)`, so moving the threshold
slider re-picks peaks from a cached probability trace instead of re-running the
network — the inference is the expensive part and the threshold does not change it.

### The four views

1. **Single record.** Pick a record and the model's input channels; get the
   overlap-stratified metric table, then four tiles: F1, non-coincident Se,
   **coincident Se** (with the delta against non-coincident — that delta *is* the
   overlap penalty of §4), and FHR MAE. Below it, a scrollable window over the
   preprocessed signal with ground truth, detections and maternal R-peaks drawn
   together, so a missed coincident beat is visible as an event rather than as a
   number, plus the probability trace and the FHR series. Detections export to CSV.
2. **Batch evaluation.** Runs a set of records and aggregates mean ± std of the
   stratified metrics, optionally with the classical arms alongside; exports CSV. On
   CinC2013 it defaults to the held-out test split, with "all records (includes
   training data)" as an explicit, labelled opt-in.
3. **Threshold sweep.** Se / PPV / F1 / coincident-Se against peak-picking threshold,
   sampled finely below 0.10 where the trade-off actually lives. This separates the
   two failure modes that look identical in a single F1 number: the model cannot see
   the beat, versus the operating point is wrong. It is the interactive form of next
   step 3 in §17.
4. **Saved training run.** Reads `results/comparison_metrics.json` — the summary
   table, the train/test split, the U-Net loss curve and the saved `fig_*.png`
   figures — so the §12 results are viewable without re-running anything.

### Provenance guards

The design rule stated at the top of `app.py`: *never show a fetal detection score
without also showing whether the record was in the model's training set and whether
the annotations are actually fetal.* An impressive F1 on a training record is not a
result. Three guards implement it, each covering a way this project's data can
silently inflate a score:

- **Train/test labelling.** Every record carries a `held-out test` /
  `SEEN IN TRAINING` badge, read from the saved run's split (falling back to the same
  deterministic split `run_full_comparison.py` uses). Training records get an explicit
  warning that the score measures memorisation.
- **ADFECGDB is not an independent dataset.** All five of its records are republished
  inside CinC2013 set-a — `r01=a05`, `r04=a25`, `r07=a19`, `r08=a17`, `r10=a03`,
  verified by cross-correlating every abdominal lead (r = 1.000, identical annotation
  samples). Three of the five were in the training split. Without this mapping,
  evaluating on ADFECGDB *looks* like cross-dataset generalisation while it is partly
  a training-set score, so the app inherits each record's split label from its twin
  and excludes the duplicates from batch runs by default.
- **NIFECGDB annotates the wrong heart.** Its `.qrs` files mark the **maternal** QRS
  (~83 bpm here), not the fetal one. Fetal detection metrics against them are not
  fetal metrics, so the app refuses to score that dataset and offers it for visual
  inspection only. A related check flags any record whose annotation rate falls in the
  maternal range.

Relatedly, the ADFECGDB direct scalp lead (`Direct_1`) is the ground-truth source, so
it is excluded from the model-input channel list — feeding it to the detector would
be leaking the answer.

### Limitations of the bench

It evaluates the U-Net and the four classical arms; the recovery arm of §9 is not
wired into it yet (use `demo_recovery.py` / `run_recovery_experiment.py`). It also
needs a trained checkpoint and the datasets present locally — with an empty `data/`
or no `results/*.pth` it stops with a message rather than fabricating a demo.

---

## 17. Suggested next steps

1. **Wire up the ADFECGDB EDF loader.** It is the only path to validating the
   coincident-beat claims against true scalp ground truth.
2. **k-fold cross-validation** to firm up the ranking.
3. **Sweep the detection threshold** — `dl_unet` is currently conservative
   (PPV 0.974, Se 0.796); lowering the threshold likely trades some precision for
   recall, and the right operating point is a clinical decision, not a default.
4. **Test the recovery arm on real data**, where the baselines actually leave
   headroom for it to demonstrate value.
