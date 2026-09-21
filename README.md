# fECG — Fetal ECG / FHR Extraction Under Maternal–Fetal QRS Overlap

Extract fetal ECG / fetal heart rate from mixed maternal–fetal abdominal ECG, with
the research focus on the case that breaks most pipelines: **when the maternal and
fetal QRS coincide in time.** Every method arm runs through the same
**overlap-stratified** evaluation harness, so the headline number is the recovery
rate on *coincident* fetal beats — not an aggregate F1 that hides them.

## Why this is structured the way it is

Overlap is fundamentally a *separability* problem. Purely temporal methods lose all
separability at exact coincidence, so the interesting question is what a
*non-temporal* separation axis buys you. The arms are organized along that idea:

| Arm | Separation axis | Needs |
|---|---|---|
| `template` | temporal (per-beat gain) | maternal R-peaks |
| `adaptive` (LMS/RLS) | reference correlation | a maternal reference lead (or a template proxy) |
| `bss_ica` | **spatial** | ≥ 2 channels |
| `kalman` | temporal + morphology tracking | maternal R-peaks |
| `dl_unet` | **learned (spatial + morphology)** | training data + PyTorch |

## Pipeline flowchart

```
          Abdominal ECG recording (4 channels)
          CinC2013 / ADFECGDB / NIFECGDB / synthetic
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │ 1. LOAD  (fecg/io.py)                 │
        │    fill NaN gaps by interpolation     │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ 2. PREPROCESS  (preprocess.py)        │
        │    band-pass + mains-noise notch      │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ 3. DETECT MATERNAL QRS                │
        │    Pan-Tompkins → maternal R-peaks    │
        └───────┬──────────────────────┬───────┘
                │                      │
      Classical methods          Deep learning
                ▼                      ▼
   ┌─────────────────────────┐  ┌──────────────────────────┐
   │ 4a. SUBTRACT MATERNAL   │  │ 4b. 1D U-NET (fecg/dl)    │
   │  template | adaptive |  │  │  all channels in →        │
   │  bss_ica  | kalman      │  │  fetal R-peak probability │
   └───────────┬─────────────┘  └────────────┬─────────────┘
               ▼                             │
   ┌─────────────────────────┐               │
   │ 5. DETECT FETAL QRS     │               │
   │  in what's left over    │               │
   │  (+ optional recovery   │               │
   │   of hidden beats)      │               │
   └───────────┬─────────────┘               │
               └──────────────┬──────────────┘
                              ▼
        ┌──────────────────────────────────────┐
        │ 6. FHR / HRV  (fetal.py)              │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ 7. EVALUATE  (evaluate.py)            │
        │  match beats to truth within ±50 ms   │
        │  Se / PPV / F1, reported separately   │
        │  for coincident and other beats       │
        │  + FHR error                          │
        └──────────────────┬───────────────────┘
                           ▼
        results/*.json, figures, Streamlit test bench
```

Only step 4 differs between the classical methods (they all plug into the same
`Suppressor` interface). Everything else is shared, so differences in the results
come from the methods themselves.

## Procedure

1. **Clone and install**
   ```bash
   git clone https://github.com/sesha003/Fetal-Project.git
   cd Fetal-Project
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt      # torch is optional (only for the DL arm)
   ```
2. **Quick check without any data:**
   ```bash
   python run_experiment.py             # runs all classical arms on a synthetic cohort
   python tests/test_smoke.py           # sanity checks
   ```
   The synthetic generator forces a controllable fraction of maternal/fetal
   coincidences (`overlap_frac`), which is how you study the overlap case directly.
   **Note:** synthetic numbers are a functional smoke test, not a finding — the
   underlying signal has only two sources, so the arms land close together and ICA
   is under-determined. Method differentiation is expected to appear on real
   multichannel recordings with genuine spatial diversity.
3. **Datasets** from PhysioNet are included in the repo under
   `data/cinc2013/set-a/`, `data/adfecgdb/` and `data/nifecgdb/`.
4. **Run the classical methods on real data:**
   ```bash
   python run_experiment.py --dataset cinc2013
   ```
5. **Run the full comparison.** Trains the U-Net (60 train / 15 test records, about
   20 minutes) and writes `results/dl_unet.pth` plus `results/comparison_metrics.json`:
   ```bash
   python run_full_comparison.py --n-test 15 --epochs 20
   ```
6. **Make the figures** (start with `results/fig_signal_anatomy.png`):
   ```bash
   python plot_comparison.py
   python plot_diagnostics.py
   ```
7. **Try the recovery method:**
   ```bash
   python demo_recovery.py --dropout 1.0
   python run_recovery_experiment.py --records 8
   ```
8. **Explore interactively:**
   ```bash
   streamlit run app.py
   ```
9. **Read further:** `EXPLANATION.md` for the full reasoning and results, and
   `RECOVERY.md` for the recovery method.

## Using real data (ADFECGDB)

1. Download ADFECGDB from PhysioNet.
2. In `run_experiment.py`, uncomment the ADFECGDB branch in `load_cohort()` and set
   the path + record IDs.
3. Confirm which annotation extension holds the scalp fetal R-peaks (the loader
   tries `fqrs` → `qrs` → `atr`). That scalp channel is your ground truth, and it is
   the only thing that tells you the truth at a coincident beat.

```python
from fecg import load_adfecgdb_record, PipelineConfig, run_pipeline, ICASuppressor
cfg = PipelineConfig(fs=1000)
rec = load_adfecgdb_record("path/to/adfecgdb/r01", name="r01")
out = run_pipeline(rec, ICASuppressor(cfg), cfg)
print(out["eval"])          # overall + coincident + non_coincident
```

## Deep-learning arm

```python
from fecg import synthetic_cohort, PipelineConfig
from fecg.dl import train_model, DLDetector
cfg = PipelineConfig()
records = synthetic_cohort(cfg, n_records=8)          # or real records
model = train_model(records[:-1], cfg, in_channels=4, epochs=20)
detector = DLDetector(model, cfg)
peaks = detector.detect(records[-1])                  # -> evaluate() as usual
```

Multichannel input lets the network exploit spatial diversity — the same property
that makes BSS work at coincidence. The output is a per-sample R-peak heatmap;
peak-picking feeds the identical FHR/HRV/evaluation code.

## Layout

```
fecg/
  config.py            PipelineConfig (all knobs)
  io.py                Record container, real loaders, synthetic generator
  preprocess.py        bandpass + notch + Pan-Tompkins QRS detection
  fetal.py             fetal QRS detection, FHR, HRV
  evaluate.py          matching + OVERLAP-STRATIFIED metrics + FHR error
  pipeline.py          orchestrator + leave-one-record-out CV
  suppression/
    base.py            Suppressor interface (the shared contract)
    template.py        adaptive template subtraction
    adaptive.py        LMS / RLS
    bss.py             FastICA
    kalman.py          beat-wise Kalman template tracker
  dl/
    model.py           1D U-Net
    dataset.py         windowing + Gaussian R-peak targets
    train.py           training loop + DLDetector inference wrapper
run_experiment.py      method comparison table
testbench.py           dataset registry + model loading + scoring helpers (no UI)
app.py                 Streamlit test bench (UI only; imports testbench.py)
tests/test_smoke.py    smoke tests
```

## Honest limitations

- The **Kalman arm is a pragmatic stand-in** for the full Sameni EKF/EKS (McSharry
  dynamic model). It tracks the maternal *template* over beats rather than a phase
  model. Swap in a McSharry-model EKS (e.g. Sameni's OSET) behind the same
  `Suppressor` interface for the full method — nothing else changes.
- The **adaptive arm** wants a real maternal reference (thoracic lead). ADFECGDB has
  none, so it builds a template-based proxy reference; pass a real chest lead when
  the dataset has one.
- **Synthetic QRS shapes are simplified** (Mexican-hat, not real P-QRS-T). Absolute
  numbers on real data will be lower and the coincident gap wider — which is the
  point.
- This is a strong **classical baseline to measure the DL model against**, not the
  final word.

## Interactive test bench (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser UI for poking at the trained detector on real recordings, instead of
re-running a script and reading a table. It loads a checkpoint from `results/`
(`dl_unet.pth` by default) and the datasets under `data/`.

Sidebar controls apply to every view: dataset, model checkpoint, **detection
threshold** (peak height on the probability trace), how many seconds of each record
to analyse, and which classical arms to run alongside the U-Net.

Four views:

- **Single record** — probability trace, detections against ground truth, FHR trace,
  maternal QRS marked so coincident beats are visible by eye. The headline tile is
  coincident Se with the non-coincident delta next to it — the overlap penalty.
- **Batch evaluation** — overlap-stratified metrics over a set of records, with the
  classical arms optionally alongside; exports CSV.
- **Threshold sweep** — Se/PPV/F1/coincident-Se vs peak-picking threshold, which
  separates "the model cannot see this beat" from "the threshold is wrong".
- **Saved training run** — the tables and figures from `run_full_comparison.py`.

Three provenance guards are built in, because each of the problems they catch
silently inflates scores:

- every record is labelled **held-out test / SEEN IN TRAINING**, and a training
  record is flagged as measuring memorisation, not generalisation;
- **ADFECGDB is not an independent dataset** — all five of its records are republished
  inside CinC2013 set-a (`r01=a05, r04=a25, r07=a19, r08=a17, r10=a03`, correlation
  1.000 with identical annotations), and three of them were in the training split.
  The app labels them and excludes them from batch runs by default;
- **NIFECGDB `.qrs` annotates the maternal QRS** (~83 bpm), not the fetal one, so fetal
  detection metrics against it are meaningless; the app refuses to score it.

The ADFECGDB direct scalp lead (`Direct_1`) is the ground-truth source, so it is also
kept out of the model-input channel list.

All the logic lives in `testbench.py`, so every number the app shows is reproducible
from a plain Python session:

```python
from testbench import load_record, load_model, dl_probability, peaks_from_probability, score
```

## Novel arm: occlusion-aware recovery

A second, novel method reframes overlap as an *occlusion/inpainting* problem instead
of a separation problem: predict the fetal beat hidden under the maternal QRS from the
fetal rhythm, using the observation only as a down-weighted (maternal-QRS-gated) hint,
and optionally inpaint its morphology with a diffusion prior. See **RECOVERY.md** for
the full write-up, and `fecg/recovery/`. Quick start: `python demo_recovery.py`.
