# Occlusion-aware fetal ECG recovery

*The novel arm of this project — reframing maternal/fetal overlap from a separation
problem into an occlusion (inpainting) problem.*

## The idea in one line

The maternal QRS **occludes** the fetal beat at known, predictable locations
(maternal R-peaks). So instead of trying to *extract* the fetal beat from the
corrupted instant, **predict** the hidden beat from the fetal rhythm — which is
independent of the mother's — and use the observation there only as a soft,
down-weighted hint.

Every classical arm trusts the overlap samples as much as any other samples. This
one knows *in advance* which samples are unreliable — they sit under the maternal
QRS — and routes around them.

## Two tiers

**Tier 1 — rhythm recovery (numpy only, runs and is tested here).**
Recovers the *timing* of occluded beats, which is what FHR and HRV depend on.

- `occlusion.py` — turns maternal R-peaks into an **occlusion mask** and per-sample
  **reliability weights** (≈1 away from maternal QRS, dropping to a floor at each
  maternal R). This is the "gated measurement model".
- `rhythm_prior.py` — estimates a robust baseline fetal RR, finds gaps in the visible
  beat train that are a near-integer multiple of it (a beat went missing), and
  predicts the hidden beat's phase location. A prediction is **kept only if an
  occlusion explains the miss** — that guardrail is what protects precision.
- `gated_recovery.py` — `OcclusionAwareRecovery`. Gates the fetal detection itself
  (reliability weights suppress maternal residual before peak-finding, so the visible
  rhythm is clean), then **adds** rhythm-predicted beats to the direct detections.
  It is **additive** — it never discards a correct direct detection, so it cannot do
  worse than the baseline by construction.

**Tier 2 — diffusion morphology inpainting (PyTorch, written and syntax-checked).**
Reconstructs the *waveform shape* under the occlusion once the timing is known.

- `diffusion/model.py` — a 1D UNet noise predictor with timestep embedding.
- `diffusion/inpaint.py` — DDPM schedule + **RePaint-style masked inpainting**: at
  each reverse step the visible samples are overwritten with the correctly-noised
  observation, so the model only generates the occluded region, consistent with the
  visible context. The mask fed in is exactly the occlusion mask from Tier 1.
- `diffusion/train.py` — trains the beat prior on clean fetal beats (self-supervised
  with random masks).

Torch does not fit in the build sandbox, so Tier 2 is syntax-checked only. It plugs
directly into your existing diffusion-model and ECG-foundation-model projects: the
diffusion model is the inpainting engine, a foundation model can serve as the fetal
prior, and this pipeline is the test bed.

## How to run

```bash
# Headline demonstration of the mechanism (numpy only)
python demo_recovery.py --dropout 1.0      # knock out ALL coincident beats, recover them
python demo_recovery.py --dropout 0.5

# Head-to-head vs classical arms in the shared harness
python run_recovery_experiment.py --records 8
```

```python
from fecg import PipelineConfig, synthesize_record
from fecg.recovery import OcclusionAwareRecovery, run_recovery_pipeline

cfg = PipelineConfig()
rec = synthesize_record(cfg, duration_s=30.0, overlap_frac=0.25)
out = run_recovery_pipeline(rec, cfg, OcclusionAwareRecovery(cfg))
print(out["eval"]["coincident"]["Se"], out["n_recovered"])
```

## An honest note on the evaluation

The simple synthetic generator here does **not** reproduce real overlap-induced
fetal dropout: template subtraction on clean synthetic maternal beats preserves the
fetal beat, so on synthetic data there is usually nothing to recover — and
`run_recovery_experiment.py` correctly shows the recovery arm staying *safe* (it
matches the baseline and recovers ≈0 beats) rather than showing a fake improvement.

To test the recovery *mechanism* in isolation, `demo_recovery.py` injects the failure
mode that real data exhibits — it removes the fetal detections under a maternal QRS —
and then measures whether the rhythm prior puts them back. It does: knocking out
**100%** of coincident beats, the prior recovers **~92%** of them with **zero** false
insertions (coincident Se 0.000 → 0.93). That is a clean unit test of the
contribution, explicitly labelled as such.

**The real magnitude of benefit must be measured on real data.** The validation path
is ADFECGDB (direct fetal scalp ground truth) and the FECGSYN simulator (realistic
maternal morphology and overlap). The generative-hallucination risk — a diffusion
prior smoothing away real HRV or inventing plausible-but-wrong beats — must be checked
adversarially there, by comparing recovered beat timing and HRV under occlusion
against the scalp ground truth.

## Why it's a distinct contribution

It changes what you build (a generative fetal-beat prior, not a better subtraction),
what you measure (recovery of *hidden* beats specifically, stratified by coincidence),
and what can go wrong (hallucinated beats — the failure mode to study). The recent
literature has diffusion-as-prior for missing ECG and ECG↔Doppler cross-modal
generation, but not occlusion-inpainting with a maternal-QRS-gated mask applied to the
maternal/fetal overlap case. That is the gap this fills.
