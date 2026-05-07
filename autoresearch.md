# Autoresearch: ML2 CV MaxViT Transfer Search

## Objective
Search whether the very strong staged MaxViT teacher can produce a better deployable student than the previous EfficientNet-B0 KD incumbent, without falling into public-leaderboard tuning.

The submitted model must remain a single PyTorch student with strictly fewer than 500,000 parameters and accept `(B,3,256,256)` float images in `[0,1]`, returning `(B,7)` logits.

## Current Pivot
Teacher diagnostics on Colab showed:

```text
MaxViT teacher: maxvit_base_tf_384.in21k_ft_in1k
staged fine-tune: 5 epoch head warmup, then full FT with backbone_lr=2e-5, head_lr=1e-3
seed 30 val acc: 0.9221
val loss: 0.3398-0.3956 depending run
Top-3: ~0.987-1.000
unlabeled pseudo counts: balanced
```

But the first MaxViT student using the old B0-tuned KD knobs did not beat the prior incumbent:

```text
T=2 alpha=0.70 labeled=0.50 unlabeled=1.00
best val acc=0.6104
best val loss=1.1946
stress mean=0.5584
stress worst=0.5065
best epoch=45
```

Interpretation: MaxViT is an excellent teacher but very sharp/overconfident, so the student likely needs softer or less teacher-dominant KD.

## Metrics
- **Primary**: `generalization_score` (higher is better)
- **Secondary**:
  - `mean_val_acc`
  - `cal_loss`
  - `stress_mean`
  - `stress_worst`
  - `selected_epoch`
  - `tau`

The benchmark still uses `autoresearch_eval.py`, which runs `experiment.run_kd_cv` and computes the robust scalar from validation accuracy, stress robustness, and calibrated loss.

## Current Baseline Candidate
`autoresearch_candidate.json` is set to the next MaxViT rescue candidate for the overnight Mac run:

```text
teacher_model=maxvit_base_tf_384.in21k_ft_in1k
teacher_epochs=15
teacher_head_warmup_epochs=5
teacher_head_lr=1e-3
teacher_backbone_lr=2e-5
teacher_label_smoothing=0.0
teacher_augment=basic
teacher_tta=false
student=plain_eca_head, head_ch=224, input_size=160
T=4 alpha=0.50 labeled=0.50 unlabeled=0.50
epochs=45
seeds=1 start_seed=30
```

This reflects the Colab grid result: `T=4 alpha=0.50 unlabeled=1.00` was the best MaxViT grid variant but still did not beat the B0 profile, and MaxViT runs peaked around epoch 45. The overnight candidate reduces unlabeled teacher pressure and stops near the observed peak. `seeds=1` is intentional for triage on the Mac because MaxViT is expensive. Do not treat a one-seed result as final proof. Promote only substantial wins to multi-seed confirmation.

## Search Space for Overnight
Prefer one global change per iteration:

- KD softness/weight:
  - `T`: 3, 4, 6
  - `alpha`: 0.40, 0.50, 0.60, 0.70
- Teacher signal weights:
  - `labeled`: 0.50, 0.75, 1.00
  - `unlabeled`: 0.50, 0.75, 1.00
- Student architecture, only if KD settings do not win:
  - `student_model=plain_eca_head` incumbent, `head_ch=224`
  - `student_model=grid_hybridse`, `dropout=0.20`, `weight_decay=1e-3`
  - `student_model=grid_tinymobilenetv2`, `lr=2e-3`, `weight_decay=1e-3`
- Training horizon:
  - `epochs`: 45, 55, 65, 75, because first MaxViT run peaked at epoch 45

## What Not To Do
- Do not use public leaderboard feedback.
- Do not repeatedly run locked confirmation seeds.
- Do not use hidden labels, outside labeled data, dataset-source hunting, or class-specific hacks.
- Do not assume the 0.92 teacher means the default student recipe is good; validate student transfer.
- Do not spend many iterations on EVA/ConvNeXtV2 for now: diagnostics showed they failed to adapt under staged fine-tuning.

## Promotion Gates
A one-seed MaxViT candidate is worth confirming only if it beats the B0/previous profile by a meaningful margin, for example:

- `mean_val_acc >= 0.63` on seed 30, or
- `stress_mean >= 0.59` with clean/loss near incumbent, or
- `cal_loss <= 1.15` while clean accuracy does not regress.

Final confirmation should restore multi-seed protocol:

```text
exploration: seed 30 single-seed triage
candidate confirmation: seeds 30,31,32 or locked 40-44 versus incumbent
final: all labeled data + legal unlabeled data with fixed recipe
```

## Files in Scope
- `autoresearch_candidate.json` — primary knob file. Prefer editing this.
- `autoresearch_eval.py` — now supports broad global candidate overrides for teacher/student/KD fields.
- `experiment.py` — only for general training capability/bug fixes.
- `maxvit_kdgrid.py` — manual Colab grid helper, not the overnight autoresearch entrypoint.

## Autoresearch Rules
- Keep only candidates that improve primary `generalization_score`.
- On discard/crash, annotate `asi` with what failed and what to avoid repeating.
- Always include why the candidate addresses MaxViT overconfidence/student transfer.
- Deferred promising ideas go to `autoresearch.ideas.md`.
