# Autoresearch: ML2 CV Generalization Search

## Objective
Find a lightweight PyTorch image classifier recipe that generalizes beyond the public leaderboard for ML2 Assignment 2. The submitted model must stay under 500,000 parameters and accept `(B,3,256,256)` float images in `[0,1]`, returning `(B,7)` logits.

This session should search only broad, defensible changes to the student/distillation/training recipe. Do not optimize to the public leaderboard and do not use hidden test feedback.

## Metrics
- **Primary**: `generalization_score` (unitless, higher is better) — robust scalar computed from paired validation accuracy, stress robustness, and calibrated loss.
- **Secondary**:
  - `mean_val_acc` — mean clean validation accuracy across split seeds.
  - `cal_loss` — mean calibrated validation cross entropy.
  - `stress_mean` — mean stress-test accuracy.
  - `stress_worst` — worst stress-test accuracy averaged across seeds.
  - `selected_epoch` — chosen checkpoint epoch.
  - `tau` — median/geomean calibration temperature.

## Generalization Protocol
Use a three-tier protocol to avoid local overfitting:

1. **Exploration seeds**: use `autoresearch_candidate.json` default seeds `30,31,32` for overnight autoresearch. These are optimization seeds, not final proof.
2. **Locked confirmation seeds**: after autoresearch proposes top candidates, run only the incumbent and <=3 finalists on locked seeds `40,41,42,43,44` once.
3. **Final training**: only after confirmation, train on all labeled data + legal unlabeled data with the fixed chosen recipe.

A candidate is submit-worthy only if it passes confirmation gates:

- mean clean acc >= incumbent + 0.012
- at least 4/5 confirmation seeds improve or tie
- no seed drops by more than 0.025
- stress mean >= incumbent - 0.005
- stress worst >= incumbent - 0.015
- calibrated NLL <= incumbent + 0.030

## How to Run
`./autoresearch.sh`

This reads `autoresearch_candidate.json`, runs a paired local KD sweep via `experiment.py`, computes metrics, prints `METRIC ...` lines, then restores `log.txt` so autoresearch commits are not polluted by benchmark logs.

## Files in Scope
- `autoresearch_candidate.json` — primary knob file for candidate configs. Prefer editing this over `experiment.py`.
- `autoresearch_eval.py` — benchmark wrapper and scoring function.
- `autoresearch.sh` — benchmark entrypoint.
- `experiment.py` — only for general, non-class-specific recipe capabilities or bug fixes. Avoid broad rewrites during autoresearch.
- `autoresearch.md` and `autoresearch.ideas.md` — session memory and backlog.

## Off Limits
- `model.pt` — preserve current submitted model unless a confirmed finalist wins.
- `data/`, `data.zip`, assignment files, and hidden/public leaderboard feedback.
- Public leaderboard submissions.
- Dataset-source searching or outside labeled data.
- Class-specific hacks based on inspected validation images.
- Directly optimizing to class 1/class 4 audit findings. Use audits only as hypothesis generation.

## Current Incumbent / Context
Current submitted model:

```text
recipe: kd_b0_tta_eca160
student epochs: 65
teacher epochs: 10
KD: T=4 alpha=0.8 labeled=0.5 unlabeled=1.0 logits TTA targets
export tau: 1.95
params: 493,651
public/client: acc 0.5816, loss 1.2370
```

Most recent local stress-finalist result found a better local candidate:

```text
T=2 alpha=0.8 labeled=0.5 unlabeled=1.0 logits targets epoch=65
mean val acc=0.5801
cal loss=1.1954
stress mean=0.5589
stress worst=0.5195
median tau=1.50
```

Image audit suggested the student under-transfers teacher semantics on falafel/mezze-like class 1 and atypical ramen/noodle class 4, but this must not be used for class-specific tuning. If acting on this, prefer global changes like labeled KD weight, crop context, or teacher-target transfer strength, validated across all classes/seeds.

## Suggested Safe Search Space
Prefer global KD/training variants:

- `T`: 1.5, 2.0, 2.5, 3.0
- `alpha`: 0.70, 0.75, 0.80, 0.85
- `labeled`: 0.50, 0.75, 1.00
- `unlabeled`: 0.50, 0.75, 1.00
- checkpoint epochs: 55, 65, 75
- possibly safer crop/context augmentation only if implemented globally and validated against stress/worst-class behavior.

## What's Been Tried
- Supervised-only was much worse than KD.
- EfficientNet-B0 teacher is reliable; ConvNeXt teacher collapsed in current setup.
- Calibration improves loss but not accuracy.
- hflip inference TTA and checkpoint averaging did not give clean local gains.
- Shared-teacher grid showed T=2/T=8 variants competitive; stress-finalist run favored T=2 epoch 65.
- Image audit found teacher/student semantic transfer gaps, especially class 1 and class 4, but class-specific optimization is off limits.

## Autoresearch Rules
- Prefer one candidate row per experiment to keep attribution clean.
- Keep only candidates that improve primary `generalization_score`; otherwise discard.
- On discard/crash, annotate `asi` with what failed and what to avoid repeating.
- Do not run confirmation seeds repeatedly. Confirmation is a separate manual step after top candidates emerge.
