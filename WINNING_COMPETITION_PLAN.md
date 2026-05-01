# ML2 Assignment 2 Competition Playbook

Objective: win the leaderboard while submitting a student model that genuinely generalizes and has **strictly fewer than 500,000 trainable/stored parameters**.

This is our shared working document. We will update it after each experiment with validation scores, leaderboard scores, and decisions.

---

## 1. Assignment constraints read from `ML2_Assignment_2.pdf`

### Hard requirements
- Submit a PyTorch image classifier as `model.pt`.
- The submitted model must take server inputs shaped `(B, 3, 256, 256)`, float32 in `[0, 1]`.
- The submitted model must return `(B, 7)` logits.
- The submitted model must have **strictly fewer than 500,000 total parameters**.
  - Count everything stored in the model: Conv/Linear weights, BatchNorm parameters, embeddings, frozen tensors, etc.
  - The starter script uses `<= 500_000`; we should change this to `< 500_000` before any final submission.
- Preprocessing must be inside the submitted TorchScript module.
- We may train large pretrained teacher models, but **only the small student is submitted**.
- We may use the provided unlabeled data for teacher prediction, pseudo-labeling, consistency training, and distillation.
- We may not use outside labeled data or try to identify the dataset source on the web.

### Report requirement
The report must include a comparison of multiple distillation temperatures `T`, with numbers and interpretation.

---

## 2. Files inspected

### Provided files
- `ML2_Assignment_2.pdf` — assignment instructions and rules.
- `data.zip` — dataset archive.
- `starter_kit.zip` — starter scripts archive.
- `starter_kit/train_baby.py` — baseline student training script.
- `starter_kit/train_teacher.py` — EfficientNet-B0 teacher fine-tuning and unlabeled-logit dump.
- `starter_kit/distill.py` — baseline knowledge distillation script.

### Extracted dataset inventory
After extracting `data.zip`:

| Split | Count | Notes |
|---|---:|---|
| `data/train/*.jpg` | 399 | all are RGB 256x256 JPEGs |
| `data/train/labels.csv` | 399 labels | exactly 57 images per class, labels 0 through 6 |
| `data/unlabeled/*.jpg` | 798 | all are RGB 256x256 JPEGs |

Class distribution is perfectly balanced:

| Class | Count |
|---:|---:|
| 0 | 57 |
| 1 | 57 |
| 2 | 57 |
| 3 | 57 |
| 4 | 57 |
| 5 | 57 |
| 6 | 57 |

Image statistics:

| Split | Mean RGB | Std RGB | Duplicate files |
|---|---|---|---:|
| train | `[0.5310, 0.4386, 0.3482]` | `[0.2357, 0.2442, 0.2391]` | 0 |
| unlabeled | `[0.5332, 0.4413, 0.3527]` | `[0.2359, 0.2441, 0.2397]` | 0 |

The train and unlabeled distributions look closely matched, so the unlabeled set should be useful for distillation.

I also generated a local visual contact sheet for orientation: `data_overview_by_label.jpg`.

---

## 3. Starter kit audit

### `train_baby.py`
Baseline student:
- `Preprocess` wrapper resizes from 256 to 64 and normalizes with ImageNet mean/std.
- `SmallCNN` has about 95K parameters.
- Trains for 20 epochs on an 80/20 random split with seed 0.
- Saves TorchScript `model.pt`.

Weaknesses:
- Uses only a single random validation split; with 399 images, this is noisy.
- Model is far below the parameter budget.
- No serious data augmentation.
- Adam without weight decay/schedule.
- No checkpoint selection by validation loss/accuracy in `train_baby.py`.

### `train_teacher.py`
Baseline teacher:
- Uses pretrained EfficientNet-B0.
- Fine-tunes for only 10 epochs.
- Saves logits on the 798 unlabeled images.

Weaknesses:
- EfficientNet-B0 may not be the best teacher.
- No augmentation, no stratified validation, no cross-validation, no test-time augmentation.
- Teacher quality dominates distillation quality, so this deserves real effort.

### `distill.py`
Baseline KD:
- Imports `SmallCNN` and `Preprocess` from `train_baby.py`.
- Uses hard CE on labeled data plus KL distillation on unlabeled data.
- Defaults: `T=4`, `ALPHA=0.7`, `EPOCHS=50`.
- Tracks best validation accuracy and saves best model.

Weaknesses:
- Uses one split only.
- Does not use labeled teacher logits; only unlabeled KD plus labeled CE.
- No strong student augmentations or consistency regularization.
- No temperature sweep yet.

### Path issue to fix before running starter scripts
The starter scripts expect:
- `starter_kit/train`
- `starter_kit/unlabeled`

The extracted dataset is currently under:
- `data/train`
- `data/unlabeled`

We should either symlink/copy the directories into `starter_kit/` or modify the scripts to point to `../data`.

---

## 4. Anti-Kaggle-trap principles

We want leaderboard performance, but the real goal is hidden-test generalization. Rules:

1. **Never tune blindly to public leaderboard only.** Treat public LB as a weak, potentially noisy signal.
2. **Use local validation discipline.** Keep at least one untouched holdout split while developing.
3. **Prefer repeated stratified splits or 7-fold CV.** The dataset has exactly 57 examples per class, which is ideal for stratified folds.
4. **Log every run.** Architecture, seed, split, augmentation, optimizer, teacher, `T`, `alpha`, validation accuracy, validation loss, calibration, and leaderboard score.
5. **Submit sparingly.** Public LB overfitting is real. Only submit models that improve robust local metrics.
6. **Do not hand-code based on visual quirks.** Visual inspection is for understanding augmentations and failure modes, not for hard-coded prediction hacks.
7. **Respect the competition ethics.** No outside labels, no dataset-source search, no hidden-test leakage.

---

## 5. Validation protocol

Because there are only 399 labeled images, one 80/20 split can lie to us.

### Phase A: fast development split
- Use a stratified 80/20 split with fixed seed.
- Keep it stable for quick architecture and bug checks.
- Track validation accuracy and cross-entropy.

### Phase B: robust model selection
Use repeated stratified validation:
- 5 or 7 folds, because each class has 57 examples.
- Suggested: 7-fold CV gives about 8 or 9 images per class per fold.
- Train each candidate on 6 folds, validate on 1 fold.
- Compare mean accuracy, std, mean loss.

### Phase C: final training
For final candidate:
1. Select architecture and hyperparameters from CV.
2. Train teacher on all labeled data, optionally with a small validation split only for early stopping if needed.
3. Generate teacher logits for unlabeled data.
4. Train student using all labeled data plus unlabeled distillation.
5. Use checkpoint chosen by CV recipe, not public LB guessing.

---

## 6. Student architecture direction

We need spend the 500K budget wisely. The likely winning student is not a plain large ConvNet; it should use modern small-model tricks.

### Candidate families

#### A. Strong plain CNN under 500K
Good baseline; easy to train.
- Conv-BN-SiLU blocks.
- Downsample progressively: 256 -> 128 -> 64 -> 32 -> 16 -> 8.
- Global average pooling.
- Small MLP classifier.
- Dropout 0.1 to 0.4.

Risk: standard convolutions spend parameters quickly and may underperform depthwise models.

#### B. Depthwise-separable CNN / MobileNet-style student
Likely best use of budget.
- Stem Conv 3x3.
- Depthwise 3x3 + pointwise 1x1 blocks.
- Squeeze-and-excitation optional if parameter count allows.
- Residual connections when channel dimensions match.
- Global average pooling.
- Classifier head.

Why promising:
- Much more depth for the same parameter count.
- Good inductive bias for image classification.
- Strong deployment story.

#### C. Tiny ResNet with bottlenecks
- Residual blocks improve optimization.
- Use widths like 32/64/128/192.
- Keep classifier cheap with global average pooling.

Risk: may waste parameters compared to depthwise separable blocks.

### Preprocessing inside model
We should test input sizes:
- 96: faster, regularizes, less detail.
- 128: likely strong compromise.
- 160: maybe better for food texture/details, still feasible.
- 224: teacher-size; student may overfit and train slower.

The submitted model can resize internally, so input-size sweep is allowed.

### Parameter-count target
Aim for **450K to 490K**, not 499,999, to leave safety margin for buffers/wrapper changes. Final assertion should be:

```python
assert count_params(model) < 500_000
```

---

## 7. Data augmentation plan

Food images should tolerate photometric and mild geometric variation, but not extreme transformations.

### Safe augmentations
- RandomResizedCrop to model input size.
- HorizontalFlip.
- ColorJitter: brightness/contrast/saturation/hue, moderate.
- RandomRotation maybe +/- 10 degrees.
- RandomAffine small translation/scale.
- RandomErasing / Cutout, moderate probability.
- MixUp and CutMix for supervised student/teacher fine-tuning.

### Be careful with
- VerticalFlip: food photos are not usually upside down; can hurt.
- Aggressive rotations: may create unrealistic plates/backgrounds.
- Too much color jitter: class cues may include food color.

### Validation preprocessing
- Deterministic resize/crop only.
- No random augmentation.
- Consider validation test-time augmentation only for teacher, not submitted student unless embedded deterministically, which is not worth it.

---

## 8. Teacher strategy

Teacher quality is leverage. Since teacher is not submitted, use capacity.

### Teachers to try
1. EfficientNet-B0 baseline.
2. EfficientNet-B2/B3 if compute allows.
3. ConvNeXt-Tiny.
4. ResNet50 as a sanity baseline.
5. Possibly an ensemble of teachers only for generating logits, if allowed by time. The ensemble is not submitted, only its soft labels are used.

### Teacher training improvements
- Use pretrained weights.
- Replace final classifier with 7-class head.
- Strong but realistic augmentation.
- AdamW with weight decay.
- Cosine LR schedule with warmup.
- Freeze backbone for a few epochs, then unfreeze all.
- Label smoothing for teacher hard-label training.
- Stratified validation.
- Save best by validation loss, not just accuracy.

### Teacher output improvements
- Dump teacher logits for both unlabeled and labeled images.
- For unlabeled data, optionally use teacher TTA to average logits over multiple crops/flips.
- Track teacher confidence distribution and pseudo-label class balance.

---

## 9. Distillation strategy

Baseline formula:

```text
loss = (1 - alpha) * CE(student_logits_labeled, labels)
     + alpha * T^2 * KL(log_softmax(student_logits_unlabeled / T), softmax(teacher_logits_unlabeled / T))
```

### Distillation variants to test
1. Unlabeled KD only, as starter.
2. Labeled + unlabeled KD: teacher logits for all images, plus CE on labels.
3. Confidence-filtered pseudo-labeling: add CE on high-confidence unlabeled predictions.
4. Soft pseudo-labels for all unlabeled, hard pseudo-label CE only for confidence > threshold.
5. MixUp/CutMix with soft targets.
6. EMA teacher for student weights, if implementation time permits.

### Required temperature sweep
We must run at least:
- `T=1`
- `T=2`
- `T=4`
- `T=8`

Optional:
- `T=6`
- `T=12`

For each temperature, record:
- Validation accuracy.
- Validation cross-entropy.
- Expected calibration error or at least mean confidence.
- Notes: too sharp, stable, too flat, etc.

### Alpha sweep
Test after choosing a reasonable `T`:
- `alpha=0.3`
- `alpha=0.5`
- `alpha=0.7`
- `alpha=0.9`

If teacher is very strong, high alpha may help. If teacher overfits or is miscalibrated, lower alpha may generalize better.

---

## 10. Experiment log

Fill this table as we run.

| ID | Date | Student | Params | Input | Teacher | Augment | Split/CV | T | Alpha | Val Acc | Val CE | LB | Decision |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| B0 | TBD | starter SmallCNN | ~95K | 64 | none | none | seed0 80/20 | - | - | TBD | TBD | TBD | smoke test |
| S1 | TBD | wider plain CNN | TBD | 96/128 | none | basic | seed0 80/20 | - | - | TBD | TBD | TBD | compare capacity |
| S2 | TBD | depthwise student | TBD | 128 | none | strong | CV | - | - | TBD | TBD | TBD | candidate |
| T0-D0 | TBD | depthwise student | TBD | 128 | EffNet-B0 | strong | CV | 1 | 0.7 | TBD | TBD | TBD | T sweep |
| T0-D1 | TBD | depthwise student | TBD | 128 | EffNet-B0 | strong | CV | 2 | 0.7 | TBD | TBD | TBD | T sweep |
| T0-D2 | TBD | depthwise student | TBD | 128 | EffNet-B0 | strong | CV | 4 | 0.7 | TBD | TBD | TBD | T sweep |
| T0-D3 | TBD | depthwise student | TBD | 128 | EffNet-B0 | strong | CV | 8 | 0.7 | TBD | TBD | TBD | T sweep |

---

## 11. Immediate implementation checklist

1. Fix data path:
   - Either symlink `data/train` and `data/unlabeled` into `starter_kit/`, or modify scripts to use `../data`.
2. Create a proper experiment script instead of editing starter scripts randomly.
3. Add reproducibility controls:
   - Python, NumPy, Torch seeds.
   - Save config JSON with every run.
   - Save metrics CSV.
4. Implement stratified split/CV.
5. Implement augmentations.
6. Build a parameter-count-safe student architecture.
7. Run baseline from scratch.
8. Train first improved teacher.
9. Dump logits for unlabeled and labeled images.
10. Run temperature sweep.
11. Submit only after local evidence is strong.

---

## 12. Submission checklist

Before uploading to `https://ml2-hw2.onrender.com/`:

- [ ] `model.pt` is TorchScript-loadable without local class definitions.
- [ ] Model accepts `(B, 3, 256, 256)` float in `[0, 1]`.
- [ ] Model returns `(B, 7)` logits.
- [ ] Parameter count is strictly `< 500_000`.
- [ ] No teacher/pretrained model included in submitted artifact.
- [ ] Preprocessing is inside submitted model.
- [ ] No outside labeled data used.
- [ ] Final metrics and settings are recorded in this document.
- [ ] Report includes the required temperature comparison.

---

## 13. Report outline

1. Problem and deployment constraint.
2. Dataset summary: 399 labeled, 798 unlabeled, 7 balanced classes.
3. Student architecture and parameter count.
4. Training setup: splits/CV, optimizer, LR schedule, augmentation, regularization.
5. Teacher model: pretrained backbone, fine-tuning setup, teacher validation performance.
6. Knowledge distillation: objective, unlabeled usage, temperature/alpha choices.
7. Temperature analysis table and interpretation.
8. Final model performance and deployment compliance.
9. Lessons learned and limitations.

---

## 14. Current best hypothesis

The likely winning path is:

1. Train a strong pretrained teacher with realistic augmentations and careful validation.
2. Use teacher TTA to produce stable logits for unlabeled images.
3. Train a MobileNet-style depthwise-separable student at input size 128 or 160, around 450K parameters.
4. Use CE on labeled data plus KD on unlabeled and labeled data.
5. Sweep `T` and `alpha` with repeated stratified validation.
6. Select by robust local validation, not leaderboard chasing.

This balances the competition goal with real generalization and stays within the production constraint.

---

## 15. Research alpha: papers and recipes to exploit

Research transcript: `.pi-sherlock/cv-small-student-alpha-sherlock.md`.

The assignment is small-data, semi-supervised, teacher-student image classification under a strict deployment cap. The highest-alpha ideas are not exotic architectures; they are disciplined combinations of: strong teacher logits, MobileNet-style student design, robust augmentation, semi-supervised consistency, and cross-validation.

### 15.1 Top 10 actionable ideas

| Rank | Idea | Why it matters here | Implementation | Submitted params? |
|---:|---|---|---|---|
| 1 | MobileNetV2/V3-style depthwise student | Best depth/accuracy per parameter; plain ConvNets waste budget | Inverted residual blocks, width tuned to 450K-490K params | Yes, but efficient |
| 2 | Strong pretrained teacher, possibly multi-teacher logits | Teacher quality is the main signal for 798 unlabeled images | Fine-tune EfficientNet/ConvNeXt/ResNet teachers; average logits only for training | No |
| 3 | Teacher TTA logits | Stabilizes pseudo-labels without changing student inference | Average teacher logits over flips/crops/color-light TTA | No |
| 4 | Distill on labeled + unlabeled images | Uses all available data; soft labels expose class similarity | CE on labeled labels + KL to teacher logits on both labeled/unlabeled | No |
| 5 | FixMatch-style consistency | Uses unlabeled images beyond static logits | Weak aug creates pseudo-label; strong aug must match if confidence high | No |
| 6 | Temperature sweep `T={1,2,4,8}` | Required by report and can materially affect student stability | Record val accuracy/loss/confidence for each T | No |
| 7 | MixUp/CutMix with soft targets | Strong regularizer for tiny labeled set | Use mostly for teacher and supervised student stages; cautious with KD | No |
| 8 | RandAugment/TrivialAugment + RandomErasing | Avoids hand-designed overfitting; improves robustness | Moderate magnitude; no vertical flips | No |
| 9 | SWA/EMA checkpoint averaging | Cheap generalization gain for single submitted model | Maintain EMA or SWA weights; save averaged student | No extra if weights replace current weights |
| 10 | Stratified CV model selection | 399 images makes one split unreliable | 5- or 7-fold CV for architecture/hparams; final train on all | No |

### 15.2 Architecture papers to mine

- **SqueezeNet** (Iandola et al., 2016) — `https://arxiv.org/abs/1602.07360`  
  Takeaway: 1x1 bottlenecks and delayed expansion reduce parameters. Useful idea, but Fire modules are older than depthwise separable blocks.

- **MobileNetV1** (Howard et al., 2017) — `https://arxiv.org/abs/1704.04861`  
  Takeaway: depthwise separable convolution is the core move for small CNNs.

- **ShuffleNet** (Zhang et al., 2017) — `https://arxiv.org/abs/1707.01083`  
  Takeaway: grouped pointwise convolutions plus channel shuffle reduce compute. Possibly useful if our pointwise layers dominate params.

- **MobileNetV2** (Sandler et al., 2018) — `https://arxiv.org/abs/1801.04381` / CVPR PDF found by Sherlock.  
  Takeaway: inverted residuals and linear bottlenecks are the default student block template.

- **ShuffleNetV2** (Ma et al., 2018) — `https://arxiv.org/abs/1807.11164`  
  Takeaway: practical mobile efficiency depends on memory access and fragmentation, not only FLOPs. Keep block design simple.

- **MnasNet** (Tan et al., 2018) — `https://arxiv.org/abs/1807.11626`  
  Takeaway: mobile architectures benefit from platform-aware depth/width/resolution tradeoffs.

- **MobileNetV3** (Howard et al., 2019) — `https://arxiv.org/abs/1905.02244`  
  Takeaway: inverted residual + SE + hard-swish is a strong small-model recipe. For this assignment, SiLU/ReLU6 may be simpler; SE/ECA is worth testing.

- **EfficientNet** (Tan and Le, 2019) — `https://arxiv.org/abs/1905.11946`  
  Takeaway: compound scaling says resolution/width/depth must be tuned together. For our student, sweep input size 96/128/160 with width multiplier.

- **GhostNet** (Han et al., 2019/2020) — `https://arxiv.org/abs/1911.11907`  
  Takeaway: many feature maps are redundant; cheap operations can generate extra channels. Possible alpha, but implementation cost is higher than MobileNet blocks.

- **Squeeze-and-Excitation Networks** (Hu et al., 2017) — `https://arxiv.org/abs/1709.01507`  
  Takeaway: channel attention can help food classes where color/texture channels matter; parameter cost is small if reduction is high.

- **ECA-Net** (Wang et al., 2019/2020) — `https://arxiv.org/abs/1910.03151`  
  Takeaway: attention without dimensionality reduction; lower-risk than SE under 500K params.

- **RepVGG** (Ding et al., 2021) — `https://arxiv.org/abs/2101.03697`  
  Takeaway: train-time multi-branch blocks can be reparameterized to simple inference convs. Interesting, but reparameterization code increases risk.

- **ConvNeXt** (Liu et al., 2022) — `https://arxiv.org/abs/2201.03545`  
  Takeaway: modernized ConvNets use large kernels, inverted bottlenecks, GELU/LayerNorm. A tiny ConvNeXt-like student could work, but BatchNorm MobileNet-style is safer for small data.

### 15.3 Distillation and teacher-student papers

- **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015) — `https://arxiv.org/abs/1503.02531`  
  Takeaway: soft teacher distributions and temperature are the foundation of the assignment's KD loss.

- **FitNets** (Romero et al., 2014/2015) — `https://arxiv.org/abs/1412.6550`  
  Takeaway: intermediate feature matching can help thin/deep students. Could add a projection head during training only, but do not submit extra heads.

- **Paying More Attention to Attention** (Zagoruyko and Komodakis, 2016) — `https://arxiv.org/abs/1612.03928`  
  Takeaway: match teacher/student attention maps, not just logits. Moderate implementation cost; potentially useful if logits alone plateau.

- **Relational Knowledge Distillation** (Park et al., 2019) — `https://arxiv.org/abs/1904.05068`  
  Takeaway: preserve sample-to-sample relations. Could help with food classes, but batch-size sensitivity makes it second priority.

- **Born Again Neural Networks** (Furlanello et al., 2018) — `https://arxiv.org/abs/1805.04770`  
  Takeaway: self-distillation can improve even same-size students. After a strong student is trained, use it as another teacher.

- **Knowledge Distillation via Route Constrained Optimization / Teacher Assistant KD** (Mirzadeh et al., 2019) — `https://arxiv.org/abs/1902.03393`  
  Takeaway: very large teacher to very small student can be hard; an intermediate teacher/student can bridge the gap. Practical version: distill from ConvNeXt/EfficientNet teacher into a ~1M assistant, then into <500K final student.

- **Noisy Student** (Xie et al., 2020) — `https://arxiv.org/abs/1911.04252`  
  Takeaway: train teacher on labeled data, generate pseudo-labels on unlabeled data, train noisy student on both. Directly maps to this assignment.

### 15.4 Semi-supervised learning papers

- **Pseudo-Label** (Lee, 2013) — `http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf`  
  Takeaway: high-confidence model predictions can become labels; simple but prone to confirmation bias.

- **Temporal Ensembling** (Laine and Aila, 2016) — `https://arxiv.org/abs/1610.02242`  
  Takeaway: consistency across training epochs improves SSL. Less convenient than EMA/FixMatch.

- **Mean Teacher** (Tarvainen and Valpola, 2017) — `https://arxiv.org/abs/1703.01780`  
  Takeaway: EMA teacher stabilizes pseudo-targets. For us, maintain EMA student and evaluate/save EMA weights.

- **MixMatch** (Berthelot et al., 2019) — `https://arxiv.org/abs/1905.02249`  
  Takeaway: guessed labels + MixUp + consistency. Good conceptual source, but more complex.

- **UDA: Unsupervised Data Augmentation** (Xie et al., 2019) — `https://arxiv.org/abs/1904.12848`  
  Takeaway: enforce predictions under strong augmentation. Use KL between weak and strong views.

- **ReMixMatch** (Berthelot et al., 2019) — `https://arxiv.org/abs/1911.09785`  
  Takeaway: distribution alignment and augmentation anchoring can help. Since our classes are balanced in labeled data, distribution alignment to uniform may be useful for unlabeled pseudo-label balance.

- **FixMatch** (Sohn et al., 2020) — `https://arxiv.org/abs/2001.07685`  
  Takeaway: weak augmentation creates high-confidence pseudo-label; strong augmentation learns from it. Sherlock found strong evidence here. This is one of the highest practical alpha sources.

### 15.5 Augmentation and regularization papers

- **MixUp** (Zhang et al., 2017) — `https://arxiv.org/abs/1710.09412`  
  Takeaway: linear interpolation of images/labels regularizes small datasets.

- **Random Erasing** (Zhong et al., 2017) — `https://arxiv.org/abs/1708.04896`  
  Takeaway: occlusion robustness; useful for dish photos with clutter.

- **AutoAugment** (Cubuk et al., 2018/2019) — `https://arxiv.org/abs/1805.09501`  
  Takeaway: learned augmentation policies work but require search; not ideal for this small assignment.

- **CutMix** (Yun et al., 2019) — `https://arxiv.org/abs/1905.04899`  
  Takeaway: patch mixing improved ImageNet ResNet-50 in the paper; Sherlock found direct evidence. Test carefully because food labels can be composition-sensitive.

- **RandAugment** (Cubuk et al., 2019/2020) — `https://arxiv.org/abs/1909.13719`  
  Takeaway: no learned policy; just tune number/magnitude. High priority for this assignment.

- **AugMix** (Hendrycks et al., 2019) — `https://arxiv.org/abs/1912.02781`  
  Takeaway: improves robustness to corruptions; useful if hidden test has lighting/camera variation.

- **TrivialAugment** (Müller and Hutter, 2021) — `https://arxiv.org/abs/2103.10158`  
  Takeaway: extremely simple augmentation policy can compete with heavier methods. Good fallback if RandAugment tuning is unstable.

- **Label smoothing / Rethinking Inception** (Szegedy et al., 2015) — `https://arxiv.org/abs/1512.00567`  
  Takeaway: reduces overconfidence; use for teacher training, but be cautious because KD already softens targets.

- **Stochastic Depth** (Huang et al., 2016) — `https://arxiv.org/abs/1603.09382`  
  Takeaway: regularizes deep residual nets. For our small MobileNet-style student, drop path may help if network is deep enough.

### 15.6 Training, averaging, and calibration papers

- **Snapshot Ensembles** (Huang et al., 2017) — `https://arxiv.org/abs/1704.00109`  
  Takeaway: use multiple checkpoints as a teacher ensemble for logits, not at submission.

- **Deep Ensembles** (Lakshminarayanan et al., 2016/2017) — `https://arxiv.org/abs/1612.01474`  
  Takeaway: ensemble teachers are better calibrated; allowed if only distilled logits are used.

- **Temperature scaling / calibration** (Guo et al., 2017) — `https://arxiv.org/abs/1706.04599`  
  Takeaway: teacher calibration matters; measure confidence and validation CE, not only accuracy.

- **SWA** (Izmailov et al., 2018) — `https://arxiv.org/abs/1803.05407`  
  Takeaway: weight averaging finds flatter optima; Sherlock found direct evidence of ImageNet/CIFAR gains. Use `torch.optim.swa_utils` or manual averaging.

- **Model Soups** (Wortsman et al., 2022) — `https://arxiv.org/abs/2203.05482`  
  Takeaway: averaging fine-tuned weights can improve robustness. For final student, try averaging compatible checkpoints from same architecture.

- **EMA of weights** (Morales-Brotons et al., 2024) — `https://arxiv.org/abs/2411.18704`  
  Takeaway: EMA is a standard low-cost stabilizer. Keep EMA during student training and compare raw vs EMA validation.

### 15.7 Ranked experiment roadmap from research

| Priority | Experiment | Expected gain | Risk | Cost |
|---:|---|---|---|---|
| 1 | Build MobileNetV2/V3-like student at 128px, 450K-490K params | High | Low | Medium |
| 2 | Add strong but sane augmentation: crop, flip, color jitter, RandAugment/TrivialAugment, RandomErasing | High | Medium if too strong | Low |
| 3 | Train strong teacher and dump TTA logits for unlabeled + labeled | High | Teacher overfit | Medium/high |
| 4 | KD sweep `T=1,2,4,8`, `alpha=0.5,0.7,0.9` | High | Time | Low/medium |
| 5 | FixMatch-style high-confidence unlabeled CE plus KD | Medium/high | Confirmation bias | Medium |
| 6 | EMA/SWA student checkpoint averaging | Medium | Minimal | Low |
| 7 | Multi-teacher or snapshot-teacher averaged logits | Medium | More compute | Medium |
| 8 | MixUp/CutMix for teacher and/or student | Medium | Can distort food semantics | Low |
| 9 | SE/ECA attention inside student | Low/medium | Slight overfit/params | Low |
| 10 | Attention/feature distillation | Low/medium | Complexity | Medium |

### 15.8 Warnings from the literature for this assignment

- **Public leaderboard overfitting:** With 399 labels, public LB can be noisier than local CV. Do not chase every submission fluctuation.
- **Teacher overconfidence:** KD with a bad or overconfident teacher can hurt. Track validation CE and confidence, not only teacher accuracy.
- **Too-high temperature:** `T=8+` may flatten useful class information. It must be tested, not assumed.
- **Pseudo-label confirmation bias:** Use confidence thresholds, class-balance checks, and strong augmentation consistency.
- **Over-augmentation:** Vertical flips, extreme rotations, or aggressive color shifts may make unrealistic food photos.
- **Architecture cleverness risk:** GhostNet/RepVGG/feature-KD may help, but the first winning implementation should be reliable MobileNet-style.
- **Rule boundary:** Teacher ensembles, TTA, and pretrained backbones are allowed only as training tools. The submitted `model.pt` must contain only the <500K student.

---

## 16. Lean workflow and anti-leaderboard protocol

We will keep the experimentation phase intentionally lean:

```text
experiment.py   # all datasets, models, recipes, training, CV, KD, export
log.txt         # append-only JSONL log of every run and manual leaderboard notes
model.pt        # current submission candidate, overwritten only by intentional exports
```

No experiment-folder sprawl. No dozens of config files. Recipes live inside `experiment.py` as Python dictionaries unless the file becomes truly unmanageable.

### Current code status

Implemented:

- `experiment.py`
- `log.txt`
- `model.pt`

Current first submission/protocol-check recipe:

| Field | Value |
|---|---|
| Recipe | `first_submit` |
| Model | conservative plain CNN |
| Params | 463,847 |
| Input | server 256 resized internally to 128 |
| Local split | stratified 80/20 seed 0 |
| Local val acc | 0.4935 |
| Local val loss | 1.4339 |
| Best epoch | 66 |
| Final export | retrained on all 399 labeled images for 66 epochs |
| Output | `model.pt` |

This model is **not** our winning strategy. It is only a server/protocol sanity check: TorchScript loads, input/output contract works, and parameter count is legal.

### The Kaggle-trap rule

We already know the main danger: tuning to the public leaderboard instead of learning a model that generalizes. Therefore:

1. Public leaderboard score is a weak external sanity check, not the optimization target.
2. We do not modify hyperparameters just because one leaderboard submission moved up or down.
3. We submit only after a local validation/CV improvement.
4. Every submission must correspond to a frozen recipe in `experiment.py` and a record in `log.txt`.
5. If local CV and public LB disagree, assume public LB may be noisy unless repeated evidence says otherwise.

### Real model-selection target

The target is:

```text
high repeated-stratified-CV accuracy
+ low validation cross-entropy
+ stable validation confidence/calibration
+ sensible unlabeled pseudo-label distribution
+ legal <500K submitted student
```

The final model should be so good locally and so well-regularized that submitting it is simply verification, not gambling.

### Required next step: trustworthy local CV

Before chasing architecture or teacher tricks, add proper cross-validation to `experiment.py`:

```bash
python3 experiment.py cv --recipe first_submit --folds 7
```

Why 7 folds:

- There are exactly 57 images per class.
- With 7 folds, each validation fold has about 8 or 9 examples per class.
- This gives much better signal than one arbitrary 80/20 split.

Each CV run must log to `log.txt`:

```json
{
  "type": "cv_run",
  "recipe": "first_submit",
  "folds": 7,
  "params": 463847,
  "fold_accs": [...],
  "mean_val_acc": 0.0,
  "std_val_acc": 0.0,
  "mean_val_loss": 0.0,
  "notes": "..."
}
```

### Development phases from here

#### Phase 1: Measurement discipline

Add and run 7-fold stratified CV for the current recipe.

Goal:

- Establish whether the current model is truly around 49%, or if the seed-0 split is pessimistic/optimistic.
- Create the baseline that all future recipes must beat.

#### Phase 2: Strong supervised student

Before teacher/KD, find a much stronger legal student by CV.

Candidate sweeps:

- `plain_cnn` vs `micro_mobilenet`
- input size 128 vs 160
- augmentation `basic` vs `strong`
- label smoothing 0, 0.03, 0.07
- MixUp off vs 0.1/0.2
- weight decay 1e-4, 5e-4, 1e-3
- dropout 0.2, 0.35, 0.5

Selection rule:

1. Highest mean CV accuracy.
2. If close, lower mean CV loss.
3. If still close, lower CV std and better calibration.

#### Phase 3: Teacher and unlabeled data

This is likely where winning performance comes from.

Use a pretrained teacher only during training:

- EfficientNet/ConvNeXt/ResNet teacher.
- Fine-tune with strong augmentations.
- Dump logits for both labeled and unlabeled images.
- Optionally average teacher TTA logits.

Then distill into the <500K student:

- CE on labeled examples.
- KL to teacher logits on labeled and unlabeled examples.
- Optional high-confidence pseudo-label CE.
- Required temperature sweep: `T = 1, 2, 4, 8`.
- Alpha sweep after T: `alpha = 0.5, 0.7, 0.9`.

Important: teacher checkpoints/logits are training artifacts only. They are not submitted.

#### Phase 4: Final recipe freeze

Once CV selects a winner, freeze:

- student architecture
- input size
- augmentation policy
- optimizer/schedule
- teacher recipe
- KD temperature
- KD alpha
- epoch policy
- seed policy

Then train final model on all labeled data plus unlabeled KD and export one final `model.pt`.

### Manual leaderboard logging

When we do submit, manually append a line to `log.txt` like:

```json
{"type":"leaderboard","recipe":"first_submit","lb_score":0.0,"notes":"protocol check only"}
```

Leaderboard scores are useful for calibration against the hidden evaluation system, but they do not drive recipe search.

### 16.1 CV baseline result after implementation

Implemented `experiment.py cv` and ran:

```bash
python3 experiment.py cv --recipe first_submit --folds 7
```

Result for the first protocol-check supervised model:

| Metric | Value |
|---|---:|
| Recipe | `first_submit` |
| Params | 463,847 |
| Folds | 7 stratified |
| Epochs per fold | 70 |
| Fold accuracies | `[0.5556, 0.4464, 0.5179, 0.5179, 0.5536, 0.4286, 0.5357]` |
| Mean CV accuracy | 0.5079 |
| Std CV accuracy | 0.0507 |
| Mean CV loss | 1.4792 |

Interpretation:

- The single seed-0 80/20 score of 0.4935 was representative.
- The current supervised-only baseline is real but nowhere near a winning solution.
- We now have the local measurement discipline needed to avoid leaderboard chasing.
- Next improvements must beat roughly **0.508 mean CV accuracy** with lower CV loss before they deserve a leaderboard submission.

---

## 17. First teacher + distillation test

Implemented the first version of the high-alpha path in `experiment.py`:

```bash
python3 experiment.py kd --recipe kd_v1_t4
```

Implementation details:

- Installed `timm` for teacher training only.
- Teacher: pretrained `efficientnet_b0`, submitted nowhere.
- Teacher input: 224px, ImageNet normalization inside teacher wrapper.
- Teacher training: 15 epochs on the seed-0 stratified train split with strong augmentation.
- Teacher logits collected **in memory only** for:
  - all 399 labeled images
  - all 798 unlabeled images
- Student: same legal `plain_cnn` as `first_submit`, 463,847 params.
- Student KD objective:
  - hard CE on labeled examples
  - KL distillation from teacher logits on labeled examples
  - KL distillation from teacher logits on unlabeled examples
- Temperature: `T=4`
- Alpha: `0.70`
- Output: `model.pt` overwritten with the best split-validation KD student.

### 17.1 Result on seed-0 split

| Metric | Supervised baseline `first_submit` | KD test `kd_v1_t4` |
|---|---:|---:|
| Student params | 463,847 | 463,847 |
| Teacher | none | EfficientNet-B0 pretrained/fine-tuned |
| Teacher best val acc | - | 0.7532 |
| Teacher best val loss | - | 0.7949 |
| Student best val acc | 0.4935 | 0.5844 |
| Student best val loss | 1.4339 | 1.2253 |
| Student best epoch | 66 | 51 |

This is the first clear evidence that teacher + unlabeled distillation is real alpha for this assignment.

### 17.2 Unlabeled teacher pseudo-label distribution

Teacher mean confidence on unlabeled images: `0.7711`.

Teacher pseudo-label counts across 798 unlabeled images:

```text
[110, 99, 103, 125, 132, 115, 114]
```

Interpretation:

- The distribution is reasonably balanced, not collapsed to one or two classes.
- This supports using the unlabeled set for KD/pseudo-labeling.
- Teacher confidence is high enough to be useful but not so high that temperature analysis becomes irrelevant.

### 17.3 What this means

The KD path improved the same student architecture from about **49.4%** to **58.4%** on the seed-0 validation split, and reduced validation loss substantially.

However, per the anti-Kaggle protocol, this is not enough to declare victory. The next required step is to validate KD with CV or repeated splits:

```bash
python3 experiment.py cv/kd-style evaluation for kd_v1_t4
```

Because full KD CV is more expensive, we should first run a small repeated-split KD test or implement a `kd_cv` command that trains teacher+student per fold and logs mean/std.

### 17.4 Immediate next experiments

1. Temperature sweep on same split:
   - `T=1`
   - `T=2`
   - `T=4`
   - `T=8`

2. Alpha sweep after choosing T:
   - `alpha=0.5`
   - `alpha=0.7`
   - `alpha=0.9`

3. CV/repeated split confirmation:
   - At minimum 3 folds/seeds for KD.
   - Ideally 7-fold KD CV once compute time is acceptable.

4. Stronger teacher:
   - `convnext_tiny` or `efficientnet_b2/b3` if compute allows.
   - Teacher ensemble logits if multiple teachers help.

---

## 18. Required validation stress tests

We will not use outside image datasets for validation or model selection. Instead, every validation run must now include robustness checks on the provided validation fold.

Implemented in `experiment.py`:

```text
STRESS_TESTS = [
  brightness_down,
  brightness_up,
  contrast_down,
  contrast_up,
  mild_blur,
  crop_90,
  jpeg_55,
  rotate_left,
  rotate_right,
]
```

These are deterministic transformations applied only to the provided labeled validation images. They simulate camera/production-line variation without introducing outside data.

### 18.1 Requirement for all future runs

Every validation-bearing run must log:

- clean validation accuracy/loss
- stress mean accuracy/loss
- stress worst-case accuracy
- per-stress-test accuracy/loss/confidence

This is now wired into:

- `experiment.py run`
- `experiment.py cv`
- `experiment.py kd`

Final-all training has no validation fold, so it cannot compute stress tests directly. The source validation run used to justify that final export must contain stress metrics.

### 18.2 Current KD model stress check

Evaluated current `model.pt` from `kd_v1_t4` on the seed-0 validation fold.

| Metric | Value |
|---|---:|
| Clean val acc | 0.5844 |
| Clean val loss | 1.2253 |
| Stress mean acc | 0.5339 |
| Stress mean loss | 1.2578 |
| Stress worst acc | 0.4416 |

Per-stress accuracy:

| Stress | Accuracy |
|---|---:|
| brightness_down | 0.5455 |
| brightness_up | 0.5065 |
| contrast_down | 0.5584 |
| contrast_up | 0.4416 |
| mild_blur | 0.5325 |
| crop_90 | 0.5325 |
| jpeg_55 | 0.5455 |
| rotate_left | 0.5844 |
| rotate_right | 0.5584 |

Interpretation:

- KD model is reasonably stable to blur, crop, JPEG, and small rotations.
- Contrast increase is currently the weakest stress case.
- Future models should improve clean CV while avoiding collapse in stress mean/worst accuracy.

---

## 19. Generalization experiment: robust augmentation KD

To improve generalization and stress-test robustness, added recipe:

```bash
python3 experiment.py kd --recipe kd_robust_t4
```

Changes vs `kd_v1_t4`:

- Student augment: `robust`
  - stronger crop range
  - stronger brightness/contrast/saturation jitter
  - small rotations
  - occasional blur
  - occasional JPEG compression
  - random erasing
- Teacher augment: `robust`
- Alpha reduced from 0.70 to 0.65
- Dropout increased from 0.35 to 0.40
- Weight decay increased from 5e-4 to 7e-4

### 19.1 Result

| Metric | `kd_v1_t4` | `kd_robust_t4` |
|---|---:|---:|
| Teacher val acc | 0.7532 | 0.7143 |
| Teacher val loss | 0.7949 | 0.8815 |
| Student clean val acc | 0.5844 | 0.4935 |
| Student clean val loss | 1.2253 | 1.3973 |
| Student stress mean acc | 0.5339 | 0.4791 |
| Student stress worst acc | 0.4416 | 0.4416 |

Interpretation:

- The robust augmentation recipe was too aggressive for this tiny labeled set.
- It hurt the teacher, then hurt the student.
- It did not improve worst-case stress accuracy.
- Conclusion: **do not use `kd_robust_t4` as a candidate**.

Action taken:

- Re-ran `kd_v1_t4` and restored `model.pt` to the better KD model.

### 19.2 Lesson

More augmentation is not automatically better. For this dataset, the teacher is the foundation; if augmentation weakens teacher validation performance, downstream distillation also weakens.

Next generalization strategy should be less blunt:

1. Keep teacher augmentation at `strong`, not `robust`.
2. Try only student-side mild contrast-focused augmentation or regularization.
3. Try temperature/alpha sweep before heavier augmentation.
4. Consider EMA/SWA for student because it can improve robustness without distorting images.

---

## 20. Three promising branch tests

Implemented and ran a shared-teacher branch runner:

```bash
python3 experiment.py branches
```

This trains the EfficientNet-B0 teacher once, collects labeled/unlabeled logits once, then trains three student variants from the same teacher signal.

Branches tested:

1. `branch_t2_a07` — lower temperature, `T=2`, `alpha=0.7`
2. `branch_t4_a05` — lower teacher weight, `T=4`, `alpha=0.5`
3. `branch_t4_a07_ema` — EMA student weights, `T=4`, `alpha=0.7`, `ema_decay=0.995`

### 20.1 Branch results

| Recipe | Clean val acc | Clean val loss | Stress mean acc | Stress worst acc | Best epoch |
|---|---:|---:|---:|---:|---:|
| `kd_v1_t4` current best | 0.5844 | 1.2253 | 0.5339 | 0.4416 | 51 |
| `branch_t2_a07` | 0.5584 | 1.3374 | 0.5137 | 0.4545 | 61 |
| `branch_t4_a05` | 0.5065 | 1.4099 | 0.4690 | 0.4156 | 65 |
| `branch_t4_a07_ema` | 0.4156 | 1.5578 | 0.4185 | 0.3766 | 70 |

### 20.2 Interpretation

- `T=2` was the best of the three new branches, and slightly improved stress worst accuracy, but it lost too much clean accuracy and stress mean accuracy relative to `kd_v1_t4`.
- `alpha=0.5` underused the teacher signal and performed worse.
- EMA with `ema_decay=0.995` performed badly; likely too sluggish for only 70 epochs / tiny dataset. If retried, use lower decay or EMA warmup, but it is not priority now.
- The current best remains `kd_v1_t4`.

Action taken:

- Restored `model.pt` by rerunning:

```bash
python3 experiment.py kd --recipe kd_v1_t4
```

So the current `model.pt` is again the best known KD candidate.

### 20.3 Lessons

- Temperature matters: `T=4` still looks better than `T=2` on clean validation and stress mean.
- The teacher is valuable: reducing alpha from 0.7 to 0.5 hurt.
- Naive EMA is not automatically useful; with tiny data, it can lag too much.

Next branches should focus on:

1. Stronger teacher, not weaker distillation.
2. Full KD CV/repeated splits for `kd_v1_t4`.
3. Possibly `T=6` or `alpha=0.8/0.9`, because lower T/alpha did not win.

---

## 21. ChatGPT Pro recommendations implemented: teacher preprocessing/TTA, 160px student, ECA head

Based on the external review, implemented several high-EV changes:

1. **Model-specific teacher preprocessing** using `timm.data.resolve_model_data_config()`.
2. **Teacher TTA logits** for labeled and unlabeled images:
   - clean
   - horizontal flip
   - 90% center crop resized back
   - contrast up 1.15
   - contrast down 0.90
3. **Normalized KD loss**:

```python
kd = (w_lab * kd_lab + w_un * kd_un) / (w_lab + w_un)
loss = (1 - alpha) * CE + alpha * kd
```

4. **Use all unlabeled images per epoch** by cycling the shorter loader and using `steps = max(len(labeled_loader), len(unlabeled_loader))`.
5. **160px student input** while keeping parameter count unchanged for `plain_cnn`.
6. **New `PlainCNNECAHead` student**:
   - current plain CNN backbone
   - ECA channel attention after each stage
   - 1x1 head expansion to 224 channels
   - 493,651 params, still under 500K

### 21.1 Results

| Recipe | Teacher | Student | Params | Clean val acc | Clean val loss | Stress mean acc | Stress worst acc | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| `kd_v1_t4` | EffNet-B0 no TTA | plain 128 | 463,847 | 0.5844 | 1.2253 | 0.5339 | 0.4416 | previous best |
| `kd_b0_tta_plain160` | EffNet-B0 + TTA | plain 160 | 463,847 | 0.6364 | 1.1707 | 0.6017 | 0.5584 | large improvement |
| `kd_b0_tta_eca160` | EffNet-B0 + TTA | ECA-head 160 | 493,651 | 0.6753 | 1.2307 | 0.6248 | 0.5844 | new best accuracy/robustness |
| `kd_convnext_tta_eca160` | ConvNeXt-Tiny + TTA | ECA-head 160 | 493,651 | 0.4935 | 1.6755 | 0.4545 | 0.4156 | teacher failed to fine-tune |

### 21.2 Interpretation

The highest-value improvements were exactly the clean ones suggested by the external review:

- 160px input helped substantially.
- TTA teacher logits helped.
- Normalized KD with higher alpha worked.
- ECA + 1x1 head improved clean validation and stress robustness.

ConvNeXt-Tiny failed badly in this setup:

```text
teacher_best_val_acc = 0.1818
unlabeled_pseudo_counts = [148, 0, 114, 450, 0, 0, 86]
```

This teacher collapsed and should not be used without debugging/freeze schedule/LR changes. EfficientNet-B0 remains the reliable teacher.

### 21.3 Current best model

Restored `model.pt` to the best recipe by rerunning:

```bash
python3 experiment.py kd --recipe kd_b0_tta_eca160
```

Current `model.pt`:

```text
recipe:            kd_b0_tta_eca160
params:            493,651
clean val acc:     0.6753
clean val loss:    1.2307
stress mean acc:   0.6248
stress worst acc:  0.5844
```

This is a major improvement over the previous KD best and over the supervised CV baseline. It still needs KD repeated-split/CV confirmation before finalizing.
