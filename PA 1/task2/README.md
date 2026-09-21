# Task 2: Unsupervised Domain Adaptation (PACS, Sketch target)

### AI Progress Report

Notes to fold into the report / README proper. Newest last.

**PACS source (mirror).** PACS is not in torchvision. Used the HuggingFace release
`flwrlabs/pacs` (single parquet, 183 MB) rather than the original Google Drive archive:
9,991 images, 4 domains x 7 classes, verified against the standard PACS counts (sketch
house = 80, person = 160). Labels are `dog, elephant, giraffe, guitar, horse, house,
person` (index order from the dataset card). Stored in the shared cache
`ATML/datasets/pacs/`, outside Git. Reading it requires `pyarrow`.

**Shared protocol (Tasks 2 and 3).** `shared/pacs.py` + `shared/pacs_protocol.py`.
Stratified 80/20 per source domain at seed 6304, written once to
`shared/splits/pacs_sketch_seed6304.json` and reused by both tasks: photo 1337/333,
art_painting 1638/410, cartoon 1875/469; target (sketch) 3929 images. One epoch = a pass
over the largest source loader = 234 steps; each step takes 8 images from each source
domain (24 total) plus 24 target images.

**BatchNorm policy verified.** After `train_mode()`, all BN modules report
`training == False` (running stats frozen at ImageNet values) while gamma/beta keep
`requires_grad == True`, as the manual requires.

**DEVIATION - domain discriminator learning rate (10x).** With every parameter at AdamW
1e-4, **DANN diverged inside epoch 1** (classification loss 376, later ~10^5; source-val
macro-F1 8.09) and **CDAN collapsed after epoch 1** (31.32). Diagnosis, in order:
(1) the gradient-reversal layer is correct - a unit test returns exactly -alpha;
(2) feature norm |f| climbed 26 -> 61 -> 8,520, i.e. the backbone was maximising the
domain loss by inflating feature magnitude, since a linear discriminator's logits scale
with |f|; (3) making the discriminator *weaker* (lr 1e-5) made it far worse
(|f| ~ 2.5e13), proving the discriminator was failing to keep up rather than pushing too
hard. **Why here specifically:** the manual's frozen BatchNorm statistics remove the
renormalisation that normally bounds activation scale, leaving the adversarial game a
free direction to run away in. **Fix:** the domain discriminator uses lr 1e-3 (10x the
backbone), which is Ganin et al.'s original practice for newly initialised layers. Over
700 steps both DANN and CDAN are then stable (|f| ~ 65, domain accuracy 0.46-0.58, low
classification loss). Gradient clipping at 1.0 fixed CDAN but **not** DANN, so it is not
used. Backbone and classifier remain at 1e-4 for **every** method, so the shared pipeline
is unchanged; the 10x rate touches only a module that exists in DANN/CDAN alone.

**Failed runs kept as evidence.** The equal-learning-rate runs are archived in
`results/failed_equal_lr/` (histories + log) rather than deleted, in case the divergence
is worth showing.

**No target metric is computed during training.** Checkpoint selection uses mean source
-validation macro-F1 only; target labels are touched for the first time in
`evaluate_final.py`, after all checkpoints are fixed.

**Timing.** The full six-run suite (source_only, dan, dann, cdan, dan lambda 0.1,
dan lambda 10) takes ~27-35 min total on the RTX 4050, ~22 s/epoch for source-only and
~33 s/epoch for the adaptation methods.
