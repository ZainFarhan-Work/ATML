# Task 4: Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknowns)

### AI Progress Report

Notes to fold into the report / README proper. Newest last.

**DEVIATION - CIFAR source (mirror).** The official host `cs.toronto.edu` stalled
completely (0 bytes in 30 s), so `torchvision.datasets.CIFAR10/100` could not download.
Used the official University of Toronto datasets as republished on HuggingFace:
`uoft-cs/cifar10` and `uoft-cs/cifar100` (parquet, ~144 MB each, fetched in ~1 min).
Verified: CIFAR-10 train 50,000 with exactly 5,000 per class; CIFAR-100 test 10,000 with
exactly 100 per fine class, so the 8 near + 8 far unknown classes give 800 images each as
the manual states. Parquet is not the format torchvision reads, so `data/cifar10.py`
implements the loader (same pattern as the PACS loader in Task 2). Stored in
`ATML/datasets/cifar/`, outside Git.

**Unknown groups (fixed, not revisable).** Near = bus(13), pickup_truck(58),
motorcycle(48), tractor(89), wolf(97), fox(34), leopard(42), camel(15). Far = bottle(9),
bowl(10), chair(20), clock(22), keyboard(39), mushroom(51), sunflower(82), wardrobe(94).
Indices are the official CIFAR-100 fine-label ordering. CIFAR-100 **test** split only;
no CIFAR-100 image is loaded by `train.py`.

**Splits.** Stratified 90/10 of the CIFAR-10 training partition, seed 6304 -> 45,000
train / 5,000 validation, saved to `results/splits/cifar10_seed6304.json`. Checkpoints
are selected on CIFAR-10 validation accuracy only.

**Performance change (no effect on the objective).** Per-item PNG decoding made training
data-loader bound at 0.183 s/step (~107 min per 100-epoch run). Images are now decoded
once into a uint8 NCHW tensor cached at `cache/cifar10_train_u8.npy` (184 MB) and the
augmentation pipeline runs on uint8 tensors instead of PIL images: 0.098 s/step, ~34
s/epoch, ~57 min per run. Same pixels, same augmentations, same order.

**Optional RPL extension: not implemented** (time). Vanilla, GCSC and PROSER only.

**PROSER implementation notes.** 5 dummy classifiers on the 512-d feature; the dummy
column used in the losses and in the detection score is the *strongest* dummy response.
Classifier-placeholder loss = CE over [10 known | dummy] + beta * CE over the same
columns with the true class masked to -inf and the dummy as target (beta = 1).
Data-placeholder loss = gamma * CE([10 known | dummy], dummy) on manifold-mixup features
mixed after layer2 with lambda ~ Beta(2,2) between *different-class* pairs (gamma = 0.1).
Each batch is split in half: first half classifier placeholders, second half data
placeholders. Initialised from the selected Vanilla checkpoint; 50 epochs, SGD lr 1e-3,
momentum 0.9, weight decay 5e-4, cosine, batch 128, seed 6304. Known-class logits are
never altered, so CSA is computed from those 10 columns alone.

**Timing.** vanilla ~57 min, gcsc ~60 min (RandAugment), proser ~35 min (50 epochs) =
~2.5 h total on the RTX 4050.
