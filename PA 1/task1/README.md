# Task 1: Inductive Biases and Representations

rejection rule

1. The content object can no longer be identified from its outline alone → shape cue destroyed.
2. The texture still resembles the content image's own texture closely enough to recognize it by that alone → style never took.
3. Artifacts render the image unrecognizable to a human eye.

### AI Progress Report

Notes to fold into the report / README proper. Newest last.

**Dataset download (mirror).** The official STL-10 host (`ai.stanford.edu`) served at
~32 KiB/s, i.e. ~23 h for the 2.6 GB archive. Measured 4.5-6 MB/s from the
HuggingFace mirror `xingslong/stl10_binary`, so `stl10_binary.tar.gz` was fetched from
there instead (~7 min). It is byte-identical to the official file: MD5
`91f7769df0f17e558f3565bffb0c7dfb`, which is the checksum torchvision itself verifies.
The dataset lives in the shared cache `ATML/datasets/`, outside Git, so later PAs reuse
it.

**CLIP config bug (affects reported numbers).** The `openai` CLIP weights were trained
with QuickGELU activations. OpenCLIP's plain `ViT-B-32` config substitutes standard
GELU and only emits a warning, which silently degraded zero-shot top-1 from **97.0 to
93.6**. We therefore build `ViT-B-32-quickgelu` with `pretrained='openai'`; this is the
same model the manual specifies, just OpenCLIP's faithful config name.

**Confidence is not comparable across backbones.** CLIP's image embedding is
L2-normalized (unit length), so its linear head's logits are small and the softmax is
flat: mean max confidence 0.27 versus ~0.91-0.97 for ResNet/ViT. This is a feature-scale
artifact, not lower certainty. Compare each model against its own clean baseline.

**Clean baseline is saturated.** All four systems score 96-97% top-1 on STL-10, and
macro-F1 tracks accuracy because the evaluation subset is exactly 50 images per class.
The signal in later steps is therefore in the *drop* from each model's own baseline, not
in absolute numbers.

**Design decision - color interventions.** Grayscale is the required common
intervention; the chosen additional one is a **fixed +40 degree hue rotation** (raised
from 30 degrees, where the recolor was too subtle on several eval images). Grayscale
*removes* color, the hue rotation *changes* it while preserving luminance, saturation
and geometry - so together they separate "needs color at all" from "needs the correct
color".

**Limitation - hue rotation is saturation-dependent.** Hue is meaningless for
near-achromatic pixels, so the rotation leaves low-saturation images (one eval image is
essentially grayscale already) effectively unchanged. Those images are unchanged by
construction, which inflates prediction consistency under the hue condition and makes
the measured hue effect a lower bound on colour reliance. Raising the angle does not fix
this.

**Design decision - intervention boundary.** All interventions are applied to the shared
224x224 RGB tensor in [0, 1], before each backbone's own normalization. Every model
therefore sees pixel-identical images, which is what makes the cross-model comparison
valid.

**External code attribution (required).** The AdaIN implementation follows Huang &
Belongie (2017); the encoder/decoder architecture and the pretrained weights
(`vgg_normalized.pth`, `decoder_final.pth`) come from the public reimplementation
`naoto0804/pytorch-AdaIN`, weights mirrored at `BhupatiNadar/adain-style-transfer-models`
on HuggingFace. Our `data/make_cue_conflicts.py` reimplements the encoder/decoder
`nn.Sequential` definitions and the AdaIN operation to load those weights.

**Design decision - style strength.** alpha = 1.0 (full stylization, the Geirhos
setting). The 0.5/0.75/1.0 sweep in `results/figures/adain_alpha_sweep.png` showed little
difference between values; style strength varies far more with *which* style image is
drawn than with alpha.

**Failed idea - style zoom (recorded, then reverted).** Centre-cropping the style image
to 25-60% and rescaling was tried to strengthen the texture cue. STL-10 sources are
natively 96x96, so a crop is ~24-58 px upscaled to 224: this only magnified blur,
weakened the style statistics further and destroyed content. Reverted; no crop is used.

**Cue-conflict generation and screening.** 5 unordered class pairs (airplane|cat,
car|horse, ship|dog, truck|bird, deer|monkey), both directions, 40 candidates per
direction = 400 total, seed 6304. Screening used a rejection rule fixed *before* any
model saw the images (see top of this file), applied by eye to content|style|conflict
triptychs. **Accepted 286, rejected 114 (28.5%).** No model predictions were used in
screening.

**Design decision - balanced scoring set.** Accepted counts per pair/direction ranged
22-38, so the primary result uses a **balanced subsample of 22 per cell = 220 images**
(seed 6304); all 286 accepted are reported as a robustness check. ResNet-50 shifts
50.9 -> 57.8 shape bias between the two, the largest of any system, so the balance
matters.

**Caveat - shape bias is inflated by a weak texture cue.** STL-10 style images are
96x96 whole-object photos, not the texture images Geirhos et al. used, so the texture
cue is weaker than in the original protocol. Absolute shape-bias values are therefore
higher than published ImageNet CNN numbers (ResNet-50 at ~51% vs Geirhos' ~20-30%); the
*ordering* across architectures (ResNet << ViT < CLIP) is the defensible claim.

**Caveat - coverage is ~47-59%.** Roughly half of predictions on cue conflicts fall
outside both intended classes, so shape bias is computed over about half the images.
This is itself a result: stylization pushes images far enough off-distribution that
models frequently answer a third class.

**Translation protocol.** Displacements 0/8/16/32 px in the four cardinal directions,
reflection padding then a shifted crop (a zero pad would add black borders, i.e. a second
intervention). Per-direction results are kept in `results/translation.csv`; the reported
curves average over directions.

**Result - ViT is the most translation-stable, ResNet is not.** Convolution provides
equivariance, not invariance, and stride/padding/pooling break it; ViT-B/16 held up best
(consistency 98.6% at 32 px vs ResNet-50's 97.7%). Zero-shot CLIP lost the most accuracy
(97.0 -> 94.8) - more than its own linear head (96.6 -> 95.7) - i.e. the same features
with a non-adapted decision rule.

**Noise floor.** The evaluation subset is 500 images, so 0.2 percentage points = 1 image.
Differences below ~1 point (5 images) are not interpretable, including the small
non-monotonic bumps at 16 px.

**Figure palette.** Charts use categorical slots 1-4 of the validated default palette in
fixed order (blue/orange/aqua/yellow), recessive grid and axis ink. The palette
validator (`node`) is not installed on this machine, so the shipped pre-validated slot
order was used unmodified rather than a hand-picked set.

**Patch shuffle protocol.** 4x4 pixel-space grid (16 patches of 56x56 on the 224 image),
one non-identity permutation per image from seed 6304, precomputed once and saved to
`results/splits/patch_permutations_seed6304.json` so every model sees pixel-identical
shuffled images. Verified by unit check: applying the inverse permutation restores the
original exactly.

**Result - patch shuffle is by far the strongest intervention.** Accuracy drops 6.8-12.6
points (vs ~2 for colour and ~1 for translation). CLIP is hit hardest (head -12.2,
zero-shot -12.6, consistency ~84%), ResNet-50 and ViT-B/16 least (-7.4/-6.8, consistency
90-93%).

**Result - the shape-bias ordering inverts under patch shuffle.** CLIP had the highest
cue-conflict shape bias (90-96%) yet degrades most when global layout is destroyed, while
the most texture-leaning model (ResNet-50, 50.9%) degrades least. Reading: CLIP's "shape"
preference is a preference for globally coherent object form, which patch shuffling
removes; texture-leaning evidence survives inside individual patches. ResNet vs ViT
differ by 0.6 points here = 3 images, i.e. within noise - do not claim a CNN/ViT split.

**Result - confidence survives the damage.** Mean max confidence falls only modestly
under shuffling (ResNet 0.91->0.87, ViT 0.97->0.89, zero-shot CLIP 0.94->0.82) while
accuracy falls 7-13 points: the models stay confident on images whose global structure is
destroyed.

**Representation stability protocol.** I_T = mean cosine between paired clean and
transformed features, for grayscale, translation (32 px, averaged over the four
directions), patch shuffle and cue conflict. Cue conflicts have two possible "clean"
counterparts, so both are recorded: **vs content** (the shape source, the headline) and
**vs style** (the texture donor, bookkeeping only). Conflict rows use the balanced
220-image set, same as the shape-bias headline. Source image indices were recovered by
replaying the generation draws (`source_indices()`), verified: 0/400 class mismatches and
a regenerated conflict matches its saved PNG to within 8-bit rounding.

**Critical caveat - raw cosines are NOT comparable across backbones.** The random-pair
floor differs hugely by feature space: ResNet-50 0.247, ViT-B/16 0.090, CLIP 0.612 (ReLU
features are non-negative, so their cosines start high; CLIP's normalized embeddings sit
in a narrow cone). Each I_T must be read against its own model's floor. The CSV therefore
carries a `normalized` column: (I_T - floor) / (1 - floor), where 0 = indistinguishable
from an unrelated image and 1 = identical.

**Result - normalized stability.** translation 0.91-0.94 (all models); grayscale
0.71-0.74 (all models); patch shuffle 0.63/0.61/0.30 (ResNet/ViT/CLIP); cue conflict vs
content 0.004/0.146/0.101; cue conflict vs style negative for all three (at or below
floor = no measurable resemblance, not "opposite").

**Points of tension for the discussion (not yet interpreted).** (a) Grayscale costs ~2
accuracy points but moves the representation to ~0.72 normalized - representation shift
without prediction change. (b) ResNet's conflict features sit *at* its floor (0.004) while
its predictions still pick the content class ~51% of the time. (c) CLIP has the lowest
patch-shuffle representation stability (0.30) and also the lowest prediction consistency
(84%) - the two measures agree there but not elsewhere.

**t-SNE settings.** sklearn TSNE, 2 components, perplexity 30, cosine metric, init='pca',
learning_rate='auto', random_state 6304. One projection per backbone, fitted to clean +
grayscale + translation32(right) + patch-shuffle + cue-conflict features combined (2220
points). Colour = ground-truth class (content class for conflicts), marker = condition.
Coordinates are never compared across backbones (separate fits). 10 classes exceed the
8-slot categorical palette, so a CVD-oriented 10-colour set is used with marker style as
secondary encoding.
