# ATML PA3 — LLM Fine-Tuning, VLM Grounding, Test-Time Compute

All commands run from this directory (`PA 3/`). Seed everywhere: **6304**. Configs live in `configs/`;
every reported number should trace to a file in `results/` (machine-readable JSON/CSV/parquet).

```
common/       shared helpers (config loading, seeding, timers, bootstrap)
configs/      task1_llm.yaml  task2_vlm.yaml  task3_ttc.yaml
scripts/      check_environment.py  download_assets.py  inspect_task3_data.py
task1_llm/    data, from-scratch LoRA/OPLoRA, trainer, lm-eval harness evaluation, diagnostics, qualitative, analysis
task2_vlm/    data builder, SigLIP feature cache, connector+Qwen3 VLM, 3-stage trainer, evaluation, analysis
task3_ttc/    answers/grading, data join, scoring+selection, offline study, signals, live generation, adaptive, analysis
splits/       saved split / example IDs        results/  metrics and per-example outputs
checkpoints/  (git-ignored)                    cached/   (git-ignored) tokenised data, SigLIP features, pools
```

## Setup
```
pip install -r requirements.txt
python -m scripts.check_environment
```
External inputs (not committed): `data/first_plot_questions.yaml` (Task 1 qualitative prompts, from the assignment),
COCO 2014 + VQAv2 complementary pairs (`python -m scripts.download_assets --coco --vqa`), and the supplied
`ATML-PA3-Data` rollouts (`--task3-repo <id from LMS>`, plus MathVista question metadata).
After download run `python -m scripts.inspect_task3_data` and compare with `COLUMN_CANDIDATES` in `task3_ttc/data.py`
(the manual's `README.MD` defines the true schema; adjust the candidate column lists if needed).

## Task 1 — LLM fine-tuning (Qwen3-0.6B, MetaMathQA)
```
for m in full_ft full_ft_l2sp lora4 lora16 oplora16; do python -m task1_llm.train --method $m; done   # add --resume to continue
python -m task1_llm.evaluate          # 7 conditions x {GSM8K, HellaSwag, BoolQ, ARC-E} via lm-evaluation-harness
python -m task1_llm.diagnostics       # rho_k, eta, activation ratio R(z) for down_proj layers 0/13/27
python -m task1_llm.qualitative       # 24 prompts x 7 conditions; writes the 42-response annotation sheet
python -m task1_llm.analyze           # tables + figures
```
LoRA / OPLoRA are implemented in `task1_llm/lora.py` (effective factors, FP32 adapters on an FP16 backbone, exported
projected factors, merged-vs-unmerged logit check in `evaluate.py`). L2-SP (λ = 1e-3) is applied as an exact gradient term
λ(θ−θ₀) after unscaling; Ω, λΩ and total loss are logged every update.

## Task 2 — VLM visual grounding (SigLIP + Qwen3-0.6B)
```
python -m task2_vlm.build_data        # COCO captions, annotation-derived presence/counting pairs, VQAv2 pairs; ID files
python -m task2_vlm.cache_features    # frozen SigLIP features, FP16 [196,768] per image
python -m task2_vlm.checks            # shapes, frozen sets, connector gradients, split isolation
python -m task2_vlm.train --run-name main            # 3 stages, 4,500 updates, resumable (--resume)
python -m task2_vlm.evaluate --run-name main         # 4 image conditions, pair-bootstrap CIs
python -m task2_vlm.analyze --run-name main
# optional Step 7 (49 tokens):  train/evaluate/analyze with  --compress --run-name tokens49
```
Inspect `splits/task2/grounding_audit.csv` by hand before training (labels come from COCO annotations only).

## Task 3 — Test-time compute
```
python -m task3_ttc.offline  --dataset math        --workers 8    # Steps 1-2 (also: --dataset mathvista)
python -m task3_ttc.signals  --dataset math                        # Step 3
python -m task3_ttc.live                                           # Step 4 (50 MATH eval questions, GPU)
python -m task3_ttc.adaptive --dataset math                        # Step 5 (dev-selected, frozen, held-out eval)
python -m task3_ttc.analyze                                        # Step 6
```
Splits: sorted question IDs permuted with `np.random.default_rng(6304)`; first 200 dev, remaining 800 held out. Live
questions are the first 50 held-out IDs in that permutation. Gold answers are used only to grade, never to group,
select, or stop.

## Attribution / notes
- LoRA init convention follows Hu et al. / PEFT (`kaiming_uniform(a=√5)`), re-implemented here; OPLoRA follows Xiong & Xie.
- `task2_vlm/vqa_norm.py` condenses the public VQA normalization/scoring code (GT-Vision-Lab/VQA).
- Benchmarks use `lm-evaluation-harness`; answer equivalence uses `math-verify`.
- Deviations / assumptions are recorded in the report; see `results/*/config.json` for the exact hyperparameters of each run.
