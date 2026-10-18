# GSM8K internalized-CoT extension

This benchmark extends the Deep 1 / Shallow 1 serial-depth experiment from
fixed-width multiplication to variable-length grade-school word problems. It
implements the GPT-2 protocol from *From Explicit CoT to Implicit CoT: Learning
to Internalize CoT Step by Step* while retaining this repository's native
checkpoint format.

The released augmented records have this format:

```text
question||GPT-4 reasoning trace #### final answer
```

The code serializes each example as `question EOS trace EOS #### answer EOS`,
masks the question from the loss, and filters training examples longer than
150 GPT-2 tokens. The dataset is not copied into this repository; obtain
`train.txt`, `valid.txt`, and `test.txt` from the authors' release and place
them under `benchmarks/gsm8k/data/` (or pass other paths explicitly).

## Stage 1: explicit CoT

Run the exact same command for Deep 1 and Shallow 1, changing only the matched
FineWeb checkpoint and output directory:

```bash
python -m benchmarks.gsm8k.train_explicit_cot \
  --checkpoint artifacts/checkpoints/deep1-fineweb/latest.pt \
  --train-path benchmarks/gsm8k/data/train.txt \
  --validation-path benchmarks/gsm8k/data/valid.txt \
  --test-path benchmarks/gsm8k/data/test.txt \
  --cache-dir artifacts/cache/gsm8k-gpt2 \
  --output-dir artifacts/checkpoints/deep1-gsm8k-explicit \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --epochs 15
```

FP32 is the default. `--bf16` is available as a deliberate protocol change.
The default 15-epoch explicit warm-up is an experiment control, not a claim
that the paper used a separate 15-epoch stage: the released paper trainer used
one combined curriculum. For a depth comparison, keep the epoch budget,
validation gate, seed, dataset hashes, and all optimization settings identical.

The trainer writes `latest.pt`, the validation-selected `best.pt`, compact
JSON/JSONL metrics, and a single final test evaluation of `best.pt`.

## Stage 2: internalization

```bash
python -m benchmarks.gsm8k.train_internalized_cot \
  --checkpoint artifacts/checkpoints/deep1-gsm8k-explicit/best.pt \
  --train-path benchmarks/gsm8k/data/train.txt \
  --validation-path benchmarks/gsm8k/data/valid.txt \
  --test-path benchmarks/gsm8k/data/test.txt \
  --cache-dir artifacts/cache/gsm8k-gpt2 \
  --output-dir artifacts/checkpoints/deep1-gsm8k-internalized \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --remove-per-epoch 1 \
  --remove-all-when-remove-beyond 39 \
  --removal-smoothing-lambda 4 \
  --epochs 200 \
  --max-runtime-hours 24
```

The GPT-2 setting removes one scheduled token per epoch; the paper README's
Mistral example instead uses eight. At a scheduled removal of 39, all remaining
trace tokens are removed, even when a particular trace is longer. AdamW is
reset at every curriculum transition, including the transition to full
removal. The exponential smoothing offset is used for training only.

Validation and test generation always begin from `question EOS` with no
ground-truth trace. Greedy generation stops after two generated EOS tokens or
800 new tokens, and accuracy is comma-insensitive exact final-answer match.
This question-only protocol is used at every curriculum step, so the saved
validation curve directly measures how much reasoning the model can perform
without being handed the omitted prefix. Metrics also report the empty-trace
rate and mean number of visible reasoning tokens, so a correct answer that was
still produced through visible CoT is not mistaken for internalization.

Only validation accuracy selects `best_fully_internalized.pt`; the test split
is evaluated once after selection. `training_metrics.jsonl` contains losses,
removal counts, timings, and validation points for a dashboard or later plots.

For a validation-only first-full-removal pilot, use one explicit epoch with
`--skip-test`, then run 43 internalization epochs with
`--eval-removals 0,8,16,24,32 --skip-test`. Epoch 39 is the first fully removed
epoch and epochs 40--42 provide three additional answer-only adaptation epochs.
This deliberately leaves the held-out test split untouched until the full
protocol and stopping rule are fixed.

## Standalone evaluation

```bash
python -m benchmarks.gsm8k.evaluate \
  --checkpoint artifacts/checkpoints/deep1-gsm8k-internalized/best_fully_internalized.pt \
  --data-path benchmarks/gsm8k/data/test.txt \
  --batch-size 16 \
  --output results/gsm8k/deep1-internalized-test.json
```

The code records all input-file SHA256 hashes in `training_config.json`. A
credible depth comparison still needs multiple seeds and capability-matched
explicit baselines; one successful pair cannot by itself isolate serial depth
from residual pretraining differences.
