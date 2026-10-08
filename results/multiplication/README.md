# Multiplication results

## Untreated shallow-model baseline

Checkpoint: 6 layers, width 1024, 8 heads, MLP width 2768, 162,201,600 parameters.
FineWeb validation loss: 3.3411. Checkpoint SHA-256:
`c5e54086d9ef419f2c43591eca0887bd5ed5e8fed52dd3a329363f807a3f49e0`.

The direct-answer baseline supplies the reversed operands, GPT-2 end-of-text separator, and
`####` answer marker. Greedy decoding then produces exactly the number of tokens occupied by the
space-separated reversed product plus the final end-of-text token. Accuracy requires the entire
continuation to match exactly.

| Test split | Correct | Examples | Exact-match accuracy |
| --- | ---: | ---: | ---: |
| 4x4 | 0 | 1,000 | 0.0% |
| 5x5 | 0 | 1,000 | 0.0% |
| 7x7 | 0 | 1,000 | 0.0% |
| 9x9 | 0 | 1,000 | 0.0% |
| 11x11 | 0 | 1,000 | 0.0% |

Machine-readable outputs and representative generations are in
[`shallow-base-direct-test.json`](shallow-base-direct-test.json).

## Shallow model after explicit-CoT fine-tuning

The 4x4 model was fine-tuned for one full pass over the authors' 808,000-example training split
(25,250 optimizer steps). Training used the paper's sequence format and multiplication
hyperparameters: FP32, AdamW, learning rate `5e-5`, batch size 32, gradient clipping at 1.0, and
seed 3456. The generation metric follows the paper and checks the final answer after the generated
trace; decoding is deterministic for this controlled comparison.

The run used a 24 GB RTX PRO 6000 Blackwell MIG at $0.59/hour. The Runpod balance delta for setup,
smoke tests, the full epoch, and evaluation was approximately $0.56. No GPU pod was left running.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 4x4 validation | 1,000 | 1,000 | 100.0% |
| 4x4 test | 1,000 | 1,000 | 100.0% |

The local model artifact is `artifacts/checkpoints/shallow-4x4-explicit-cot/` and is excluded from
Git because it is 619 MB. Checkpoint SHA-256:
`75c8284fc6b087acd34cbfff1f37cae8f4ec3b342b83a5c696744fb7d8a26874`.

- [`shallow-4x4-explicit-cot-validation.json`](shallow-4x4-explicit-cot-validation.json)
- [`shallow-4x4-explicit-cot-test.json`](shallow-4x4-explicit-cot-test.json)
- [`shallow-4x4-explicit-cot-training-config.json`](shallow-4x4-explicit-cot-training-config.json)
- [`4x4-explicit-cot-fp32.log`](4x4-explicit-cot-fp32.log)

## Shallow model after 5x5 explicit-CoT fine-tuning

Starting again from the untreated shallow FineWeb checkpoint—not from either
4x4 checkpoint—the model was fine-tuned for one full pass over the paper's
808,000-example 5x5 training split. The run used the same paper-matched FP32
settings as 4x4: AdamW, learning rate `5e-5`, effective batch size 32, gradient
clipping at 1.0, and seed 3456. The training set SHA-256 is
`e41642536746292598fb5fa53bf8ebb01358a161a6ba78892d0579ccc1489253`.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 5x5 validation | 1,000 | 1,000 | 100.0% |
| 5x5 test | 1,000 | 1,000 | 100.0% |

The explicit checkpoint was produced after 25,250 optimizer steps. Its verified
SHA-256 is
`c9a95971305960d4d8152b257088df723f8447c7064bcea35ea90b2a01d585ee`.
The local artifact is stored under
`artifacts/checkpoints/shallow-5x5-explicit-cot/` and excluded from Git. The
75-token internalization curriculum is in progress from this checkpoint.

- [`shallow-5x5-explicit-cot-validation.json`](shallow-5x5-explicit-cot-validation.json)
- [`shallow-5x5-explicit-cot-test.json`](shallow-5x5-explicit-cot-test.json)
- [`shallow-5x5-explicit-cot-training-config.json`](shallow-5x5-explicit-cot-training-config.json)

## Shallow-model 5x5 CoT internalization curriculum

This run starts from the 100%-accurate 5x5 explicit-CoT checkpoint and removes
the leftmost reasoning tokens at eight tokens per epoch. The training remains
paper-matched FP32 with effective batch size 32. The held-out test split remains
untouched until the curriculum and stopping decision are complete.

| Curriculum checkpoint | Removed CoT tokens | Correct | Validation examples | Exact-match accuracy |
| ---: | ---: | ---: | ---: | ---: |
| Explicit stage | 0 / 75 | 1,000 | 1,000 | 100.0% |
| Epoch 0 | 8 / 75 | 915 | 1,000 | 91.5% |

The first published removal checkpoint is below both the 99% primary threshold
and 95% sensitivity threshold, bracketing both frontiers between 0 and 8 removed
tokens. Because removal advances continuously during each epoch, this coarse
checkpoint does not identify an individual-token boundary and is confounded by
adaptation time. The curriculum is still running, and a later recovery would be
scientifically relevant. This is a within-model result only; it cannot establish
a causal depth effect without the matched deep-model run on identical examples.

- [`shallow-5x5-internalized-cot-validation.json`](shallow-5x5-internalized-cot-validation.json)
- [`shallow-5x5-internalized-cot-training-config.json`](shallow-5x5-internalized-cot-training-config.json)
- [`shallow-5x5-internalized-cot-progress.json`](shallow-5x5-internalized-cot-progress.json)

## Shallow-model 4x4 CoT internalization curriculum

This run starts from the 100%-accurate explicit-CoT checkpoint and removes the
leftmost reasoning tokens at the paper's rate of eight tokens per epoch. The
table is updated after each completed epoch; the held-out test split remains
untouched until the curriculum and stopping decision are complete.

| Curriculum checkpoint | Removed CoT tokens | Correct | Validation examples | Exact-match accuracy |
| ---: | ---: | ---: | ---: | ---: |
| Explicit stage | 0 / 47 | 1,000 | 1,000 | 100.0% |
| Epoch 0 | 8 / 47 | 852 | 1,000 | 85.2% |
| Epoch 1 | 16 / 47 | 177 | 1,000 | 17.7% |
| Epoch 2 | 24 / 47 | 74 | 1,000 | 7.4% |
| Epoch 3 | 32 / 47 | 7 | 1,000 | 0.7% |
| Epoch 4 | 40 / 47 | 791 | 1,000 | 79.1% |
| Epoch 5 | 47 / 47 | 651 | 1,000 | 65.1% |
| Epoch 6 | 47 / 47 | 799 | 1,000 | 79.9% |
| Epoch 7 | 47 / 47 | 923 | 1,000 | 92.3% |

The first published removal checkpoint already falls below both the
preregistered 99% primary threshold and 95% sensitivity threshold. Thus the
current shallow-model frontier is bracketed between 0 and 8 removed tokens; the
epoch-level protocol does not identify which individual token inside that
interval caused the crossing. Accuracy then declines sharply to 17.7% at 16
removed tokens, 7.4% at 24, and 0.7% at 32. Accuracy then rebounds sharply to
79.1% at 40 removed tokens. The non-monotonic curve means token count alone does
not explain intermediate performance: the identity and structure of the
remaining trace suffix, as well as adaptation to each curriculum boundary, may
matter. At full removal, accuracy is 65.1% immediately after the transition and
recovers across two additional fully internalized epochs to 79.9% and then
92.3%. The final held-out test also scores 923/1,000 (92.3%). This misses both
the preregistered 99% primary target and 95% sensitivity threshold, so the
shallow model did not fully preserve its explicit-CoT capability under the
fixed eight-epoch budget.

The first observed failing checkpoint for both thresholds is 8/47 removed
tokens, bracketing the frontier between 0 and 8. That label should be read with
care: the released curriculum advances continuously, so epoch-boundary
evaluations can occur shortly after a removal transition. The recovery across
repeated 47/47 evaluations demonstrates this adaptation-time confound directly.
This is a within-model result only. It is not evidence for a causal depth effect
until the parameter-matched deep model is run with the identical schedule and
examples.

The FP32 run used 202,000 optimizer steps and cost approximately $2.16 by
Runpod balance delta, including setup and evaluation. Wall-clock pod time was
about 3 hours 40 minutes. The GPU pod was deleted after artifact verification;
the persistent network volume remains. The 619 MB final checkpoint is stored
locally under `artifacts/checkpoints/shallow-4x4-internalized-cot/` and excluded
from Git. SHA-256:
`a6a23d9de0639b6329bea159cfd320849cd30739d7a731a3f3a98a6cc325f094`.

- [`shallow-4x4-internalized-cot-validation.json`](shallow-4x4-internalized-cot-validation.json)
- [`shallow-4x4-internalized-cot-test.json`](shallow-4x4-internalized-cot-test.json)
- [`shallow-4x4-internalized-cot-training-config.json`](shallow-4x4-internalized-cot-training-config.json)
- [`shallow-4x4-internalized-cot-summary.json`](shallow-4x4-internalized-cot-summary.json)
- [`4x4-internalized-cot-fp32.log`](4x4-internalized-cot-fp32.log)

### Out-of-distribution 5x5 test

The final fully internalized checkpoint was evaluated without further training
on the deterministic 1,000-example 5x5 test split. Evaluation used the same
paper-format prompt, greedy decoding, FP32 parameters, and final-answer exact
match as the original 4x4 result. A same-session 4x4 control exactly reproduced
the published score.

| Test split | Correct | Examples | Exact-match accuracy |
| --- | ---: | ---: | ---: |
| 4x4 control | 923 | 1,000 | 92.3% |
| 5x5 transfer | 0 | 1,000 | 0.0% |

Representative 5x5 generations emit an eight-digit answer—the output shape
learned from 4x4 multiplication—where the benchmark target contains ten digits.
The predicted low-order digits also diverge from the target, so this is not just
a harmless padding or parsing mismatch. This result measures zero-shot transfer
from 4x4 to 5x5; it does not measure how readily the model could learn 5x5
multiplication with additional fine-tuning.

- [`shallow-4x4-internalized-cot-5x5-test.json`](shallow-4x4-internalized-cot-5x5-test.json)
