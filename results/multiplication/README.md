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

## Shallow-model CoT internalization curriculum

This run starts from the 100%-accurate explicit-CoT checkpoint and removes the
leftmost reasoning tokens at the paper's rate of eight tokens per epoch. The
table is updated after each completed epoch; the held-out test split remains
untouched until the curriculum and stopping decision are complete.

| Removed CoT tokens | Correct | Validation examples | Exact-match accuracy |
| ---: | ---: | ---: | ---: |
| 0 / 47 | 1,000 | 1,000 | 100.0% |
| 8 / 47 | 852 | 1,000 | 85.2% |

The first published removal checkpoint already falls below both the
preregistered 99% primary threshold and 95% sensitivity threshold. Thus the
current shallow-model frontier is bracketed between 0 and 8 removed tokens; the
epoch-level protocol does not identify which individual token inside that
interval caused the crossing. This is a within-model result only. It is not
evidence for a causal depth effect until the parameter-matched deep model is run
with the identical curriculum and examples.

- [`shallow-4x4-internalized-cot-validation.json`](shallow-4x4-internalized-cot-validation.json)
