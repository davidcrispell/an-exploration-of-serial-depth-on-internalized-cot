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

## Matched-loss Shallow 1 on custom 3x3 explicit CoT

This probe starts from the final matched-loss Shallow 1 checkpoint (FineWeb
validation loss `3.3054311275`) rather than the earlier `3.3411` checkpoint.
Because the paper's released multiplication task starts at 4x4, the 3x3 data
is a deterministic custom extension of its exact long-multiplication format.
The 806,000-example training set contains every ordered three-digit pair except
both orientations of the 2,000 held-out commutative pairs. Validation and test
contain 1,000 examples each (seed 3456), and the explicit trace occupies 25
GPT-2 tokens.

After one paper-matched FP32 explicit-CoT epoch (25,187 optimizer steps),
input-only greedy evaluation required the model to generate the complete trace
and final answer; no reference CoT was supplied at evaluation time.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 3x3 validation | 1,000 | 1,000 | 100.0% |
| 3x3 test | 1,000 | 1,000 | 100.0% |

The explicit gate therefore passed and the stepwise 25-token internalization
curriculum began from this checkpoint. This establishes that Shallow 1 can
learn the visible 3x3 algorithm perfectly; it does not yet measure fully hidden
reasoning or a depth effect.

### 3x3 internalization progress

The first epoch removes eight leftmost CoT tokens over the course of the full
training pass. At its epoch-boundary input-only validation, accuracy fell to
757/1,000 (75.7%). The first observed checkpoint below both the 99% primary
threshold and 95% sensitivity threshold is therefore 8/25 removed tokens,
bracketing both frontiers between 0 and 8. This coarse checkpoint is affected
by which trace suffix remains and by adaptation time; later recovery would not
erase the observed crossing. Accuracy declined further to 613/1,000 (61.3%)
at the 16/25-token checkpoint, then rebounded to 842/1,000 (84.2%) at 24/25.
As in the earlier 4x4 curriculum, the curve is non-monotonic, so intermediate
declines cannot be extrapolated to the first or final full-removal result. At
the first 25/25 checkpoint, accuracy recovered further to 965/1,000 (96.5%).
This misses the preregistered 99% target but clears the 95% sensitivity
threshold, so training continues through the remaining full-removal adaptation
budget. One additional full-removal epoch reached 990/1,000 (99.0%), satisfying
the stopping target. The resulting held-out test score is 988/1,000 (98.8%).

| Curriculum checkpoint | Removed CoT tokens | Correct | Examples | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| Explicit stage | 0 / 25 | 1,000 | 1,000 | 100.0% |
| Epoch 0 | 8 / 25 | 757 | 1,000 | 75.7% |
| Epoch 1 | 16 / 25 | 613 | 1,000 | 61.3% |
| Epoch 2 | 24 / 25 | 842 | 1,000 | 84.2% |
| Epoch 3 (first full removal) | 25 / 25 | 965 | 1,000 | 96.5% |
| Epoch 4 (full-removal adaptation 1) | 25 / 25 | 990 | 1,000 | 99.0% |

Thus Shallow 1 successfully internalizes the custom 3x3 task under the primary
validation criterion after one full-removal adaptation epoch, with 98.8% held-out
test accuracy. The first observed checkpoint below both 99% and 95% remains
8/25, but that frontier records the transient removal curriculum rather than a
hard capacity limit: the same model later recovers above both thresholds at
full removal. The non-monotonic curve and adaptation-time confound are therefore
substantive, not merely formal caveats.

The run used a secure 24 GB RTX PRO 6000 Blackwell MIG at $0.69/hour. Pod
lifecycle time was approximately 2.5 hours, including setup, artifact transfer,
explicit training, internalization, evaluation, and download. The observed
Runpod balance delta was **$1.8729679308**. The GPU pod was deleted after local
checkpoint hashes were verified; network volume `vpzu3qptxw` remains.

Local Git-ignored artifacts:

- `artifacts/checkpoints/shallow1-3x3-explicit-cot/explicit_cot_epoch_000.pt`
  (`7ce513979bf7f9423991250828824c9f5709fe16fe778e908ef9e014c2f6ce80`)
- `artifacts/checkpoints/shallow1-3x3-internalized-cot/latest.pt`
  (`7d06b82ede81dd82f238602ae9271c8a1ad48ea1aa50b500fa3d215a97d8a7c8`)

- [`3x3-dataset.json`](3x3-dataset.json)
- [`shallow1-3x3-explicit-cot-validation.json`](shallow1-3x3-explicit-cot-validation.json)
- [`shallow1-3x3-explicit-cot-test.json`](shallow1-3x3-explicit-cot-test.json)
- [`shallow1-3x3-explicit-cot-training-config.json`](shallow1-3x3-explicit-cot-training-config.json)
- [`shallow1-3x3-internalized-cot-validation.json`](shallow1-3x3-internalized-cot-validation.json)
- [`shallow1-3x3-internalized-cot-training-config.json`](shallow1-3x3-internalized-cot-training-config.json)
- [`shallow1-3x3-internalized-cot-test.json`](shallow1-3x3-internalized-cot-test.json)
- [`shallow1-3x3-internalized-cot-summary.json`](shallow1-3x3-internalized-cot-summary.json)

## Matched-loss Deep 1 on custom 3x3 explicit CoT

Deep 1 has 12 layers at width 768 (6 attention heads and MLP width 3072),
while Shallow 1 has 6 layers at width 1024. Both models have exactly
162,201,600 parameters. Their FineWeb validation losses are closely matched:
`3.3050160408` for Deep 1 and `3.3054311275` for Shallow 1 (absolute gap
`0.0004151`).

Deep 1 was trained on the identical custom 3x3 examples and paper-matched
explicit-CoT schedule used for Shallow 1. Input-only greedy evaluation required
generation of the complete reasoning trace and final answer; no reference CoT
was supplied at evaluation time.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 3x3 validation | 1,000 | 1,000 | 100.0% |
| 3x3 test | 1,000 | 1,000 | 100.0% |

The explicit gate passed, and Deep 1 completed the same 25-token stepwise
internalization curriculum. This establishes equal visible-algorithm accuracy
before reasoning tokens are hidden; it does not by itself establish a depth
effect. The evaluator records correctness for every ordered example.

At the first curriculum boundary, Deep 1 scored 897/1,000 (89.7%) with 8/25
CoT tokens removed. Shallow 1 scored 757/1,000 (75.7%) at the identical
checkpoint, giving an observed deep-minus-shallow gap of **14.0 percentage
points**. This is the first observed checkpoint at which the preregistered gap
is at least five points. Both models are below the 99% and 95% thresholds at
this checkpoint, so each first observed frontier is bracketed between 0 and 8
removed tokens.

A fresh recovery run from the independently preserved explicit-CoT checkpoint
reproduced every archived aggregate and per-example result exactly: 897/1,000
(89.7%) at 8/25, 767/1,000 (76.7%) at 16/25, 873/1,000 (87.3%) at 24/25,
975/1,000 (97.5%) at the first 25/25 checkpoint, 990/1,000 (99.0%) after one
full-removal adaptation epoch, and 992/1,000 (99.2%) on test. Its final
checkpoint was downloaded and verified at SHA256
`e3eedc9de3bc1a98f9567662cd1a29e2bedd990f8bbea6396cec1446de0c5868`.
The serialized bytes differ from the lost prior checkpoint, whose remote hash
was `91ef1a680f14912be4e53e163643b02fcd96d536697d33079b6cb92e4613b729`,
but all five validation correctness vectors and the test correctness vector are
identical. This is an exact behavioral reproduction on the recorded examples.

| Curriculum checkpoint | Deep 1 | Shallow 1 | Deep minus shallow |
| --- | ---: | ---: | ---: |
| Explicit stage (0 / 25) | 100.0% | 100.0% | 0.0 pp |
| Epoch 0 (8 / 25) | 89.7% | 75.7% | +14.0 pp |
| Epoch 1 (16 / 25) | 76.7% | 61.3% | +15.4 pp |
| Epoch 2 (24 / 25) | 87.3% | 84.2% | +3.1 pp |
| Epoch 3 (first full removal) | 97.5% | 96.5% | +1.0 pp |
| Epoch 4 (one full-removal adaptation epoch) | 99.0% | 99.0% | 0.0 pp |
| Held-out test after epoch 4 | 99.2% | 98.8% | +0.4 pp |

At 16/25 removed tokens, Deep 1 scored 767/1,000 (76.7%) versus
613/1,000 (61.3%) for Shallow 1, widening the observed gap slightly to
15.4 points. Both models declined from their 8-token checkpoint, reinforcing
that checkpoint number, remaining trace suffix, and adaptation time cannot be
collapsed into a simple monotonic capacity axis.

At 24/25 removed tokens, both models rebounded: Deep 1 reached 873/1,000
(87.3%) and Shallow 1 reached 842/1,000 (84.2%). The observed gap therefore
narrowed to 3.1 points, below the preregistered five-point threshold. The
earlier large separation does not persist uniformly across removal checkpoints.

At the first 25/25 checkpoint, Deep 1 reached 975/1,000 (97.5%) and Shallow 1
reached 965/1,000 (96.5%), leaving only a 1.0-point gap. Both models therefore
clear the 95% sensitivity threshold at first full removal, while neither clears
the 99% primary threshold. After one additional full-removal adaptation epoch,
both reached exactly 990/1,000 (99.0%) on validation. Their archived CUDA test
scores were 992/1,000 (99.2%) for Deep 1 and 988/1,000 (98.8%) for Shallow 1,
only a 0.4-point final gap.

The archived Shallow result predates per-example correctness recording. A
rerun of its identical checkpoint and examples on local CPU produced 986/1,000
(98.6%), two examples below the archived CUDA score. Pairing that CPU vector
with Deep's CUDA vector gives a **cross-backend sensitivity analysis** of +0.6
points (paired-bootstrap 95% interval -0.3 to +1.5; 14 Deep-only versus 8
Shallow-only successes; exact McNemar `p = 0.2863`). This does not detect a
final-test difference, but it is not a definitive same-backend paired test.
The small reproducibility discrepancy is consistent with backend-sensitive
low-margin greedy decisions; both checkpoints should be rerun on the same CUDA
backend before making a primary paired claim.

The intermediate separation is real as an observed training-curve result, but
it is not a clean capacity frontier: both curves are strongly non-monotonic,
each checkpoint differs in adaptation time, and the gap collapses near full
removal. Deep 1 also outperformed Shallow 1 on LAMBADA by 3.12 points despite
the parameter/loss match, leaving residual broad-capability mismatch as a
causal confound. These data therefore do not establish a causal depth-only
effect.

- [`deep1-3x3-explicit-cot-validation.json`](deep1-3x3-explicit-cot-validation.json)
- [`deep1-3x3-explicit-cot-test.json`](deep1-3x3-explicit-cot-test.json)
- [`deep1-3x3-explicit-cot-training-config.json`](deep1-3x3-explicit-cot-training-config.json)
- [`deep1-3x3-internalized-cot-validation.json`](deep1-3x3-internalized-cot-validation.json)
- [`deep1-3x3-internalized-cot-test.json`](deep1-3x3-internalized-cot-test.json)
- [`deep1-3x3-internalized-cot-training-config.json`](deep1-3x3-internalized-cot-training-config.json)
- [`deep1-3x3-internalized-cot-progress.json`](deep1-3x3-internalized-cot-progress.json)
- [`deep1-3x3-internalized-cot-recovery-progress.json`](deep1-3x3-internalized-cot-recovery-progress.json)
- [`deep1-3x3-internalized-cot-recovery-validation.json`](deep1-3x3-internalized-cot-recovery-validation.json)
- [`deep1-3x3-internalized-cot-recovery-test.json`](deep1-3x3-internalized-cot-recovery-test.json)
- [`deep1-3x3-internalized-cot-recovery-training-config.json`](deep1-3x3-internalized-cot-recovery-training-config.json)
- [`deep1-vs-shallow1-3x3-paired-sensitivity.json`](deep1-vs-shallow1-3x3-paired-sensitivity.json)
- [`shallow1-3x3-internalized-cot-test-paired-rerun.json`](shallow1-3x3-internalized-cot-test-paired-rerun.json)

The Deep run used a 24 GB RTX PRO 6000 Blackwell MIG at $0.69/hour. From pod
creation through its stop after completion, runtime was about 2 hours 53
minutes and the observed Runpod balance delta was $2.0199. The explicit
checkpoint is verified locally (SHA256
`d71a3f994619d6f8276dcfeb51a790a3fb376a1f8dfb3450c43f73f945b58ba0`).
The exactly reproduced final internalized checkpoint is also verified locally
(SHA256 `e3eedc9de3bc1a98f9567662cd1a29e2bedd990f8bbea6396cec1446de0c5868`).

## Matched-loss Deep 1 on 4x4 explicit CoT

Starting from the untouched matched-loss Deep 1 FineWeb checkpoint, one
paper-matched FP32 epoch on the released 808,000-example 4x4 training set
reached perfect input-only greedy accuracy. Evaluation required generation of
the full visible trace and final answer; no ground-truth CoT was supplied.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 4x4 validation | 1,000 | 1,000 | 100.0% |
| 4x4 test | 1,000 | 1,000 | 100.0% |

The explicit checkpoint is downloaded under
`artifacts/checkpoints/deep1-4x4-explicit-cot/` and verified at SHA256
`f7df6b802eca06ce78c5db280ab21dc01206745cd67d1057a910941659195daf`.
The 47-token internalization curriculum is now running from this checkpoint.
This establishes equal visible-algorithm accuracy for the matched Deep and
Shallow models before the trace is hidden; it is not evidence of a depth-only
effect.

- [`deep1-4x4-explicit-cot-validation.json`](deep1-4x4-explicit-cot-validation.json)
- [`deep1-4x4-explicit-cot-test.json`](deep1-4x4-explicit-cot-test.json)
- [`deep1-4x4-explicit-cot-training-config.json`](deep1-4x4-explicit-cot-training-config.json)

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
| Epoch 1 | 16 / 75 | 323 | 1,000 | 32.3% |
| Epoch 2 | 24 / 75 | 188 | 1,000 | 18.8% |
| Epoch 3 | 32 / 75 | 672 | 1,000 | 67.2% |
| Epoch 4 | 40 / 75 | 146 | 1,000 | 14.6% |
| Epoch 5 | 48 / 75 | 104 | 1,000 | 10.4% |
| Epoch 6 | 56 / 75 | 0 | 1,000 | 0.0% |
| Epoch 7 | 64 / 75 | 2 | 1,000 | 0.2% |
| Epoch 8 | 72 / 75 | 0 | 1,000 | 0.0% |
| Epoch 9 (first full removal) | 75 / 75 | 2 | 1,000 | 0.2% |
| Epoch 10 (full-removal adaptation 1) | 75 / 75 | 4 | 1,000 | 0.4% |
| Epoch 11 (full-removal adaptation 2) | 75 / 75 | 2 | 1,000 | 0.2% |
| Epoch 12 (full-removal adaptation 3) | 75 / 75 | 5 | 1,000 | 0.5% |
| Epoch 13 (full-removal adaptation 4) | 75 / 75 | 3 | 1,000 | 0.3% |
| Epoch 14 (full-removal adaptation 5) | 75 / 75 | 6 | 1,000 | 0.6% |

As a retention control, the epoch-0 checkpoint was also evaluated with the
complete ground-truth CoT supplied in the prompt and greedy generation beginning
at the final-answer field. It scored **999/1,000 (99.9%)**. The single failure
produced nine correct answer digits before reverting to trace-like syntax instead
of emitting the tenth digit. This oracle-CoT control shows that the 91.5% result
mostly reflects difficulty operating with the shortened trace, rather than broad
catastrophic forgetting of answer readout when the complete correct trace is
available. It is an auxiliary oracle control, not the paper's input-only
curriculum metric, and does not test whether the model can independently
regenerate the removed CoT prefix.

The first published removal checkpoint is below both the 99% primary threshold
and 95% sensitivity threshold, bracketing both frontiers between 0 and 8 removed
tokens. Because removal advances continuously during each epoch, this coarse
checkpoint does not identify an individual-token boundary and is confounded by
adaptation time. Accuracy declined further to 32.3% at 16 removed tokens and
18.8% at 24 removed tokens, rebounded to 67.2% at 32 removed tokens, and then
fell to 14.6% at 40 removed tokens and 10.4% at 48. This non-monotonicity makes
the identity of the remaining trace suffix and adaptation time important confounds;
accuracy then reached 0.0% at 56 removed tokens. Removed-token count alone does
not explain the intermediate curve; accuracy recovered only negligibly to 0.2%
at 64 removed tokens, returned to 0.0% at 72, and was 0.2% at the first 75/75
evaluation. Because that full-removal checkpoint did not meet the 99% stopping
target, the curriculum used all five remaining full-removal adaptation epochs.
Accuracy was 0.4%, 0.2%, 0.5%, 0.3%, and 0.6% across those epochs. The final
held-out test scored **3/1,000 (0.3%)**. Thus the shallow model learned the
explicit algorithm perfectly but did not preserve it after the full 75-token
trace was hidden under this curriculum and training budget.

Here, “75 steps” would be misleading. The experiment operationalizes trace
length as **75 GPT-2 tokens**, and the curriculum removes those tokens—not
independently labeled semantic reasoning steps. A 5x5 trace contains five
shifted partial-product blocks and three explicitly parenthesized intermediate
cumulative-sum blocks. The final answer supplies the result after incorporating
the fifth partial product. Thus there are eight visible arithmetic blocks in
the CoT, or nine arithmetic actions if the final accumulation represented by
the answer is counted; neither count has a one-to-one mapping to the 75-token
removal axis.

The first observed failing checkpoint for both the 99% primary threshold and
95% sensitivity threshold is 8/75 removed tokens, bracketing each frontier
between 0 and 8. The curve is strongly non-monotonic, and every checkpoint also
has a different amount of adaptation time, so removed-token count alone should
not be treated as a clean difficulty axis. This is a within-model result only;
it cannot establish a causal depth effect without the matched deep-model run on
identical examples and schedule.

The full pipeline used 378,750 internalization optimizer steps. It ran for about
8 hours 21 minutes from pod creation through final testing (about 8 hours 23
minutes through pod deletion) and cost **$4.949346536** by Runpod balance delta,
including setup, explicit-CoT training, evaluation, internalization, and storage.
The GPU pod was deleted after artifact verification; network volume
`vpzu3qptxw` remains. The 648.8 MB final checkpoint is stored locally under
`artifacts/checkpoints/shallow-5x5-internalized-cot/` and excluded from Git.
SHA-256: `cce7abad55f8ebb545dd2f8f29bb3b3cdbe5e39f99b7771552497ab6fdb4c3fc`.

- [`shallow-5x5-internalized-cot-validation.json`](shallow-5x5-internalized-cot-validation.json)
- [`shallow-5x5-internalized-cot-test.json`](shallow-5x5-internalized-cot-test.json)
- [`shallow-5x5-internalized-cot-training-config.json`](shallow-5x5-internalized-cot-training-config.json)
- [`shallow-5x5-internalized-cot-progress.json`](shallow-5x5-internalized-cot-progress.json)
- [`shallow-5x5-internalized-cot-summary.json`](shallow-5x5-internalized-cot-summary.json)
- [`shallow-5x5-internalized-cot-8-removed-explicit-cot-validation.json`](shallow-5x5-internalized-cot-8-removed-explicit-cot-validation.json)

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
