# Multiplication results

## Paper-style Shallow 1.1 recurrent 5x5 run (in progress)

The replacement Shallow 1.1 experiment adapts the recurrent-depth construction
from [Geiping et al. (2025)](https://arxiv.org/abs/2502.05171) to the pretrained
six-layer Shallow 1 checkpoint. It partitions the model as one prelude block,
four recurrent core blocks, and one coda block. Unlike the completed frozen-stack
pilot below, all 164,299,777 parameters train; the recurrent additions comprise a
learned 2h-to-h state/input adapter, a core normalization weight, and a scalar
normalization interpolation. Recurrence is sampled with the paper's
log-normal--Poisson rule around target depth four, and gradients are retained
through the final four recurrent iterations.

The first explicit-CoT recurrent warmup epoch passed the primary r=4 gate. The
same checkpoint was evaluated at four inference recurrence counts:

| Recurrences | Effective layers | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 6 | 999 / 1,000 | 99.9% |
| 2 | 10 | 998 / 1,000 | 99.8% |
| 4 | 18 | 996 / 1,000 | 99.6% |
| 8 | 34 | 992 / 1,000 | 99.2% |

At the first internalization boundary (8/75 removed), validation accuracy was
63.3% at r=1, 88.8% at r=2, 86.0% at the primary r=4 recurrence, and 84.5% at
r=8. At the second boundary (16/75 removed), accuracy was 37.0% at r=1, 55.7%
at r=2, 66.9% at r=4, and 69.0% at r=8. The first boundary is non-monotonic in
inference recurrence count, while the second is monotonic across the evaluated
recurrences. At the third boundary (24/75 removed), accuracy rebounded to 44.0%
at r=1, 84.1% at r=2, 86.9% at r=4, and 86.8% at r=8. At the fourth boundary
(32/75 removed), it fell to 11.9%, 24.2%, 36.0%, and
36.3%, respectively. At the fifth boundary (40/75 removed), accuracy was 11.4%,
25.4%, 29.6%, and 29.5%. At the sixth boundary (48/75 removed), it dropped to
1.0%, 3.4%, 4.4%, and 4.6%. At the seventh boundary (56/75 removed), every
evaluated recurrence scored 0/1,000 (0.0%). At the eighth boundary (64/75
removed), r=1 remained near zero at 2/1,000 (0.2%), while r=2, r=4, and r=8
rebounded to 29.2%, 34.3%, and 33.9%. At the ninth boundary (72/75 removed),
accuracy fell back to 0.0%, 0.1%, 0.3%, and 0.2%. The full
recurrence-by-removal surface then rebounded at the first full-removal boundary
to 22.4%, 67.7%, 70.9%, and 71.0%. It is being preserved rather than summarized
as a single depth curve. After one full-removal adaptation epoch, accuracy rose
again to 41.5%, 77.4%, 78.8%, and 78.9%. The epoch-10 checkpoint SHA256 is
`9fd51d14bec451272e972370bd6d7dbcba52cfce465f7c4ac1b79517e0149c79`;
the epoch-9 checkpoint SHA256 is
`5d9c561ee93014b4c0bf1eda9d263790fa934cdd5d3408c4554f4b58a5e201ee`;
the epoch-8 checkpoint SHA256 is
`62c2c21075b1fa9fd26c56b6e3dbc955c1669a6d1e7c26ae80956bf00772eee2`;
the epoch-7 checkpoint SHA256 is
`18a009385e07b93014cb94ec73e8bffb33383c85d3b7e7fd9bf434ffc68c06e4`;
the epoch-6 checkpoint SHA256 is
`53a0fac5ca552abbf24c5c2e0eeb36f5bc957c264f22b7d1aff20e9cf1781414`;
the epoch-5 checkpoint SHA256 is
`d07c82d09340a80fa27d3a2062a0c771cf4fbeb9315cbbda62f1d6e775e5662d`;
the epoch-4 checkpoint SHA256 is
`b28a6984966f1fe915f2913b68b33734499801e81c7760ecb7e18fb9e8f4ead7`;
the epoch-3 checkpoint SHA256 is
`b26895da0bc278d2e4e32b1f84165bb7f6ef74c8a6cf0159eb6994a37c393e6b`;
the epoch-2 checkpoint SHA256 is
`6ca6b610de60b36212fddf7228ead134bcb868d496169189751ed3b0a770f0e7`;
the epoch-1 checkpoint SHA256 is
`28d822cc0c7aace5f996b413347c569480d442a282d685a5e069fdaf5252e03b`, and
the epoch-0 checkpoint SHA256 is
`ccdce4c73939a636194b526956262167baf23517aa85995a117eec66fc80c753`.
The curriculum continues from that checkpoint. The warmup checkpoint is
preserved on the run's network volume at SHA256
`e0765193ad23199c2360ac7be529db2142f9a64a74ad576fcc7e5e1fa0c5af7d`.
These one-seed measurements do not yet establish whether recurrence improves
hidden-reasoning acquisition.

- [`shallow1-1-paper-recurrent-5x5-progress.json`](shallow1-1-paper-recurrent-5x5-progress.json)
- [Validation comparison figure (PNG)](figures/paper-recurrent-validation-comparison.png)
- [Validation comparison figure (SVG)](figures/paper-recurrent-validation-comparison.svg)
- [Validation comparison figure (PDF)](figures/paper-recurrent-validation-comparison.pdf)

## Completed Shallow 1.1 recurrent 5x5 pilot

Shallow 1.1 initializes from Shallow 1's verified 100%-accurate explicit-CoT
checkpoint and executes its frozen six-layer stack twice. Only two full-width
1024×1024 recurrent matrices train: 2,097,152 trainable parameters on top of
the frozen 162,201,600-parameter base (164,298,752 total).

The explicit-CoT recurrent-adapter warmup completed one epoch and scored
994/1,000 (99.4%) on input-only greedy validation. The run then encountered a
PyTorch rotary-cache transition bug when switching from inference-mode
validation back to training. The cache fix is regression-tested, and the run
resumed from the saved warmup checkpoint without repeating the warmup. The
first completed internalization boundary scored 67/1,000 (6.7%) with 8/75 CoT
tokens removed. At the identical boundary, Shallow 1 scored 91.5% and Deep 1
scored 95.3%.
At the second boundary, Shallow 1.1 scored 64/1,000 (6.4%) with 16/75 tokens
removed, versus 32.3% for Shallow 1 and 10.0% for Deep 1. At 24/75 removed,
Shallow 1.1 fell to 1/1,000 (0.1%), versus 18.8% for Shallow 1 and 91.3% for
Deep 1. At 32/75 removed, Shallow 1.1 scored 0/1,000 (0.0%), versus 67.2%
for Shallow 1 and 83.7% for Deep 1. At 40/75 removed, Shallow 1.1 remained at
0/1,000 (0.0%), versus 14.6% for Shallow 1 and 45.2% for Deep 1. It again
scored 0/1,000 (0.0%) at 48/75 removed, versus 10.4% for Shallow 1 and 14.4%
for Deep 1. At 56/75 removed, Shallow 1.1, Shallow 1, and Deep 1 all scored
0/1,000 (0.0%). At 64/75 removed, Shallow 1.1 remained at 0/1,000 (0.0%),
versus 0.2% for Shallow 1 and Deep 1's sharp rebound to 89.9%. At 72/75
removed, Shallow 1.1 and Shallow 1 both scored 0/1,000 (0.0%), while Deep 1
scored 3.3%. At the first 75/75 full-removal boundary, Shallow 1.1 remained at
0/1,000 (0.0%), versus 0.2% for Shallow 1 and 82.1% for Deep 1. After one
full-removal adaptation epoch it remained at 0/1,000 (0.0%), versus 0.4% for
Shallow 1 and 91.3% for Deep 1. After a second adaptation epoch it again
scored 0/1,000 (0.0%), versus 0.2% for Shallow 1 and 91.0% for Deep 1. After a
third adaptation epoch it remained at 0/1,000 (0.0%), versus 0.5% for Shallow
1 and 96.5% for Deep 1. After a fourth adaptation epoch it again scored
0/1,000 (0.0%), versus 0.3% for Shallow 1 and 96.6% for Deep 1. The fifth and
final adaptation epoch remained at 0/1,000 (0.0%), versus 0.6% for Shallow 1
and 96.3% for Deep 1. Held-out test was also 0/1,000 (0.0%), versus 0.3% for
Shallow 1 and 95.8% for Deep 1.

| Curriculum checkpoint | Shallow 1.1 recurrent | Shallow 1 | Deep 1 |
| --- | ---: | ---: | ---: |
| Explicit warmup (0 / 75) | 99.4% | 100.0% | 100.0% |
| 8 / 75 | 6.7% | 91.5% | 95.3% |
| 16 / 75 | 6.4% | 32.3% | 10.0% |
| 24 / 75 | 0.1% | 18.8% | 91.3% |
| 32 / 75 | 0.0% | 67.2% | 83.7% |
| 40 / 75 | 0.0% | 14.6% | 45.2% |
| 48 / 75 | 0.0% | 10.4% | 14.4% |
| 56 / 75 | 0.0% | 0.0% | 0.0% |
| 64 / 75 | 0.0% | 0.2% | 89.9% |
| 72 / 75 | 0.0% | 0.0% | 3.3% |
| First 75 / 75 | 0.0% | 0.2% | 82.1% |
| Adaptation 1 | 0.0% | 0.4% | 91.3% |
| Adaptation 2 | 0.0% | 0.2% | 91.0% |
| Adaptation 3 | 0.0% | 0.5% | 96.5% |
| Adaptation 4 | 0.0% | 0.3% | 96.6% |
| Adaptation 5 (final validation) | 0.0% | 0.6% | 96.3% |
| Held-out test | 0.0% | 0.3% | 95.8% |

The full run lasted 7:10:22.743 (2026-10-19 08:38:43Z through
15:49:05.743Z). Account balance changed from $56.4655598618 before setup to
$48.3601044425 after teardown, an exact $8.1054554193 account-level delta that
includes other concurrent Runpod work. The pod-specific billing read recorded
$4.0673626156 through 15:00 UTC; its final partial bucket was still pending.
Pod `vpp9263s4vctvi` was deleted after local artifact verification.

Git-ignored artifacts:

- `artifacts/checkpoints/shallow1-1-recurrent-5x5-warmup/recurrent_warmup_latest.pt`
  (657,214,613 bytes; SHA256 `e1856b58f0fc349ee07431f54ad17aa050865026207522a2b767569e5c7ff765`)
- `artifacts/checkpoints/shallow1-1-recurrent-5x5-internalized-cot/latest.pt`
  (657,213,895 bytes; SHA256 `45f8d1de55f3f957b1cc593bb2a927f170831764889450dee7020a09a7ebd6a4`)

This one-seed pilot does not identify a causal recurrence or depth effect: the
architectures differ, only the recurrent adapters trained, and the baseline
models retain a 3.12-point LAMBADA capability difference.

- [`shallow1-1-recurrent-5x5-internalized-cot-progress.json`](shallow1-1-recurrent-5x5-internalized-cot-progress.json)
- [`shallow1-1-recurrent-5x5-internalized-cot-training-config.json`](shallow1-1-recurrent-5x5-internalized-cot-training-config.json)
- [`shallow1-1-recurrent-5x5-internalized-cot-test-metrics.json`](shallow1-1-recurrent-5x5-internalized-cot-test-metrics.json)
- [`shallow1-1-recurrent-5x5-internalized-cot-summary.json`](shallow1-1-recurrent-5x5-internalized-cot-summary.json)

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
The 47-token internalization curriculum completed all eight scheduled epochs.
At its first epoch boundary, Deep 1 scored 933/1,000 (93.3%) with 8/47 CoT
tokens removed, versus 852/1,000 (85.2%) for Shallow 1 at the identical
boundary, an observed gap of +8.1 percentage points. At 16/47, Deep 1 fell to
692/1,000 (69.2%) while Shallow 1 had fallen further to 177/1,000 (17.7%), an
observed gap of +51.5 points. At 24/47, Deep 1 declined again to 173/1,000
(17.3%), versus 74/1,000 (7.4%) for Shallow 1, a +9.9-point gap. At 32/47,
Deep 1 scored 82/1,000 (8.2%) versus 7/1,000 (0.7%) for Shallow 1, a +7.5-point
gap. At 40/47, both curves rebounded: Deep 1 scored 899/1,000 (89.9%) versus
791/1,000 (79.1%) for Shallow 1, a +10.8-point gap. At the first 47/47
checkpoint—before any full-removal adaptation—Deep 1 scored 802/1,000 (80.2%)
versus Shallow 1's 651/1,000 (65.1%), a +15.1-point gap. After one full-removal
adaptation epoch, Deep 1 reached 931/1,000 (93.1%) versus Shallow 1's 799/1,000
(79.9%), a +13.2-point gap. The final Deep 1 checkpoint reached 939/1,000
(93.9%) validation and 919/1,000 (91.9%) test, versus Shallow 1's 923/1,000
(92.3%) on both splits: +1.6 points on validation but -0.4 points on test.
Both models are below the
99% primary and 95% sensitivity thresholds, so both first observed frontiers
are bracketed between 0 and 8 removed tokens. The curve is non-monotonic and
the full-removal checkpoints differ in adaptation time. Together with the
slightly lower Deep test accuracy and Deep 1's 3.12-point LAMBADA advantage,
this prevents a causal depth-only interpretation.

| Curriculum checkpoint | Deep 1 | Shallow 1 | Deep minus shallow |
| --- | ---: | ---: | ---: |
| Explicit stage (0 / 47) | 100.0% | 100.0% | 0.0 pp |
| Epoch 0 (8 / 47) | 93.3% | 85.2% | +8.1 pp |
| Epoch 1 (16 / 47) | 69.2% | 17.7% | +51.5 pp |
| Epoch 2 (24 / 47) | 17.3% | 7.4% | +9.9 pp |
| Epoch 3 (32 / 47) | 8.2% | 0.7% | +7.5 pp |
| Epoch 4 (40 / 47) | 89.9% | 79.1% | +10.8 pp |
| Epoch 5, first full removal (47 / 47) | 80.2% | 65.1% | +15.1 pp |
| Epoch 6, one adaptation epoch (47 / 47) | 93.1% | 79.9% | +13.2 pp |
| Epoch 7, final validation (47 / 47) | 93.9% | 92.3% | +1.6 pp |
| Final test (47 / 47) | 91.9% | 92.3% | -0.4 pp |

The final checkpoint is stored locally at
`artifacts/checkpoints/deep1-4x4-internalized-cot/latest.pt` (648,838,501
bytes), SHA256
`d55d5dcd70edb9a7b9e895d55d36933d1556be5355692b2921c4fba38205d348`.
The serial pipeline ran from 2026-10-10 07:33:11Z through 14:21:32Z
(6:48:21). The Runpod balance changed from $13.5142354785 to $8.3638714864,
an exact delta of $5.1503639921 through artifact verification. Both pods were
deleted after verification; network volume `vpzu3qptxw` was retained.

- [`deep1-4x4-explicit-cot-validation.json`](deep1-4x4-explicit-cot-validation.json)
- [`deep1-4x4-explicit-cot-test.json`](deep1-4x4-explicit-cot-test.json)
- [`deep1-4x4-explicit-cot-training-config.json`](deep1-4x4-explicit-cot-training-config.json)
- [`deep1-4x4-internalized-cot-progress.json`](deep1-4x4-internalized-cot-progress.json)
- [`deep1-4x4-internalized-cot-training-config.json`](deep1-4x4-internalized-cot-training-config.json)
- [`deep1-4x4-internalized-cot-validation.json`](deep1-4x4-internalized-cot-validation.json)
- [`deep1-4x4-internalized-cot-test.json`](deep1-4x4-internalized-cot-test.json)
- [`deep1-4x4-internalized-cot-summary.json`](deep1-4x4-internalized-cot-summary.json)

## Matched-loss Deep 1 on 5x5 explicit CoT

Starting from the untouched matched-loss Deep 1 FineWeb checkpoint, the
12-layer, width-768 model completed one paper-matched FP32 epoch on the released
808,000-example 5x5 training set (25,250 optimizer steps). The run used AdamW
at `5e-5`, batch size 32, gradient clipping at 1.0, and seed 3456. Input-only
greedy evaluation supplied no ground-truth reasoning tokens and required the
model to generate the full visible trace and final answer.

| Split | Correct | Examples | Final-answer exact match |
| --- | ---: | ---: | ---: |
| 5x5 validation | 1,000 | 1,000 | 100.0% |
| 5x5 test | 1,000 | 1,000 | 100.0% |

The checkpoint is archived locally at
`artifacts/checkpoints/deep1-5x5-explicit-cot/explicit_cot_epoch_000.pt`
(648,839,285 bytes) and verified at SHA256
`50e220806977ed7ff6f19ed413473e88bb1df574875f835e6410ee8b7f0f65bb`.
The standard 75-token internalization curriculum completed all 15 prescribed
epochs from this checkpoint. It did not meet the 99% early-stop target.

The first completed internalization checkpoint retained 953/1,000 (95.3%)
validation accuracy with 8/75 reasoning tokens removed. Shallow 1 scored
915/1,000 (91.5%) at the identical boundary, an observed Deep-minus-Shallow
gap of +3.8 percentage points. This is below the predeclared 5-point reporting
threshold. At 16/75, Deep 1 then fell to 100/1,000 (10.0%), below Shallow 1's
323/1,000 (32.3%) by 22.3 points. At 24/75, Deep rebounded to 913/1,000
(91.3%), while Shallow reached 188/1,000 (18.8%), an observed +72.5-point
gap. The curve is therefore sharply non-monotonic and does not support a
uniformly greater Deep internalization capacity. At 32/75, Deep retained
837/1,000 (83.7%) versus Shallow's 672/1,000 (67.2%), an observed +16.5-point
gap. At 40/75, Deep fell to 452/1,000 (45.2%) but remained above Shallow's
146/1,000 (14.6%) by 30.6 points. At 48/75, Deep fell further to 144/1,000
(14.4%) versus Shallow's 104/1,000 (10.4%), narrowing the observed gap to
+4.0 points. At 56/75, both models scored 0/1,000 (0.0%), so the observed gap
collapsed completely at that boundary. At 64/75, Deep rebounded dramatically to
899/1,000 (89.9%), while Shallow scored 2/1,000 (0.2%), an observed +89.7-point
gap. This second rebound further emphasizes that the curriculum curve is highly
non-monotonic. At 72/75, Deep fell again to 33/1,000 (3.3%), while Shallow
scored 0/1,000 (0.0%), narrowing the observed gap to +3.3 points. The first
fully internalized checkpoint then rebounded to 821/1,000 (82.1%), versus
Shallow's 2/1,000 (0.2%), an observed +81.9-point gap. One full-removal
adaptation epoch raised Deep to 913/1,000 (91.3%), while Shallow's matched point
was 4/1,000 (0.4%), an observed +90.9-point gap. Deep has not yet met the 99%
stopping threshold. A second adaptation epoch held Deep at 910/1,000 (91.0%),
versus Shallow's matched 2/1,000 (0.2%), an observed +90.8-point gap, so
full-removal adaptation continued. A third adaptation epoch then raised Deep to
965/1,000 (96.5%), versus Shallow's matched 5/1,000 (0.5%), an observed
+96.0-point gap. A fourth adaptation epoch reached 966/1,000 (96.6%) versus
Shallow's 3/1,000 (0.3%), a +96.3-point gap. The fifth and final adaptation
epoch ended at 963/1,000 (96.3%) versus Shallow's 6/1,000 (0.6%), a +95.7-point
gap. On the held-out test set, Deep scored 958/1,000 (95.8%) and Shallow scored
3/1,000 (0.3%), an observed +95.5-point gap.

| Curriculum checkpoint | Deep 1 | Shallow 1 | Deep minus shallow |
| --- | ---: | ---: | ---: |
| Explicit stage (0 / 75) | 100.0% | 100.0% | 0.0 pp |
| Epoch 0 (8 / 75) | 95.3% | 91.5% | +3.8 pp |
| Epoch 1 (16 / 75) | 10.0% | 32.3% | -22.3 pp |
| Epoch 2 (24 / 75) | 91.3% | 18.8% | +72.5 pp |
| Epoch 3 (32 / 75) | 83.7% | 67.2% | +16.5 pp |
| Epoch 4 (40 / 75) | 45.2% | 14.6% | +30.6 pp |
| Epoch 5 (48 / 75) | 14.4% | 10.4% | +4.0 pp |
| Epoch 6 (56 / 75) | 0.0% | 0.0% | 0.0 pp |
| Epoch 7 (64 / 75) | 89.9% | 0.2% | +89.7 pp |
| Epoch 8 (72 / 75) | 3.3% | 0.0% | +3.3 pp |
| Epoch 9 (first 75 / 75) | 82.1% | 0.2% | +81.9 pp |
| Epoch 10 (adaptation 1) | 91.3% | 0.4% | +90.9 pp |
| Epoch 11 (adaptation 2) | 91.0% | 0.2% | +90.8 pp |
| Epoch 12 (adaptation 3) | 96.5% | 0.5% | +96.0 pp |
| Epoch 13 (adaptation 4) | 96.6% | 0.3% | +96.3 pp |
| Epoch 14 (adaptation 5, final) | 96.3% | 0.6% | +95.7 pp |
| Final test | 95.8% | 0.3% | +95.5 pp |

Deep's first fully internalized checkpoint was 82.1%, and its final test score
was 95.8%; the matched Shallow values were 0.2% and 0.3%. This large observed
gap is not a causal depth-only estimate: the comparison is one seed, both
curves are sharply non-monotonic, and Deep retains a 3.12-point LAMBADA
advantage (29.42% versus 26.30%).

The final checkpoint is archived locally at
`artifacts/checkpoints/deep1-5x5-internalized-cot/latest.pt` (648,838,501
bytes), SHA256
`f5b0300d4d1d54e681b7975c458d37261438b6b74f7b0168884a32d5839cda28`.
The pipeline ran from 2026-10-10 23:59:23Z through 2026-10-11 02:52:11Z
(2:52:48). Including the user's $10 mid-run account top-up, adjusted available
balance changed from $18.3130256562 to $10.1883324393, an exact experiment
delta of $8.1246932169 through artifact verification. The pod was deleted;
network volume `vpzu3qptxw` was retained and the remaining spend rate is
$0.002/hour.

The localhost dashboard overlays Deep 1 and Shallow 1 for explicit-CoT loss,
internalization loss, and validation accuracy on 3x3, 4x4, and 5x5. Every
refresh also writes a self-contained HTML snapshot, complete JSON payload, and
per-series CSV files under `artifacts/dashboards/deep1-5x5/` so the view and raw
chart data remain available after the live run ends.

- [`deep1-5x5-explicit-cot-validation.json`](deep1-5x5-explicit-cot-validation.json)
- [`deep1-5x5-explicit-cot-test.json`](deep1-5x5-explicit-cot-test.json)
- [`deep1-5x5-explicit-cot-training-config.json`](deep1-5x5-explicit-cot-training-config.json)
- [`deep1-5x5-internalized-cot-progress.json`](deep1-5x5-internalized-cot-progress.json)
- [`deep1-5x5-internalized-cot-training-config.json`](deep1-5x5-internalized-cot-training-config.json)
- [`deep1-5x5-internalized-cot-validation.json`](deep1-5x5-internalized-cot-validation.json)
- [`deep1-5x5-internalized-cot-test.json`](deep1-5x5-internalized-cot-test.json)
- [`deep1-5x5-internalized-cot-summary.json`](deep1-5x5-internalized-cot-summary.json)

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
