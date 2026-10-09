# LAMBADA capability check

Shallow 1, its same-recipe matched-loss Deep 1 control, a community-uploaded
12-layer modded-nanoGPT checkpoint, and OpenAI GPT-2 Small were evaluated on all 5,153 examples in the
`EleutherAI/lambada_openai` test split. The scorer follows lm-evaluation-harness
task version 1.0: the final space-delimited word is the target, accuracy requires
teacher-forced greedy exact match over every target token, and perplexity is
aggregated from target-word log likelihood. Both models used the GPT-2 tokenizer,
FP32 parameters and the Apple M4 GPU backend. Deep 1 used batch size 8 to fit
the optimizer-bearing checkpoint in memory; the other runs used batch size 16.
Batch size affects timing but not the teacher-forced predictions.

| Model | Correct | Accuracy | Target-word perplexity | M4 elapsed | Logical input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow 1 | 1,322 / 5,153 | 25.65% | 70.34 | 54.1 s | 7,862 |
| Shallow 1 continued (step 8,703) | 1,355 / 5,153 | 26.30% | 68.48 | 53.4 s | 7,953 |
| Deep 1 matched (step 4,200) | 1,516 / 5,153 | 29.42% | 55.42 | 84.9 s | 5,008 |
| Community Deep 3242 | 1,628 / 5,153 | 31.59% | 45.50 | 78.9 s | 5,390 |
| OpenAI GPT-2 Small | 1,678 / 5,153 | 32.56% | 40.06 | 79.2 s | 5,365 |

Deep 1 leads the matched-loss Shallow 1 checkpoint by **3.12 percentage
points** (1,516 versus 1,355 correct). The paired 20,000-replicate bootstrap
95% interval is 2.06–4.19 points. Deep 1 alone is correct on 470 examples,
Shallow 1 alone on 309, both on 1,046, and neither on 3,328; the exact
two-sided McNemar test gives `p = 8.80e-9`. Target-word perplexity also improves
from 68.48 to 55.42.

This is an important qualification to the experimental control. The models
have exactly 162,201,600 parameters, use the same pretraining implementation,
data, tokenizer, and validation set, and their FineWeb losses differ by only
0.000415. Nevertheless, matching one aggregate loss does not make broad
capability identical: Deep 1 remains measurably better on LAMBADA. The paired
multiplication experiment is still necessary, and any internalization gap must
be interpreted alongside this residual capability difference.

GPT-2 Small leads Deep 1 by 3.14 points (paired 95% interval 1.98–4.29;
McNemar `p = 9.35e-8`). Community Deep 3242 leads Deep 1 by 2.17 points
(1.09–3.26; `p = 8.14e-5`).

The constant-rate continuation improves Shallow 1 by 33 correct examples, or
0.64 percentage points. On paired examples, the continued checkpoint alone is
correct 195 times, the original alone is correct 162 times, both are correct
1,160 times, and both are wrong 3,636 times. The 20,000-replicate paired
bootstrap interval is -0.08 to 1.36 percentage points and the exact two-sided
McNemar test gives `p = 0.090`; the observed gain is therefore suggestive but
not statistically conclusive. Target-word perplexity improves from 70.34 to
68.48.

GPT-2 Small still leads the continued checkpoint by 6.27 percentage points
(paired 95% interval 5.08–7.45), and Community Deep 3242 leads it by 5.30
points (4.25–6.35). Thus the continuation narrows but does not close the broad
capability gap.

Community Deep 3242 leads Shallow 1 by 5.94 percentage points. A
20,000-replicate paired bootstrap gives a 95% interval of 4.87–7.03 points.
The community model alone is correct 556 times, Shallow 1 alone is correct 250
times, both are correct 1,072 times, and both are wrong 3,275 times. An exact
two-sided McNemar test gives `p = 1.48e-27`.

OpenAI GPT-2 Small leads Community Deep 3242 by 0.97 points, but the paired 95%
interval is -0.17–2.10 points and the exact McNemar test gives `p = 0.097`.

GPT-2 Small leads by 6.91 percentage points. A 20,000-replicate paired
bootstrap gives a 95% interval of 5.69–8.09 points. On the paired examples,
GPT-2 alone is correct 679 times, Shallow 1 alone is correct 323 times, both are
correct 999 times, and both are wrong 3,152 times. An exact two-sided McNemar
test gives `p = 8.62e-30`.

These scores are not equal, but they are comparable enough to rule out the
uninteresting picture that Shallow 1 is simply a nonfunctional language model.
That makes the multiplication contrast informative: the same Shallow 1
checkpoint can be fine-tuned to 100.0% explicit-CoT accuracy on held-out 5x5
multiplication, yet its final fully internalized checkpoint reaches only
3/1,000 (0.3%) test accuracy after all 75 CoT tokens are hidden. Broad language
capability and perfect acquisition of the visible algorithm therefore coexist
with failure of this internalized-computation protocol. This cross-task contrast
does not isolate depth; it motivates the still-needed matched deep 5x5 run.

Shallow 1 is 1.47x faster than GPT-2 Small for this batched M4 scoring workload,
consistent with its six rather than twelve serial transformer blocks. This is a
throughput result, not autoregressive decoding speed. On a separate 128-example
batch-one probe, Shallow 1 processed 1,775 logical input tokens/s and GPT-2 Small
processed 2,065 tokens/s; small batches do not saturate Shallow 1's wider layers
as effectively.

Community Deep 3242 has 12 layers × 768 width and 162,201,636 parameters; Shallow
1 has 6 layers × 1,024 width and 162,201,600 parameters. The 36-parameter
difference comes from the community model's learned residual and value-mixing
gates. Despite this excellent parameter match, the comparison is not a causal
depth result: Community Deep 3242 reached reported FineWeb validation loss
3.2766 while Shallow 1 reached 3.3411, and the community run used cross-layer
value sharing, learned residual mixing, a logit soft-cap, fewer training steps,
and a later optimization recipe. It is therefore a useful capability control,
but not the same-training-recipe Deep 1 control required by the experiment.

The checkpoint is a community upload, not the unpublished official 10.8-minute
checkpoint. Its local 1.0 GB artifact is excluded from Git and has SHA256
`fd7c4e06f79fb0dfea83b6e370aec80b49ae741dd158fc2ad6d141334049ac10`.

- [`comparison.json`](comparison.json)
- [`shallow-1.json`](shallow-1.json)
- [`shallow-1-continuation-step8703.json`](shallow-1-continuation-step8703.json)
- [`deep-1.json`](deep-1.json)
- [`community-deep-3242.json`](community-deep-3242.json)
- [`gpt2-small.json`](gpt2-small.json)
