# LAMBADA capability check

Shallow 1, a community-uploaded 12-layer modded-nanoGPT checkpoint, and OpenAI
GPT-2 Small were evaluated on all 5,153 examples in the
`EleutherAI/lambada_openai` test split. The scorer follows lm-evaluation-harness
task version 1.0: the final space-delimited word is the target, accuracy requires
teacher-forced greedy exact match over every target token, and perplexity is
aggregated from target-word log likelihood. Both models used the GPT-2 tokenizer,
FP32 parameters, batch size 16, and the Apple M4 GPU backend.

| Model | Correct | Accuracy | Target-word perplexity | M4 elapsed | Logical input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow 1 | 1,322 / 5,153 | 25.65% | 70.34 | 54.1 s | 7,862 |
| Community Deep 3242 | 1,628 / 5,153 | 31.59% | 45.50 | 78.9 s | 5,390 |
| OpenAI GPT-2 Small | 1,678 / 5,153 | 32.56% | 40.06 | 79.2 s | 5,365 |

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
- [`community-deep-3242.json`](community-deep-3242.json)
- [`gpt2-small.json`](gpt2-small.json)
