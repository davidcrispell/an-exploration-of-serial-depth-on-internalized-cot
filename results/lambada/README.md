# LAMBADA capability check

Shallow 1 and OpenAI GPT-2 Small were evaluated on all 5,153 examples in the
`EleutherAI/lambada_openai` test split. The scorer follows lm-evaluation-harness
task version 1.0: the final space-delimited word is the target, accuracy requires
teacher-forced greedy exact match over every target token, and perplexity is
aggregated from target-word log likelihood. Both models used the GPT-2 tokenizer,
FP32 parameters, batch size 16, and the Apple M4 GPU backend.

| Model | Correct | Accuracy | Target-word perplexity | M4 elapsed | Logical input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow 1 | 1,322 / 5,153 | 25.65% | 70.34 | 54.1 s | 7,862 |
| OpenAI GPT-2 Small | 1,678 / 5,153 | 32.56% | 40.06 | 79.2 s | 5,365 |

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

The capability result is not a causal depth comparison. OpenAI GPT-2 Small and
Shallow 1 differ in pretraining corpus, architecture, parameter accounting, and
optimizer. The required control remains Deep 1: the parameter-matched 12-layer
modded-nanoGPT checkpoint evaluated on these identical examples.

- [`comparison.json`](comparison.json)
- [`shallow-1.json`](shallow-1.json)
- [`gpt2-small.json`](gpt2-small.json)
