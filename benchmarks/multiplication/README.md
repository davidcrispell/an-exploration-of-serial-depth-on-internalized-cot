# NxN multiplication held-out benchmarks

This directory vendors the canonical validation and test splits released with
*From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step*.
They are the controlled multiplication benchmarks used to measure how far a
model can internalize an explicit long-multiplication trace.

## Included splits

Each size contains 1,000 validation and 1,000 test examples:

| Directory | Validation | Test | Intended role |
| --- | ---: | ---: | --- |
| `data/3x3/` | 1,000 | 1,000 | custom Shallow 1 probe |
| `data/4x4/` | 1,000 | 1,000 | pipeline sanity check |
| `data/5x5/` | 1,000 | 1,000 | easy transition point |
| `data/7x7/` | 1,000 | 1,000 | intermediate internalization |
| `data/9x9/` | 1,000 | 1,000 | primary full-internalization target |
| `data/11x11/` | 1,000 | 1,000 | partial-internalization stress test |

Do model selection and curriculum decisions with `validation.txt`. Evaluate
`test.txt` only after the procedure and checkpoint are fixed.

## Record format

Digits are written least-significant first:

```text
[reversed operands]||[long-multiplication trace] #### [reversed product]
```

For example, `9 1 7 3 * 9 4 3 3` represents `3719 * 3349`. The output is
zero-padded to `2N` digits and reversed. The trace consists of shifted partial
products and parenthesized cumulative sums.

Validate every split with:

```bash
python -m benchmarks.multiplication.validate
```

The validator checks all 10,000 records for:

- exact operand digit lengths;
- the final multiplication result;
- every shifted partial product;
- every intermediate cumulative sum;
- duplicate commutative operand pairs; and
- leakage between validation and test splits.

## Training stages

`train_explicit_cot.py` first teaches the full visible long-multiplication trace.
`train_internalized_cot.py` then removes the trace from left to right at the
paper's rate of eight GPT-2 tokens per epoch. The uniform traces contain 47
tokens for 4x4 and 75 tokens for 5x5. These are tokenizer positions, not labeled
semantic reasoning steps. The removal boundary uses exponential smoothing with
lambda 4, and AdamW is reset whenever one more token is scheduled for removal.
Both stages use the same GPT-2 tokenization and held-out evaluator.

### Custom 3x3 extension

The authors' released benchmark starts at 4x4. For the Shallow 1 probe,
`generate.py` deterministically extends the same format to 3x3. Seed 3456
selects 1,000 validation and 1,000 test pairs without commutative duplicates,
and both orientations of every held-out pair are excluded from training. The
training split contains all remaining ordered three-digit pairs (806,000
examples), making 808,000 examples across the three splits. Its CoT is 25
GPT-2 tokens. This is a custom extrapolation, not a paper-released split.

## Provenance

Source: [`da03/Internalize_CoT_Step_by_Step`](https://github.com/da03/Internalize_CoT_Step_by_Step),
commit `c2e3891d8d577beeaff03fa3b076d1e4db96cfe8`.

For the 4x4 through 11x11 data, the original files are named `valid.txt` and
`test_bigbench.txt`; only their filenames have been normalized here. Those
files are redistributed under the authors' MIT license in `SOURCE_LICENSE`.
