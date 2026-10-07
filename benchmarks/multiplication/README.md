# NxN multiplication held-out benchmarks

This directory vendors the canonical validation and test splits released with
*From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step*.
They are the controlled multiplication benchmarks used to measure how far a
model can internalize an explicit long-multiplication trace.

## Included splits

Each size contains 1,000 validation and 1,000 test examples:

| Directory | Validation | Test | Intended role |
| --- | ---: | ---: | --- |
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

## Provenance

Source: [`da03/Internalize_CoT_Step_by_Step`](https://github.com/da03/Internalize_CoT_Step_by_Step),
commit `c2e3891d8d577beeaff03fa3b076d1e4db96cfe8`.

The original files are named `valid.txt` and `test_bigbench.txt`; only their
filenames have been normalized here. The data is redistributed under the
authors' MIT license in `SOURCE_LICENSE`.
