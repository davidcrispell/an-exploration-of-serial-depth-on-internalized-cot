# NLS-depth measurements with Google DeepMind's serial-depth calculator

These measurements use the unmodified calculator from
[`google-deepmind/serial_depth`](https://github.com/google-deepmind/serial_depth)
at commit
[`9d9f53c`](https://github.com/google-deepmind/serial_depth/commit/9d9f53c7f1b27ad692ece5e48f9c586d2f17984c).
The calculator assigns depth zero to data movement, depth one to elementwise
operations, logarithmic depth to dot products and reductions, and cumulative
depth to sequentially dependent operations.

Throughout this project, **NLS depth** means the forward-pass serial depth
returned by this implementation. It depends on input length, so every value is
reported together with its sequence length.

Because the released calculator traces JAX rather than PyTorch, the project
script [`scripts/measure_google_serial_depth.py`](../../scripts/measure_google_serial_depth.py)
contains an operation-faithful JAX transcription of the inference paths in
`model.py`, `recurrent.py`, and `paper_recurrent.py`. Parameters are abstract
shape inputs, so the computation graph is measured without loading or
allocating a checkpoint. All traces below have **zero unsupported primitives**.

## Results

| Model / inference mode | Parameters | Effective transformer blocks | Prompt length 13 | 5x5 train length 103 | Pretrain length 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow 1 | 162,201,600 | 6 | 769 | 823 | 877 |
| Deep 1 | 162,201,600 | 12 | 1,495 | 1,603 | 1,711 |
| Deeper 1 | 162,201,600 | 24 | 2,947 | 3,163 | 3,379 |
| Shallow 1.1 frozen pilot | 164,298,752 | 12 | 1,552 | 1,660 | 1,768 |
| Paper recurrent Shallow 1.1, r=1 | 164,299,777 | 6 | 800 | 854 | 908 |
| Paper recurrent Shallow 1.1, r=2 | 164,299,777 | 10 | 1,315 | 1,405 | 1,495 |
| Paper recurrent Shallow 1.1, r=4 | 164,299,777 | 18 | 2,345 | 2,507 | 2,669 |
| Paper recurrent Shallow 1.1, r=8 | 164,299,777 | 34 | 4,405 | 4,711 | 5,017 |

At the actual 103-token 5x5 training length, the three fixed-depth models obey

```text
serial depth = 43 + 130 × transformer blocks.
```

Their widths differ, but the calculator uses ceiling-logarithmic reduction
depths, and all three configurations land in the same relevant power-of-two
buckets. Therefore parameter-matched Deep 1 is 1.95× Shallow 1's measured
serial depth, while Deeper 1 is 3.84× Shallow 1's. The small non-layer overhead
prevents the ratios from being exactly 2× and 4×.

For paper-recurrent Shallow 1.1 at the same sequence length,

```text
serial depth = 303 + 551 × recurrence count.
```

The extra 31 units per recurrence beyond four ordinary 130-depth blocks come
from the learned 2h-to-h state/input adapter and recurrent output-normalization
path. At the primary r=4 setting, its measured serial depth is 2,507: 1.56×
Deep 1, 3.05× Shallow 1, and lower than the 24-block Deeper 1 value of 3,163.

These are semantic forward-pass graph measurements under the released rules,
not wall-clock latency, FLOPs, bits stored in the weights, or a measurement of
the backward pass. The PyTorch implementation uses fused attention kernels,
but the transcription exposes the equivalent dot-product, masking, softmax,
and value-reduction operations so the calculator can apply its stated rules.

- [`google-serial-depth.json`](google-serial-depth.json) contains every trace,
  operation count, architecture field, and unsupported-primitive audit.
