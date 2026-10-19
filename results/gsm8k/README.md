# Paired GSM8K internalized-CoT pilot

This directory records the completed one-seed paired pilot (`seed=3456`). Both models have 162,201,600 parameters. The deep model is 12x768 with 6 heads and MLP 3072; the shallow model is 6x1024 with 8 heads and MLP 2768.

The held-out test split was provenance-hashed but **not evaluated**. All reported accuracies use the 500-example validation split.

## Results

| Stage | Deep | Shallow |
|---|---:|---:|
| Explicit CoT, epoch 0 | 16.0% (80/500) | 14.2% (71/500) |
| Internalization, removal 0 | 20.8% | 21.0% |
| Removal 8 | 22.8% | 19.6% |
| Removal 16 | 16.4% | 12.8% |
| Removal 24 | 17.6% | 14.4% |
| Removal 32 | 18.8% | 14.8% |
| First full removal, epoch 39 | 17.8% | 13.6% |
| Full-removal adaptation, epoch 40 | 17.8% | 11.8% |
| Full-removal adaptation, epoch 41 | 17.6% | 14.2% |
| Full-removal adaptation, epoch 42 | 18.2% | 11.6% |

The pipeline ran for 34h 21m 36s. The A40 pod existed for 35h 6m 19.355s at $0.59/hour, giving an isolated pilot cost of $20.7122. Raw account-balance movement is confounded by another unrelated pod and is not attributed to this pilot.

Machine-readable results, protocol details, dataset/base hashes, checkpoint hashes, runtimes, and cost are in [`paired-pilot.json`](paired-pilot.json). Loss and validation curves are in [`curves/`](curves/), and exported PNG/SVG/PDF figures are in [`figures/`](figures/).

## Interpretation boundary

This one-seed pilot is descriptive. It does not establish a causal effect of depth, especially because residual baseline capability was not fully matched.
