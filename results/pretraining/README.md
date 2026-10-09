# Shallow 1 pretraining continuation

The continuation starts from the untouched 6-layer Shallow 1 FineWeb
checkpoint at total step 4,578 and validation loss 3.3411. The parent weights
have SHA256
`c5e54086d9ef419f2c43591eca0887bd5ed5e8fed52dd3a329363f807a3f49e0`.
Optimizer state is reset because the original distributed checkpoint preserves
only rank zero's Muon momentum shards.

The active Runpod phase uses one RTX PRO 6000 Blackwell Server Edition GPU,
unseen FineWeb shards 25-35, an effective batch of 512 sequences of 1,024
tokens, 100 warmup updates, 500 warmdown updates, and a 0.2 learning-rate
scale. It is capped at 2,000 updates and stops early at validation loss 3.25.
Validation runs every 125 updates and `latest.pt` is overwritten every 250
updates.

## Local feasibility probe

An M4 MacBook Air with 8 GPU cores and 16 GB unified memory completed a
single MPS optimizer update with an effective batch of 8 in 5.840 seconds.
Post-update MPS allocation was 1,676 MiB. Linear extrapolation to the matched
effective batch of 512 is 373.8 seconds per update, or about 8.65 days for
2,000 updates. This is only a throughput estimate: larger device batches could
improve utilization, but local training is not a cost- or time-matched control.

The one-GPU Runpod probe sustained 1.512 seconds per matched update after
compilation and used 23,257 MiB peak GPU memory. Its projected optimizer time
for 2,000 updates is 50.4 minutes, before validation and checkpoint overhead.

The continuation reproduced the parent validation loss of 3.3411 at step 0.
At continuation step 125 (total step 4,703), validation loss was 3.3457. This
small initial regression occurred just after the 100-update warmup and does not
establish the eventual direction of the continuation. Subsequent validation
losses were 3.3440 at step 250, 3.3414 at step 375, and 3.3390 at step 500.
The post-warmup curve is improving: step 500 is 0.0021 below the parent loss of
3.3411. Validation loss then reached 3.3368 at step 625, 3.3348 at step 750,
and 3.3335 at step 875. The latest checkpoint is 0.0076 below the parent while
remaining 0.0835 above the 3.25 target. Validation loss continued to 3.3319 at
step 1,000, 3.3307 at step 1,125, and 3.3288 at step 1,250. The latest point is
0.0123 below the parent and 0.0788 above the target.
