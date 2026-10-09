# Shallow 1 pretraining continuation

The continuation starts from the untouched 6-layer Shallow 1 FineWeb
checkpoint at total step 4,578 and validation loss 3.3411. The parent weights
have SHA256
`c5e54086d9ef419f2c43591eca0887bd5ed5e8fed52dd3a329363f807a3f49e0`.
Optimizer state is reset because the original distributed checkpoint preserves
only rank zero's Muon momentum shards.

The completed Runpod phase used one RTX PRO 6000 Blackwell Server Edition GPU,
unseen FineWeb shards 25-35, an effective batch of 512 sequences of 1,024
tokens, 100 warmup updates, 500 warmdown updates, and a 0.2 learning-rate
scale. It was capped at 2,000 updates with an early-stop target of validation
loss 3.25. Validation ran every 125 updates and `latest.pt` was overwritten every 250
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
0.0123 below the parent and 0.0788 above the target. Validation loss then
reached 3.3280 at step 1,375, 3.3265 at step 1,500, and 3.3223 at step 1,625.
The final 500-update cooldown begins at step 1,500; the latest checkpoint is
0.0188 below the parent and 0.0723 above the target. The last three validation
losses were 3.3170 at step 1,750, 3.3120 at step 1,875, and **3.3092277 at step
2,000**. The continuation therefore improved the parent by 0.0318723 but ended
0.0592277 above the 3.25 target and 0.0326277 above the community Deep 3242
checkpoint's reported 3.2766. It processed 1,048,576,000 additional training
tokens.

The measured training timer was 3,041.318 seconds (50m 41s), excluding initial
compilation, validation, and checkpoint I/O. Pod creation through the final
checkpoint took 1h 01m 26s, and artifact recovery plus pod deletion brought the
total billed lifecycle to approximately 1h 04m. The observed Runpod balance
delta was **$2.4993515471**. The GPU pod was deleted and network volume
`vpzu3qptxw` was retained.

The verified final checkpoint is stored locally at
`artifacts/checkpoints/shallow1-fineweb-continuation/latest.pt` with SHA256
`27964d91b2a36700340db5a89227dcdb8f226ee2f7404d6ef06d235480a82b6b`.
The checkpoint, training configuration, and full logs are Git-ignored; compact
metrics are tracked in `shallow1-continuation-start.json`.

## Constant-rate follow-on phase

Because the first continuation spent its final 500 updates linearly reducing
the learning rate to zero, a follow-on phase starts from its verified final
checkpoint and restores the complete single-GPU optimizer state. This phase
uses `lr_scale=0.2` with **zero warmup and zero warmdown**, so the learning rate
is constant and nonzero for every optimizer update. It uses unseen FineWeb
shards 36-50, validates and overwrites its rolling checkpoint every 125
updates, stops early at validation loss 3.25, and has a budget-safe maximum of
2,200 updates. Step-zero validation reproduced the parent loss of 3.3092.
