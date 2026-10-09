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

The first launch reached continuation step 125 with validation loss 3.3195,
then failed while writing its rolling checkpoint because the 20 GB network
volume had reached its quota. PyTorch could not load the resulting truncated
file, so that observation is retained only as a failed-attempt diagnostic and
is not part of the active run's learning curve. The partial checkpoint was
removed, the logs were preserved under
`/workspace/experiment-results/shallow1-fineweb-constant-lr/failed-attempt-001`,
and the run restarted from the verified step-6,578 parent with identical
optimizer and schedule settings. Active rolling checkpoints now write to the
pod's otherwise-empty local disk at `/root/shallow1-fineweb-constant-lr`.

The restarted run recorded validation losses of 3.3194 at continuation step
125 and 3.3201711 at step 250. The step-250 rolling checkpoint loads
successfully with model and optimizer state. Both measurements are above the
3.3092277 parent loss, so the immediate full-rate response is a small
regression; two points are not enough to determine whether the constant-rate
phase will reverse direction later.

Validation then improved modestly to 3.3181 at continuation step 375 and
3.3175735 at step 500. This reverses part of the initial jump, but step 500
remains 0.0083458 above the 3.3092277 parent loss and 0.0675735 above the 3.25
target. The step-500 rolling checkpoint was load-verified.

The gradual recovery continued to 3.3165 at step 625 and 3.3160462 at step
750. Step 750 is still 0.0068185 above the parent and 0.0660462 above the
target. Its checkpoint was load-verified with the restored optimizer state.

At continuation steps 875 and 1,000, validation loss reached 3.3149 and
3.3142254 respectively. The monotonic recovery since step 250 continues, but
step 1,000 remains 0.0049977 above the parent and 0.0642254 above the target.
The step-1,000 checkpoint was load-verified.

Validation loss subsequently reached 3.3132, 3.3129, 3.3125, and 3.3103549 at
steps 1,125, 1,250, 1,375, and 1,500. The latest value is only 0.0011272
above the 3.3092277 parent, although it remains 0.0603549 above the 3.25
target. The step-1,500 checkpoint was load-verified.

At continuation step 1,625, validation loss reached 3.3088481. This is the
first constant-rate checkpoint to improve on the 3.3092277 parent loss, by
0.0003796, while remaining 0.0588481 above the 3.25 target. The checkpoint was
load-verified.

Validation loss continued down to 3.3079 at step 1,750 and 3.3071771 at step
1,875. The latter is 0.0020506 below the parent and 0.0571771 above the target.
Its load-verified checkpoint was copied to the local Git-ignored artifacts
directory before the Runpod balance entered its final safety margin; the local
file has SHA256
`9c73f0cf6a817d7782976362640f3b680bf320adb6b13caac215757f90d11cdc`.

The final two validations were 3.3061 at step 2,000 and **3.3054311 at step
2,125**. Step 2,125 is 0.0037966 below the parent loss, but still 0.0554311
above the 3.25 target. With the account entering its safety margin, a live
guard stopped the trainer immediately after that validation and rolling
checkpoint were written; the resulting SIGINT traceback in `launcher.log` is
therefore an intentional budget stop, not a training failure. The run processed
1,114,112,000 additional tokens, and its training timer recorded 3,217.113
seconds (53m 37s), excluding compilation, validation, checkpoint I/O, and the
earlier failed launch.

The verified final checkpoint, configuration, and logs are stored locally in
`artifacts/checkpoints/shallow1-fineweb-constant-lr/`. `latest.pt` is
1,709,766,045 bytes with SHA256
`3ebb9d8a6189390c779e499957d3b44e4e9a1ae13a41ac8f91f61749beabf27c`.
The observed balance delta from the phase's initial $3.3188042684 was
**$3.0282304941**, leaving $0.2905737743. GPU pod `v65fbs8pswgrui` was deleted;
network volume `vpzu3qptxw` was retained.

## Deep 1 matched-loss control

Deep 1 was trained from scratch with the same fork, FineWeb binary data,
tokenizer, global batch of 512 sequences, sequence length 1,024, and learning-rate
schedule used for the original Shallow 1 run. Its architecture is 12 layers,
width 768, 6 attention heads, and MLP width 3,072. It contains exactly
**162,201,600 parameters**, identical to Shallow 1 rather than merely rounded to
the same parameter scale.

Before launch, the loss-matching rule was fixed as an absolute FineWeb validation
loss difference of at most 0.002 from Shallow 1's 3.3054311275. Validation and a
rolling checkpoint were produced every 25 updates, and the trainer stopped at the
first validation at or below 3.3074311275. It stopped at step 4,200 with exact
validation loss **3.3050160408**. The deep-minus-shallow difference is
**-0.0004150867**, so the preregistered match succeeded by a comfortable margin.
The pair also has zero parameter-count difference.

The run processed 2,202,009,600 training tokens. Its measured training timer was
8,044.599 seconds (2h 14m 05s); dense validation and repeated 1.6 GB rolling
checkpoint writes increased total pod lifetime to approximately 3h 07m. The
Runpod balance delta was **$7.6257225656**. The GPU pod was deleted after artifact
verification and network volume `vpzu3qptxw` was retained.

The load-verified checkpoint is stored locally at
`artifacts/checkpoints/deep1-fineweb-matched/latest.pt`, with SHA256
`b7d1e1c596839a8d3da0fd27eea21ad9f4f9b9537d5a0dd527190594fd14b300`.
Checkpoint weights and full logs are Git-ignored; compact configuration, cost,
and validation-curve data are tracked in [`deep1-matched.json`](deep1-matched.json).
