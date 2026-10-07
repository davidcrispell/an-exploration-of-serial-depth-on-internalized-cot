# Modded-NanoGPT

## Configurable-depth experimental fork

This branch is pinned to upstream commit
[`dec62cd89dea97f7b13ab544715eb0e755e5a93c`](https://github.com/KellerJordan/modded-nanogpt/tree/dec62cd89dea97f7b13ab544715eb0e755e5a93c),
the homogeneous-block 10.8-minute record, and exposes the model architecture on the command line.

The defaults reproduce the upstream architecture:

| Option | Meaning | Default |
| --- | --- | ---: |
| `--n-layer`, `--layers` | Transformer block count | 12 |
| `--n-embd`, `--width` | Residual/model width | 768 |
| `--n-head`, `--heads` | Attention head count | 6 |
| `--n-ff`, `--mlp-width` | MLP hidden width | 3072 |

The default attention head width is therefore 128. Running without architecture arguments preserves
the original model:

```bash
./run.sh
```

An exactly parameter-matched shallow/deep pair, counting every trainable parameter in the untied
embedding/head model, is:

```bash
# Deep control: 12 layers, 162,201,600 parameters
./run.sh --layers 12 --width 768 --heads 6 --mlp-width 3072

# Shallow treatment: 6 layers, 162,201,600 parameters
./run.sh --layers 6 --width 1024 --heads 8 --mlp-width 2768
```

Both configurations retain 128-dimensional attention heads. The trainer prints the resolved
architecture and verifies its analytical parameter count before moving the model to CUDA.

Run the CPU-only architecture tests with:

```bash
python -m unittest discover -s tests -v
```

## Held-out multiplication benchmarks

The canonical validation and test splits from
[From Explicit CoT to Implicit CoT](https://arxiv.org/abs/2405.14838) are vendored under
[`benchmarks/multiplication`](benchmarks/multiplication/README.md). The collection contains 1,000
validation and 1,000 test examples for each of 4x4, 5x5, 7x7, 9x9, and 11x11 multiplication,
including the explicit reversed-digit long-multiplication traces.

Validate all 10,000 examples, their intermediate arithmetic, and split isolation with:

```bash
python -m benchmarks.multiplication.validate
```

### Baseline and explicit-CoT evaluation

The untreated checkpoint can be evaluated with deterministic greedy decoding at the answer field:

```bash
python -m benchmarks.multiplication.evaluate \
  --checkpoint /path/to/state_step004578.pt \
  --digits 4 5 7 9 11 \
  --split test \
  --protocol direct \
  --output results/multiplication/shallow-base-direct-test.json
```

Explicit-CoT fine-tuning follows the multiplication setup released with the paper: GPT-2
tokenization; `input EOS CoT EOS #### answer EOS`; loss masking through the first EOS; AdamW with
learning rate `5e-5`; effective batch size 32; gradient clipping at 1.0; and seed 3456. The paper's
GPT-2 multiplication command uses FP32, which is therefore the default here. The training split is
the authors' Git-LFS file and is intentionally not vendored in this repository.

```bash
python -m benchmarks.multiplication.train_explicit_cot \
  --checkpoint /path/to/state_step004578.pt \
  --train-path /path/to/4_by_4_mult/train.txt \
  --validation-path benchmarks/multiplication/data/4x4/validation.txt \
  --test-path benchmarks/multiplication/data/4x4/test.txt \
  --output-dir /path/to/4x4-explicit-cot \
  --cache-dir /path/to/token-cache \
  --digits 4 --epochs 1 --batch-size 32 --accumulate 1 \
  --lr 5e-5 --max-grad-norm 1.0 --seed 3456
```

This is a deliberately separate explicit-CoT stage. It establishes that each architecture can
perform the visible algorithm before the stepwise token-removal curriculum tests internalization.

Starting from that checkpoint, reproduce the paper's left-to-right internalization curriculum with:

```bash
python -m benchmarks.multiplication.train_internalized_cot \
  --checkpoint /path/to/4x4-explicit-cot/explicit_cot_epoch_000.pt \
  --train-path /path/to/4_by_4_mult/train.txt \
  --validation-path benchmarks/multiplication/data/4x4/validation.txt \
  --test-path benchmarks/multiplication/data/4x4/test.txt \
  --output-dir /path/to/4x4-internalized-cot \
  --cache-dir /path/to/token-cache \
  --digits 4 --epochs 8 --batch-size 32 --accumulate 1 \
  --lr 5e-5 --max-grad-norm 1.0 --remove-per-epoch 8 \
  --removal-smoothing-lambda 4 --seed 3456
```

The trainer removes eight CoT tokens per epoch, samples the paper's exponential smoothing offset,
resets AdamW each time the scheduled removal advances, and stops once the whole trace is hidden and
validation exact-answer accuracy reaches 99%. It keeps a single atomic `latest.pt` checkpoint plus
the complete epoch-level validation history, avoiding multi-gigabyte checkpoint accumulation.

This is a fast variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo, which attains the same final validation loss in:
* 2.4B tokens instead of 10B
* 10.8 minutes on 8xH100 instead of 45

It uses the following techniques:
* Modernized architecture: Rotary embeddings, QK-Norm, and ReLU^2.
* Projection layers initialized to zero (muP-like).
* Untied head from embedding and init to zero.
* New optimizer: Muon - Momentum Orthogonalized by Newton-schulz.

To execute the training, run the following three commands.
They should all complete within <20min on an 8xH100 with decent internet connection.
```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 24 # downloads only the first 2.4B training tokens to save time
./run.sh
```

The result will be a transformer with 124M active parameters trained for 4768 steps on 2.4B tokens of Fineweb [1], achieving ~3.275 validation loss.
For comparison, the default llm.c PyTorch trainer yields [>3.28 validation loss after training for 19560 steps on 10B tokens](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

## World record history

The following is the progression of world records for the task of *training a model with 124M active parameters to 3.28 validation loss on FineWeb in the minimal amount of time on an 8xH100 machine.*

1. [45 minutes: llm.c baseline](https://github.com/karpathy/llm.c/discussions/481) (05/28/24) [[training log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101324_llmc/main.log)] (note: the 90 minute time is on 8xA100; it's 45 minutes on 8xH100)
2. [31.4 minutes: Architectural modernizations and learning rate tuning](https://x.com/kellerjordan0/status/1798863559243513937) (06/06/24) [[training log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/060624_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log)] (note: this uses half the tokens as the baseline but isn't yet twice as fast since it's slower PyTorch code rather than raw CUDA. also note: by far the biggest improvement here came from simply tripling the learning rate.)
3. [24.9 minutes: Introduced the Muon optimizer](https://x.com/kellerjordan0/status/1842300916864844014) (10/04/24)
4. [22.3 minutes: Muon improvements](https://x.com/kellerjordan0/status/1844820919061287009) (10/11/24) [[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt)]
5. [15.2 minutes: Pad embeddings & architectural modernizations](https://x.com/kellerjordan0/status/1845865698532450646) (10/14/24) [[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101424_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)]
6. [13.1 minutes: Distributed the overhead of Muon](https://x.com/kellerjordan0/status/1847291684016783746) (10/18/24) [[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt)]
7. [12.0 minutes: Upgraded PyTorch from 2.4.1 to 2.5.0](https://x.com/kellerjordan0/status/1847358578686152764) (10/18/24) [[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101824_PyTorch25/d4bfb25f-688d-4da5-8743-33926fad4842.txt)] (note: this now runs at the same speed per step as the CUDA llm.c trainer!)

Direct contributors to these records: @Grad62304977, @bozavlado, myself

Note: The original llm.c baseline is intended to be closer to a replication of GPT-2 than to an optimized LLM training.
So it's no surprise that there is room to improve; as @karpathy has said, 'llm.c still has a lot of pending optimizations.'
In addition, many of the techniques used in these records are completely standard, such as rotary embeddings.
The goal of this benchmark/speedrun is simply to find out which techniques actually work, and maybe come up with some new ones.
<!--The goal of this benchmark is simply to find out all the techniques which actually work, because I'm going crazy reading all these
LLM training papers
which claim a huge benefit but then use their own idiosyncratic non-competitive benchmark and therefore no one in the community has any idea if it's legit for months.-->
<!--[LLM](https://arxiv.org/abs/2305.14342) [training](https://arxiv.org/abs/2402.17764) [papers](https://arxiv.org/abs/2410.01131)-->
<!--I mean hello??? We're in a completely empirical field; it is insane to not have a benchmark. Ideally everyone uses the same LLM training benchmark,
and then reviewing LLM training papers becomes as simple as checking if they beat the benchmark. It's not like this would be unprecedented, that's how things
were in the ImageNet days.
The only possible 'benefit' I can think of for any empirical field to abandon benchmarks is that it would make it easier to publish false results. Oh, I guess that's why it happened.
Hilarious to think about how, in the often-commented-upon and ongoing collapse of the peer review system, people blame the *reviewers* --
yeah, those guys doing free labor who everyone constantly musters all of their intelligence to lie to, it's *their* fault! My bad, you caught me monologuing.-->

### Q: What makes "NanoGPT speedrunning" not just another idiosyncratic benchmark?

A: Because it is a *competitive* benchmark. In particular, if you attain a new speed record (using whatever method you want), there is an open invitation for you
to post that record (on arXiv or X) and thereby vacuum up all the clout for yourself. I will even help you do it by reposting you as much as I can.

<!--On the contrary, for example, the benchmark used in the [Sophia](https://arxiv.org/abs/2305.14342) paper does *not* have this property.
There is no such open invitation for anyone to compete on the benchmark they used. In particular, if, for a random and definitely not weirdly specific example, you happen to find better AdamW hyperparameters for their training setup than
the ones they used which significantly close the gap between AdamW and their proposed optimizer,
then there is no clear path for you to publish that result in *any* form.
You could try posting it on X.com, but then you would be risking being perceived as aggressive/confrontational, which is *not a good look* in this racket.
So if you're rational, the result probably just dies with you and no one else learns anything
(unless you're in a frontier lab, in which case you can do a nice internal writeup. Boy I'd love to get my hands on those writeups).-->

["Artificial intelligence advances by inventing games and gloating to goad others to play" - Professor Ben Recht](https://www.argmin.net/p/too-much-information)

### Q: NanoGPT speedrunning is cool and all, but meh it probably won't scale and is just overfitting to val loss

A: Ok, well, "at scale" is an infinite category (what if the methods stop working only for >100T models?), so it's impossible for me to conclusively refute the allegation that whatever we're doing here doesn't work at scale.
But if you care about 1.5B models, then you might be convinced by this result:

*Straightforwardly scaling up the speedrun to 1.5B parameters yields a model with GPT-2 (1.5B)-level quality 2.5x more cheaply than [@karpathy's baseline](https://github.com/karpathy/llm.c/discussions/677) ($233 instead of $576):*

![](img/nanogpt_speedrun51.png)
[[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/102024_ScaleUp1B/ad8d7ae5-7b2d-4ee9-bc52-f912e9174d7a.txt)]
![](img/nanogpt_speedrun52.png)

## Muon optimizer

Muon is defined as follows:

![](img/algo_optimizer.png)

Where NewtonSchulz5 is the following Newton-Schulz iteration [2, 3], which approximately replaces `G` with `U @ V.T` where `U, S, V = G.svd()`.
```python
@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T 
    for _ in range(steps):
        A = X @ X.T 
        B = A @ X 
        X = a * X + b * B + c * A @ B 
    if G.size(0) > G.size(1):
        X = X.T 
    return X.to(G.dtype)
```

For this training scenario, Muon has the following favorable properties:
* Less memory usage than Adam
* ~1.5x faster training
* <2% wallclock overhead


### Provenance

Many of the choices made to generate this optimizer were obtained experimentally by our pursuit of [CIFAR-10 speedrunning](https://github.com/KellerJordan/cifar10-airbench).
In particular, we experimentally obtained the following practices:
* Using Nesterov momentum inside the update, with orthogonalization applied after momentum.
* Using a specifically quintic Newton-Schulz iteration as the method of orthogonalization.
* Using non-convergent coefficients for the quintic polynomial in order to maximize slope at zero, and thereby minimize the number of necessary Newton-Schulz iterations.
It turns out that the variance doesn't actually matter that much, so we end up with a quintic that (rapidly) converges to the range 0.68, 1.13 upon repeated application, rather than to 1.
* Running the Newton-Schulz iteration in bfloat16 (whereas Shampoo implementations often compute the preconditioners via inverse-pth-roots in fp32 or fp64).

Our use of a Newton-Schulz iteration for orthogonalization traces to [Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325),
who suggested it as a way to compute Shampoo [5, 6] preconditioners, and theoretically explored Shampoo without preconditioner accumulation.
In particular, Jeremy Bernstein @jxbz sent us the draft, which caused us to experiment with various Newton-Schulz iterations as the
orthogonalization method for this optimizer.
If we had used SVD instead of a Newton-Schulz iteration, this optimizer would have been too slow to be useful.
Bernstein & Newhouse also pointed out that Shampoo without preconditioner accumulation is equivalent to steepest descent in the spectral norm,
and therefore Shampoo can be thought of as a way to smooth out spectral steepest descent.
The proposed optimizer can be thought of as a second way of smoothing spectral steepest descent, with a different set of memory and runtime tradeoffs
compared to Shampoo.

---

## Startup script

Here's a good startup script for a fresh instance. If you get `torchrun not found` after this upon running then just close and reopen your tmux tab.

```
sudo apt-get update
sudo apt-get install vim tmux python3-pip python-is-python3 -y
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
tmux

pip install numpy==1.23.5 huggingface-hub tqdm
pip install --upgrade torch &
python data/cached_fineweb10B.py 30
```

## Running on fewer GPUs or with less memory

To run on fewer GPUs, just modify the 1-liner `run.sh` to have a different `--nproc_per_node`. If you don't have enough memory to fit the batch size, then
go into `train_gpt2.py` and scale down the `device_batch_size` by either 1/2 or 1/4.
Both of these changes will have no effect on the training - you should get the exact same loss curve as the most recent record, because the training code
will automatically adjust the gradient accumulation in order to have the same total batch size.

---

## References

1. [Penedo, Guilherme, et al. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)
2. Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics, 2008. Equation 5.22.
3. Günther Schulz. Iterative Berechnung der reziproken Matrix. Z. Angew. Math. Mech., 13:57–59, 1933.
4. [Jeremy Bernstein and Laker Newhouse. "Old Optimizer, New Norm: An Anthology." arxiv preprint arXiv:2409.20325 (2024).](https://arxiv.org/abs/2409.20325)
5. [Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned stochastic tensor optimization." International Conference on Machine Learning. PMLR, 2018.](https://arxiv.org/abs/1802.09568)
6. [Anil, Rohan, et al. "Scalable second order optimization for deep learning." arXiv preprint arXiv:2002.09018 (2020).](https://arxiv.org/abs/2002.09018)
7. [Hägele, Alexander, et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." arXiv preprint arXiv:2405.18392 (2024).](https://arxiv.org/abs/2405.18392)

<img src="img/dofa.jpg" alt="itsover_wereback" style="width:100%;">
