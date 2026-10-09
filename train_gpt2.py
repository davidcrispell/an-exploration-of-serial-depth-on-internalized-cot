import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
with open(os.path.join(os.path.dirname(__file__), "model.py")) as f:
    code += "\n\n# ===== model.py =====\n" + f.read()
with open(os.path.join(os.path.dirname(__file__), "training_utils.py")) as f:
    code += "\n\n# ===== training_utils.py =====\n" + f.read()
import argparse
import json
import uuid
import glob
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig, add_architecture_arguments, config_from_args
from training_utils import (
    linear_warmup_warmdown_factor,
    restore_optimizer_states,
    strip_compiled_prefix,
)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_via_newtonschulz5_compiled = torch.compile(zeropower_via_newtonschulz5)
zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5_compiled)
zeropower_backends_eager = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            parameter_device = group['params'][0].device
            backends = zeropower_backends if parameter_device.type == "cuda" else zeropower_backends_eager
            zeropower_backend = backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            updates_flat = torch.zeros(total_params, device=parameter_device, dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % world_size == rank:
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, file_offset=0, device='cuda'):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.device = device

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))[file_offset:]
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.to(self.device), y.to(self.device)

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 4578 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 1308 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()
parser = argparse.ArgumentParser(description="Train configurable-depth modded-nanoGPT")
add_architecture_arguments(parser)
parser.add_argument("--resume", help="model checkpoint to continue from")
parser.add_argument(
    "--restore-optimizer",
    action="store_true",
    help="restore optimizer moments from --resume, then apply the requested learning rates",
)
parser.add_argument("--num-iterations", type=int, default=args.num_iterations)
parser.add_argument("--warmup-iters", type=int, default=args.warmup_iters)
parser.add_argument("--warmdown-iters", type=int, default=args.warmdown_iters)
parser.add_argument("--lr-scale", type=float, default=1.0)
parser.add_argument("--target-val-loss", type=float)
parser.add_argument("--val-loss-every", type=int, default=args.val_loss_every)
parser.add_argument("--save-every", type=int, default=args.save_every)
parser.add_argument("--input-bin", default=args.input_bin)
parser.add_argument("--input-val-bin", default=args.input_val_bin)
parser.add_argument("--train-shard-offset", type=int, default=0)
parser.add_argument("--output-dir")
parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
parser.add_argument("--batch-size", type=int, default=args.batch_size)
parser.add_argument("--device-batch-size", type=int, default=args.device_batch_size)
parser.add_argument("--sequence-length", type=int, default=args.sequence_length)
parser.add_argument("--val-tokens", type=int, default=args.val_tokens)
parser.add_argument("--no-compile", action="store_true")
parser.add_argument("--no-save", action="store_true")
architecture_args = parser.parse_args()
model_config = config_from_args(architecture_args, vocab_size=50304)
args.num_iterations = architecture_args.num_iterations
args.warmup_iters = architecture_args.warmup_iters
args.warmdown_iters = architecture_args.warmdown_iters
args.val_loss_every = architecture_args.val_loss_every
args.save_every = architecture_args.save_every
args.input_bin = architecture_args.input_bin
args.input_val_bin = architecture_args.input_val_bin
args.batch_size = architecture_args.batch_size
args.device_batch_size = architecture_args.device_batch_size
args.sequence_length = architecture_args.sequence_length
args.val_tokens = architecture_args.val_tokens
if architecture_args.lr_scale <= 0:
    parser.error("--lr-scale must be positive")
if architecture_args.train_shard_offset < 0:
    parser.error("--train-shard-offset must be nonnegative")
if architecture_args.restore_optimizer and not architecture_args.resume:
    parser.error("--restore-optimizer requires --resume")
linear_warmup_warmdown_factor(
    0,
    num_iterations=args.num_iterations,
    warmup_iters=args.warmup_iters,
    warmdown_iters=args.warmdown_iters,
)

resume_payload = None
resume_optimizer_states = None
parent_step = 0
if architecture_args.resume:
    torch.serialization.add_safe_globals([GPTConfig])
    resume_payload = torch.load(
        architecture_args.resume, map_location="cpu", weights_only=True
    )
    checkpoint_config = resume_payload.get("model_config")
    if checkpoint_config != model_config:
        parser.error(
            f"checkpoint architecture {checkpoint_config!r} does not match "
            f"requested architecture {model_config!r}"
        )
    parent_step = int(resume_payload["step"])
    if architecture_args.restore_optimizer:
        resume_optimizer_states = resume_payload.get("optimizers")
        if resume_optimizer_states is None:
            parser.error("--restore-optimizer requested but checkpoint has no optimizer states")

# Set up DDP for CUDA; MPS and CPU continuation probes are single-process.
if architecture_args.device == "cuda":
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    if architecture_args.device == "mps":
        assert torch.backends.mps.is_available()
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    device = architecture_args.device
print(f"using device: {device}")

def synchronize_device():
    if architecture_args.device == "cuda":
        torch.cuda.synchronize()
    elif architecture_args.device == "mps":
        torch.mps.synchronize()
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(
    args.input_bin,
    B,
    T,
    ddp_rank,
    ddp_world_size,
    file_offset=architecture_args.train_shard_offset,
    device=device,
)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, device=device)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = model_config.vocab_size
model = GPT(model_config)
if resume_payload is not None:
    model.load_state_dict(strip_compiled_prefix(resume_payload["model"]), strict=True)
    del resume_payload
actual_parameter_count = sum(p.numel() for p in model.parameters())
assert actual_parameter_count == model_config.expected_parameter_count()
if master_process:
    print(
        f"architecture: layers={model_config.n_layer} width={model_config.n_embd} "
        f"heads={model_config.n_head} head_dim={model_config.head_dim} "
        f"mlp_width={model_config.n_ff} parameters={actual_parameter_count}"
    )
model = model.to(device)
if architecture_args.device == "cuda" and hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
if not architecture_args.no_compile:
    model = torch.compile(model)
# Here we wrap CUDA models in DDP; local MPS/CPU probes stay single-process.
if architecture_args.device == "cuda":
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
else:
    raw_model = model
    ctx = nullcontext()

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
if architecture_args.device == "cuda":
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(True)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

# init the optimizer(s)
use_fused_adam = architecture_args.device == "cuda"
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.3 * architecture_args.lr_scale,   betas=(0.9, 0.95), fused=use_fused_adam)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.002 * architecture_args.lr_scale, betas=(0.9, 0.95), fused=use_fused_adam)
optimizer3 = Muon(raw_model.transformer.h.parameters(),           lr=0.02 * architecture_args.lr_scale,  momentum=0.95)
optimizers = [optimizer1, optimizer2, optimizer3]
optimizer_learning_rates = [
    0.3 * architecture_args.lr_scale,
    0.002 * architecture_args.lr_scale,
    0.02 * architecture_args.lr_scale,
]
optimizer_state_mode = "reset_at_continuation_start" if architecture_args.resume else "fresh"
if resume_optimizer_states is not None:
    restore_optimizer_states(
        optimizers, resume_optimizer_states, optimizer_learning_rates
    )
    optimizer_state_mode = "restored_from_parent"
    del resume_optimizer_states
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    return linear_warmup_warmdown_factor(
        it,
        num_iterations=args.num_iterations,
        warmup_iters=args.warmup_iters,
        warmdown_iters=args.warmdown_iters,
    )
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = architecture_args.output_dir or ('logs/%s/' % run_id)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "train.log") if architecture_args.output_dir else 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(
            json.dumps(
                {
                    "resume": architecture_args.resume,
                    "parent_step": parent_step,
                    "optimizer_state": optimizer_state_mode,
                    "num_iterations": args.num_iterations,
                    "warmup_iters": args.warmup_iters,
                    "warmdown_iters": args.warmdown_iters,
                    "lr_scale": architecture_args.lr_scale,
                    "target_val_loss": architecture_args.target_val_loss,
                    "train_shard_offset": architecture_args.train_shard_offset,
                    "model_config": vars(model_config),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} on {device}; CUDA build {torch.version.cuda}\n")
        import subprocess
        if architecture_args.device == "cuda":
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'nvidia-smi:\n{result.stdout}\n')
        f.write('='*100 + '\n')
    if architecture_args.output_dir:
        with open(os.path.join(logdir, "training_config.json"), "w") as f:
            json.dump(
                {
                    "resume": architecture_args.resume,
                    "parent_step": parent_step,
                    "optimizer_state": optimizer_state_mode,
                    "num_iterations": args.num_iterations,
                    "warmup_iters": args.warmup_iters,
                    "warmdown_iters": args.warmdown_iters,
                    "lr_scale": architecture_args.lr_scale,
                    "target_val_loss": architecture_args.target_val_loss,
                    "val_loss_every": args.val_loss_every,
                    "save_every": args.save_every,
                    "input_bin": args.input_bin,
                    "input_val_bin": args.input_val_bin,
                    "train_shard_offset": architecture_args.train_shard_offset,
                    "model_config": vars(model_config),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

training_time_ms = 0
# start the clock
synchronize_device()
t0 = time.time()
# begin training
train_loader.reset()
x, y = train_loader.next_batch()
last_val_loss = None
for step in range(args.num_iterations + 1):
    scheduled_last_step = (step == args.num_iterations)
    target_reached = False
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (scheduled_last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        synchronize_device()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        if dist.is_initialized():
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        last_val_loss = float(val_loss.item())
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} total_step:{parent_step + step} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} total_step:{parent_step + step} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        target_reached = (
            architecture_args.target_val_loss is not None
            and val_loss.item() <= architecture_args.target_val_loss
        )
        # start the clock again
        synchronize_device()
        t0 = time.time()

    last_step = scheduled_last_step or target_reached
    if master_process and not architecture_args.no_save and (
        last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0)
    ):
        # stop the clock
        synchronize_device()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(
            step=parent_step + step,
            continuation_step=step,
            parent_step=parent_step,
            parent_checkpoint=architecture_args.resume,
            optimizer_state=optimizer_state_mode,
            validation_loss=last_val_loss,
            target_val_loss=architecture_args.target_val_loss,
            code=code,
            model_config=model_config,
            model=raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
        )
        checkpoint_path = (
            os.path.join(logdir, "latest.pt")
            if architecture_args.output_dir
            else 'logs/%s/state_step%06d.pt' % (run_id, parent_step + step)
        )
        torch.save(log, checkpoint_path)
        # start the clock again
        synchronize_device()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with (model.no_sync() if isinstance(model, DDP) else nullcontext()):
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} total_step:{parent_step + step + 1} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} total_step:{parent_step + step + 1} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    if architecture_args.device == "cuda":
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    elif architecture_args.device == "mps":
        print(f"current MPS memory: {torch.mps.current_allocated_memory() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if dist.is_initialized():
    dist.destroy_process_group()
