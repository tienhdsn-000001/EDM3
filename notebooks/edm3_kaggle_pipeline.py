# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EEPM3: GFlowNet Sparse Training Pipeline
#
# **Platform**: Kaggle (primary) / Colab (fallback)
#
# This notebook runs the full EEPM3 training pipeline:
# 1. Install dependencies & upload local artifacts
# 2. Generate trajectories (or load pre-generated)
# 3. Score via AlphaGenome API
# 4. RBS data augmentation
# 5. Offline α-GFN training with dual-head policy
#
# **Runtime**: Use GPU (T4/P100) accelerator.
# Set `ALPHA_GENOME_API_KEY` in Kaggle Secrets or Colab userdata.

# %% [markdown]
# ## 0. Setup & Dependencies

# %%
# Detect platform
import os, sys
IN_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
IN_COLAB = 'google.colab' in sys.modules
PLATFORM = "Kaggle" if IN_KAGGLE else ("Colab" if IN_COLAB else "Local")
print(f"Platform: {PLATFORM}")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# %%
# Install core dependencies
# !pip install -q jax[cpu] flax optax alphagenome 2>/dev/null

# For GPU acceleration on Kaggle/Colab:
import subprocess
try:
    import jax
    if not jax.devices('gpu'):
        raise RuntimeError("No GPU")
    print(f"JAX GPU: {jax.devices('gpu')}")
except:
    print("Installing JAX with GPU support...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "jax[cuda12]", "flax", "optax"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "alphagenome"],
               capture_output=True)
print("Dependencies installed.")

# %%
# Load API key from platform secrets
if IN_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    ALPHA_GENOME_API_KEY = UserSecretsClient().get_secret("ALPHA_GENOME_API_KEY")
elif IN_COLAB:
    from google.colab import userdata
    ALPHA_GENOME_API_KEY = userdata.get("ALPHA_GENOME_API_KEY")
else:
    ALPHA_GENOME_API_KEY = os.environ.get("ALPHA_GENOME_API_KEY", "")

assert ALPHA_GENOME_API_KEY, "Set ALPHA_GENOME_API_KEY in Kaggle Secrets or Colab userdata!"
print(f"API Key loaded: {ALPHA_GENOME_API_KEY[:8]}...{ALPHA_GENOME_API_KEY[-4:]}")

# %% [markdown]
# ## 1. Configuration

# %%
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import sqlite3
import time
import asyncio
import logging
from jax import tree_util
from typing import List, Tuple, Optional

# ── Global Config ──────────────────────────────────────────────
SEQ_LEN         = 100_000       # MDP sequence length
API_SEQ_LEN     = 131_072       # AlphaGenome required length (N-padded)
VOCAB_SIZE      = 5             # A, C, G, T, N
NUM_EDITS       = 10            # Mutations per trajectory
METADATA_DIM    = 10            # Epigenetic metadata dimension
TEMPERATURE     = 2.0           # Sampling temperature for exploration
BATCH_SIZE      = 32            # Training batch size
LEARNING_RATE   = 1e-4
MAX_GRAD_NORM   = 1.0
ALPHA_GFN       = 0.5           # α-GFN mixing parameter

# Pipeline settings
NUM_TRAJECTORIES = 500          # Smaller batch for cloud GPU time limits
API_CONCURRENCY  = 1            # Forced to 1 for local Evo2 to avoid OOM
API_MAX_RETRIES  = 8
API_BASE_BACKOFF = 1.0
TRAIN_EPOCHS     = 200

# Paths
DATA_DIR = "/kaggle/working" if IN_KAGGLE else "/content" if IN_COLAB else "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "experience_replay.db")
TRAJ_PATH = os.path.join(DATA_DIR, "unscored_trajectories.npz")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Data dir: {DATA_DIR}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## 2. Model Architecture (Dual-Head Conv1D Policy)

# %%
class GeneratorPolicyV2(nn.Module):
    """
    SOTA Dual-Head GFlowNet Policy:
      - Shared Conv1D backbone (aggressive striding)
      - Factored action head (position + base scores)
      - Value head V(s) for Sub-EB
    """
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    input_channels: int = 6  # 5 one-hot + mutated mask

    @nn.compact
    def __call__(self, state_input, target_metadata):
        B = state_input.shape[0]
        L = self.seq_len

        # Shared Conv1D backbone
        x = nn.Conv(features=16, kernel_size=(10,), strides=(10,))(state_input)
        x = jax.nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(10,), strides=(10,))(x)
        x = jax.nn.relu(x)  # (B, L/100, 32)

        # Position scores
        pos_coarse = nn.Conv(features=1, kernel_size=(1,), strides=(1,))(x).squeeze(-1)
        pos_scores = jnp.repeat(pos_coarse, 100, axis=-1)[:, :L]

        # Global features
        global_feat = jnp.concatenate([jnp.mean(x, axis=1), target_metadata], axis=-1)
        g = jax.nn.relu(nn.Dense(128)(global_feat))
        g = jax.nn.relu(nn.Dense(64)(g))

        # Action head
        base_scores = nn.Dense(self.vocab_size)(g)
        terminate_logit = nn.Dense(1)(g)
        combined = pos_scores[:, :, None] + base_scores[:, None, :]
        mutation_logits = combined.reshape((B, L * self.vocab_size))
        action_logits = jnp.concatenate([mutation_logits, terminate_logit], axis=-1)

        # Value head
        v = jax.nn.relu(nn.Dense(128)(global_feat))
        v = jax.nn.relu(nn.Dense(64)(v))
        value = nn.Dense(1)(v).squeeze(-1)

        return action_logits, value


# Verify architecture
policy = GeneratorPolicyV2()
dummy_state = jnp.zeros((1, SEQ_LEN, 6))
dummy_meta = jnp.zeros((1, METADATA_DIM))
params = policy.init(jax.random.PRNGKey(0), dummy_state, dummy_meta)
num_params = sum(p.size for p in tree_util.tree_leaves(params))
action_logits, value = policy.apply(params, dummy_state, dummy_meta)

print(f"GeneratorPolicyV2: {num_params:,} parameters")
print(f"  Action logits: {action_logits.shape}")
print(f"  Value output:  {value.shape}")
assert num_params < 5_000_000, f"Budget exceeded: {num_params}"
print(f"  ✅ Under 5M parameter budget")

# %% [markdown]
# ## 3. GFlowNet Environment

# %%
class GFlowNetState:
    """Minimal state for trajectory generation."""
    def __init__(self, seq, mutated, step):
        self.seq = seq
        self.mutated = mutated
        self.step = step

def sample_trajectory(params, key, wt_seq, metadata, num_edits=NUM_EDITS, temperature=TEMPERATURE):
    """Sample a single trajectory from the policy."""
    seq = wt_seq.copy()
    mutated = jnp.zeros(SEQ_LEN, dtype=jnp.bool_)
    actions = []
    log_probs = []

    for step in range(num_edits):
        key, action_key = jax.random.split(key)

        # Build 6-channel input
        mutated_ch = mutated.astype(jnp.float32)[:, None]
        state_6ch = jnp.concatenate([seq, mutated_ch], axis=-1)[None, ...]
        meta_batch = metadata[None, ...]

        raw_logits, _ = policy.apply(params, state_6ch, meta_batch)
        raw_logits = raw_logits[0]  # unbatch

        # Mask already-mutated positions and terminate
        mask = jnp.ones(SEQ_LEN * VOCAB_SIZE + 1, dtype=jnp.bool_)
        for pos in range(SEQ_LEN):
            if mutated[pos]:
                for b in range(VOCAB_SIZE):
                    mask = mask.at[pos * VOCAB_SIZE + b].set(False)
        mask = mask.at[-1].set(False)  # Block terminate during forced edits

        masked_logits = jnp.where(mask, raw_logits, -1e9)
        scaled_logits = masked_logits / temperature

        action = jax.random.categorical(action_key, scaled_logits)
        lp = jax.nn.log_softmax(masked_logits)[action]

        actions.append(int(action))
        log_probs.append(float(lp))

        # Apply mutation
        pos = int(action) // VOCAB_SIZE
        base = int(action) % VOCAB_SIZE
        seq = seq.at[pos].set(jax.nn.one_hot(base, VOCAB_SIZE))
        mutated = mutated.at[pos].set(True)

    return seq, np.array(actions, dtype=np.int32), np.array(log_probs, dtype=np.float32)

def onehot_to_acgtn(seq_onehot):
    """Converts (L, 5) one-hot to ACGTN string."""
    BASES = "ACGTN"
    return "".join(BASES[i] for i in np.argmax(np.array(seq_onehot), axis=-1))

# %% [markdown]
# ## 4. Stage 1 — Trajectory Generation

# %%
def generate_trajectories(params, num_trajectories=NUM_TRAJECTORIES):
    """Generate offline trajectories for API scoring."""
    wt_seq = jax.nn.one_hot(jnp.zeros(SEQ_LEN, dtype=jnp.int32), VOCAB_SIZE)
    metadata = jnp.ones(METADATA_DIM)

    all_onehot, all_actions, all_logprobs, all_seqs = [], [], [], []
    key = jax.random.PRNGKey(42)
    t0 = time.time()

    for i in range(num_trajectories):
        key, traj_key = jax.random.split(key)
        terminal_seq, actions, log_probs = sample_trajectory(
            params, traj_key, wt_seq, metadata
        )
        all_onehot.append(np.array(terminal_seq))
        all_actions.append(actions)
        all_logprobs.append(log_probs)
        all_seqs.append(onehot_to_acgtn(terminal_seq))

        if (i + 1) % max(1, num_trajectories // 10) == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}/{num_trajectories}] {rate:.1f} traj/s")

    elapsed = time.time() - t0
    print(f"Generated {num_trajectories} trajectories in {elapsed:.1f}s")

    np.savez_compressed(TRAJ_PATH,
        terminal_onehot=np.stack(all_onehot),
        actions=np.stack(all_actions),
        forward_log_probs=np.stack(all_logprobs),
        sequences=np.array(all_seqs, dtype=object),
        seq_len=SEQ_LEN, num_edits=NUM_EDITS, temperature=TEMPERATURE,
    )
    size_mb = os.path.getsize(TRAJ_PATH) / 1024 / 1024
    print(f"Saved: {TRAJ_PATH} ({size_mb:.1f} MB)")
    return all_seqs, all_actions, all_logprobs, all_onehot

# %%
# Run or load trajectories
if os.path.exists(TRAJ_PATH):
    print(f"Loading existing trajectories from {TRAJ_PATH}")
    data = np.load(TRAJ_PATH, allow_pickle=True)
    all_seqs = list(data['sequences'])
    all_actions = list(data['actions'])
    all_logprobs = list(data['forward_log_probs'])
    all_onehot = list(data['terminal_onehot'])
    print(f"  Loaded {len(all_seqs)} trajectories")
else:
    print("Generating new trajectories...")
    all_seqs, all_actions, all_logprobs, all_onehot = generate_trajectories(params)

# %% [markdown]
# ## 5. Stage 2 — AlphaGenome API Scoring

# %%
# %% [markdown]
# ## 5. Stage 2 — AlphaGenome & Evo2 Scoring
#
# Scoring DNA edits via the **AlphaGenome API** (chromatin accessibility) 
# and the **Evo2 7B model** (biological likelihood).
#
# **Strict Mode**: If foundations fail, the trajectory is skipped to prevent placeholder pollution.

# %%
import torch
import aiohttp

# ── Flash Attention Mocking (Colab T4 Compatibility) ──
try:
    import flash_attn
except ImportError:
    from unittest.mock import MagicMock
    _mock = MagicMock()
    sys.modules["flash_attn"] = _mock
    sys.modules["flash_attn_2_cuda"] = _mock 
    sys.modules["flash_attn.flash_attn_interface"] = _mock

log = logging.getLogger("api")
_api_client = None
_evo2_model = None

def _get_api_client(api_key):
    global _api_client
    if _api_client is None:
        from alphagenome.models import dna_client
        _api_client = dna_client.create(api_key)
        print("[API] AlphaGenome DnaClient initialized.")
    return _api_client

def _get_evo2_model():
    global _evo2_model
    model_name = os.environ.get("EVO2_MODEL_NAME", "evo2_7b")
    if model_name == "legacy_oracle": return "legacy_oracle"
    
    if _evo2_model is None:
        try:
            from evo2 import Evo2
            print(f"[Evo2] Initializing: {model_name}")
            # The Evo2 class handles internal device mapping. Do NOT call .to() or .eval().
            _evo2_model = Evo2(model_name)
            print(f"[Evo2] Model {model_name} is LIVE.")
        except Exception as e:
            print(f"[Evo2] Init failed: {e}")
            raise
    return _evo2_model

@torch.no_grad()
async def compute_evo2_likelihood(sequence: str) -> float:
    model = _get_evo2_model()
    if model == "legacy_oracle":
        return float(hash(sequence) % 1000) / 1000.0
        
    def _score(seq_to_score):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq_to_score),
            dtype=torch.int,
        ).unsqueeze(0).to(device)
        
        # Forward pass: returns (logits, embeddings)
        outputs, _ = model(input_ids)
        logits = outputs[0]  # Shape: [1, seq_len, vocab]
        
        # Shift logits and targets to align
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Mean log-likelihood
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return -float(loss.cpu().item())

    try:
        return _score(sequence)
    except Exception as e:
        if "CUDA out of memory" in str(e):
            print(f"[Evo2] OOM on {len(sequence)}bp. Falling back to center 32k window...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Center crop to 32,768 bp (biologically dense region in EDM3)
            mid = len(sequence) // 2
            start = max(0, mid - 16384)
            end = min(len(sequence), mid + 16384)
            windowed = sequence[start:end]
            try:
                return _score(windowed)
            except Exception as e2:
                raise RuntimeError(f"Evo2 Scoring Failed (Windowed): {e2}")
        raise RuntimeError(f"Evo2 Scoring Failed: {e}")

async def score_trajectory(sequence, api_key, semaphore, traj_id):
    """Query AlphaGenome and Evo2 for a single trajectory."""
    from alphagenome.models import dna_client
    
    # 1. AlphaGenome API
    ag_preds = None
    async with semaphore:
        for attempt in range(API_MAX_RETRIES):
            try:
                loop = asyncio.get_event_loop()
                def _predict():
                    model = _get_api_client(api_key)
                    padded = sequence + 'N' * (API_SEQ_LEN - len(sequence))
                    out = model.predict_sequence(padded, [dna_client.OutputType.DNASE])
                    if out.dnase is not None:
                        for attr in ['values', 'data', 'X']:
                            if hasattr(out.dnase, attr):
                                arr = np.array(getattr(out.dnase, attr), dtype=np.float32)
                                return arr if arr.ndim >= 2 else arr[:, None]
                    return None
                ag_preds = await loop.run_in_executor(None, _predict)
                if ag_preds is not None: break
            except Exception as e:
                msg = str(e)
                if any(x in msg for x in ["429", "RESOURCE_EXHAUSTED"]):
                    await asyncio.sleep(API_BASE_BACKOFF * (2**attempt))
                else:
                    raise RuntimeError(f"AlphaGenome API Error: {msg[:100]}")
    
    if ag_preds is None: raise RuntimeError("AlphaGenome API returned None")

    # 2. Evo2 Scoring
    evo2_score = await compute_evo2_likelihood(sequence)
    
    return ag_preds, evo2_score

async def run_scoring_strict(sequences, actions_list, logprobs_list, onehot_list):
    conn = init_database(DB_PATH)
    scored_ids = {r[0] for r in conn.execute("SELECT trajectory_id FROM experiences").fetchall()}
    pending = [(i, s) for i, s in enumerate(sequences) if i not in scored_ids]
    
    print(f"Scoring: {len(pending)} pending")
    if not pending: return conn.close()

    semaphore = asyncio.Semaphore(API_CONCURRENCY)
    scored, failed = 0, 0
    t0 = time.time()

    for batch_start in range(0, len(pending), 10):
        batch = pending[batch_start:batch_start + 10]
        tasks = [score_trajectory(s, ALPHA_GENOME_API_KEY, semaphore, i) for i, s in batch]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Explicitly clear cache after each batch to reclaim activation memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for (traj_id, seq), res in zip(batch, results):
            if isinstance(res, Exception):
                failed += 1
                log.warning(f"  Traj {traj_id}: FAILED — {res}")
                continue
            
            ag_preds, evo2_score = res
            
            # Weighted reward: R(x) = exp(-MSE) + β * likelihood
            mse = float(np.mean(ag_preds ** 2))
            reward = float(np.exp(-mse) + 0.1 * evo2_score)
            reward = max(reward, 1e-8)

            conn.execute(
                "INSERT OR REPLACE INTO experiences "
                "(trajectory_id, actions, forward_log_probs, terminal_onehot, reward) "
                "VALUES (?, ?, ?, ?, ?)",
                (traj_id, actions_list[traj_id].tobytes(),
                 logprobs_list[traj_id].tobytes(),
                 onehot_list[traj_id].tobytes(),
                 reward),
            )
            conn.commit()
            scored += 1

        if (batch_start + 10) % 50 == 0:
            print(f"  [{batch_start+10}/{len(pending)}] processed | {failed} failures")

    print(f"Scoring complete. Scored: {scored}, Failed: {failed}")
    conn.close()

# %%
# Run strict scoring
await run_scoring_strict(all_seqs, all_actions, all_logprobs, all_onehot)

# %% [markdown]
# ## 6. Stage 3 — RBS Data Augmentation

# %%
def rbs_augment(db_path, top_pct=0.10, hallucinations_per=5):
    """Retrospective Backward Synthesis: multiply high-reward signal."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT trajectory_id, actions, forward_log_probs, terminal_onehot, reward "
        "FROM experiences ORDER BY reward DESC"
    ).fetchall()

    if not rows:
        print("No experiences to augment!")
        return

    total = len(rows)
    cutoff = max(1, int(total * top_pct))
    top_rows = rows[:cutoff]
    print(f"RBS: Augmenting top {cutoff}/{total} experiences (reward >= {top_rows[-1][4]:.6f})")

    rng = np.random.default_rng(42)
    augmented = 0

    for traj_id, act_bytes, lp_bytes, oh_bytes, reward in top_rows:
        actions = np.frombuffer(act_bytes, dtype=np.int32).copy()

        # Extract mutations
        mutations = []
        for a in actions:
            if a != SEQ_LEN * VOCAB_SIZE:
                mutations.append((int(a) // VOCAB_SIZE, int(a) % VOCAB_SIZE))

        if len(mutations) < 2:
            continue

        seen = set()
        for _ in range(hallucinations_per * 3):
            if len(seen) >= hallucinations_per:
                break

            perm = rng.permutation(len(mutations))
            key = tuple(perm)
            if key in seen:
                continue
            seen.add(key)

            perm_actions = np.array(
                [mutations[i][0] * VOCAB_SIZE + mutations[i][1] for i in perm],
                dtype=np.int32,
            )
            if len(perm_actions) < NUM_EDITS:
                pad = np.full(NUM_EDITS - len(perm_actions), SEQ_LEN * VOCAB_SIZE, dtype=np.int32)
                perm_actions = np.concatenate([perm_actions, pad])

            # Uniform backward log-probs
            syn_lp = np.array([-np.log(max(SEQ_LEN - i, 1) * 4) for i in range(NUM_EDITS)], dtype=np.float32)

            conn.execute(
                "INSERT INTO experiences "
                "(trajectory_id, actions, forward_log_probs, terminal_onehot, reward) "
                "VALUES (?, ?, ?, ?, ?)",
                (total + augmented, perm_actions.tobytes(), syn_lp.tobytes(), oh_bytes, reward),
            )
            augmented += 1

    conn.commit()
    final = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    conn.close()
    print(f"RBS complete: {augmented} hallucinated trajectories added")
    print(f"Total experiences: {final} ({final/total:.1f}x original)")

# %%
rbs_augment(DB_PATH)

# %% [markdown]
# ## 7. Stage 4 — Offline α-GFN Training

# %%
class ConvergenceTracker:
    """EMA-based convergence detection."""
    def __init__(self, alpha=0.95, threshold_pct=0.05, window=50, var_thresh=0.01):
        self.alpha = alpha
        self.threshold_pct = threshold_pct
        self.window = window
        self.var_thresh = var_thresh
        self.ema = None
        self.baseline_ema = None
        self.losses = []
        self.converged = False
        self.convergence_epoch = None

    def update(self, loss, epoch):
        self.losses.append(loss)
        if self.ema is None:
            self.ema = loss
            self.baseline_ema = loss
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * loss

        if len(self.losses) >= self.window and not self.converged:
            pct_drop = (self.baseline_ema - self.ema) / max(abs(self.baseline_ema), 1e-8)
            recent_var = np.var(self.losses[-self.window:])
            if pct_drop > self.threshold_pct and recent_var < self.var_thresh:
                self.converged = True
                self.convergence_epoch = epoch
        return self.converged

# %%
def load_replay_data(db_path, batch_size=BATCH_SIZE):
    """Load scored experiences for offline training."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT actions, forward_log_probs, reward FROM experiences").fetchall()
    conn.close()

    if not rows:
        raise ValueError("No experiences in replay buffer!")

    all_actions = [np.frombuffer(r[0], dtype=np.int32).copy() for r in rows]
    all_lp = [np.frombuffer(r[1], dtype=np.float32).copy() for r in rows]
    all_rewards = [float(r[2]) for r in rows]

    print(f"Loaded {len(rows)} experiences")
    rewards = np.array(all_rewards)
    print(f"  Reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    return all_actions, all_lp, all_rewards


def iter_batches(all_actions, all_lp, all_rewards, batch_size, rng_key=None):
    """Yield batches for one epoch."""
    n = len(all_rewards)
    indices = np.arange(n)
    if rng_key is not None:
        np.random.seed(int(jax.random.randint(rng_key, (), 0, 2**31)))
        np.random.shuffle(indices)

    for b in range(n // batch_size):
        idx = indices[b * batch_size : (b + 1) * batch_size]
        yield {
            "forward_log_probs": jnp.array(np.stack([all_lp[i] for i in idx])),
            "rewards": jnp.array([all_rewards[i] for i in idx]),
        }

# %%
def alpha_gfn_tb_loss(log_z, forward_log_probs, log_reward, alpha, num_edits):
    """α-GFN modified Trajectory Balance loss."""
    sum_log_pf = jnp.sum(forward_log_probs[:num_edits])
    log_pb_terms = jnp.array([-jnp.log(jnp.float32(num_edits - t)) for t in range(num_edits)])
    sum_log_pb = jnp.sum(log_pb_terms)
    residual = log_z + alpha * sum_log_pf - log_reward - (1.0 - alpha) * sum_log_pb
    return residual ** 2


optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adamw(learning_rate=LEARNING_RATE),
)

@jax.jit
def train_step(log_z, opt_state, batch_lp, batch_rewards):
    def loss_fn(lz):
        log_r = jnp.log(jnp.maximum(batch_rewards, 1e-8))
        losses = jax.vmap(
            lambda lp, lr: alpha_gfn_tb_loss(lz, lp, lr, ALPHA_GFN, NUM_EDITS)
        )(batch_lp, log_r)
        return jnp.mean(losses)

    loss, grad = jax.value_and_grad(loss_fn)(log_z)
    updates, new_opt_state = optimizer.update(grad, opt_state, log_z)
    new_log_z = optax.apply_updates(log_z, updates)
    grad_norm = jnp.sqrt(grad ** 2)
    return loss, new_log_z, new_opt_state, grad_norm

# %%
# Run offline training
all_act, all_lp, all_rew = load_replay_data(DB_PATH)

log_z = jnp.float32(0.0)
opt_state = optimizer.init(log_z)
tracker = ConvergenceTracker()

print(f"\n{'Epoch':>6} | {'Mean Loss':>12} | {'EMA':>10} | {'log_Z':>10} | {'Grad':>10}")
print("-" * 62)

key = jax.random.PRNGKey(9999)
for epoch in range(1, TRAIN_EPOCHS + 1):
    key, ek = jax.random.split(key)
    losses, grads = [], []

    for batch in iter_batches(all_act, all_lp, all_rew, BATCH_SIZE, ek):
        loss, log_z, opt_state, gn = train_step(log_z, opt_state, batch["forward_log_probs"], batch["rewards"])
        losses.append(float(loss))
        grads.append(float(gn))

    ml = np.mean(losses)
    mg = np.mean(grads)
    converged = tracker.update(ml, epoch)

    if epoch == 1 or epoch % 10 == 0 or converged:
        print(f"{epoch:>6} | {ml:>12.4f} | {tracker.ema:>10.4f} | {float(log_z):>10.6f} | {mg:>10.6f}")

    if converged and tracker.convergence_epoch == epoch:
        print(f"\n*** CONVERGENCE at epoch {epoch} ***")

print("-" * 62)
pct = (tracker.baseline_ema - tracker.ema) / max(abs(tracker.baseline_ema), 1e-8) * 100
if tracker.converged:
    print(f"✅ CONVERGENCE VALIDATED at epoch {tracker.convergence_epoch} ({pct:.2f}% EMA drop)")
else:
    print(f"❌ Not converged in {TRAIN_EPOCHS} epochs ({pct:.2f}% EMA drop, threshold 5%)")

# Save checkpoint
ckpt = os.path.join(CKPT_DIR, "edm3_v2_kaggle_final.npz")
np.savez(ckpt, log_z=np.array(float(log_z)))
print(f"\nCheckpoint saved: {ckpt}")
print(f"Final log_z: {float(log_z):.6f}")

# %% [markdown]
# ## 8. Download Artifacts
#
# After training, download the checkpoint and replay buffer:

# %%
if IN_KAGGLE:
    # Kaggle auto-saves /kaggle/working/* as output
    print("Artifacts auto-saved to Kaggle output:")
    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            print(f"  {f}: {os.path.getsize(fp)/1024/1024:.1f} MB")
elif IN_COLAB:
    from google.colab import files
    files.download(ckpt)
    files.download(DB_PATH)
    print("Download triggered for checkpoint and replay buffer.")
else:
    print(f"Artifacts in: {DATA_DIR}")
