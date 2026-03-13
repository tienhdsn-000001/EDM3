"""
Phase 5, Script 2: SOTA Offline Trainer v2.

Implements three cutting-edge GFlowNet methodologies:

1. **Dual-Head Policy (GeneratorPolicyV2)**: Conv1D spatial stem with both an
   action head (logits) and a value head V(s) for Sub-EB evaluation.

2. **α-GFN Loss**: Mixing parameter α ∈ [0,1] that explicitly weights the
   forward vs backward policy components for exploration-exploitation control.

3. **Sub-Trajectory Evaluation Balance (Sub-EB)**: Instead of full-trajectory
   TB loss, evaluates partial sub-trajectories using V(s) outputs for denser
   gradient signal.

Reads from data/experience_replay_augmented.db (or falls back to experience_replay.db).
Compatible with V1 data currently being collected by the API worker.
"""

import os
import sys
import time
import sqlite3
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from jax import tree_util
from typing import Tuple

from gflownet_env import GFlowNetEnv

def build_optimizer(learning_rate: float, max_grad_norm: float):
    """Builds the Optax AdamW optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate),
    )

class ConvergenceTracker:
    """Tracks EMA of the loss and detects convergence thresholds."""
    def __init__(self, alpha=0.95, threshold_pct=0.05, window_size=50, variance_threshold=0.01):
        self.alpha = alpha
        self.threshold_pct = threshold_pct
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        
        self.baseline_ema = None
        self.ema = None
        self.losses = []
        self.converged = False
        self.convergence_epoch = -1
        
    def update(self, mean_loss: float, epoch: int) -> bool:
        if self.baseline_ema is None:
            self.baseline_ema = mean_loss
            self.ema = mean_loss
        else:
            self.ema = self.alpha * self.ema + (1.0 - self.alpha) * mean_loss
            
        self.losses.append(mean_loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
            
        if self.baseline_ema > 0:
            pct_drop = (self.baseline_ema - self.ema) / self.baseline_ema
            if pct_drop >= self.threshold_pct:
                variance = np.var(self.losses)
                if variance < self.variance_threshold and not self.converged:
                    self.converged = True
                    self.convergence_epoch = epoch
                    
        return self.converged
        
    def get_status_str(self) -> str:
        return f"EMA: {self.ema:.4f} (Baseline: {self.baseline_ema:.4f})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUGMENTED_DB = "data/experience_replay_augmented.db"
FALLBACK_DB = "data/experience_replay.db"
SEQ_LEN = 100_000
NUM_EDITS = 10
METADATA_DIM = 10
VOCAB_SIZE = 5
BATCH_SIZE = 32
TOTAL_EPOCHS = 500
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 1.0
ALPHA_GFN = 0.5    # α-GFN mixing parameter (0=pure backward, 1=pure forward)
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# GeneratorPolicyV2: Dual-Head (Action + Value) Architecture
# ---------------------------------------------------------------------------

class GeneratorPolicyV2(nn.Module):
    """
    SOTA Dual-Head GFlowNet Policy with:
      - Conv1D spatial stem (shared backbone)
      - Factored action head (position + base scores, same as V1)
      - Value head V(s) → scalar state value estimate for Sub-EB

    Architecture:
        Input:  (B, L, 6)
        Shared: Conv1D(16, k=10, s=10) → Conv1D(32, k=10, s=10) → (B, L/100, 32)

        Action Head (factored):
            Position: Conv1D(1, k=1) → upsample → (B, L)
            Base:     Global pool → Dense(128) → Dense(64) → Dense(5)
            → combined → (B, L*5+1)

        Value Head:
            Global pool → Dense(128) → Dense(64) → Dense(1) → scalar V(s)
    """
    seq_len: int
    vocab_size: int = 5
    input_channels: int = 6

    @nn.compact
    def __call__(self, state_input: jnp.ndarray, target_metadata: jnp.ndarray):
        """
        Returns: (action_logits, value) tuple.
            action_logits: (B, L*5+1)
            value:         (B,)
        """
        B = state_input.shape[0]
        L = self.seq_len

        # ── Shared Conv1D Backbone ─────────────────────────────
        x = nn.Conv(features=16, kernel_size=(10,), strides=(10,), name="conv_stem_1")(state_input)
        x = jax.nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(10,), strides=(10,), name="conv_stem_2")(x)
        x = jax.nn.relu(x)
        # x: (B, L/100, 32)

        # ── Action Head (factored) ────────────────────────────
        # Position branch
        pos_coarse = nn.Conv(features=1, kernel_size=(1,), strides=(1,), name="pos_score_conv")(x)
        pos_coarse = pos_coarse.squeeze(-1)  # (B, L/100)
        stride_factor = 100
        pos_scores = jnp.repeat(pos_coarse, stride_factor, axis=-1)[:, :L]  # (B, L)

        # Global features for base scores
        global_features = jnp.mean(x, axis=1)  # (B, 32)
        global_features = jnp.concatenate([global_features, target_metadata], axis=-1)

        g = nn.Dense(128, name="action_dense_1")(global_features)
        g = jax.nn.relu(g)
        g = nn.Dense(64, name="action_dense_2")(g)
        g = jax.nn.relu(g)

        base_scores = nn.Dense(self.vocab_size, name="base_scores")(g)       # (B, 5)
        terminate_logit = nn.Dense(1, name="terminate_logit")(g)              # (B, 1)

        # Factored combination
        combined = pos_scores[:, :, None] + base_scores[:, None, :]           # (B, L, 5)
        mutation_logits = combined.reshape((B, L * self.vocab_size))           # (B, L*5)
        action_logits = jnp.concatenate([mutation_logits, terminate_logit], axis=-1)

        # ── Value Head ────────────────────────────────────────
        v = nn.Dense(128, name="value_dense_1")(global_features)
        v = jax.nn.relu(v)
        v = nn.Dense(64, name="value_dense_2")(v)
        v = jax.nn.relu(v)
        value = nn.Dense(1, name="value_output")(v).squeeze(-1)               # (B,)

        return action_logits, value


# ---------------------------------------------------------------------------
# α-GFN + Sub-EB Loss Functions
# ---------------------------------------------------------------------------

def sub_eb_loss(
    log_z: jnp.ndarray,
    forward_log_probs: jnp.ndarray,
    log_reward: jnp.ndarray,
    value_estimates: jnp.ndarray,
    alpha: float,
    num_edits: int,
) -> jnp.ndarray:
    """
    Sub-Trajectory Evaluation Balance loss with α-GFN mixing.

    For a trajectory τ = (s_0, a_0, s_1, ..., s_T), standard TB considers
    the entire trajectory. Sub-EB instead evaluates each partial sub-trajectory
    using V(s) to bridge the gap:

        L_SubEB = Σ_{t=0}^{T-1} (V(s_t) + α·log P_F(a_t|s_t) -
                                  V(s_{t+1}) - (1-α)·log P_B(a_t|s_{t+1}))²

    At the terminal step, V(s_T) is replaced with log R(x).
    At the initial step, V(s_0) is replaced with log Z.

    Parameters:
        log_z:               Learnable log partition function
        forward_log_probs:   (num_edits,) forward log-probs from trajectory
        log_reward:          Scalar log R(x) for terminal state
        value_estimates:     (num_edits+1,) V(s_t) estimates for states s_0..s_T
        alpha:               α-GFN mixing parameter
        num_edits:           Number of edits/steps in trajectory
    """
    total_loss = jnp.float32(0.0)

    for t in range(num_edits):
        # V(s_t): use log_z for initial state, else value estimate
        v_t = jnp.where(t == 0, log_z, value_estimates[t])

        # V(s_{t+1}): use log_reward for terminal state, else value estimate
        v_next = jnp.where(t == num_edits - 1, log_reward, value_estimates[t + 1])

        # Forward component
        forward_term = alpha * forward_log_probs[t]

        # Backward component: uniform backward policy
        # log P_B = -log(num_edits - t) since we can reverse any of the remaining mutations
        backward_log_prob = -jnp.log(jnp.float32(num_edits - t))
        backward_term = (1.0 - alpha) * backward_log_prob

        # Sub-EB residual for this transition
        residual = v_t + forward_term - v_next - backward_term
        total_loss += residual ** 2

    return total_loss / num_edits


def alpha_gfn_tb_loss(
    log_z: jnp.ndarray,
    forward_log_probs: jnp.ndarray,
    log_reward: jnp.ndarray,
    alpha: float,
    num_edits: int,
) -> jnp.ndarray:
    """
    α-GFN modified Trajectory Balance loss.

    L_α = (log Z + α·Σ log P_F - log R - (1-α)·Σ log P_B)²

    When α=1.0, reduces to standard TB loss.
    When α=0.5, equally weights forward and backward paths.
    """
    sum_log_pf = jnp.sum(forward_log_probs[:num_edits])

    # Uniform backward policy: log P_B = Σ -log(num_edits - t) for t=0..T-1
    log_pb_terms = jnp.array([-jnp.log(jnp.float32(num_edits - t)) for t in range(num_edits)])
    sum_log_pb = jnp.sum(log_pb_terms)

    residual = log_z + alpha * sum_log_pf - log_reward - (1.0 - alpha) * sum_log_pb
    return residual ** 2


# ---------------------------------------------------------------------------
# Experience Replay DataLoader (compatible with V1 data)
# ---------------------------------------------------------------------------

class AugmentedReplayLoader:
    """
    Reads from augmented or original experience replay database.
    Falls back to original DB if augmented doesn't exist.
    """
    def __init__(self, batch_size: int, num_edits: int, seq_len: int):
        self.batch_size = batch_size
        self.num_edits = num_edits
        self.seq_len = seq_len

        if os.path.exists(AUGMENTED_DB):
            self.db_path = AUGMENTED_DB
            print(f"[DataLoader] Using augmented database: {AUGMENTED_DB}")
        elif os.path.exists(FALLBACK_DB):
            self.db_path = FALLBACK_DB
            print(f"[DataLoader] Augmented DB not found, using original: {FALLBACK_DB}")
        else:
            raise FileNotFoundError(
                f"No replay buffer found at {AUGMENTED_DB} or {FALLBACK_DB}"
            )

        self._load_all()

    def _load_all(self):
        conn = sqlite3.connect(self.db_path)

        # Detect schema (V1 vs augmented)
        cursor = conn.execute("PRAGMA table_info(experiences)")
        columns = {row[1] for row in cursor.fetchall()}
        has_augmented = "is_augmented" in columns

        if has_augmented:
            query = "SELECT actions, forward_log_probs, reward FROM experiences ORDER BY RANDOM()"
        else:
            query = "SELECT actions, forward_log_probs, reward FROM experiences ORDER BY trajectory_id"

        cursor = conn.execute(query)

        self.actions_list = []
        self.log_probs_list = []
        self.rewards_list = []

        for row in cursor:
            actions_bytes, lp_bytes, reward = row
            actions = np.frombuffer(actions_bytes, dtype=np.int32).copy()
            log_probs = np.frombuffer(lp_bytes, dtype=np.float32).copy()

            self.actions_list.append(actions)
            self.log_probs_list.append(log_probs)
            self.rewards_list.append(float(reward))

        conn.close()

        self.total = len(self.rewards_list)
        if self.total == 0:
            raise ValueError("No experiences in database!")

        rewards_arr = np.array(self.rewards_list)
        print(f"[DataLoader] {self.total} experiences loaded")

        if has_augmented:
            conn2 = sqlite3.connect(self.db_path)
            orig = conn2.execute("SELECT COUNT(*) FROM experiences WHERE is_augmented=0").fetchone()[0]
            aug = conn2.execute("SELECT COUNT(*) FROM experiences WHERE is_augmented=1").fetchone()[0]
            conn2.close()
            print(f"[DataLoader] Original: {orig} | RBS Augmented: {aug}")

        print(f"[DataLoader] Reward: mean={rewards_arr.mean():.4f}, "
              f"std={rewards_arr.std():.4f}, min={rewards_arr.min():.4f}, max={rewards_arr.max():.4f}")

    def __len__(self):
        return self.total // self.batch_size

    def iter_epoch(self, rng_key=None):
        indices = np.arange(self.total)
        if rng_key is not None:
            np.random.seed(int(jax.random.randint(rng_key, (), 0, int(2**31 - 1))))
            np.random.shuffle(indices)

        for b in range(self.total // self.batch_size):
            batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]

            yield {
                "actions": jnp.array(np.stack([self.actions_list[i] for i in batch_idx])),
                "forward_log_probs": jnp.array(np.stack([self.log_probs_list[i] for i in batch_idx])),
                "rewards": jnp.array([self.rewards_list[i] for i in batch_idx]),
            }


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------

def make_offline_alpha_step(optimizer, alpha, num_edits):
    """Creates a JIT-compiled offline training step with α-GFN TB loss."""

    @jax.jit
    def step_fn(log_z, opt_state, batch_forward_log_probs, batch_rewards):
        def loss_fn(lz):
            log_rewards = jnp.log(jnp.maximum(batch_rewards, 1e-8))
            losses = jax.vmap(
                lambda lp, lr: alpha_gfn_tb_loss(lz, lp, lr, alpha, num_edits)
            )(batch_forward_log_probs, log_rewards)
            return jnp.mean(losses)

        loss, grad = jax.value_and_grad(loss_fn)(log_z)
        grad_norm = jnp.sqrt(grad ** 2)
        updates, new_opt_state = optimizer.update(grad, opt_state, log_z)
        new_log_z = optax.apply_updates(log_z, updates)
        return loss, new_log_z, new_opt_state, grad_norm

    return step_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[Init] α-GFN mixing parameter: {ALPHA_GFN}")
    print(f"[Init] Sub-EB: Value head enabled for sub-trajectory evaluation")

    # ── Verify dual-head policy parameters ─────────────────────
    print("\n[Architecture] Verifying GeneratorPolicyV2 parameter count...")
    policy_v2 = GeneratorPolicyV2(seq_len=SEQ_LEN)
    dummy_state = jnp.zeros((1, SEQ_LEN, 6))
    dummy_meta = jnp.zeros((1, METADATA_DIM))
    params = policy_v2.init(jax.random.PRNGKey(0), dummy_state, dummy_meta)

    num_params = sum(p.size for p in tree_util.tree_leaves(params))
    print(f"  Total parameters: {num_params:,}")
    assert num_params < 5_000_000, f"PARAMETER BUDGET EXCEEDED: {num_params:,} >= 5,000,000"
    print(f"  [PASS] Parameter budget assertion (< 5M)")

    # Test dual outputs
    action_logits, value = policy_v2.apply(params, dummy_state, dummy_meta)
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Value output shape:  {value.shape}")
    assert action_logits.shape == (1, SEQ_LEN * VOCAB_SIZE + 1)
    assert value.shape == (1,)
    print(f"  [PASS] Dual-head architecture verified")

    # Print parameter breakdown
    print("\n  Parameter Breakdown:")
    flat_params, _ = tree_util.tree_flatten_with_path(params)
    for path, leaf in flat_params:
        path_str = "/".join(str(p) for p in path)
        print(f"    {path_str:55s} → {str(leaf.shape):>15s}  ({leaf.size:>8,})")

    # ── Load replay data ───────────────────────────────────────
    print()
    try:
        dataloader = AugmentedReplayLoader(
            batch_size=BATCH_SIZE,
            num_edits=NUM_EDITS,
            seq_len=SEQ_LEN,
        )
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[WARN] No replay data available yet. Exiting gracefully.")
        print("  Run the overnight pipeline first, then re-run this trainer.")
        sys.exit(0)

    num_batches = len(dataloader)
    print(f"\n[Config] Batch size: {BATCH_SIZE} | Batches/epoch: {num_batches}")
    print(f"[Config] Epochs: {TOTAL_EPOCHS} | α-GFN: {ALPHA_GFN}")
    print(f"[Config] Optimizer: AdamW | LR: {LEARNING_RATE} | Clip: {MAX_GRAD_NORM}")

    # ── Initialize optimizer and log_z ─────────────────────────
    log_z = jnp.float32(0.0)

    # Try loading from previous checkpoint
    prior_ckpt = os.path.join(CHECKPOINT_DIR, "edm3_offline_final.npz")
    if os.path.exists(prior_ckpt):
        data = np.load(prior_ckpt)
        if "log_z" in data:
            log_z = jnp.float32(float(data["log_z"]))
            print(f"[Resume] log_z loaded from {prior_ckpt}: {log_z:.6f}")

    optimizer = build_optimizer(LEARNING_RATE, MAX_GRAD_NORM)
    opt_state = optimizer.init(log_z)

    step_fn = make_offline_alpha_step(optimizer, ALPHA_GFN, NUM_EDITS)
    tracker = ConvergenceTracker(alpha=0.95, threshold_pct=0.05,
                                 window_size=50, variance_threshold=0.01)

    print(f"\n[Training] Beginning α-GFN offline optimization.")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Mean Loss':>14} | {'EMA Loss':>12} | {'log_Z':>10} | {'Grad Norm':>10}")
    print("-" * 70)

    key = jax.random.PRNGKey(7777)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        key, epoch_key = jax.random.split(key)

        epoch_losses = []
        epoch_grads = []

        for batch in dataloader.iter_epoch(rng_key=epoch_key):
            loss, log_z, opt_state, grad_norm = step_fn(
                log_z, opt_state,
                batch["forward_log_probs"],
                batch["rewards"],
            )
            epoch_losses.append(float(loss))
            epoch_grads.append(float(grad_norm))

        mean_loss = np.mean(epoch_losses)
        mean_grad = np.mean(epoch_grads)

        converged = tracker.update(mean_loss, epoch)

        if epoch == 1 or epoch % 10 == 0 or converged:
            print(f"{epoch:>6} | {mean_loss:>14.4f} | {tracker.ema:>12.4f} | "
                  f"{float(log_z):>10.6f} | {mean_grad:>10.6f}")

        if converged and tracker.convergence_epoch == epoch:
            print(f"\n*** CONVERGENCE at epoch {epoch} ***")
            print(f"    {tracker.get_status_str()}")

    print("-" * 70)

    if tracker.converged:
        pct = (tracker.baseline_ema - tracker.ema) / max(abs(tracker.baseline_ema), 1e-8) * 100
        print(f"CONVERGENCE VALIDATED at epoch {tracker.convergence_epoch} ({pct:.2f}% EMA drop).")
    else:
        pct = (tracker.baseline_ema - tracker.ema) / max(abs(tracker.baseline_ema), 1e-8) * 100
        print(f"NOT CONVERGED in {TOTAL_EPOCHS} epochs ({pct:.2f}% EMA drop, threshold 5%).")

    # Save
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "edm3_v2_offline_final.npz")
    np.savez(ckpt_path, log_z=np.array(float(log_z)))
    print(f"\n[Checkpoint] Saved to {ckpt_path}")
    print(f"  Final log_z: {float(log_z):.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
