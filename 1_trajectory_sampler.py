"""
Phase 4, Script 1: Offline Trajectory Sampler.

Loads the trained Conv1D GeneratorPolicy checkpoint, generates N=5,000 complete
trajectories (100kb sequences, 10 edits each) with temperature scaling T=2.0
for high-variance exploration, and exports terminal sequences + actions to
data/unscored_trajectories.npz.

This script is JAX-only and operates entirely offline.
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util

from gflownet_env import GFlowNetEnv, GeneratorPolicy
from gflownet_trainer import init_train_state


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEQ_LEN = 100_000         # 100kb
NUM_EDITS = 10            # Actions per trajectory
TEMPERATURE = 2.0         # High temperature for exploration diversity
METADATA_DIM = 10
OUTPUT_PATH = "data/unscored_trajectories.npz"
CHECKPOINT_PATH = "checkpoints/edm3_epoch_500.npz"


def load_checkpoint_params(checkpoint_path: str, seq_len: int, metadata_dim: int):
    """
    Loads TrainState from a numpy checkpoint file.
    Falls back to freshly initialized random parameters if no checkpoint exists.
    """
    if os.path.exists(checkpoint_path):
        print(f"[Load] Found checkpoint: {checkpoint_path}")
        # We initialize a fresh train state to get the PyTree structure,
        # then populate it from the checkpoint leaf values.
        key = jax.random.PRNGKey(0)
        template_state = init_train_state(key, seq_len=seq_len, metadata_dim=metadata_dim)

        data = np.load(checkpoint_path, allow_pickle=True)
        # Reconstruct leaves from checkpoint
        ts_leaves, ts_struct = tree_util.tree_flatten(template_state)
        loaded_leaves = []
        for i, leaf in enumerate(ts_leaves):
            checkpoint_key = f"ts_leaf_{i}"
            if checkpoint_key in data:
                loaded_leaves.append(jnp.array(data[checkpoint_key]))
            else:
                print(f"  [Warn] Leaf {i} not found in checkpoint, using initialization.")
                loaded_leaves.append(leaf)

        train_state = tree_util.tree_unflatten(ts_struct, loaded_leaves)
        print(f"[Load] TrainState restored from checkpoint (epoch {int(data.get('epoch', -1))}).")
        return train_state
    else:
        print(f"[Load] No checkpoint at {checkpoint_path}. Initializing fresh parameters.")
        key = jax.random.PRNGKey(42)
        return init_train_state(key, seq_len=seq_len, metadata_dim=metadata_dim)


def onehot_to_acgtn(seq_onehot: np.ndarray) -> str:
    """
    Converts a (L, 5) one-hot array to an ACGTN string.
    Vocab: index 0=A, 1=C, 2=G, 3=T, 4=N
    """
    BASES = "ACGTN"
    indices = np.argmax(seq_onehot, axis=-1)  # (L,)
    return "".join(BASES[i] for i in indices)


def sample_single_trajectory(
    env: GFlowNetEnv,
    policy: GeneratorPolicy,
    gen_params,
    wt_seq: jnp.ndarray,
    target_metadata: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float,
    num_edits: int,
):
    """
    Runs a single trajectory through the MDP using the trained policy with
    temperature-scaled sampling. Returns (terminal_seq, actions, forward_log_probs).
    """
    state = env.reset(wt_seq)

    def scan_step(carry, step_key):
        current_state = carry

        # Build 6-channel input
        mutated_channel = current_state.mutated.astype(jnp.float32)[:, None]
        state_6ch = jnp.concatenate([current_state.seq, mutated_channel], axis=-1)

        # Forward policy
        seq_batched = state_6ch[None, ...]
        meta_batched = target_metadata[None, ...]
        raw_logits = policy.apply(gen_params, seq_batched, meta_batched)[0]

        # Mask invalid actions and block terminate during forced edits
        valid_mask = env.get_valid_actions(current_state)
        masked_logits = jnp.where(valid_mask, raw_logits, -1e9)
        masked_logits = masked_logits.at[env.terminate_action].set(-1e9)

        # Temperature scaling for exploration diversity
        scaled_logits = masked_logits / temperature

        # Sample action
        action = jax.random.categorical(step_key, scaled_logits)

        # Log prob under original (unscaled) logits for TB loss
        log_probs = jax.nn.log_softmax(masked_logits)
        log_p_f = log_probs[action]

        # Step environment
        next_state, _ = env.step(current_state, action)
        return next_state, (action, log_p_f)

    keys = jax.random.split(key, num_edits)
    final_state, (actions, forward_log_probs) = jax.lax.scan(
        scan_step, state, keys,
    )

    return final_state.seq, actions, forward_log_probs


def main():
    # Parse optional arguments
    num_trajectories = 5000
    if len(sys.argv) > 1:
        num_trajectories = int(sys.argv[1])

    print("=" * 70)
    print("EEPM3 Phase 4: Offline Trajectory Sampler")
    print("=" * 70)
    print(f"[Config] Trajectories: {num_trajectories}")
    print(f"[Config] Sequence Length: {SEQ_LEN} bp")
    print(f"[Config] Edits per trajectory: {NUM_EDITS}")
    print(f"[Config] Temperature: {TEMPERATURE}")
    print(f"[Config] Output: {OUTPUT_PATH}")

    # Load trained policy
    train_state = load_checkpoint_params(CHECKPOINT_PATH, SEQ_LEN, METADATA_DIM)
    gen_params = train_state.gen_params

    num_params = sum(p.size for p in tree_util.tree_leaves(gen_params))
    print(f"[Init] Generator parameters: {num_params:,}")

    # Initialize environment and policy
    env = GFlowNetEnv(seq_len=SEQ_LEN, max_edits=NUM_EDITS)
    policy = GeneratorPolicy(seq_len=SEQ_LEN)

    # Wild-type sequence (uniform A's as baseline)
    wt_seq = jax.nn.one_hot(
        jnp.zeros((SEQ_LEN,), dtype=jnp.int32), 5, dtype=jnp.float32
    )
    target_metadata = jnp.ones((METADATA_DIM,))

    # JIT-compile the trajectory sampler
    print("[Init] JIT compiling trajectory sampler...")
    jitted_sample = jax.jit(
        lambda params, wt, meta, k: sample_single_trajectory(
            env, policy, params, wt, meta, k,
            temperature=TEMPERATURE, num_edits=NUM_EDITS,
        )
    )

    # Warm-up compilation
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    _ = jitted_sample(gen_params, wt_seq, target_metadata, key)
    t1 = time.time()
    print(f"[Init] Compilation completed in {t1 - t0:.2f}s")

    # Generate trajectories
    print(f"\n[Sampling] Generating {num_trajectories} trajectories...")
    os.makedirs("data", exist_ok=True)

    all_terminal_seqs = []   # Will store ACGTN strings
    all_terminal_onehot = [] # Will store (L, 5) arrays for offline training
    all_actions = []         # Will store (num_edits,) int arrays
    all_log_probs = []       # Will store (num_edits,) float arrays

    master_key = jax.random.PRNGKey(1234)

    report_interval = max(1, num_trajectories // 20)
    t_start = time.time()

    for i in range(num_trajectories):
        master_key, traj_key = jax.random.split(master_key)

        terminal_seq, actions, log_probs = jitted_sample(
            gen_params, wt_seq, target_metadata, traj_key,
        )

        # Convert to numpy for storage
        terminal_np = np.array(terminal_seq)
        actions_np = np.array(actions)
        log_probs_np = np.array(log_probs)

        all_terminal_onehot.append(terminal_np)
        all_actions.append(actions_np)
        all_log_probs.append(log_probs_np)

        # Convert to ACGTN string for API submission
        all_terminal_seqs.append(onehot_to_acgtn(terminal_np))

        if (i + 1) % report_interval == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (num_trajectories - i - 1) / max(rate, 0.01)
            print(f"  [{i+1:>5}/{num_trajectories}] Rate: {rate:.1f} traj/s | ETA: {eta:.0f}s")

    t_end = time.time()
    total_time = t_end - t_start
    print(f"\n[Done] Generated {num_trajectories} trajectories in {total_time:.1f}s "
          f"({num_trajectories / total_time:.1f} traj/s)")

    # Save to npz
    terminal_onehot_arr = np.stack(all_terminal_onehot, axis=0)  # (N, L, 5)
    actions_arr = np.stack(all_actions, axis=0)                   # (N, num_edits)
    log_probs_arr = np.stack(all_log_probs, axis=0)               # (N, num_edits)

    # Save ACGTN strings as object array
    seq_strings = np.array(all_terminal_seqs, dtype=object)

    np.savez_compressed(
        OUTPUT_PATH,
        terminal_onehot=terminal_onehot_arr,
        actions=actions_arr,
        forward_log_probs=log_probs_arr,
        sequences=seq_strings,
        seq_len=SEQ_LEN,
        num_edits=NUM_EDITS,
        temperature=TEMPERATURE,
    )

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"[Save] Wrote {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    print(f"  terminal_onehot: {terminal_onehot_arr.shape}")
    print(f"  actions:         {actions_arr.shape}")
    print(f"  forward_log_probs: {log_probs_arr.shape}")
    print(f"  sequences:       {len(seq_strings)} ACGTN strings ({len(seq_strings[0])} bp each)")
    print("=" * 70)


if __name__ == "__main__":
    main()
