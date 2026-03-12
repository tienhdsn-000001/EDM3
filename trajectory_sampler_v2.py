"""
Phase 5, Script 3: Partial GFlowNet Trajectory Sampler v2.

Implements a "Planner" heuristic that restricts the GFlowNet's action space to
a randomly-selected 10kb continuous window within the 100kb sequence. This
reduces the action space from 500,001 to ~50,001 per trajectory, dramatically
improving sample efficiency and exploration density within local neighborhoods.

Features:
  - Random 10kb window selection per trajectory
  - Strict action masking outside the window
  - Compatibility with V1 data format for the API worker
  - Temperature-scaled sampling for exploration
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

SEQ_LEN = 100_000
WINDOW_SIZE = 10_000          # 10kb local exploration window
NUM_EDITS = 10
TEMPERATURE = 2.0
METADATA_DIM = 10
VOCAB_SIZE = 5
OUTPUT_PATH = "data/unscored_trajectories_v2.npz"
CHECKPOINT_PATH = "checkpoints/edm3_epoch_500.npz"


# ---------------------------------------------------------------------------
# Windowed Action Masking
# ---------------------------------------------------------------------------

def create_window_mask(
    env: GFlowNetEnv,
    state,
    window_start: int,
    window_end: int,
) -> jnp.ndarray:
    """
    Creates an action mask that restricts mutations to positions within
    [window_start, window_end). Positions outside the window are zeroed out.

    This works by taking the standard valid-action mask and ANDing it with
    the window constraint.
    """
    base_mask = env.get_valid_actions(state)

    # Build position-level window mask
    # Action index = position * VOCAB_SIZE + base, for all bases 0-4
    # So positions in [window_start, window_end) have indices in
    # [window_start*5, window_end*5)
    action_space_size = SEQ_LEN * VOCAB_SIZE + 1
    window_mask = jnp.zeros(action_space_size, dtype=jnp.bool_)

    # Enable actions within window
    start_action = window_start * VOCAB_SIZE
    end_action = window_end * VOCAB_SIZE
    window_mask = window_mask.at[start_action:end_action].set(True)

    # Always allow terminate action (last index)
    window_mask = window_mask.at[-1].set(True)

    # Combined mask: valid AND within window
    return base_mask & window_mask


# ---------------------------------------------------------------------------
# Windowed Trajectory Sampling
# ---------------------------------------------------------------------------

def sample_windowed_trajectory(
    env: GFlowNetEnv,
    policy: GeneratorPolicy,
    gen_params,
    wt_seq: jnp.ndarray,
    target_metadata: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float,
    num_edits: int,
    window_start: int,
    window_end: int,
):
    """
    Samples a trajectory with mutations restricted to [window_start, window_end).
    
    Returns (terminal_seq, actions, forward_log_probs, window_start, window_end).
    """
    state = env.reset(wt_seq)

    def scan_step(carry, step_key):
        current_state = carry

        # Build 6-channel input
        mutated_channel = current_state.mutated.astype(jnp.float32)[:, None]
        state_6ch = jnp.concatenate([current_state.seq, mutated_channel], axis=-1)

        # Forward policy (full sequence input, windowed output masking)
        seq_batched = state_6ch[None, ...]
        meta_batched = target_metadata[None, ...]
        raw_logits = policy.apply(gen_params, seq_batched, meta_batched)[0]

        # Apply windowed mask
        windowed_mask = create_window_mask(env, current_state, window_start, window_end)
        masked_logits = jnp.where(windowed_mask, raw_logits, -1e9)

        # Block terminate during forced edits
        masked_logits = masked_logits.at[env.terminate_action].set(-1e9)

        # Temperature scaling
        scaled_logits = masked_logits / temperature

        # Sample action
        action = jax.random.categorical(step_key, scaled_logits)

        # Log prob under unscaled logits
        log_probs = jax.nn.log_softmax(masked_logits)
        log_p_f = log_probs[action]

        # Step
        next_state, _ = env.step(current_state, action)
        return next_state, (action, log_p_f)

    keys = jax.random.split(key, num_edits)
    final_state, (actions, forward_log_probs) = jax.lax.scan(
        scan_step, state, keys,
    )

    return final_state.seq, actions, forward_log_probs


def onehot_to_acgtn(seq_onehot: np.ndarray) -> str:
    """Converts (L, 5) one-hot to ACGTN string."""
    BASES = "ACGTN"
    return "".join(BASES[i] for i in np.argmax(seq_onehot, axis=-1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_trajectories = 5000
    if len(sys.argv) > 1:
        num_trajectories = int(sys.argv[1])

    print("=" * 70)
    print("EEPM3 Phase 5: Partial GFlowNet Trajectory Sampler v2")
    print("=" * 70)
    print(f"[Config] Trajectories: {num_trajectories}")
    print(f"[Config] Sequence: {SEQ_LEN} bp | Window: {WINDOW_SIZE} bp")
    print(f"[Config] Effective action space: ~{WINDOW_SIZE * (VOCAB_SIZE - 1) + 1:,} (vs {SEQ_LEN * VOCAB_SIZE + 1:,} full)")
    print(f"[Config] Reduction: {((SEQ_LEN * VOCAB_SIZE + 1) / (WINDOW_SIZE * (VOCAB_SIZE - 1) + 1)):.1f}x")
    print(f"[Config] Edits: {NUM_EDITS} | Temperature: {TEMPERATURE}")
    print(f"[Config] Output: {OUTPUT_PATH}")

    # Load policy
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[Load] Loading checkpoint: {CHECKPOINT_PATH}")
        key = jax.random.PRNGKey(0)
        train_state = init_train_state(key, seq_len=SEQ_LEN, metadata_dim=METADATA_DIM)

        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        ts_leaves, ts_struct = tree_util.tree_flatten(train_state)
        loaded_leaves = []
        for i, leaf in enumerate(ts_leaves):
            ck = f"ts_leaf_{i}"
            loaded_leaves.append(jnp.array(data[ck]) if ck in data else leaf)
        train_state = tree_util.tree_unflatten(ts_struct, loaded_leaves)
        print(f"[Load] Restored from epoch {int(data.get('epoch', -1))}")
    else:
        print("[Load] No checkpoint found, using fresh initialization.")
        train_state = init_train_state(jax.random.PRNGKey(42), seq_len=SEQ_LEN, metadata_dim=METADATA_DIM)

    gen_params = train_state.gen_params

    env = GFlowNetEnv(seq_len=SEQ_LEN, max_edits=NUM_EDITS)
    policy = GeneratorPolicy(seq_len=SEQ_LEN)

    wt_seq = jax.nn.one_hot(
        jnp.zeros((SEQ_LEN,), dtype=jnp.int32), 5, dtype=jnp.float32
    )
    target_metadata = jnp.ones((METADATA_DIM,))

    # Generate with random windows
    print(f"\n[Sampling] Generating {num_trajectories} windowed trajectories...")
    os.makedirs("data", exist_ok=True)

    all_terminal_onehot = []
    all_actions = []
    all_log_probs = []
    all_terminal_seqs = []
    all_windows = []

    master_key = jax.random.PRNGKey(5678)
    report_interval = max(1, num_trajectories // 20)
    t_start = time.time()

    # We don't JIT the outer loop since window_start varies per trajectory
    # Instead we JIT-compile with static window args
    jit_cache = {}

    for i in range(num_trajectories):
        master_key, traj_key, window_key = jax.random.split(master_key, 3)

        # Random 10kb window
        max_start = SEQ_LEN - WINDOW_SIZE
        window_start = int(jax.random.randint(window_key, (), 0, max_start))
        window_end = window_start + WINDOW_SIZE

        # Quantize window_start to nearest 1000 for JIT cache efficiency
        cache_key = (window_start // 1000) * 1000
        if cache_key not in jit_cache:
            jit_cache[cache_key] = jax.jit(
                lambda params, wt, meta, k, ws=cache_key, we=cache_key + WINDOW_SIZE: (
                    sample_windowed_trajectory(
                        env, policy, params, wt, meta, k,
                        temperature=TEMPERATURE, num_edits=NUM_EDITS,
                        window_start=ws, window_end=we,
                    )
                )
            )

        jitted_fn = jit_cache[cache_key]
        terminal_seq, actions, log_probs = jitted_fn(
            gen_params, wt_seq, target_metadata, traj_key,
        )

        terminal_np = np.array(terminal_seq)
        all_terminal_onehot.append(terminal_np)
        all_actions.append(np.array(actions))
        all_log_probs.append(np.array(log_probs))
        all_terminal_seqs.append(onehot_to_acgtn(terminal_np))
        all_windows.append((window_start, window_end))

        if (i + 1) % report_interval == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (num_trajectories - i - 1) / max(rate, 0.01)
            w = all_windows[-1]
            print(f"  [{i+1:>5}/{num_trajectories}] Rate: {rate:.1f}/s | ETA: {eta:.0f}s | "
                  f"Window: [{w[0]}, {w[1]})")

    t_end = time.time()
    print(f"\n[Done] {num_trajectories} windowed trajectories in {t_end-t_start:.1f}s")
    print(f"  JIT compilations: {len(jit_cache)} (one per 1kb-quantized window start)")

    # Save
    terminal_arr = np.stack(all_terminal_onehot, axis=0)
    actions_arr = np.stack(all_actions, axis=0)
    log_probs_arr = np.stack(all_log_probs, axis=0)
    seq_strings = np.array(all_terminal_seqs, dtype=object)
    windows_arr = np.array(all_windows, dtype=np.int32)

    np.savez_compressed(
        OUTPUT_PATH,
        terminal_onehot=terminal_arr,
        actions=actions_arr,
        forward_log_probs=log_probs_arr,
        sequences=seq_strings,
        windows=windows_arr,
        seq_len=SEQ_LEN,
        window_size=WINDOW_SIZE,
        num_edits=NUM_EDITS,
        temperature=TEMPERATURE,
    )

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n[Save] {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    print(f"  terminal_onehot: {terminal_arr.shape}")
    print(f"  actions:         {actions_arr.shape}")
    print(f"  windows:         {windows_arr.shape}")
    print(f"  sequences:       {len(seq_strings)} ACGTN strings")

    # Window distribution stats
    starts = windows_arr[:, 0]
    print(f"\n[Window Stats]")
    print(f"  Mean start: {starts.mean():.0f} bp")
    print(f"  Std start:  {starts.std():.0f} bp")
    print(f"  Min start:  {starts.min()} bp")
    print(f"  Max start:  {starts.max()} bp")
    print("=" * 70)


if __name__ == "__main__":
    main()
