import jax
import jax.numpy as jnp
from jax import tree_util
from gflownet_env import GFlowNetEnv, EnvState, GeneratorPolicy

def run_test():
    print("--------------------------------------------------")
    print("EEPM3 GFlowNet Environment Verification Log")
    print("--------------------------------------------------")

    seq_len = 100000
    env = GFlowNetEnv(seq_len=seq_len, max_edits=5)

    key = jax.random.PRNGKey(1337)

    # Initialize a dummy wild-type sequence containing purely Base0 everywhere
    wt_seq = jax.nn.one_hot(jnp.zeros((seq_len,), dtype=jnp.int32), 5, dtype=jnp.float32)
    state = env.reset(wt_seq)

    print(f"Initialized Mock State (Seq Length: {seq_len}, Vocab: 5)")
    print("Executing 5 independent edits using jax.lax.scan...")

    def step_fn(carry_state, step_key):
        valid_mask = env.get_valid_actions(carry_state)

        # We manually bypass the policy neural network just for this environment logic test.
        # We sample purely randomly from the *valid* actions via logits masking.
        logits = jnp.where(valid_mask, 0.0, -jnp.inf)

        # Explicitly prevent the terminate index from triggering to guarantee 5 forced edits
        logits = logits.at[env.terminate_action].set(-jnp.inf)

        action = jax.random.categorical(step_key, logits)
        next_state, is_terminal = env.step(carry_state, action)

        return next_state, action

    keys = jax.random.split(key, 5)

    # Pure generic JIT compilation
    compiled_scan = jax.jit(lambda s, ks: jax.lax.scan(step_fn, s, ks))
    final_state, actions = compiled_scan(state, keys)

    print("\n[Executing Required Assertions]")
    assert actions.shape == (5,)
    print(f"Randomly Sampled Valid Actions: {actions}")

    pos = actions // 5
    bases = actions % 5

    for i in range(5):
        p = pos[i]
        b = bases[i]
        print(f" -> Edit {i+1}: Pos {p}, Base {b}")
        assert final_state.seq[p, b] == 1.0, f"[FAIL] Base {b} not set at position {p}"
        assert final_state.mutated[p] == True, f"[FAIL] Mutated mask not True at position {p}"

    total_mutations = jnp.sum(final_state.mutated)
    assert total_mutations == 5, f"[FAIL] Expected precisely 5 total mutations, got {total_mutations}"

    # Validate no identical indices via unique
    unique_positions = jnp.unique(pos)
    assert len(unique_positions) == 5, "[FAIL] Masking failed: duplicate positions were edited!"

    print("[PASS] next_state correctly reflects exactly 5 unique base changes.")
    print("[PASS] Action mask successfully prevented duplicate edits dynamically within scan compilation.")

    # --- Test new Conv1D-based GeneratorPolicy ---
    print("\nInitializing Conv1D-based Forward Policy P_F...")
    policy = GeneratorPolicy(seq_len=seq_len)

    # Build 6-channel input: 5 one-hot + 1 mutated mask
    mutated_channel = final_state.mutated.astype(jnp.float32)[:, None]  # (L, 1)
    state_6ch = jnp.concatenate([final_state.seq, mutated_channel], axis=-1)  # (L, 6)
    state_batched = jnp.stack([state_6ch, state_6ch], axis=0)  # (2, L, 6)

    # Fake target metadata shape B=2, Metadata_Size=10
    dummy_meta = jnp.ones((2, 10))

    pkey, init_key = jax.random.split(key)
    params = policy.init(init_key, state_batched, dummy_meta)

    out_logits = policy.apply(params, state_batched, dummy_meta)
    expected_out_shape = (2, seq_len * 5 + 1)

    assert out_logits.shape == expected_out_shape, f"[FAIL] Logits shape expected {expected_out_shape}, got {out_logits.shape}"
    print(f"[PASS] Conv1D policy compiled. Emitting {out_logits.shape} action logits.")

    # --- Parameter Count Assertion ---
    num_params = sum(p.size for p in tree_util.tree_leaves(params))
    print(f"\n[Parameter Count] Total: {num_params:,}")
    assert num_params < 5_000_000, f"[FAIL] Parameter budget exceeded: {num_params:,} >= 5,000,000"
    print(f"[PASS] Parameter count {num_params:,} is under 5M budget.")

    # Print parameter breakdown
    print("\n    Parameter Breakdown:")
    flat_params, tree_def = tree_util.tree_flatten_with_path(params)
    for path, leaf in flat_params:
        path_str = "/".join(str(p) for p in path)
        print(f"      {path_str:55s} → {str(leaf.shape):>15s}  ({leaf.size:>10,} params)")

    print("\n--------------------------------------------------")
    print("All MDP environment phase specifications passed!")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_test()
