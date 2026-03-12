# EEPM3: Expandable Epigenetic Profile Mimicry Module by Mutation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Powered-blue.svg)](https://github.com/google/jax)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## 🧬 Abstract: The Inverse Epigenetic Problem

Traditional genomic predictive models (like AlphaGenome) focus on the **Forward Problem**: predicting biological and epigenetic functions from a fixed DNA sequence ($X \rightarrow Y$). 

**EEPM3** solves the **Inverse Problem**. Utilizing a Generative Flow Network (GFlowNet) architecture, EEPM3 identifies the precise, optimal sequence of mutations ($X_{mutated}$) required to transition a genomic state toward any desired epigenetic target profile ($Y_{target}$). It acts as a universal epigenetic state-matcher, routing high-value biological edits without temporal confounding.

---

## 🚀 The Architecture (SOTA 2026 GFlowNets)

Operating on a massive 100kb sequence space introduces significant memory bottlenecks. The `GeneratorPolicyV2` utilizes a highly optimized **dual-head Conv1D architecture** comprising exactly **34,136 parameters**. By leveraging 1D convolutions and dense positional projections instead of heavy attention mechanisms, it completely bypasses the catastrophic $O(N \times V)$ memory explosion typical in $100kb \times 5$ state spaces.

### Sub-EB & $\alpha$-GFN Objective

To solve the credit assignment problem across hundreds of mutation steps, EEPM3 replaces standard terminal-only evaluation with **Sub-Trajectory Evaluation Balance (Sub-EB)**. The dual-head policy outputs both action logits and a continuous state Value $V(s)$.

The flow is optimized using a hybrid $\alpha$-GFN equation that mixes on-policy flow limits with off-policy exploration:

$$
\mathcal{L}_{\alpha\text{-GFN}} = \alpha \cdot \mathcal{L}_{TB} + (1 - \alpha) \cdot \mathcal{L}_{Sub-EB}
$$

The core **Trajectory Balance (TB)** loss ensures global flow consistency:

$$
L_{TB} = \left(\log Z + \sum_{t=0}^{T-1} \log P_F(a_t|s_t) - \log R(x) - \sum_{t=0}^{T-1} \log P_B(s_t|s_{t+1})\right)^2
$$

### Retrospective Backward Synthesis (RBS)

Because API scoring is the most expensive operational bottleneck, EEPM3 implements **RBS** (Retrospective Backward Synthesis). After a successful trajectory achieves a high reward, the augmenter "hallucinates" alternative, valid mutation permutations that arrive at the same terminal state. This synthetically increases the high-reward training signal density by 1.5x at zero API cost.

---

## 🧬 Biological Priors & Reward Function

The environment's composite reward $R(x)$ enforces both epigenetic target similarity and biological viability:

$$
R(x) = \exp(-\alpha \cdot \mathcal{L}_{mask}(AG(x), T, M)) + \beta \cdot \log P_{Evo}(x)
$$

1. **Masked Modality Loss ($\mathcal{L}_{mask}$)**: Evaluates the delta between the AlphaGenome inference $AG(x)$ and the target tensor $T$. Because clinical and API tracks are highly sparse, the loss operates behind a boolean mask $M$, strictly preventing NaN leakage from unmeasured epigenetic tracks.
2. **Evo-2 Zero-Shot Prior ($\log P_{Evo}$)**: Neural networks are prone to finding adversarial vulnerabilities. To prevent the GFlowNet from proposing biologically lethal "garbage DNA" that artificially inflates the API score, we regularize the reward using the log-likelihood of a foundational DNA language model (Evo-2). This acts as a gravitational pull within the manifold of evolutionarily viable DNA.

---

## ⚙️ The Decoupled "Fire & Forget" Pipeline

EEPM3 is designed for high-concurrency cloud execution, orchestrated across three fault-tolerant stages:

1. **`1_trajectory_sampler.py` (GPU)**: Rapidly explores the sequence space using vectorized JAX sampling to generate tens of thousands of candidate trajectories.
2. **`2_api_worker.py` (CPU)**: An async, multi-threaded API polling worker that scores candidates against AlphaGenome. It features exponential backoff to handle `RESOURCE_EXHAUSTED` errors and commits results to a local SQLite database for crash-safe resumption.
3. **`3_offline_trainer.py` (GPU)**: Imports the augmented replay buffer, JIT-compiles the Sub-EB gradient steps, and optimizes the policy over 100-500 epochs.

---

## ⚡ Quick Start / Usage

EEPM3 is optimized for Colab / Kaggle environments using T4 or L4 GPUs.

```bash
# Clone the repository
git clone https://github.com/tienhdsn-000001/EDM3.git
cd EDM3

# Install dependencies
pip install -r requirements.txt

# Export your AlphaGenome Auth Key
export ALPHA_GENOME_API_KEY="your_api_key_here"

# Execute the decoupled pipeline
bash run_overnight.sh
```

---

## 📊 Benchmarks (March 2026)

In the latest production execution on a Colab T4 instance:
- **Sequence Target**: 100,000 base pairs (N-padded to 131,072 to meet specific API constraints).
- **Target Modality**: DNASE Accessibility.
- **Optimization Speed**: Reached Statistical Convergence at **Epoch 82**.
- **Model Efficiency**: 34,136 parameters achieved a mathematically validated **14.3% EMA loss drop** across the augmented offline replay buffer.

---

## 🤝 Contributing & Disclaimer

### Fork & Pull Request Rules
Direct pushes to the `main` branch are restricted. If you wish to contribute to the engine:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/optimization-name`).
3. Submit a Pull Request targeting validation checks.

### 🔬 Scientific Disclaimer
**EEPM3 is a pre-alpha computational architecture designed to solve the VRAM and latency bottlenecks of inverse genomic design. It currently demonstrates mathematical optimization convergence against proxy and unvalidated API targets. It is not yet clinically validated. Do not use this architecture for clinical decision-making or real-world biological synthesis without rigorous wet-lab validation.**
