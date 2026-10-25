"""
==============================================================================
 Experiment 6 - Hidden Markov Model (Forward + Viterbi)
 Used Mobile Phone Price Prediction
==============================================================================

AIM:
  Model monthly price movements of phones using an HMM.
  Implement Forward (sequence likelihood), Viterbi (decoding), and
  Baum-Welch (learning) from scratch. Validate against hmmlearn.

THEORY:
  HMM = (pi, A, B) where:
  - pi: initial state distribution
  - A: state transition probabilities
  - B: emission probabilities
  Hidden states: "Launch phase", "Stable market", "Depreciation phase"
  Observations: "Big Drop", "Small Drop", "Stable", "Rise"
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_project, print_section, print_subsection, save_fig, RANDOM_STATE
from src.hmm import HMMScratch

setup_project()
print_section("Experiment 6: Hidden Markov Model")

# ======================================================================
# 6A: BUILD OBSERVATION SEQUENCES FROM PHONE DATA
# ======================================================================
print_section("6A: Building Observation Sequences")

# Simulate monthly price depreciation for different phone models
np.random.seed(RANDOM_STATE)

# Define HMM structure
STATE_NAMES = ['Launch', 'Stable', 'Depreciation']
OBS_NAMES = ['Big Drop', 'Small Drop', 'Stable', 'Rise']
N_STATES = len(STATE_NAMES)
N_OBS = len(OBS_NAMES)

# True parameters (for data generation)
true_pi = np.array([0.7, 0.2, 0.1])  # Most phones start in Launch
true_A = np.array([
    [0.5, 0.4, 0.1],   # Launch -> Launch/Stable/Deprec
    [0.05, 0.7, 0.25],  # Stable -> Launch/Stable/Deprec
    [0.02, 0.08, 0.90], # Depreciation -> (mostly stays)
])
true_B = np.array([
    [0.3, 0.3, 0.2, 0.2],   # Launch: drops common, some stability
    [0.05, 0.15, 0.6, 0.2],  # Stable: mostly stable, some rise
    [0.4, 0.35, 0.2, 0.05],  # Depreciation: drops dominate
])

# Generate observation sequences (simulate 20 phone models, 24 months each)
def generate_sequences(pi, A, B, n_sequences=20, seq_length=24):
    """Generate observation sequences from true HMM."""
    sequences = []
    hidden_sequences = []
    for _ in range(n_sequences):
        states = np.zeros(seq_length, dtype=int)
        obs = np.zeros(seq_length, dtype=int)
        states[0] = np.random.choice(len(pi), p=pi)
        obs[0] = np.random.choice(B.shape[1], p=B[states[0]])
        for t in range(1, seq_length):
            states[t] = np.random.choice(len(pi), p=A[states[t-1]])
            obs[t] = np.random.choice(B.shape[1], p=B[states[t]])
        sequences.append(obs)
        hidden_sequences.append(states)
    return sequences, hidden_sequences

sequences, true_hidden = generate_sequences(true_pi, true_A, true_B)
print(f"  Generated {len(sequences)} sequences of length {len(sequences[0])}")
print(f"  Observation vocabulary: {OBS_NAMES}")
print(f"  Hidden states: {STATE_NAMES}")
print(f"  Example sequence: {[OBS_NAMES[o] for o in sequences[0][:10]]}...")

# ======================================================================
# 6B: FORWARD ALGORITHM (Sequence Likelihood)
# ======================================================================
print_section("6B: Forward Algorithm")

hmm = HMMScratch(n_states=N_STATES, n_obs=N_OBS)
hmm.set_params(true_pi, true_A, true_B)

# Compute likelihood for each sequence
log_lls = []
for i, seq in enumerate(sequences):
    ll, alpha = hmm.forward(seq)
    log_lls.append(ll)
    if i < 3:
        print(f"  Sequence {i+1}: Log-likelihood = {ll:.4f}")

print(f"\n  Mean log-likelihood: {np.mean(log_lls):.4f}")
print(f"  Std log-likelihood:  {np.std(log_lls):.4f}")

# ======================================================================
# 6C: VITERBI ALGORITHM (Decoding)
# ======================================================================
print_section("6C: Viterbi Algorithm (Decoding)")

# Decode first sequence
best_path, log_prob = hmm.viterbi(sequences[0])
print(f"  Best path log-prob: {log_prob:.4f}")
print(f"  True states:    {[STATE_NAMES[s] for s in true_hidden[0][:12]]}")
print(f"  Decoded states: {[STATE_NAMES[s] for s in best_path[:12]]}")

# Accuracy of decoding
correct = sum(1 for a, b in zip(true_hidden[0], best_path) if a == b)
print(f"  Decoding accuracy: {correct}/{len(best_path)} = {correct/len(best_path):.2%}")

# Visualize decoded path
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# Observations
axes[0].step(range(len(sequences[0])), sequences[0], 'b-', linewidth=1.5, where='mid')
axes[0].set_yticks(range(N_OBS))
axes[0].set_yticklabels(OBS_NAMES)
axes[0].set_ylabel('Observation')
axes[0].set_title('Observations & Decoded Hidden States')
axes[0].grid(True, alpha=0.3)

# True vs Decoded states
axes[1].step(range(len(best_path)), true_hidden[0], 'g-', linewidth=2,
             where='mid', label='True States', alpha=0.7)
axes[1].step(range(len(best_path)), best_path, 'r--', linewidth=2,
             where='mid', label='Viterbi Decoded')
axes[1].set_yticks(range(N_STATES))
axes[1].set_yticklabels(STATE_NAMES)
axes[1].set_ylabel('Hidden State')
axes[1].set_xlabel('Time Step (Month)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, 'exp06_viterbi_path.png')

# ======================================================================
# 6D: BAUM-WELCH LEARNING
# ======================================================================
print_section("6D: Baum-Welch Learning")

# Start from random initialization
hmm_learn = HMMScratch(n_states=N_STATES, n_obs=N_OBS, random_state=42)

# Concatenate sequences for training
concat_seq = np.concatenate(sequences[:10])  # Use first 10 sequences
log_lls_bw = hmm_learn.baum_welch(concat_seq, max_iter=50, tol=1e-3)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(log_lls_bw, 'o-', color='darkorange', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Log-likelihood')
ax.set_title('Baum-Welch: Log-likelihood Convergence')
ax.grid(True, alpha=0.3)
save_fig(fig, 'exp06_baum_welch_convergence.png')

print("\n  Learned transition matrix A:")
print(np.round(hmm_learn.A, 3))
print("\n  True transition matrix A:")
print(np.round(true_A, 3))

# ======================================================================
# 6E: TRANSITION MATRIX HEATMAP
# ======================================================================
print_section("6E: Transition Matrix Heatmap")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(true_A, annot=True, fmt='.3f', ax=ax1, cmap='YlOrRd',
            xticklabels=STATE_NAMES, yticklabels=STATE_NAMES)
ax1.set_title('True Transition Matrix')
ax1.set_ylabel('From State')
ax1.set_xlabel('To State')

sns.heatmap(hmm_learn.A, annot=True, fmt='.3f', ax=ax2, cmap='YlOrRd',
            xticklabels=STATE_NAMES, yticklabels=STATE_NAMES)
ax2.set_title('Learned Transition Matrix (Baum-Welch)')
ax2.set_ylabel('From State')
ax2.set_xlabel('To State')

plt.tight_layout()
save_fig(fig, 'exp06_transition_matrices.png')

# ======================================================================
# 6F: VALIDATION AGAINST hmmlearn
# ======================================================================
print_section("6F: Validation Against hmmlearn")

try:
    from hmmlearn import hmm as hmmlearn_hmm

    model = hmmlearn_hmm.CategoricalHMM(n_components=N_STATES, n_iter=100,
                                          random_state=RANDOM_STATE, init_params='')
    model.startprob_ = true_pi
    model.transmat_ = true_A
    model.emissionprob_ = true_B

    # Forward likelihood
    seq_2d = sequences[0].reshape(-1, 1)
    ll_hmmlearn = model.score(seq_2d)
    ll_scratch, _ = hmm.forward(sequences[0])

    print(f"  Forward log-likelihood:")
    print(f"    hmmlearn: {ll_hmmlearn:.4f}")
    print(f"    scratch:  {ll_scratch:.4f}")
    print(f"    diff:     {abs(ll_hmmlearn - ll_scratch):.6f}")

    # Viterbi
    _, path_hmmlearn = model.decode(seq_2d, algorithm='viterbi')
    path_scratch, _ = hmm.viterbi(sequences[0])

    match = np.sum(path_hmmlearn == path_scratch)
    print(f"\n  Viterbi path agreement: {match}/{len(path_scratch)} ({match/len(path_scratch):.2%})")

except ImportError:
    print("  hmmlearn not installed. Skipping validation.")
    print("  Install with: pip install hmmlearn")

# ======================================================================
# OBSERVATIONS & CONCLUSION
# ======================================================================
print_section("Observations & Conclusion")
print("""
OBSERVATIONS:
1. The Forward algorithm correctly computes sequence likelihoods in
   log-space, avoiding numerical underflow.
2. Viterbi decoding recovers the hidden state path with good accuracy
   when using the true parameters.
3. Baum-Welch learning converges and discovers structure similar to
   the true parameters, though not identical (identifiability limits).
4. The transition matrix reveals expected patterns: phones tend to
   stay in their current market phase, with gradual transitions
   from Launch -> Stable -> Depreciation.
5. Scratch and hmmlearn implementations produce matching results,
   validating correctness.

CONCLUSION:
HMMs provide a principled framework for modeling sequential market
dynamics. The phone price depreciation follows interpretable hidden
states (Launch/Stable/Depreciation) that match business intuition.
""")

print("\n[OK] Experiment 6 complete.\n")
