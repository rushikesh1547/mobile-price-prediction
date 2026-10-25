"""
hmm.py - Hidden Markov Model from scratch (NumPy, log-space).
Implements Forward, Viterbi, and Baum-Welch algorithms.
"""

import numpy as np


class HMMScratch:
    """Hidden Markov Model with Forward, Viterbi, and Baum-Welch.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_obs : int
        Number of possible observation symbols.
    """

    def __init__(self, n_states, n_obs, random_state=42):
        np.random.seed(random_state)
        self.n_states = n_states
        self.n_obs = n_obs

        # Initialize parameters randomly
        self.pi = np.random.dirichlet(np.ones(n_states))          # initial state probs
        self.A = np.random.dirichlet(np.ones(n_states), n_states) # transition matrix
        self.B = np.random.dirichlet(np.ones(n_obs), n_states)    # emission matrix

    def set_params(self, pi, A, B):
        """Set HMM parameters directly."""
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)

    def _log_safe(self, x):
        """Log that handles zeros."""
        return np.log(np.clip(x, 1e-300, None))

    # ── Forward Algorithm (log-space) ────────────────────────────────────
    def forward(self, obs):
        """Compute log-likelihood of observation sequence using Forward algorithm.

        Parameters
        ----------
        obs : array-like of int, shape (T,)
            Observation sequence (indices 0..n_obs-1).

        Returns
        -------
        log_likelihood : float
        alpha : ndarray, shape (T, n_states) — scaled forward variables
        """
        T = len(obs)
        N = self.n_states
        log_alpha = np.full((T, N), -np.inf)

        # Initialization
        log_alpha[0] = self._log_safe(self.pi) + self._log_safe(self.B[:, obs[0]])

        # Induction
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = (
                    np.logaddexp.reduce(log_alpha[t-1] + self._log_safe(self.A[:, j]))
                    + self._log_safe(self.B[j, obs[t]])
                )

        log_likelihood = np.logaddexp.reduce(log_alpha[-1])
        return log_likelihood, log_alpha

    # ── Backward Algorithm (log-space) ───────────────────────────────────
    def backward(self, obs):
        """Compute backward variables in log-space."""
        T = len(obs)
        N = self.n_states
        log_beta = np.full((T, N), -np.inf)
        log_beta[-1] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = np.logaddexp.reduce(
                    self._log_safe(self.A[i]) + self._log_safe(self.B[:, obs[t+1]]) + log_beta[t+1]
                )
        return log_beta

    # ── Viterbi Algorithm ────────────────────────────────────────────────
    def viterbi(self, obs):
        """Find most probable hidden state sequence using Viterbi.

        Returns
        -------
        best_path : ndarray of int, shape (T,)
        log_prob : float — log probability of the best path
        """
        T = len(obs)
        N = self.n_states

        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        log_delta[0] = self._log_safe(self.pi) + self._log_safe(self.B[:, obs[0]])

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = log_delta[t-1] + self._log_safe(self.A[:, j])
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = candidates[psi[t, j]] + self._log_safe(self.B[j, obs[t]])

        # Termination
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(log_delta[-1])
        log_prob = log_delta[-1, best_path[-1]]

        # Backtracking
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path, log_prob

    # ── Baum-Welch (EM) ─────────────────────────────────────────────────
    def baum_welch(self, obs, max_iter=100, tol=1e-4, verbose=True):
        """Learn HMM parameters from observation sequence using Baum-Welch.

        Returns
        -------
        log_likelihoods : list of floats
        """
        T = len(obs)
        N = self.n_states
        log_likelihoods = []

        for iteration in range(max_iter):
            # E-step
            ll, log_alpha = self.forward(obs)
            log_beta = self.backward(obs)
            log_likelihoods.append(ll)

            # Gamma: P(state=i at time t | observations)
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Xi: P(state=i at t, state=j at t+1 | observations)
            xi = np.zeros((T - 1, N, N))
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = np.exp(
                            log_alpha[t, i] + self._log_safe(self.A[i, j])
                            + self._log_safe(self.B[j, obs[t+1]]) + log_beta[t+1, j] - ll
                        )

            # M-step
            self.pi = gamma[0] / gamma[0].sum()

            for i in range(N):
                denom = gamma[:-1, i].sum()
                if denom > 0:
                    self.A[i] = xi[:, i, :].sum(axis=0) / denom
                for k in range(self.n_obs):
                    mask = (np.array(obs) == k)
                    numer = gamma[mask, i].sum()
                    full_denom = gamma[:, i].sum()
                    self.B[i, k] = numer / full_denom if full_denom > 0 else 1.0 / self.n_obs

            if verbose and (iteration + 1) % 10 == 0:
                print(f"    BW Iter {iteration+1}: Log-likelihood = {ll:.4f}")

            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration+1}")
                break

        return log_likelihoods
