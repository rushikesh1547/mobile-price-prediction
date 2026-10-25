"""
Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models (GMM)
Implemented from scratch with full vectorized operations and closed-form
conditional expectation imputation for missing data.
"""

import numpy as np
from scipy.stats import multivariate_normal


class GMMScratch:
    """
    Gaussian Mixture Model trained via Expectation-Maximization.
    
    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence threshold on log-likelihood improvement.
    reg_covar : float, default=1e-6
        Regularization added to covariance diagonal for numerical stability.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        
        self.weights_ = None       # shape: (K,)
        self.means_ = None         # shape: (K, D)
        self.covariances_ = None   # shape: (K, D, D)
        self.log_likelihood_history_ = []
        self.converged_ = False

    def _initialize_parameters(self, X):
        """Initialize weights uniformly, means via random points or k-means, and covariances."""
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # Uniform mixture weights
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Random sample points as initial means
        random_idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx].copy()
        
        # Initial covariances: global covariance for all components
        base_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array([base_cov.copy() for _ in range(self.n_components)])

    def _multivariate_gaussian_log_pdf(self, X, mean, cov):
        """Vectorized log probability density function for multivariate normal."""
        n_samples, n_features = X.shape
        diff = X - mean
        try:
            # Cholesky decomposition for stability and speed
            L = np.linalg.cholesky(cov)
            # solve L * y = diff.T
            y = np.linalg.solve(L, diff.T)
            maha = np.sum(y ** 2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse / regularized cov
            cov_reg = cov + self.reg_covar * np.eye(n_features)
            inv_cov = np.linalg.pinv(cov_reg)
            maha = np.sum(diff @ inv_cov * diff, axis=1)
            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                log_det = 0.0

        log_2pi = n_features * np.log(2.0 * np.pi)
        log_prob = -0.5 * (log_2pi + log_det + maha)
        return log_prob

    def _e_step(self, X):
        """
        E-Step: Compute responsibilities gamma_{ik} = P(Z_i = k | x_i, theta).
        Returns responsibilities (N, K) and current total log-likelihood.
        """
        n_samples = X.shape[0]
        log_weighted_probs = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_pdf = self._multivariate_gaussian_log_pdf(X, self.means_[k], self.covariances_[k])
            log_weighted_probs[:, k] = np.log(self.weights_[k] + 1e-15) + log_pdf

        # Log-sum-exp trick for numerical stability
        max_log = np.max(log_weighted_probs, axis=1, keepdims=True)
        sum_exp = np.sum(np.exp(log_weighted_probs - max_log), axis=1, keepdims=True)
        log_likelihood_per_sample = max_log + np.log(sum_exp + 1e-15)
        log_likelihood = np.sum(log_likelihood_per_sample)

        responsibilities = np.exp(log_weighted_probs - log_likelihood_per_sample)
        # Normalize responsibilities across components
        responsibilities = responsibilities / (np.sum(responsibilities, axis=1, keepdims=True) + 1e-15)
        
        return responsibilities, log_likelihood

    def _m_step(self, X, responsibilities):
        """
        M-Step: Update mixture weights, means, and covariance matrices.
        """
        n_samples, n_features = X.shape
        Nk = np.sum(responsibilities, axis=0) + 1e-10  # effective cluster counts
        
        # Update weights
        self.weights_ = Nk / n_samples
        
        # Update means
        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]  # (N, D)
            weighted_diff = responsibilities[:, k:k+1] * diff  # (N, D)
            cov_k = (weighted_diff.T @ diff) / Nk[k]
            # Regularize
            self.covariances_[k] = cov_k + self.reg_covar * np.eye(n_features)

    def fit(self, X):
        """Fit the GMM on data X."""
        X = np.asarray(X, dtype=float)
        self._initialize_parameters(X)
        self.log_likelihood_history_ = []

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            responsibilities, log_ll = self._e_step(X)
            self.log_likelihood_history_.append(log_ll)
            
            # Check convergence
            if iteration > 0 and abs(log_ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = log_ll
            
            self._m_step(X, responsibilities)
            
        return self

    def predict_proba(self, X):
        """Predict posterior probability distribution over components."""
        X = np.asarray(X, dtype=float)
        responsibilities, _ = self._e_step(X)
        return responsibilities

    def predict(self, X):
        """Predict cluster index for each sample."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X):
        """Compute per-sample average log-likelihood."""
        X = np.asarray(X, dtype=float)
        _, log_ll = self._e_step(X)
        return log_ll / len(X)

    def impute_missing(self, X_with_nans):
        """
        Perform EM conditional expectation imputation for missing entries.
        
        For each sample x_i with missing indices M and observed indices O:
        E[x_M | x_O] = sum_k gamma_k(x_O) * [ mu_M^k + Sigma_{MO}^k (Sigma_{OO}^k)^{-1} (x_O - mu_O^k) ]
        """
        X_imputed = np.asarray(X_with_nans, dtype=float).copy()
        n_samples, n_features = X_imputed.shape
        
        for i in range(n_samples):
            row = X_imputed[i]
            missing_mask = np.isnan(row)
            if not np.any(missing_mask):
                continue
                
            obs_mask = ~missing_mask
            if not np.any(obs_mask):
                # If everything is missing, impute with global means weighted by component priors
                X_imputed[i] = np.sum(self.weights_[:, np.newaxis] * self.means_, axis=0)
                continue
                
            x_obs = row[obs_mask]
            d_obs = np.sum(obs_mask)
            
            # Calculate component responsibilities given observed features
            log_resp = np.zeros(self.n_components)
            mu_cond_list = []
            
            for k in range(self.n_components):
                mu_k_obs = self.means_[k][obs_mask]
                mu_k_mis = self.means_[k][missing_mask]
                
                cov_k_oo = self.covariances_[k][np.ix_(obs_mask, obs_mask)]
                cov_k_mo = self.covariances_[k][np.ix_(missing_mask, obs_mask)]
                
                # Marginal likelihood of x_obs under component k
                diff_obs = x_obs - mu_k_obs
                try:
                    inv_cov_oo = np.linalg.pinv(cov_k_oo + self.reg_covar * np.eye(d_obs))
                    sign, logdet = np.linalg.slogdet(cov_k_oo + self.reg_covar * np.eye(d_obs))
                    maha = diff_obs @ inv_cov_oo @ diff_obs
                    log_pdf_obs = -0.5 * (d_obs * np.log(2 * np.pi) + logdet + maha)
                except Exception:
                    inv_cov_oo = np.eye(d_obs)
                    log_pdf_obs = -0.5 * np.sum(diff_obs ** 2)

                log_resp[k] = np.log(self.weights_[k] + 1e-15) + log_pdf_obs
                
                # Conditional mean: mu_M + Sigma_MO * Sigma_OO^-1 * (x_O - mu_O)
                cond_mean = mu_k_mis + cov_k_mo @ inv_cov_oo @ diff_obs
                mu_cond_list.append(cond_mean)
                
            # Normalize responsibilities
            max_log = np.max(log_resp)
            resp = np.exp(log_resp - max_log)
            resp /= np.sum(resp) + 1e-15
            
            # Weighted expected value
            imputed_mis = np.zeros(np.sum(missing_mask))
            for k in range(self.n_components):
                imputed_mis += resp[k] * mu_cond_list[k]
                
            X_imputed[i, missing_mask] = imputed_mis
            
        return X_imputed
