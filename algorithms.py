"""
Bandit algorithms for learning to target with network interference.

This module implements the four algorithms described in the paper:
  - BaselineBandit    : Network-agnostic linear bandit (Algorithm 3)
  - NSEBandit         : Network Successive Elimination with partial support knowledge (Algorithm 1)
  - NSEFSBandit       : NSE with Full Support knowledge (Algorithm 4)
  - NETCBandit        : Network Explore Then Commit, no support information (Algorithm 2)

It also provides data-generation utilities for simulated and semi-synthetic
(Indian village) network experiments.

"""

import math
import warnings
from statistics import NormalDist
from typing import Sequence

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from numpy.linalg import norm
from sklearn.linear_model import Lasso, MultiTaskLassoCV, Ridge

# ---------------------------------------------------------------------------
# Gurobi global settings
# ---------------------------------------------------------------------------
gp.setParam("OutputFlag", 0)
MAX_THREADS = 1
MAX_TIME = 10
MAX_TIME_BASELINE = 10

# Minimum batch size before elimination is attempted in batched algorithms.
MIN_ELIM_BATCH = 64

# Estimation method used in batched algorithms ("ridge" or "lasso").
ESTIMATION_METHOD = "ridge"

# Valid village numbers from the Indian village dataset (Banerjee et al., 2013).
VILLAGE_NUMBERS = np.array(
    [num for num in np.arange(1, 77) if num not in [13, 22]]
)


# ============================================================================
# Data generation
# ============================================================================

def generate_network(seed, d, s, signal_strength):
    """Generate a random sparse network treatment effect matrix X*.

    The matrix X is d x d. Each diagonal entry X[i,i] represents the direct
    treatment effect on individual i. Off-diagonal entry X[i,j] represents
    the spillover effect of treating individual j on individual i.

    Sparsity is controlled by s: each row has at most s non-zero off-diagonal
    entries in expectation (sparsity_level = s / d). Non-zero entries are
    drawn from a mixed-signal model where half the effects are strong
    (multiplied by 100) and half are weak (multiplied by 0.01), scaled by
    signal_strength.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    d : int
        Population size (number of individuals).
    s : int
        Row sparsity level.
    signal_strength : float
        Scaling factor for treatment effect magnitudes.

    Returns
    -------
    X : np.ndarray of shape (d, d)
        The network treatment effect matrix.
    """
    sparsity_level = s / d
    np.random.seed(seed)
    X = np.random.uniform(-0.01, 0.01, (d, d)) * signal_strength

    for i in range(d):
        for j in range(d):
            if np.random.uniform() > 0.5:
                X[i, j] *= 100
            if np.random.uniform() > sparsity_level and i != j:
                X[i, j] = 0

    return X


def generate_village_network(seed, village_number, signal_strength):
    """Generate a treatment effect matrix from Indian village adjacency data.

    Uses household-level adjacency matrices from the dataset published in
    Banerjee et al. (2013). The adjacency structure is preserved, and
    heterogeneous treatment effects are generated for each edge using a
    mixed-signal model.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    village_number : int
        Village identifier (must be in VILLAGE_NUMBERS).
    signal_strength : float
        Scaling factor for treatment effect magnitudes.

    Returns
    -------
    X : np.ndarray of shape (d, d)
        The network treatment effect matrix, where d is the number of
        households in the village.
    """
    np.random.seed(seed)
    if village_number not in VILLAGE_NUMBERS:
        raise ValueError(
            f"Village number {village_number} is not valid. "
            f"Must be one of {VILLAGE_NUMBERS.tolist()}."
        )
    # Village adjacency data lives at <repo_root>/datav4.0/... (one level
    # above this algorithms.py file, which is at <repo_root>/Final_code/).
    # Anchor the path to the module file so the loader works regardless of
    # where the script is run from.
    import os
    module_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.normpath(os.path.join(module_dir, os.pardir))
    folder_path = os.path.join(
        data_root, "datav4.0", "Data", "1. Network Data", "Adjacency Matrices"
    )
    file_name = f"adj_allVillageRelationships_HH_vilno_{village_number}.csv"
    adj_matrix = pd.read_csv(os.path.join(folder_path, file_name), header=None)
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix for village {village_number} is not square.")

    d = adj_matrix.shape[0]
    X = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            if i != j and adj_matrix.iloc[i, j] == 0:
                X[i, j] = 0
            else:
                X[i, j] = np.random.uniform(-0.01, 0.01) * signal_strength
                if np.random.uniform() > 0.5:
                    X[i, j] *= 100
    return X


# ============================================================================
# Estimation helpers
# ============================================================================

def _fit_ridge_matrix(actions, rewards, alpha):
    """Fit a multi-output ridge regression and return coefficient / variance info.

    Parameters
    ----------
    actions : np.ndarray of shape (n, d)
    rewards : np.ndarray of shape (n, d)
    alpha : float
        Ridge regularization parameter (must be positive).

    Returns
    -------
    dict with keys:
        coef       : np.ndarray of shape (d, d) -- coefficient matrix
        common_diag: np.ndarray of shape (d,)    -- diagonal of the sandwich matrix
        sigma2     : np.ndarray of shape (d,)    -- per-target residual variance
    """
    actions = np.asarray(actions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(actions, rewards)
    coef = ridge.coef_
    residuals = rewards - ridge.predict(actions)
    XtX = actions.T @ actions
    identity = np.identity(actions.shape[1])
    ridge_matrix = XtX + alpha * identity
    # Use np.linalg.solve (LU decomposition) instead of np.linalg.pinv (SVD)
    # for robustness: ridge_matrix is positive definite (alpha > 0), so LU
    # is appropriate and avoids SVD convergence failures that can occur on
    # certain near-singular patterns (observed on large village networks).
    try:
        ridge_inv = np.linalg.solve(ridge_matrix, identity)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse with slight extra regularization.
        ridge_inv = np.linalg.pinv(ridge_matrix + 1e-8 * identity)
    gram = ridge_inv @ XtX @ ridge_inv
    common_diag = np.diag(gram)
    dof = max(actions.shape[0] - actions.shape[1], 1)
    sigma2 = np.maximum(np.sum(residuals ** 2, axis=0) / dof, 1e-12)
    return {
        "coef": coef,
        "common_diag": np.nan_to_num(common_diag, nan=0.0, posinf=0.0, neginf=0.0),
        "sigma2": sigma2,
    }


def _fit_lasso_matrix(actions, rewards, alpha):
    """Fit per-outcome-row Lasso with a fixed alpha and return coef / variance info.

    Runs d independent Lasso regressions (one per outcome coordinate), giving
    each outcome row its own sparsity pattern. Uses fit_intercept=True so both
    actions and rewards are centered before the L1 fit.
    """
    actions = np.asarray(actions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    n, d = actions.shape
    if n < 2:
        coef = (np.linalg.pinv(actions) @ rewards).T
        residuals = rewards - actions @ coef.T
    else:
        coef = np.zeros((rewards.shape[1], d))
        residuals = np.zeros_like(rewards)
        for i in range(rewards.shape[1]):
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
            model.fit(actions, rewards[:, i])
            coef[i, :] = model.coef_
            residuals[:, i] = rewards[:, i] - model.predict(actions)
    XtX = actions.T @ actions
    identity = np.identity(d)
    reg_matrix = XtX + alpha * identity
    reg_inv = np.linalg.pinv(reg_matrix)
    gram = reg_inv @ XtX @ reg_inv
    common_diag = np.diag(gram)
    dof = max(n - d, 1)
    sigma2 = np.maximum(np.sum(residuals ** 2, axis=0) / dof, 1e-12)
    return {
        "coef": coef,
        "common_diag": np.nan_to_num(common_diag, nan=0.0, posinf=0.0, neginf=0.0),
        "sigma2": sigma2,
    }


def _fit_estimator_matrix(actions, rewards, alpha, method=None):
    """Dispatch to ridge or lasso. `method` overrides the global ESTIMATION_METHOD flag."""
    chosen = method if method is not None else ESTIMATION_METHOD
    if chosen == "ridge":
        return _fit_ridge_matrix(actions, rewards, alpha)
    return _fit_lasso_matrix(actions, rewards, alpha)


# ============================================================================
# Base bandit class
# ============================================================================

class BaseBandit:
    """Common utilities shared by all bandit algorithms.

    Attributes
    ----------
    X : np.ndarray
        The true d x d network treatment effect matrix.
    d : int
        Population size.
    tau : int
        Time horizon.
    b : int
        Budget constraint (sum of actions <= b).
    """

    def __init__(self, X, tau, b, noise_std=1.0, random_state=None):
        X = np.asarray(X, dtype=float)
        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square matrix.")
        self.X = X
        self.d = X.shape[0]
        self.tau = tau
        self.b = b
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_state)
        self._true_action_cache = None

    def sample_random_action(self):
        """Draw a uniformly random action in [-1, 1]^d."""
        return self.rng.uniform(-1, 1, self.d)

    def observe(self, action, noise_scale=None):
        """Observe reward vector Y_t = X * a_t + epsilon_t."""
        noise = self.rng.normal(0, noise_scale or self.noise_std, self.d)
        return self.X @ action + noise

    def _solve_true_action(self):
        """Compute the optimal action a* = argmax 1^T X a subject to constraints."""
        if self._true_action_cache is not None:
            return self._true_action_cache
        model = gp.Model("true_opt")
        model.setParam("Threads", MAX_THREADS)
        a_vars = model.addVars(self.d, lb=-1, ub=1, name="a")
        model.addConstr(gp.quicksum(a_vars[i] for i in range(self.d)) <= self.b, "budget")
        objective = gp.LinExpr()
        for row in range(self.d):
            objective += gp.quicksum(self.X[row, col] * a_vars[col] for col in range(self.d))
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        solution = np.array([a_vars[i].X for i in range(self.d)])
        self._true_action_cache = solution
        return solution

    def _regret_from_actions(self, actions):
        """Compute per-round regret for a sequence of actions."""
        optimal = self._solve_true_action()
        regrets = [float(np.sum(self.X @ (optimal - act))) for act in actions]
        return regrets, optimal

    def _instant_regret(self, action):
        """Compute single-round regret for a given action."""
        optimal = self._solve_true_action()
        return float(np.sum(self.X @ (optimal - action)))

    def run(self):
        raise NotImplementedError


# ============================================================================
# Lasso exploration mixin (used by NETCBandit)
# ============================================================================

class _LassoExplorationMixin:
    """Provides the Lasso-based exploration phase for NETC (Algorithm 2).

    During exploration, actions are sampled uniformly at random and
    row-wise Lasso regression is used to estimate each row of X*.
    """

    def _run_lasso_exploration(self, rounds, lasso_alpha, noise_scale=0.1):
        """Run the exploration phase and return estimated X, actions, rewards.

        Parameters
        ----------
        rounds : int
            Number of exploration rounds (T_1 in the paper).
        lasso_alpha : float
            L1 regularization parameter for Lasso.
        noise_scale : float
            Standard deviation of observation noise during exploration.

        Returns
        -------
        X_hat : np.ndarray of shape (d, d)
            Lasso estimate of X*.
        A : np.ndarray of shape (rounds, d)
            Exploration actions.
        Y : np.ndarray of shape (rounds, d)
            Observed rewards.
        """
        print(f"[Exploration] Starting {rounds} exploratory rounds (lambda={lasso_alpha}).")
        A = np.empty((rounds, self.d))
        Y = np.empty((rounds, self.d))
        for t in range(rounds):
            action = self.sample_random_action()
            reward = self.observe(action, noise_scale=noise_scale)
            A[t] = action
            Y[t] = reward

        X_hat = np.empty((self.d, self.d))
        warnings.filterwarnings("ignore")
        for i in range(self.d):
            model = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=10000)
            model.fit(A, Y[:, i])
            X_hat[i] = model.coef_
        print("[Exploration] Exploration phase complete.")
        return X_hat, A, Y


# ============================================================================
# Algorithm 2: NETC -- Network Explore Then Commit (no support information)
# ============================================================================

class NETCBandit(_LassoExplorationMixin, BaseBandit):
    """Network Explore Then Commit (NETC) -- Algorithm 2 in the paper.

    This algorithm is designed for the setting where no structural
    information about the network is available. It uses an explore-then-commit
    strategy:
      1. Exploration: sample random actions for T_1 rounds and estimate X*
         via row-wise Lasso regression (Eq. 5 in the paper).
      2. Exploitation: solve for the best action using the estimated X_hat
         and commit to it for the remaining rounds (Eq. 6).

    Under Assumptions 1 and 2, NETC achieves expected regret
    O~(d (s * tau)^{2/3}).

    Parameters
    ----------
    X : np.ndarray
        True treatment effect matrix (for simulation/regret computation).
    tau : int
        Time horizon T.
    b : int
        Budget constraint.
    exploration_rounds : int
        Number of exploration rounds T_1.
    lambda_explore : float
        Lasso regularization parameter.
    """

    def __init__(self, X, tau, b, exploration_rounds, lambda_explore,
                 noise_std=1.0, random_state=None):
        super().__init__(X, tau, b, noise_std=noise_std, random_state=random_state)
        self.t1 = min(exploration_rounds, tau)
        self.lambda_explore = lambda_explore

    def _solve_commit_action(self, X_hat):
        """Solve for the best action given estimated X_hat (Eq. 6)."""
        model = gp.Model("netc_commit")
        model.setParam("Threads", MAX_THREADS)
        a_vars = model.addVars(self.d, lb=-1, ub=1, name="a")
        model.addConstr(gp.quicksum(a_vars[i] for i in range(self.d)) <= self.b, "budget")
        objective = gp.LinExpr()
        for row in range(self.d):
            for col in range(self.d):
                objective += X_hat[row, col] * a_vars[col]
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        return np.array([a_vars[i].X for i in range(self.d)])

    def run(self):
        """Execute the NETC algorithm and return per-round regrets."""
        print("[NETC] Running exploration phase.")
        X_hat, A_explore, _ = self._run_lasso_exploration(self.t1, self.lambda_explore)

        a_commit = self._solve_commit_action(X_hat)
        exploitation_rounds = self.tau - self.t1
        print(f"[NETC] Committing to estimated optimal action for {exploitation_rounds} rounds.")

        if exploitation_rounds > 0:
            actions_commit = np.repeat(a_commit[np.newaxis, :], exploitation_rounds, axis=0)
            actions = np.vstack((A_explore, actions_commit))
        else:
            actions = A_explore

        regrets, _ = self._regret_from_actions(actions)
        print(f"[NETC] Completed -- total_regret={sum(regrets):.2f}.")
        return {"regret": regrets}


# ============================================================================
# Algorithm 1: NSE -- Network Successive Elimination (partial support knowledge)
# ============================================================================

class NSEBandit(BaseBandit):
    """Network Successive Elimination (NSE) -- Algorithm 1 in the paper.

    This algorithm is designed for the setting where the column support
    sizes {rho_j} are known. It operates in logarithmically many batches
    of geometrically increasing size. In each batch:
      1. For uncertain coordinates j in U_m, sample a_{t,j} uniformly from
         {-1, +1}; for resolved coordinates, commit to sign(theta_hat_j).
      2. At the end of each batch, update the estimator X_hat and theta_hat.
      3. Eliminate coordinate j from the uncertainty set if
         |theta_hat_j| > 2 * rho_j * tau_m.

    Under Assumptions 1 and 2 with known column support sizes, NSE achieves
    regret O~(sqrt(T) * sum_j rho_j), which is at most O~(d * s * sqrt(T)).

    Parameters
    ----------
    X : np.ndarray
        True treatment effect matrix.
    tau : int
        Time horizon T.
    tau_constant : float or list of float
        Scaling constant(s) for the elimination threshold tau_m.
    estimation_alpha : float
        Ridge regularization parameter for estimation.
    """

    def __init__(self, X, tau, noise_std=1.0, random_state=None,
                 tau_constant=0.1, estimation_alpha=1.0, estimation_method=None):
        super().__init__(X, tau, b=100, noise_std=noise_std, random_state=random_state)
        self.tau_constant = tau_constant
        self.estimation_alpha = float(estimation_alpha)
        self.estimation_method = estimation_method

    def _rho_vector(self):
        """Compute column support sizes rho_j = |{i : X*_{ij} != 0}|."""
        return np.sum(self.X != 0, axis=0).astype(float)

    def _batch_targets(self):
        """Compute batch boundary indices: T_1, T_2, ..., T_M = T."""
        if self.tau <= 0:
            return [0]
        if self.tau == 1:
            return [1]
        M = max(1, math.ceil(math.log2(self.tau / 2 + 1)))
        targets = []
        for i in range(1, M):
            target = min(self.tau, max(1, 2 * (2 ** i - 1)))
            if targets:
                target = max(target, targets[-1] + 1)
            targets.append(target)
        targets.append(self.tau)
        return targets

    def run(self):
        """Execute the NSE algorithm and return per-round regrets."""
        batch_targets = self._batch_targets()
        rho_vec = self._rho_vector()

        # theta_j = sum_i X*_{ij} is the column sum we want to estimate.
        theta_hat = np.zeros(self.d)
        uncertain_mask = np.ones(self.d, dtype=bool)
        actions_history = []
        rewards_history = []
        samples_collected = 0
        X_hat_estimates = np.zeros((self.d, self.d))

        for batch_idx, target in enumerate(batch_targets):
            uncertain_indices = np.where(uncertain_mask)[0]

            # If all coordinates are resolved, commit for remaining rounds.
            if uncertain_indices.size == 0:
                remaining = self.tau - samples_collected
                if remaining <= 0:
                    break
                resolved_actions = np.where(theta_hat >= 0, 1, -1)
                for _ in range(remaining):
                    action = resolved_actions.copy()
                    reward = self.observe(action)
                    actions_history.append(action)
                    rewards_history.append(reward)
                samples_collected += remaining
                break

            batch_size = target - samples_collected
            if batch_size <= 0:
                continue

            # Within each batch: explore uncertain, exploit resolved.
            resolved_actions = np.where(theta_hat >= 0, 1, -1)
            for t in range(batch_size):
                random_draws = self.rng.choice([-1, 1], size=self.d)
                action = np.where(uncertain_mask, random_draws, resolved_actions)
                reward = self.observe(action)
                actions_history.append(action)
                rewards_history.append(reward)
                global_round = samples_collected + t + 1
                if global_round % 500 == 0 or global_round == self.tau:
                    print(
                        f"[NSE] Round {global_round}/{self.tau}; "
                        f"uncertain={np.sum(uncertain_mask)}; batch={batch_idx + 1}."
                    )

            # End-of-batch estimation and elimination.
            total_samples = samples_collected + batch_size
            tau_m = self.tau_constant * np.sqrt(
                2 * np.log(2 * max(self.tau, 1)) / total_samples
            )

            if self.estimation_method == "onehot":
                # Paper Algorithm 2, line 16: one-hot estimator using only the
                # current batch D_m (not accumulated history):
                #   X_hat_ij = (1/|D_m|) sum_{t in D_m} Y_{t,i} * a_{t,j}
                batch_start = total_samples - batch_size
                A_m = np.asarray(
                    actions_history[batch_start:total_samples], dtype=float
                )  # shape (|D_m|, d)
                Y_m = np.asarray(
                    rewards_history[batch_start:total_samples], dtype=float
                )  # shape (|D_m|, d)
                Dm = A_m.shape[0]
                if uncertain_indices.size and Dm > 0:
                    # X_hat[i, j] = (1/|D_m|) sum_t Y_m[t, i] * A_m[t, j]
                    coef_uncertain = (
                        Y_m.T @ A_m[:, uncertain_indices]
                    ) / Dm  # shape (d, len(uncertain_indices))
                    X_hat_estimates[:, uncertain_indices] = coef_uncertain
            else:
                actions_array = np.asarray(actions_history, dtype=float)
                rewards_array = np.asarray(rewards_history, dtype=float)
                estimation_info = _fit_estimator_matrix(
                    actions_array, rewards_array, self.estimation_alpha,
                    method=self.estimation_method,
                )
                coef_matrix = estimation_info["coef"]
                if uncertain_indices.size:
                    X_hat_estimates[:, uncertain_indices] = coef_matrix[:, uncertain_indices]
            # Hard-thresholded column sum:
            # theta_hat_j = sum_i X_hat_ij * 1{|X_hat_ij| >= tau_m}.
            # Entries below the noise floor tau_m are zeroed out so only
            # confidently non-zero estimates contribute.
            truncated = np.where(np.abs(X_hat_estimates) >= tau_m, X_hat_estimates, 0.0)
            theta_hat[:] = truncated.sum(axis=0)

            # Elimination step (Eq. following Algorithm 1, line 20).
            if batch_size >= MIN_ELIM_BATCH:
                uncertain_mask = uncertain_mask & (
                    np.abs(theta_hat) <= rho_vec * tau_m
                )
            samples_collected = target
            print(
                f"[NSE] Batch {batch_idx + 1}/{len(batch_targets)} done; "
                f"uncertain={np.sum(uncertain_mask)}; tau_m={tau_m:.4f}."
            )

        actions = np.array(actions_history) if actions_history else np.empty((0, self.d))
        regrets, _ = self._regret_from_actions(actions)
        print(f"[NSE] Completed -- total_regret={sum(regrets):.2f}.")
        return {"regret": regrets}


# ============================================================================
# Algorithm 4: NSE-FS -- NSE with Full Support knowledge
# ============================================================================

class NSEFSBandit(BaseBandit):
    """NSE with Full Support knowledge (NSE-FS) -- Algorithm 4 in the paper.

    This algorithm is designed for the setting where the full support S of
    X* is known (but effect magnitudes are unknown). It uses the same
    successive elimination framework as NSE, but leverages the known support
    to construct tighter confidence intervals for each theta_j.

    Under Assumptions 1 and 2 with known support, NSE-FS achieves regret
    O~(sqrt(T) * sum_j rho_j), matching the lower bound up to log factors.

    Parameters
    ----------
    X : np.ndarray
        True treatment effect matrix.
    tau : int
        Time horizon T.
    alpha : float
        Significance level for confidence intervals (e.g. 0.05).
    estimation_alpha : float
        Ridge regularization parameter for estimation.
    """

    def __init__(self, X, tau, noise_std=1.0, random_state=None,
                 alpha=0.05, estimation_alpha=1.0, estimation_method=None):
        super().__init__(X, tau, b=100, noise_std=noise_std, random_state=random_state)
        self.alpha = float(alpha)
        self.z_alpha = NormalDist().inv_cdf(1 - self.alpha / 2)
        self.support_sets = [np.flatnonzero(self.X[:, j]) for j in range(self.d)]
        self.support_cardinality = np.array([len(s) for s in self.support_sets], dtype=int)
        self.estimation_alpha = float(estimation_alpha)
        self.estimation_method = estimation_method

    def _batch_targets(self):
        """Compute batch boundary indices."""
        if self.tau <= 0:
            return [0]
        if self.tau == 1:
            return [1]
        M = max(1, math.ceil(math.log2(self.tau / 2 + 1)))
        targets = []
        for i in range(1, M):
            target = min(self.tau, max(1, 2 * (2 ** i - 1)))
            if targets:
                target = max(target, targets[-1] + 1)
            targets.append(target)
        targets.append(self.tau)
        return targets

    def run(self):
        """Execute the NSE-FS algorithm and return per-round regrets."""
        batch_targets = self._batch_targets()

        # Initialize tracking.
        self.theta_estimates = np.zeros(self.d)
        self.ci_halfwidth = np.full(self.d, np.inf)
        uncertain_mask = np.ones(self.d, dtype=bool)

        # Coordinates with empty support are immediately resolved.
        zero_support = np.where(self.support_cardinality == 0)[0]
        if zero_support.size:
            uncertain_mask[zero_support] = False
            self.ci_halfwidth[zero_support] = 0.0

        actions_history = []
        rewards_history = []
        samples_collected = 0
        X_hat_estimates = np.zeros((self.d, self.d))

        for batch_idx, target in enumerate(batch_targets):
            uncertain_indices = np.where(uncertain_mask)[0]

            # If all coordinates are resolved, commit for remaining rounds.
            if uncertain_indices.size == 0:
                remaining = self.tau - samples_collected
                if remaining <= 0:
                    break
                resolved_actions = np.where(self.theta_estimates >= 0, 1.0, -1.0)
                for _ in range(remaining):
                    action = resolved_actions.copy()
                    reward = self.observe(action)
                    actions_history.append(action)
                    rewards_history.append(reward)
                samples_collected += remaining
                break

            batch_size = target - samples_collected
            if batch_size <= 0:
                continue

            # Within each batch: explore uncertain, exploit resolved.
            resolved_actions = np.where(self.theta_estimates >= 0, 1.0, -1.0)
            for t in range(batch_size):
                random_draws = self.rng.choice([-1.0, 1.0], size=self.d)
                action = np.where(uncertain_mask, random_draws, resolved_actions)
                reward = self.observe(action)
                actions_history.append(action)
                rewards_history.append(reward)
                global_round = samples_collected + t + 1
                if global_round % 500 == 0 or global_round == self.tau:
                    print(
                        f"[NSE-FS] Round {global_round}/{self.tau}; "
                        f"uncertain={np.sum(uncertain_mask)}; batch={batch_idx + 1}."
                    )

            # End-of-batch estimation.
            actions_array = np.asarray(actions_history, dtype=float)
            rewards_array = np.asarray(rewards_history, dtype=float)
            estimation_info = _fit_estimator_matrix(
                actions_array, rewards_array, self.estimation_alpha,
                method=self.estimation_method,
            )
            coef_matrix = estimation_info["coef"]
            if uncertain_indices.size:
                X_hat_estimates[:, uncertain_indices] = coef_matrix[:, uncertain_indices]
            theta_vector = X_hat_estimates.sum(axis=0)
            if uncertain_indices.size:
                self.theta_estimates[uncertain_indices] = theta_vector[uncertain_indices]

            # Confidence interval construction using known support.
            sigma2_sum = float(np.sum(estimation_info["sigma2"]))
            var_theta_full = estimation_info["common_diag"] * max(sigma2_sum, 1e-12)
            if uncertain_indices.size:
                self.ci_halfwidth[uncertain_indices] = self.z_alpha * np.sqrt(
                    np.maximum(var_theta_full[uncertain_indices], 1e-12)
                )
            if zero_support.size:
                self.theta_estimates[zero_support] = 0.0
                self.ci_halfwidth[zero_support] = 0.0

            # Elimination: remove j if CI for theta_j excludes zero.
            if batch_size >= MIN_ELIM_BATCH:
                for j in np.where(uncertain_mask)[0]:
                    lower = self.theta_estimates[j] - self.ci_halfwidth[j]
                    upper = self.theta_estimates[j] + self.ci_halfwidth[j]
                    if not (lower <= 0.0 <= upper):
                        uncertain_mask[j] = False

            samples_collected = target
            print(
                f"[NSE-FS] Batch {batch_idx + 1}/{len(batch_targets)} done; "
                f"uncertain={np.sum(uncertain_mask)}."
            )

        actions = np.array(actions_history) if actions_history else np.empty((0, self.d))
        regrets, _ = self._regret_from_actions(actions)
        print(f"[NSE-FS] Completed -- total_regret={sum(regrets):.2f}.")
        return {"regret": regrets}


# ============================================================================
# Algorithm 3: Baseline -- Network-agnostic linear bandit (LinUCB on Z_t)
# ============================================================================

class BaselineBandit(BaseBandit):
    """Baseline network-agnostic algorithm -- Algorithm 3 in the paper.

    This algorithm ignores individual reward observations and instead works
    with the aggregated scalar reward Z_t = 1^T Y_t = theta^T a_t + eta_t
    where theta = X*^T 1. It applies a standard LinUCB strategy on this
    reduced d-dimensional linear bandit.

    This baseline achieves regret O~(d * sqrt(d * T)), which is
    fundamentally inefficient compared to algorithms that exploit
    network structure.

    Parameters
    ----------
    X : np.ndarray
        True treatment effect matrix.
    tau : int
        Time horizon T.
    b : int
        Budget constraint.
    lambda_confidence : float
        Regularization parameter for the confidence ellipsoid.
    reg_param : float
        Additional regularization scaling per round.
    sparsity : int
        Sparsity level s (used for R_max scaling).
    R_max : float
        Upper bound on ||theta||.
    """

    def __init__(self, X, tau, b, lambda_confidence, reg_param, sparsity,
                 R_max, noise_std=None, random_state=None):
        super().__init__(X, tau, b, noise_std=noise_std or X.shape[0], random_state=random_state)
        self.lambda_confidence = lambda_confidence
        self.reg_param = reg_param
        self.sparsity = sparsity
        self.R_max = R_max
        self.X_row_sum = np.sum(self.X, axis=0)  # theta = X*^T 1

    def _baseline_beta(self, t, m2):
        """Confidence radius for the baseline LinUCB ellipsoid."""
        numerator = self.d * self.lambda_confidence + t + 1
        denominator = self.d * self.lambda_confidence
        return (np.sqrt(self.lambda_confidence) * m2
                + np.sqrt(2 * np.log(t + 1) + self.d * np.log(numerator / denominator)))

    def _optimize_baseline_round(self, t, estimate, L, m2):
        """Solve the optimistic optimization for one round of LinUCB."""
        model = gp.Model(f"baseline_round_{t}")
        model.setParam("Threads", MAX_THREADS)
        model.setParam("TimeLimit", MAX_TIME_BASELINE)

        a_vars = model.addVars(self.d, lb=-1, ub=1, name="a")
        x_vars = model.addVars(self.d, lb=-self.R_max, ub=self.R_max, name="x")
        model.addConstr(gp.quicksum(a_vars[i] for i in range(self.d)) <= self.b, "budget")

        objective = gp.quicksum(a_vars[i] * x_vars[i] for i in range(self.d))
        model.setObjective(objective, GRB.MAXIMIZE)

        # Confidence ellipsoid constraint: (x - theta_hat)^T L (x - theta_hat) <= beta^2
        diff_terms = [x_vars[j] - estimate[j] for j in range(self.d)]
        quad = gp.QuadExpr()
        for row in range(self.d):
            for col in range(self.d):
                coeff = L[row, col]
                if coeff == 0:
                    continue
                quad += coeff * diff_terms[row] * diff_terms[col]
        model.addQConstr(quad <= self._baseline_beta(t, m2) ** 2, name="conf")

        model.optimize()
        return model

    def run(self):
        """Execute the baseline LinUCB algorithm and return per-round regrets."""
        A_hist = np.empty((0, self.d))
        Y_hist = []
        estimate = np.zeros(self.d)
        L = np.diag(np.full(self.d, self.lambda_confidence))
        m2 = norm(self.X_row_sum) * 2
        error_count = 0
        print("[Baseline] Starting run.")

        for t in range(self.tau):
            try:
                model = self._optimize_baseline_round(t + 1, estimate, L, m2)
                action = np.array([model.getVarByName(f"a[{i}]").X for i in range(self.d)])
            except AttributeError:
                error_count += 1
                if A_hist.shape[0] == 0:
                    action = self.sample_random_action()
                else:
                    action = A_hist[self.rng.integers(A_hist.shape[0])]

            # Observe aggregated reward Z_t = theta^T a_t + eta_t.
            reward = np.dot(self.X_row_sum, action) + self.rng.normal(0, self.noise_std)
            A_hist = np.vstack((A_hist, action))
            Y_hist.append(reward)

            # Update confidence ellipsoid.
            L = (L + np.outer(action, action)
                 + (t / max(self.tau, 1)) * np.identity(self.d) * self.reg_param)

            # Update estimate via ridge regression.
            ridge = Ridge(alpha=self.lambda_confidence, fit_intercept=False)
            ridge.fit(A_hist, Y_hist)
            estimate = ridge.coef_

            if (t + 1) % 500 == 0 or t + 1 == self.tau:
                print(f"[Baseline] Round {t + 1}/{self.tau}; error_count={error_count}.")

        print(f"[Baseline] Finished. error_count={error_count}.")
        regrets, _ = self._regret_from_actions(A_hist)
        print(f"[Baseline] Total regret={sum(regrets):.2f}.")
        return {"regret": regrets, "error_count": error_count}


# ---------------------------------------------------------------------------
# Backward-compatible aliases for old import names used in simulation scripts.
# ---------------------------------------------------------------------------
ESTCBandit = NETCBandit
PartialBandit = NSEBandit
fullBandit = NSEFSBandit

# For generate_X -> generate_network compatibility
generate_X = generate_network
generate_village = generate_village_network
