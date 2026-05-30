"""Domain 11 — reinforcement learning (CartPole, GridWorld)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_SEED,
    finalize_domain,
    make_regressor,
    reg_metrics,
    rng,
)


class CartPoleEnv:
    def __init__(self, rng_seed: int = DEFAULT_SEED):
        self.rng = np.random.default_rng(rng_seed)
        self.g, self.mc, self.mp, self.l = 9.8, 1.0, 0.1, 0.5
        self.dt, self.force = 0.02, 10.0
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force = self.force if action == 1 else -self.force
        costheta, sintheta = np.cos(theta), np.sin(theta)
        tmp = (force + self.mp * self.l * theta_dot**2 * sintheta) / (self.mc + self.mp)
        theta_acc = (self.g * sintheta - costheta * tmp) / (
            self.l * (4 / 3 - self.mp * costheta**2 / (self.mc + self.mp))
        )
        x_acc = tmp - self.mp * self.l * theta_acc * costheta / (self.mc + self.mp)
        self.state = np.array(
            [
                x + self.dt * x_dot,
                x_dot + self.dt * x_acc,
                theta + self.dt * theta_dot,
                theta_dot + self.dt * theta_acc,
            ]
        )
        done = (abs(self.state[2]) > 0.2094) or (abs(self.state[0]) > 2.4)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


class GridWorldEnv:
    def __init__(self, size: int = 4, rng_seed: int = DEFAULT_SEED):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.rng = np.random.default_rng(rng_seed)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._encode()

    def step(self, action: int):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        r = max(0, min(self.size - 1, self.pos[0] + dr))
        c = max(0, min(self.size - 1, self.pos[1] + dc))
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 1.0 if done else 0.0
        return self._encode(), reward, done

    def _encode(self) -> np.ndarray:
        vec = np.zeros(self.size * self.size, dtype=np.float32)
        vec[self.pos[0] * self.size + self.pos[1]] = 1.0
        return vec


def _collect_cartpole_returns(n_episodes: int = 1000, gamma: float = 0.99):
    env = CartPoleEnv()
    states, returns = [], []
    for _ in range(n_episodes):
        s = env.reset()
        G, traj = 0.0, []
        for _ in range(200):
            action = env.rng.integers(0, 2)
            s_next, r, done = env.step(action)
            traj.append((s.copy(), r))
            s = s_next
            if done:
                break
        G = 0.0
        for _, r in reversed(traj):
            G = r + gamma * G
        for s, _ in traj:
            states.append(s.astype(np.float32))
            returns.append(G)
    return np.stack(states), np.array(returns, dtype=np.float32)


def _true_q_gridworld(gamma: float = 0.9):
    size = 4
    n_states = size * size
    n_actions = 4
    goal = n_states - 1
    Q_true = np.zeros((n_states, n_actions), dtype=np.float64)
    for _ in range(1000):
        Q_new = Q_true.copy()
        for s in range(n_states):
            if s == goal:
                continue
            r, c = divmod(s, size)
            for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nr = max(0, min(size - 1, r + dr))
                nc = max(0, min(size - 1, c + dc))
                ns = nr * size + nc
                reward = 1.0 if ns == goal else 0.0
                Q_new[s, a] = reward + gamma * Q_true[ns].max()
        Q_true = Q_new
    return Q_true


def experiment_11a_cartpole_value():
    X, y = _collect_cartpole_returns()
    split = int(0.8 * len(X))
    reg = make_regressor(X.shape[1])
    for _pass in range(4):
        g = rng(DEFAULT_SEED + _pass)
        order = g.permutation(split)
        for i in order:
            reg.train_step(X[i], float(y[i]))
    cypha = reg_metrics(reg, X[split:], y[split:])
    ridge = Ridge(alpha=1.0).fit(X[:split], y[:split])
    y_hat = ridge.predict(X[split:])
    cypha["ridge_rmse"] = float(np.sqrt(np.mean((y_hat - y[split:]) ** 2)))
    cypha["ridge_r2"] = float(
        1.0 - np.sum((y_hat - y[split:]) ** 2) / (np.sum((y[split:] - y[split:].mean()) ** 2) + 1e-12)
    )
    near_tip = np.abs(X[split:, 2]) > 0.15
    if near_tip.any():
        unc = []
        for x in X[split:][near_tip]:
            _, u = reg.predict(x)
            unc.append(u)
        cypha["mean_epistemic_near_tip"] = float(np.mean(np.square(unc)))
    return cypha


def experiment_11b_gridworld_q():
    Q_true = _true_q_gridworld()
    env = GridWorldEnv()
    X, y = [], []
    g = rng(DEFAULT_SEED + 1)
    for s in range(16):
        state_vec = np.zeros(16, dtype=np.float32)
        state_vec[s] = 1.0
        for a in range(4):
            X.append(np.concatenate([state_vec, np.eye(4, dtype=np.float32)[a]]))
            y.append(Q_true[s, a])
    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    split = int(0.8 * len(X))
    reg = make_regressor(X.shape[1], seed=DEFAULT_SEED + 1)
    for _pass in range(5):
        g = rng(DEFAULT_SEED + _pass)
        order = g.permutation(split)
        for i in order:
            reg.train_step(X[i], float(y[i]))
    preds = []
    for x in X[split:]:
        y_hat, _ = reg.predict(x)
        preds.append(float(np.ravel(y_hat)[0]))
    mae = float(np.mean(np.abs(np.array(preds) - y[split:])))
    return {"q_value_mae": mae, "n_pairs": len(y) - split}


def experiment_11c_preference():
    env = CartPoleEnv(rng_seed=DEFAULT_SEED + 2)
    pairs_X, pairs_y = [], []
    for _ in range(500):
        s0 = env.reset()
        traj_a, ret_a = [], 0.0
        traj_b, ret_b = [], 0.0
        s = s0.copy()
        for _ in range(50):
            a = env.rng.integers(0, 2)
            s, r, done = env.step(a)
            traj_a.append(s.copy())
            ret_a += r
            if done:
                break
        env.state = s0.copy()
        s = s0.copy()
        for _ in range(50):
            a = env.rng.integers(0, 2)
            s, r, done = env.step(a)
            traj_b.append(s.copy())
            ret_b += r
            if done:
                break
        if not traj_a or not traj_b:
            continue
        sa = np.stack(traj_a).mean(axis=0)
        sb = np.stack(traj_b).mean(axis=0)
        feat = np.concatenate([sa, sb, [len(traj_a), len(traj_b), np.std(traj_a), np.std(traj_b)]])
        pairs_X.append(feat.astype(np.float32))
        pairs_y.append(1 if ret_a > ret_b else 0)

    X = np.stack(pairs_X)
    y = np.array(pairs_y, dtype=int)
    from bench_common import make_classifier, train_classifier_online, clf_metrics

    split = int(0.8 * len(X))
    clf = make_classifier(X.shape[1], seed=DEFAULT_SEED + 2)
    train_classifier_online(clf, X[:split], y[:split], label_fmt=str, passes=4)
    return clf_metrics(clf, X[split:], y[split:])


def run() -> dict:
    experiments = {
        "11A_cartpole_value_regression": experiment_11a_cartpole_value(),
        "11B_gridworld_q_estimation": experiment_11b_gridworld_q(),
        "11C_trajectory_preference": experiment_11c_preference(),
    }
    return finalize_domain("d11", experiments)


if __name__ == "__main__":
    print(run())
