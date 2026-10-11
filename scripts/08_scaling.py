"""
08_scaling.py
Experimento de escalabilidade: Q-Learning e DQN em instâncias de 4, 8 e 12 estações.

Objetivo científico:
  Demonstrar que Q-Learning tabular converge apenas em instâncias pequenas (4 estações)
  e colapsa à medida que o espaço de estados cresce (8 e 12 estações), enquanto o DQN
  mantém desempenho consistente graças à generalização via função de valor aproximada.

Uso:
    # Q-Learning em todas as escalas (rápido, ~30s)
    python scripts/08_scaling.py --mode qlearning

    # DQN em uma escala específica (blocos de 3k episódios)
    python scripts/08_scaling.py --mode dqn --stations 4
    python scripts/08_scaling.py --mode dqn --stations 8
    python scripts/08_scaling.py --mode dqn --stations 12

    # DQN continuar de checkpoint
    python scripts/08_scaling.py --mode dqn --stations 8 --resume results/scaling/dqn_8st_blk1.npz --ep-offset 3000

    # Ablação (12 estações)
    python scripts/08_scaling.py --mode ablation --stations 12

    # Avaliação final multi-seed
    python scripts/08_scaling.py --mode eval --stations 12 --weights results/scaling/dqn_12st_best.npz
"""

import os, sys, json, argparse, time
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "scaling")

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning
QL_ALPHA      = 0.1
QL_GAMMA      = 1.0
QL_EPS_START  = 1.0
QL_EPS_END    = 0.05
QL_EPS_DECAY  = 0.80    # fração de episódios para decair ε
QL_N_TRAIN    = 20_000
QL_EVAL_EVERY = 1_000

# DQN
DQN_LR          = 3e-4
DQN_GAMMA       = 1.0
DQN_EPS_START   = 1.0
DQN_EPS_END     = 0.05
DQN_EPS_DECAY   = 0.70
DQN_BATCH_SIZE  = 64
DQN_BUFFER_SIZE = 10_000
DQN_TARGET_SYNC = 50
DQN_TRAIN_EVERY = 4
DQN_N_TRAIN     = 9_500    # por bloco padrão
DQN_EVAL_EVERY  = 500
DQN_EVAL_EPS    = 20

DELTAS     = [-10, -5, 0, 5, 10]
EVAL_SEEDS = [123, 456, 789]   # 3 seeds para avaliação multi-seed
TRAIN_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO SIMULADOR
# ─────────────────────────────────────────────────────────────────────────────
def load_sim_and_env(n_stations: int):
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader(
        "sim", os.path.join(os.path.dirname(__file__), "04_simulator.py"))
    spec = importlib.util.spec_from_loader("sim", loader)
    sim  = importlib.util.module_from_spec(spec)
    loader.exec_module(sim)

    profiles = os.path.join(BASE_DIR, f"demand_profiles_{n_stations}stations_normalized.csv")
    stations_csv = os.path.join(BASE_DIR, f"station_selected_{n_stations}.csv")

    stations = sim.load_environment_data(profiles_path=profiles, stations_path=stations_csv)
    env_tr = sim.BSRPEnv(stations, rng=np.random.default_rng(TRAIN_SEED))
    env_ev = sim.BSRPEnv(stations, rng=np.random.default_rng(EVAL_SEEDS[0]))
    return sim, stations, env_tr, env_ev


def baselines(sim, env_ev, n_ep=100, seed=123):
    rng = np.random.default_rng(seed)
    out = {}
    for name, pol in [("Aleatória", sim.RandomPolicy(np.random.default_rng(seed))),
                      ("Gulosa",    sim.GreedyPolicy())]:
        rs, us = [], []
        for _ in range(n_ep):
            res = sim.run_episode(env_ev, pol)
            rs.append(res["total_reward"])
            us.append(res["unmet_dep"] + res["unmet_arr"])
        out[name] = {"reward": float(np.mean(rs)), "std": float(np.std(rs)),
                     "unmet": float(np.mean(us))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING TABULAR
# ─────────────────────────────────────────────────────────────────────────────
class QTable:
    def __init__(self):
        self._q  = {}

    def get(self, key, n_actions):
        if key not in self._q:
            self._q[key] = np.zeros(n_actions, dtype=np.float32)
        return self._q[key]

    def update(self, key, a_idx, target, alpha):
        q = self._q[key]
        q[a_idx] += alpha * (target - q[a_idx])

    def n_states(self):
        return len(self._q)


def obs_to_key(obs):
    return tuple(obs.tolist())


def ql_valid_indices(env, all_actions):
    valid_set = set(map(tuple, env.valid_actions()))
    return [i for i, a in enumerate(all_actions) if tuple(a) in valid_set]


def train_qlearning(sim, env_tr, env_ev, n_stations, n_episodes=QL_N_TRAIN):
    all_actions = [(d, delta)
                   for d in range(n_stations)
                   for delta in DELTAS]
    n_actions = len(all_actions)

    qt  = QTable()
    eps = QL_EPS_START
    eps_end = QL_EPS_END
    decay_end = int(n_episodes * QL_EPS_DECAY)
    eps_delta = (QL_EPS_START - QL_EPS_END) / max(decay_end, 1)

    history = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, state = env_tr.reset()
        key = obs_to_key(obs)

        while not state.done:
            valid_idxs = ql_valid_indices(env_tr, all_actions)
            q_vals = qt.get(key, n_actions)

            if np.random.random() < eps:
                a_idx = np.random.choice(valid_idxs)
            else:
                a_idx = valid_idxs[np.argmax(q_vals[valid_idxs])]

            action = all_actions[a_idx]
            obs2, r, done, _ = env_tr.step(action)
            key2 = obs_to_key(obs2)

            # TD update
            if done:
                target = r
            else:
                valid2 = ql_valid_indices(env_tr, all_actions)
                q2     = qt.get(key2, n_actions)
                target = r + QL_GAMMA * np.max(q2[valid2])

            qt.update(key, a_idx, target, QL_ALPHA)
            obs, key, state = obs2, key2, env_tr._state

        if eps > eps_end and ep < decay_end:
            eps -= eps_delta

        # Avaliação periódica
        if (ep + 1) % QL_EVAL_EVERY == 0:
            rs, us = [], []
            for _ in range(20):
                obs_e, st_e = env_ev.reset()
                ep_r = 0.0
                while not st_e.done:
                    k_e = obs_to_key(obs_e)
                    vi  = ql_valid_indices(env_ev, all_actions)
                    qv  = qt.get(k_e, n_actions)
                    a_e = vi[np.argmax(qv[vi])]
                    obs_e, re, de, _ = env_ev.step(all_actions[a_e])
                    ep_r += re
                    st_e = env_ev._state
                rs.append(ep_r)
                us.append(env_ev._state.unmet_dep + env_ev._state.unmet_arr
                          if hasattr(env_ev._state, 'unmet_dep') else 0)

            elapsed = time.time() - t0
            history.append({"ep": ep+1, "eval_r": float(np.mean(rs)),
                            "eps": round(eps, 4), "states": qt.n_states(),
                            "elapsed": round(elapsed, 1)})
            print(f"  ep {ep+1:6d} | eval_r={np.mean(rs):7.1f} | ε={eps:.3f} "
                  f"| states={qt.n_states():,} | t={elapsed:.0f}s")

    return qt, history


def eval_qlearning_multiseed(sim, n_stations, qt, n_ep=100):
    all_actions = [(d, delta) for d in range(n_stations) for delta in DELTAS]
    n_actions   = len(all_actions)
    results     = {}
    for seed in EVAL_SEEDS:
        stations = sim.load_environment_data(
            profiles_path=os.path.join(BASE_DIR, f"demand_profiles_{n_stations}stations_normalized.csv"),
            stations_path=os.path.join(BASE_DIR, f"station_selected_{n_stations}.csv"))
        env_s = sim.BSRPEnv(stations, rng=np.random.default_rng(seed))
        rs, us = [], []
        for _ in range(n_ep):
            obs, state = env_s.reset()
            ep_r = 0.0
            while not state.done:
                vi  = ql_valid_indices(env_s, all_actions)
                qv  = qt.get(obs_to_key(obs), n_actions)
                a_e = vi[np.argmax(qv[vi])]
                obs, r, done, _ = env_s.step(all_actions[a_e])
                ep_r += r
                state = env_s._state
            rs.append(ep_r)
            us.append(getattr(env_s._state, 'unmet_dep', 0) +
                      getattr(env_s._state, 'unmet_arr', 0))
        results[seed] = {"reward": float(np.mean(rs)), "std": float(np.std(rs)),
                         "unmet": float(np.mean(us))}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DQN — REDE NEURAL (NUMPY PURO)
# ─────────────────────────────────────────────────────────────────────────────
def relu(x):     return np.maximum(0, x)
def he(n_in, n_out, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.standard_normal((n_in, n_out)).astype(np.float32) * np.sqrt(2.0 / n_in)

def init_weights(n_in, n_out, rng=None):
    rng = rng or np.random.default_rng(0)
    return {
        "W1": he(n_in, 64, rng), "b1": np.zeros(64,    dtype=np.float32),
        "W2": he(64,   64, rng), "b2": np.zeros(64,    dtype=np.float32),
        "W3": he(64, n_out, rng),"b3": np.zeros(n_out, dtype=np.float32),
    }

def copy_weights(W):
    return {k: v.copy() for k, v in W.items()}

def forward(x, W):
    h1 = relu(x @ W["W1"] + W["b1"])
    h2 = relu(h1 @ W["W2"] + W["b2"])
    q  = h2 @ W["W3"] + W["b3"]
    return q, h1, h2

def backward(x, h1, h2, q_pred, q_target, W):
    batch = x.shape[0]
    dq   = (q_pred - q_target) / batch

    dW3  = h2.T @ dq;        db3  = dq.sum(0)
    dh2  = dq @ W["W3"].T;   dh2 *= (h2 > 0)

    dW2  = h1.T @ dh2;       db2  = dh2.sum(0)
    dh1  = dh2 @ W["W2"].T;  dh1 *= (h1 > 0)

    dW1  = x.T @ dh1;        db1  = dh1.sum(0)
    return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}


class Adam:
    def __init__(self, lr=DQN_LR, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = self.t = None

    def step(self, W, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in W.items()}
            self.v = {k: np.zeros_like(v) for k, v in W.items()}
            self.t = 0
        self.t += 1
        for k in W:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k]**2
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            W[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return W


class ReplayBuffer:
    def __init__(self, capacity=DQN_BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s  = np.array([b[0] for b in batch], dtype=np.float32)
        a  = np.array([b[1] for b in batch], dtype=np.int32)
        r  = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.array([b[3] for b in batch], dtype=np.float32)
        d  = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def get_features(env, n_stations) -> np.ndarray:
    state = env._state
    occ   = state.occupancy
    occ_ratio  = np.array([occ[i] / max(env.stations[i].cap, 1)
                            for i in range(n_stations)], dtype=np.float32)
    pos_onehot = np.zeros(n_stations, dtype=np.float32)
    pos_onehot[state.vehicle_pos] = 1.0
    load_ratio = np.float32(state.vehicle_load / 20.0)
    time_norm  = np.float32(state.t / 32.0)
    return np.concatenate([occ_ratio, pos_onehot, [load_ratio, time_norm]])


def get_valid_mask(env, all_actions, n_actions):
    valid_set = set(map(tuple, env.valid_actions()))
    mask = np.array([tuple(a) in valid_set for a in all_actions], dtype=bool)
    return mask


def eval_dqn(env_ev, W, all_actions, n_stations, n_ep=20,
             rng=None, use_replay=True, use_target=True):
    """Avaliação greedy (ε=0) do DQN."""
    n_actions = len(all_actions)
    rewards, unmets = [], []
    for _ in range(n_ep):
        obs, state = env_ev.reset()
        feat = get_features(env_ev, n_stations)
        ep_r = 0.0
        while not state.done:
            mask = get_valid_mask(env_ev, all_actions, n_actions)
            q, _, _ = forward(feat[np.newaxis], W)
            q[0][~mask] = -np.inf
            a_idx  = int(np.argmax(q[0]))
            obs, r, done, _ = env_ev.step(all_actions[a_idx])
            feat   = get_features(env_ev, n_stations)
            state  = env_ev._state
            ep_r  += r
        rewards.append(ep_r)
        u = getattr(env_ev._state, 'unmet_dep', 0) + getattr(env_ev._state, 'unmet_arr', 0)
        unmets.append(u)
    return float(np.mean(rewards)), float(np.mean(unmets))


def train_dqn(sim, env_tr, env_ev, n_stations,
              n_episodes=DQN_N_TRAIN,
              W_start=None, ep_offset=0,
              eps_fixed=None,          # se não None, usa este ε fixo
              lr=DQN_LR,
              use_replay=True,         # False = sem replay buffer (online)
              use_target=True,         # False = sem target network
              use_double=True,         # False = DQN padrão (não Double)
              verbose=True):

    all_actions = [(d, delta) for d in range(n_stations) for delta in DELTAS]
    n_actions   = len(all_actions)
    n_features  = n_stations * 2 + 2

    rng = np.random.default_rng(TRAIN_SEED + ep_offset)

    W     = copy_weights(W_start) if W_start else init_weights(n_features, n_actions, rng)
    W_tgt = copy_weights(W)
    opt   = Adam(lr=lr)
    buf   = ReplayBuffer(DQN_BUFFER_SIZE)

    eps = eps_fixed if eps_fixed is not None else DQN_EPS_START
    if eps_fixed is None:
        decay_end = int(n_episodes * DQN_EPS_DECAY)
        eps_delta = (DQN_EPS_START - DQN_EPS_END) / max(decay_end, 1)
    else:
        eps_delta = 0.0
        decay_end = 0

    history = []
    step = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs, state = env_tr.reset()
        feat = get_features(env_tr, n_stations)
        ep_r = 0.0

        while not state.done:
            mask = get_valid_mask(env_tr, all_actions, n_actions)
            if rng.random() < eps:
                valid_idxs = np.where(mask)[0]
                a_idx = int(rng.choice(valid_idxs))
            else:
                q, _, _ = forward(feat[np.newaxis], W)
                q[0][~mask] = -np.inf
                a_idx = int(np.argmax(q[0]))

            obs2, r, done, _ = env_tr.step(all_actions[a_idx])
            feat2 = get_features(env_tr, n_stations)
            ep_r += r

            if use_replay:
                buf.push(feat.copy(), a_idx, r, feat2.copy(), float(done))
            else:
                # Online (sem replay): treina direto com a transição atual
                s  = feat[np.newaxis]
                s2 = feat2[np.newaxis]
                q_p, h1, h2 = forward(s, W)
                if done:
                    td_target = r
                else:
                    if use_double:
                        q2_main, _, _ = forward(s2, W)
                        a_best = int(np.argmax(q2_main[0]))
                        q2_tgt, _, _ = forward(s2, W_tgt if use_target else W)
                        td_target = r + DQN_GAMMA * q2_tgt[0, a_best]
                    else:
                        q2, _, _ = forward(s2, W_tgt if use_target else W)
                        td_target = r + DQN_GAMMA * np.max(q2[0])
                q_t = q_p.copy()
                q_t[0, a_idx] = td_target
                grads = backward(s, h1, h2, q_p, q_t, W)
                W = opt.step(W, grads)

            feat = feat2
            state = env_tr._state
            step += 1

            # Treino com replay
            if use_replay and len(buf) >= DQN_BATCH_SIZE and step % DQN_TRAIN_EVERY == 0:
                s_b, a_b, r_b, s2_b, d_b = buf.sample(DQN_BATCH_SIZE)
                q_p_b, h1_b, h2_b = forward(s_b, W)

                if use_double:
                    q2_main, _, _ = forward(s2_b, W)
                    a_best_b = np.argmax(q2_main, axis=1)
                    q2_tgt, _, _ = forward(s2_b, W_tgt if use_target else W)
                    q_next = q2_tgt[np.arange(DQN_BATCH_SIZE), a_best_b]
                else:
                    q2, _, _ = forward(s2_b, W_tgt if use_target else W)
                    q_next = np.max(q2, axis=1)

                td = r_b + DQN_GAMMA * q_next * (1 - d_b)
                q_t_b = q_p_b.copy()
                q_t_b[np.arange(DQN_BATCH_SIZE), a_b] = td
                grads = backward(s_b, h1_b, h2_b, q_p_b, q_t_b, W)
                W = opt.step(W, grads)

        if use_replay and eps_fixed is None and ep < decay_end:
            eps -= eps_delta
            eps  = max(eps, DQN_EPS_END)

        # Sincroniza target
        if use_target and (ep + 1) % DQN_TARGET_SYNC == 0:
            W_tgt = copy_weights(W)

        # Avaliação periódica
        if (ep + 1) % DQN_EVAL_EVERY == 0:
            eval_r, eval_u = eval_dqn(env_ev, W, all_actions, n_stations,
                                       n_ep=DQN_EVAL_EPS)
            elapsed = time.time() - t0
            history.append({"ep": ep + 1 + ep_offset,
                             "eval_r": eval_r, "eval_u": eval_u,
                             "eps": round(eps, 4), "elapsed": round(elapsed, 1)})
            if verbose:
                print(f"  ep {ep+1+ep_offset:6d} | eval_r={eval_r:7.1f} "
                      f"| unmet={eval_u:.1f} | ε={eps:.3f} | t={elapsed:.0f}s")

    return W, history


def eval_dqn_multiseed(sim, n_stations, W, n_ep=100):
    all_actions = [(d, delta) for d in range(n_stations) for delta in DELTAS]
    results = {}
    for seed in EVAL_SEEDS:
        stations = sim.load_environment_data(
            profiles_path=os.path.join(BASE_DIR, f"demand_profiles_{n_stations}stations_normalized.csv"),
            stations_path=os.path.join(BASE_DIR, f"station_selected_{n_stations}.csv"))
        env_s = sim.BSRPEnv(stations, rng=np.random.default_rng(seed))
        rs, us = [], []
        for _ in range(n_ep):
            obs, state = env_s.reset()
            feat = get_features(env_s, n_stations)
            ep_r = 0.0
            while not state.done:
                mask = get_valid_mask(env_s, all_actions, len(all_actions))
                q, _, _ = forward(feat[np.newaxis], W)
                q[0][~mask] = -np.inf
                a_idx = int(np.argmax(q[0]))
                obs, r, done, _ = env_s.step(all_actions[a_idx])
                feat = get_features(env_s, n_stations)
                state = env_s._state
                ep_r += r
            rs.append(ep_r)
            u = getattr(env_s._state, 'unmet_dep', 0) + getattr(env_s._state, 'unmet_arr', 0)
            us.append(u)
        results[seed] = {"reward": float(np.mean(rs)), "std": float(np.std(rs)),
                         "unmet": float(np.mean(us))}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["qlearning","dqn","ablation","eval"], required=True)
    parser.add_argument("--stations",  type=int, default=12, choices=[4, 8, 12])
    parser.add_argument("--episodes",  type=int, default=None)
    parser.add_argument("--resume",    type=str, default=None, help="Checkpoint .npz para continuar DQN")
    parser.add_argument("--ep-offset", type=int, default=0,   help="Episódios já treinados (para resume)")
    parser.add_argument("--eps-fixed", type=float, default=None, help="ε fixo (sem decay)")
    parser.add_argument("--lr",        type=float, default=DQN_LR)
    parser.add_argument("--weights",   type=str, default=None, help="Pesos DQN para eval-only")
    parser.add_argument("--ablation-variant", type=str, default=None,
                        choices=["no_replay","no_target","no_double","full"],
                        help="Variante para ablação")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n = args.stations
    sim, stations, env_tr, env_ev = load_sim_and_env(n)

    # ── BASELINES ────────────────────────────────────────────────────────────
    print(f"\n=== BASELINES — {n} estações (50 ep, seed=123) ===")
    bl = baselines(sim, env_ev, n_ep=50)
    for name, d in bl.items():
        print(f"  {name:<12}: reward={d['reward']:7.1f} ±{d['std']:.1f}  unmet={d['unmet']:.1f}")

    # ── Q-LEARNING ───────────────────────────────────────────────────────────
    if args.mode == "qlearning":
        for n_st in [4, 8, 12]:
            print(f"\n{'='*60}")
            print(f"Q-Learning — {n_st} estações")
            sim2, _, env_tr2, env_ev2 = load_sim_and_env(n_st)
            n_ep = args.episodes or QL_N_TRAIN
            qt, hist = train_qlearning(sim2, env_tr2, env_ev2, n_st, n_ep)
            ms_results = eval_qlearning_multiseed(sim2, n_st, qt, n_ep=100)
            mean_r = np.mean([v["reward"] for v in ms_results.values()])
            std_r  = np.mean([v["std"]    for v in ms_results.values()])
            mean_u = np.mean([v["unmet"]  for v in ms_results.values()])
            print(f"\nQ-Learning {n_st} est (multi-seed): reward={mean_r:.1f} ±{std_r:.1f}  unmet={mean_u:.1f}")
            out = {"n_stations": n_st, "n_episodes": n_ep,
                   "multiseed": {str(k): v for k, v in ms_results.items()},
                   "history": hist}
            path = os.path.join(RESULTS_DIR, f"ql_{n_st}st.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Salvo: {path}")
        return

    # ── DQN ─────────────────────────────────────────────────────────────────
    if args.mode == "dqn":
        n_ep = args.episodes or DQN_N_TRAIN
        W_start = None
        if args.resume:
            data = np.load(args.resume)
            W_start = {k: data[k] for k in data.files}
            print(f"Carregado checkpoint: {args.resume} (ep_offset={args.ep_offset})")

        W, hist = train_dqn(sim, env_tr, env_ev, n,
                            n_episodes=n_ep,
                            W_start=W_start, ep_offset=args.ep_offset,
                            eps_fixed=args.eps_fixed, lr=args.lr)

        total_ep = args.ep_offset + n_ep
        save_path = os.path.join(RESULTS_DIR, f"dqn_{n}st_ep{total_ep}.npz")
        np.savez(save_path, **W)

        ms_results = eval_dqn_multiseed(sim, n, W, n_ep=100)
        mean_r = np.mean([v["reward"] for v in ms_results.values()])
        std_r  = np.mean([v["std"]    for v in ms_results.values()])
        mean_u = np.mean([v["unmet"]  for v in ms_results.values()])
        print(f"\nDQN {n} est ep{total_ep} (multi-seed): reward={mean_r:.1f} ±{std_r:.1f}  unmet={mean_u:.1f}")

        out = {"n_stations": n, "n_episodes": total_ep,
               "multiseed": {str(k): v for k, v in ms_results.items()},
               "history": hist, "weights_path": save_path}
        json_path = os.path.join(RESULTS_DIR, f"dqn_{n}st_ep{total_ep}.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Pesos: {save_path}\nJSON: {json_path}")
        return

    # ── ABLAÇÃO ──────────────────────────────────────────────────────────────
    if args.mode == "ablation":
        variants = [
            ("full",       True,  True,  True),
            ("no_replay",  False, True,  True),
            ("no_target",  True,  False, True),
            ("no_double",  True,  True,  False),
        ]
        if args.ablation_variant:
            variants = [v for v in variants if v[0] == args.ablation_variant]

        abl_results = {}
        for vname, use_rep, use_tgt, use_dbl in variants:
            print(f"\n{'─'*50}")
            print(f"Ablação: {vname} (replay={use_rep}, target={use_tgt}, double={use_dbl})")
            n_ep = args.episodes or 5_000
            sim2, _, env_tr2, env_ev2 = load_sim_and_env(12)
            W, hist = train_dqn(sim2, env_tr2, env_ev2, 12,
                                n_episodes=n_ep, lr=DQN_LR,
                                use_replay=use_rep,
                                use_target=use_tgt,
                                use_double=use_dbl)
            ms = eval_dqn_multiseed(sim2, 12, W, n_ep=100)
            mean_r = np.mean([v["reward"] for v in ms.values()])
            std_r  = np.mean([v["std"]    for v in ms.values()])
            mean_u = np.mean([v["unmet"]  for v in ms.values()])
            print(f"  [{vname}] reward={mean_r:.1f} ±{std_r:.1f}  unmet={mean_u:.1f}")
            abl_results[vname] = {"reward": mean_r, "std": std_r,
                                  "unmet": mean_u,
                                  "multiseed": {str(k): v for k, v in ms.items()},
                                  "history": hist}
            wp = os.path.join(RESULTS_DIR, f"abl_{vname}.npz")
            np.savez(wp, **W)

        out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
        with open(out_path, "w") as f:
            json.dump(abl_results, f, indent=2)
        print(f"\nResultados ablação: {out_path}")
        return

    # ── EVAL ONLY ────────────────────────────────────────────────────────────
    if args.mode == "eval":
        if not args.weights:
            print("Forneça --weights caminho/para/pesos.npz")
            return
        data = np.load(args.weights)
        W = {k: data[k] for k in data.files}
        ms = eval_dqn_multiseed(sim, n, W, n_ep=100)
        print(f"\nAvaliação multi-seed DQN {n} estações:")
        for seed, d in ms.items():
            print(f"  seed={seed}: reward={d['reward']:.1f} ±{d['std']:.1f}  unmet={d['unmet']:.1f}")
        mean_r = np.mean([v["reward"] for v in ms.values()])
        mean_u = np.mean([v["unmet"]  for v in ms.values()])
        print(f"  MÉDIA: reward={mean_r:.1f}  unmet={mean_u:.1f}")


if __name__ == "__main__":
    main()
