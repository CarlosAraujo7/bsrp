"""
09_reinforce.py  —  REINFORCE (Policy Gradient) para o BSRP
============================================================
Diferenças vs DQN (08_scaling.py):
  * On-policy: sem replay buffer, gradientes calculados sobre episódios completos
  * Retorno G_t normalizado por episódio (reduz variância do Poisson)
  * Baseline: EMA do retorno total do episódio (reduz variância do gradiente)
  * Masked softmax: ações inválidas recebem logit -1e9 antes do softmax
  * Suporte a resume via --resume / --ep-offset

Uso:
    python scripts/09_reinforce.py --stations 4
    python scripts/09_reinforce.py --stations 8
    python scripts/09_reinforce.py --stations 12
    python scripts/09_reinforce.py --stations 8 --resume results/reinforce/pg_8st_ep3000.npz --ep-offset 3000
    python scripts/09_reinforce.py --mode eval --stations 12 --weights results/reinforce/pg_12st_ep9500.npz
"""
import os, sys, json, argparse, time
import numpy as np
import importlib.machinery, importlib.util

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results", "reinforce")
os.makedirs(RESULTS_DIR, exist_ok=True)

loader = importlib.machinery.SourceFileLoader(
    "sim", os.path.join(BASE, "scripts", "04_simulator.py"))
spec = importlib.util.spec_from_loader("sim", loader)
sim  = importlib.util.module_from_spec(spec)
loader.exec_module(sim)

# ── hyperparameters ────────────────────────────────────────────────────────
TRAIN_SEED  = 42
EVAL_SEEDS  = [123, 456, 789]
DELTAS      = [-10, -5, -2, 0, 2, 5, 10, 20]
GAMMA       = 1.0        # episódio finito
LR          = 3e-4       # mesma LR do DQN
BASELINE_DECAY = 0.95    # EMA para baseline de retorno
EVAL_EVERY  = 500
EVAL_EP     = 30
N_TRAIN     = 3000       # episódios por bloco padrão

# ── utilidades ────────────────────────────────────────────────────────────────
def relu(x):  return np.maximum(0.0, x)
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def load_sim_and_env(n_stations):
    profiles = os.path.join(BASE, f"demand_profiles_{n_stations}stations_normalized.csv")
    stations_csv = os.path.join(BASE, f"station_selected_{n_stations}.csv")
    stations = sim.load_environment_data(profiles_path=profiles, stations_path=stations_csv)
    env_tr = sim.BSRPEnv(stations, rng=np.random.default_rng(TRAIN_SEED))
    env_ev = sim.BSRPEnv(stations, rng=np.random.default_rng(EVAL_SEEDS[0]))
    return sim, stations, env_tr, env_ev

def get_features(env, n_stations):
    s = env._state
    occ_ratio  = np.array([s.occupancy[i] / max(env.stations[i].cap, 1)
                           for i in range(n_stations)], dtype=np.float32)
    pos_onehot = np.zeros(n_stations, dtype=np.float32)
    pos_onehot[s.vehicle_pos] = 1.0
    load_ratio = np.float32(s.vehicle_load / 20.0)
    time_norm  = np.float32(s.t / 32.0)
    return np.concatenate([occ_ratio, pos_onehot, [load_ratio, time_norm]])

def get_valid_mask(env, all_actions, n_actions):
    valid_set = set(map(tuple, env.valid_actions()))
    return np.array([tuple(a) in valid_set for a in all_actions], dtype=bool)

# ── rede de política ──────────────────────────────────────────────────────────
def he(fan_in, fan_out, rng):
    return (rng.standard_normal((fan_in, fan_out)) *
            np.sqrt(2.0 / fan_in)).astype(np.float32)

def init_weights(n_features, n_actions, rng):
    return {
        "W1": he(n_features, 64, rng), "b1": np.zeros(64,       dtype=np.float32),
        "W2": he(64,         64, rng), "b2": np.zeros(64,       dtype=np.float32),
        "W3": he(64, n_actions, rng),  "b3": np.zeros(n_actions, dtype=np.float32),
    }

def copy_weights(W):
    return {k: v.copy() for k, v in W.items()}

def forward(x, W):
    """x: (1, n_features) → logits (1, n_actions), also returns intermediates."""
    h1 = relu(x @ W["W1"] + W["b1"])
    h2 = relu(h1 @ W["W2"] + W["b2"])
    logits = h2 @ W["W3"] + W["b3"]
    return logits, h1, h2

def policy_probs(feat, W, mask):
    """Compute masked softmax probability distribution over actions."""
    logits, _, _ = forward(feat[np.newaxis], W)
    logits = logits[0].copy()
    logits[~mask] = -1e9
    return softmax(logits)

# ── Adam optimizer ────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0; self.m = {}; self.v = {}
    def step(self, W, grads):
        self.t += 1
        for k in W:
            if k not in self.m:
                self.m[k] = np.zeros_like(W[k])
                self.v[k] = np.zeros_like(W[k])
            self.m[k] = self.b1 * self.m[k] + (1-self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1-self.b2) * grads[k]**2
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            W[k] += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ── REINFORCE backward ────────────────────────────────────────────────────────
def policy_gradient(feat, W, a_idx, advantage, mask):
    """
    Gradient of log π(a|s) × advantage, backprop through MLP.
    Returns grad dict matching W keys.
    """
    x = feat[np.newaxis]          # (1, n_in)
    logits, h1, h2 = forward(x, W)
    logits_masked = logits[0].copy(); logits_masked[~mask] = -1e9
    probs = softmax(logits_masked)

    # δ = advantage × (e_a - π)  — gradient of log π(a|s) w.r.t. logits
    delta = -advantage * probs.copy()
    delta[a_idx] += advantage
    delta = delta[np.newaxis]     # (1, n_actions)

    dW3 = h2.T @ delta;          db3 = delta.sum(0)
    dh2 = delta @ W["W3"].T
    dh2 *= (h2 > 0)              # ReLU gate
    dW2 = h1.T @ dh2;            db2 = dh2.sum(0)
    dh1 = dh2 @ W["W2"].T
    dh1 *= (h1 > 0)
    dW1 = x.T @ dh1;             db1 = dh1.sum(0)

    return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}

def clip_grads(grads, max_norm=1.0):
    total = sum(np.sum(g**2) for g in grads.values()) ** 0.5
    if total > max_norm:
        scale = max_norm / total
        return {k: v * scale for k, v in grads.items()}
    return grads

# ── avaliação ─────────────────────────────────────────────────────────────────
def eval_policy(env_ev, W, all_actions, n_stations, n_ep=30):
    n_actions = len(all_actions)
    rewards = []
    for _ in range(n_ep):
        obs, state = env_ev.reset()
        feat = get_features(env_ev, n_stations)
        ep_r = 0.0
        while not state.done:
            mask = get_valid_mask(env_ev, all_actions, n_actions)
            probs = policy_probs(feat, W, mask)
            a_idx = int(np.argmax(probs))    # greedy durante eval
            obs, r, done, _ = env_ev.step(all_actions[a_idx])
            feat = get_features(env_ev, n_stations)
            state = env_ev._state
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards))

def eval_multiseed(W, n_stations, all_actions, n_ep=100):
    results = {}
    for seed in EVAL_SEEDS:
        profiles = os.path.join(BASE, f"demand_profiles_{n_stations}stations_normalized.csv")
        stations_csv = os.path.join(BASE, f"station_selected_{n_stations}.csv")
        stations = sim.load_environment_data(profiles_path=profiles, stations_path=stations_csv)
        env = sim.BSRPEnv(stations, rng=np.random.default_rng(seed))
        n_actions = len(all_actions)
        ep_rewards = []
        for _ in range(n_ep):
            obs, state = env.reset()
            feat = get_features(env, n_stations)
            ep_r = 0.0
            while not state.done:
                mask = get_valid_mask(env, all_actions, n_actions)
                probs = policy_probs(feat, W, mask)
                a_idx = int(np.argmax(probs))
                obs, r, done, _ = env.step(all_actions[a_idx])
                feat = get_features(env, n_stations)
                state = env._state
                ep_r += r
            ep_rewards.append(ep_r)
        arr = np.array(ep_rewards)
        results[seed] = {"reward": float(arr.mean()), "std": float(arr.std())}
    return results

# ── treinamento REINFORCE ─────────────────────────────────────────────────────
def train_reinforce(sim, env_tr, env_ev, n_stations,
                    n_episodes=N_TRAIN, W_start=None, ep_offset=0, lr=LR):
    all_actions = [(d, delta) for d in range(n_stations) for delta in DELTAS]
    n_actions   = len(all_actions)
    n_features  = n_stations * 2 + 2

    rng = np.random.default_rng(TRAIN_SEED + ep_offset)
    W   = copy_weights(W_start) if W_start else init_weights(n_features, n_actions, rng)
    opt = Adam(lr=lr)

    baseline   = 0.0          # EMA do retorno total do episódio
    history    = []
    t0         = time.time()

    for ep in range(n_episodes):
        # ── coleta de trajetória ──────────────────────────────────────────
        obs, state = env_tr.reset()
        trajectory = []          # lista de (feat, a_idx, r, mask)

        while not state.done:
            feat = get_features(env_tr, n_stations)
            mask = get_valid_mask(env_tr, all_actions, n_actions)
            probs = policy_probs(feat, W, mask)
            # amostra estocástica durante treinamento
            a_idx = int(rng.choice(n_actions, p=probs))
            obs, r, done, _ = env_tr.step(all_actions[a_idx])
            trajectory.append((feat.copy(), a_idx, r, mask.copy()))
            state = env_tr._state

        # ── calcula retornos G_t ─────────────────────────────────────────
        T = len(trajectory)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + GAMMA * G
            returns[t] = G

        ep_return = float(returns[0])

        # ── normaliza retornos (reduz variância do Poisson) ──────────────
        ret_mean = returns.mean()
        ret_std  = returns.std() + 1e-8
        returns_norm = (returns - ret_mean) / ret_std

        # ── atualiza baseline EMA ────────────────────────────────────────
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_return

        # ── acumula gradientes ao longo da trajetória ────────────────────
        accum = {k: np.zeros_like(v) for k, v in W.items()}
        for t, (feat, a_idx, r, mask) in enumerate(trajectory):
            advantage = float(returns_norm[t])
            grads = policy_gradient(feat, W, a_idx, advantage, mask)
            for k in accum:
                accum[k] += grads[k]

        # média por passo + gradient clipping
        for k in accum:
            accum[k] /= T
        accum = clip_grads(accum, max_norm=1.0)

        opt.step(W, accum)

        # ── avaliação periódica ──────────────────────────────────────────
        if (ep + 1) % EVAL_EVERY == 0:
            eval_r = eval_policy(env_ev, W, all_actions, n_stations, n_ep=EVAL_EP)
            elapsed = time.time() - t0
            history.append({"ep": ep+1+ep_offset, "eval_r": eval_r,
                            "baseline": round(baseline, 2)})
            print(f"  ep {ep+1+ep_offset:6d} | eval_r={eval_r:7.1f} "
                  f"| baseline={baseline:7.1f} | t={elapsed:.0f}s")

    return W, history

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["train","eval"], default="train")
    parser.add_argument("--stations",  type=int, required=True)
    parser.add_argument("--resume",    type=str, default=None)
    parser.add_argument("--ep-offset", type=int, default=0)
    parser.add_argument("--episodes",  type=int, default=N_TRAIN)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--weights",   type=str, default=None)
    args = parser.parse_args()
    n = args.stations

    sim_mod, stations, env_tr, env_ev = load_sim_and_env(n)
    all_actions = [(d, delta) for d in range(n) for delta in DELTAS]

    # baselines
    print(f"\n=== BASELINES — {n} estações ===")
    for name, pol in [("Aleatória", sim.RandomPolicy(np.random.default_rng(123))),
                      ("Gulosa",    sim.GreedyPolicy())]:
        rs = [sim.run_episode(env_ev, pol)["total_reward"] for _ in range(100)]
        print(f"  {name}: {np.mean(rs):.2f} ±{np.std(rs):.2f}")

    if args.mode == "eval":
        assert args.weights, "--weights obrigatório no modo eval"
        data = np.load(args.weights)
        W = {k: data[k] for k in data.files}
        ms = eval_multiseed(W, n, all_actions, n_ep=100)
        mean_r = np.mean([v["reward"] for v in ms.values()])
        std_r  = np.mean([v["std"]    for v in ms.values()])
        print(f"REINFORCE {n}st (multi-seed): {mean_r:.2f} ±{std_r:.2f}")
    else:
        W_start = None
        if args.resume:
            data = np.load(args.resume)
            W_start = {k: data[k] for k in data.files}
            print(f"Carregado: {args.resume} (ep_offset={args.ep_offset})")

        W, hist = train_reinforce(sim_mod, env_tr, env_ev, n,
                                  n_episodes=args.episodes,
                                  W_start=W_start,
                                  ep_offset=args.ep_offset,
                                  lr=args.lr)

        total = args.ep_offset + args.episodes
        save  = os.path.join(RESULTS_DIR, f"pg_{n}st_ep{total}.npz")
        np.savez(save, **W)

        ms = eval_multiseed(W, n, all_actions, n_ep=100)
        mean_r = np.mean([v["reward"] for v in ms.values()])
        std_r  = np.mean([v["std"]    for v in ms.values()])
        print(f"\nREINFORCE {n}st ep{total} (multi-seed): {mean_r:.2f} ±{std_r:.2f}")

        out = {"n_stations": n, "n_episodes": total,
               "multiseed": {str(k): v for k, v in ms.items()},
               "history": hist}
        json_path = os.path.join(RESULTS_DIR, f"pg_{n}st_ep{total}.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Pesos: {save}\nJSON : {json_path}")
