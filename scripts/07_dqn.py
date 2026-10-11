"""
07_dqn.py
Deep Q-Network (DQN) para o BSRP — implementação em numpy puro.

Diferenciais em relação ao Q-Learning tabular (05_qlearning.py):
  - State features contínuas: occupancy ratio por estação (12), one-hot de
    posição (12), load ratio (1), tempo normalizado (1) → 26 entradas
  - Rede 26→64→64→60: generaliza para estados nunca visitados
  - Experience replay (10k transições): quebra correlação temporal
  - Target network: estabiliza os alvos de treinamento
  - Espaço de ações fixo (60) com máscara de inválidas

Uso:
    python scripts/07_dqn.py                     # 10k episódios
    python scripts/07_dqn.py --episodes 20000
    python scripts/07_dqn.py --eval-only results/dqn_weights.npz
"""

import os, sys, time, argparse, json
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────
LR            = 3e-4      # taxa de aprendizado (Adam)
GAMMA         = 1.0       # desconto (episódio finito)
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY_EP  = 0.80
BATCH_SIZE    = 64
BUFFER_SIZE   = 10_000
TARGET_SYNC   = 50        # sincronizar target network a cada N episódios
TRAIN_EVERY   = 4         # atualizar rede a cada N passos
N_TRAIN       = 10_000
EVAL_EVERY    = 500
EVAL_EPISODES = 30
SEED_TRAIN    = 42
SEED_EVAL     = 123
N_STATIONS    = 12
DELTAS        = [-10, -5, 0, 5, 10]
ALL_ACTIONS   = [(d, delta) for d in range(N_STATIONS) for delta in DELTAS]
N_ACTIONS     = len(ALL_ACTIONS)   # 60
N_FEATURES    = 26  # 12 occ_ratio + 12 pos_onehot + 1 load_ratio + 1 time_norm


# ─────────────────────────────────────────────────────────────────────────────
# EXTRAÇÃO DE FEATURES CONTÍNUAS
# ─────────────────────────────────────────────────────────────────────────────
def get_features(env) -> np.ndarray:
    """
    Extrai vetor de features contínuas do estado atual do ambiente.
    Não usa discretização — passa valores reais para a rede.
    """
    state = env._state
    occ   = state.occupancy          # array[12]: bikes em cada estação

    # 12 occupancy ratios (0..1)
    occ_ratio = np.array([occ[i] / max(env.stations[i].cap, 1)
                          for i in range(N_STATIONS)], dtype=np.float32)

    # 12 posição do veículo (one-hot)
    pos_onehot = np.zeros(N_STATIONS, dtype=np.float32)
    pos_onehot[state.vehicle_pos] = 1.0

    # 1 carga do veículo normalizada (0..1)
    load_ratio = np.float32(state.vehicle_load / 20.0)  # VEHICLE_CAP=20

    # 1 tempo normalizado (0..1)
    time_norm = np.float32(state.t / 32.0)  # N_INTERVALS=32

    return np.concatenate([occ_ratio, pos_onehot, [load_ratio, time_norm]])


def get_valid_mask(env) -> np.ndarray:
    valid_set = set(map(tuple, env.valid_actions()))
    return np.array([a in valid_set for a in ALL_ACTIONS], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# REDE NEURAL (numpy): 26 → 64 → 64 → 60
# ─────────────────────────────────────────────────────────────────────────────
def init_weights(rng: np.random.Generator) -> dict:
    """Inicializa pesos com He (para ReLU)."""
    def he(fan_in, fan_out):
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)

    return {
        "W1": he(N_FEATURES, 64), "b1": np.zeros(64, dtype=np.float32),
        "W2": he(64, 64),         "b2": np.zeros(64, dtype=np.float32),
        "W3": he(64, N_ACTIONS),  "b3": np.zeros(N_ACTIONS, dtype=np.float32),
    }


def copy_weights(src: dict) -> dict:
    return {k: v.copy() for k, v in src.items()}


def forward(x: np.ndarray, W: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass. Retorna (q_values, h1, h2) para backprop."""
    h1 = np.maximum(0.0, x @ W["W1"] + W["b1"])
    h2 = np.maximum(0.0, h1 @ W["W2"] + W["b2"])
    q  = h2 @ W["W3"] + W["b3"]
    return q, h1, h2


def backward(x: np.ndarray, h1: np.ndarray, h2: np.ndarray,
             q_pred: np.ndarray, q_target: np.ndarray,
             W: dict) -> dict:
    """Backprop MSE loss. Retorna gradientes."""
    N  = x.shape[0]
    dq = 2.0 * (q_pred - q_target) / N          # dL/dq, shape [N, 60]

    dW3 = h2.T @ dq                              # [64, 60]
    db3 = dq.sum(axis=0)                         # [60]

    dh2 = (dq @ W["W3"].T) * (h2 > 0)           # [N, 64]
    dW2 = h1.T @ dh2                             # [64, 64]
    db2 = dh2.sum(axis=0)                        # [64]

    dh1 = (dh2 @ W["W2"].T) * (h1 > 0)          # [N, 64]
    dW1 = x.T @ dh1                              # [26, 64]
    db1 = dh1.sum(axis=0)                        # [64]

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


# ─────────────────────────────────────────────────────────────────────────────
# ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr: float = LR, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m: dict = {}
        self.v: dict = {}

    def step(self, W: dict, grads: dict) -> dict:
        self.t += 1
        for k in W:
            if k not in self.m:
                self.m[k] = np.zeros_like(W[k])
                self.v[k] = np.zeros_like(W[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            W[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return W


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a_idx, r, s_next, done, mask_next):
        self.buf.append((s.copy(), a_idx, r, s_next.copy(), done, mask_next.copy()))

    def sample(self, batch_size: int, rng: np.random.Generator):
        idxs = rng.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, sn, d, mn = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(sn, dtype=np.float32),
                np.array(d,  dtype=bool),
                np.array(mn, dtype=bool))

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO DQN
# ─────────────────────────────────────────────────────────────────────────────
def train(env_train, env_eval, n_episodes: int = N_TRAIN,
          verbose: bool = True) -> tuple[dict, dict]:

    rng   = np.random.default_rng(SEED_TRAIN)
    W     = init_weights(rng)          # pesos principais
    W_tgt = copy_weights(W)            # target network
    opt   = Adam(lr=LR)
    buf   = ReplayBuffer(BUFFER_SIZE)

    decay_end = int(n_episodes * EPS_DECAY_EP)
    eps_delta = (EPS_START - EPS_END) / max(decay_end, 1)
    epsilon   = EPS_START

    history = {"train_rewards": [], "eval_rewards": [], "eval_unmet": [],
               "eval_episodes": [], "epsilon_log": [], "losses": []}

    step_count = 0
    t_start    = time.time()

    if verbose:
        print(f"\nDQN: {n_episodes:,} episódios | lr={LR} batch={BATCH_SIZE} "
              f"buf={BUFFER_SIZE} target_sync={TARGET_SYNC}")
        print(f"  Arquitetura: {N_FEATURES}→64→64→{N_ACTIONS} | "
              f"parâmetros: {N_FEATURES*64+64+64*64+64+64*N_ACTIONS+N_ACTIONS:,}\n")

    for ep in range(n_episodes):
        obs, state = env_train.reset()
        feat       = get_features(env_train)
        ep_reward  = 0.0

        while not state.done:
            mask = get_valid_mask(env_train)

            # epsilon-greedy com espaço de ações mascarado
            if rng.random() < epsilon:
                valid_idxs = np.where(mask)[0]
                a_idx = int(rng.choice(valid_idxs))
            else:
                q, _, _ = forward(feat[np.newaxis], W)
                q_masked = q[0].copy()
                q_masked[~mask] = -np.inf
                a_idx = int(np.argmax(q_masked))

            action = ALL_ACTIONS[a_idx]
            obs_next, reward, done, _ = env_train.step(action)
            feat_next  = get_features(env_train)
            mask_next  = get_valid_mask(env_train) if not done else np.zeros(N_ACTIONS, dtype=bool)
            state      = env_train._state

            buf.push(feat, a_idx, reward, feat_next, done, mask_next)
            feat       = feat_next
            ep_reward += reward
            step_count += 1

            # Treina rede a cada TRAIN_EVERY passos
            if len(buf) >= BATCH_SIZE and step_count % TRAIN_EVERY == 0:
                s_b, a_b, r_b, sn_b, d_b, mn_b = buf.sample(BATCH_SIZE, rng)

                # Predict Q-values para estados atuais e próximos
                q_pred, h1, h2    = forward(s_b, W)
                q_next_tgt, _, _  = forward(sn_b, W_tgt)   # target network

                # Double DQN: seleciona ação com W principal, avalia com W_tgt
                q_next_main, _, _ = forward(sn_b, W)

                # Alvo de Bellman
                q_target = q_pred.copy()
                for i in range(BATCH_SIZE):
                    if d_b[i]:
                        td_target = r_b[i]
                    else:
                        # Máscara de ações válidas no próximo estado
                        q_nm = q_next_main[i].copy()
                        q_nm[~mn_b[i]] = -np.inf
                        best_next_a = int(np.argmax(q_nm))
                        td_target = r_b[i] + GAMMA * q_next_tgt[i, best_next_a]

                    q_target[i, a_b[i]] = td_target

                grads = backward(s_b, h1, h2, q_pred, q_target, W)
                W     = opt.step(W, grads)

                loss = float(np.mean((q_pred - q_target) ** 2))
                history["losses"].append(loss)

        history["train_rewards"].append(ep_reward)

        # Atualiza target network
        if (ep + 1) % TARGET_SYNC == 0:
            W_tgt = copy_weights(W)

        # Decai epsilon
        if ep < decay_end:
            epsilon = max(EPS_END, epsilon - eps_delta)
        else:
            epsilon = EPS_END

        # Avaliação periódica
        if (ep + 1) % EVAL_EVERY == 0:
            eval_r, eval_u = evaluate(env_eval, W, EVAL_EPISODES)
            history["eval_rewards"].append(eval_r)
            history["eval_unmet"].append(eval_u)
            history["eval_episodes"].append(ep + 1)
            history["epsilon_log"].append(epsilon)

            if verbose:
                elapsed = time.time() - t_start
                pct     = (ep + 1) / n_episodes * 100
                avg_loss = np.mean(history["losses"][-100:]) if history["losses"] else 0
                print(f"  ep {ep+1:>6,}/{n_episodes:,} ({pct:4.0f}%)  "
                      f"eps={epsilon:.3f}  "
                      f"train_r={np.mean(history['train_rewards'][-EVAL_EVERY:]):6.1f}  "
                      f"eval_r={eval_r:6.1f}  unmet={eval_u:.1f}  "
                      f"loss={avg_loss:.4f}  t={elapsed:.0f}s")

    return W, history


# ─────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(env, W: dict, n_episodes: int,
             rng: np.random.Generator = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(SEED_EVAL)
    rewards, unmets = [], []
    for _ in range(n_episodes):
        obs, state = env.reset()
        feat = get_features(env)
        ep_r, ep_u = 0.0, 0
        while not state.done:
            mask = get_valid_mask(env)
            q, _, _ = forward(feat[np.newaxis], W)
            q_masked = q[0].copy()
            q_masked[~mask] = -np.inf
            a_idx  = int(np.argmax(q_masked))
            action = ALL_ACTIONS[a_idx]
            obs, r, done, info = env.step(action)
            feat   = get_features(env)
            state  = env._state
            ep_r  += r
            ep_u  += info.unmet_departures + info.unmet_arrivals
        rewards.append(ep_r)
        unmets.append(ep_u)
    return float(np.mean(rewards)), float(np.mean(unmets))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: carrega simulador
# ─────────────────────────────────────────────────────────────────────────────
def load_env():
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader(
        "sim", os.path.join(os.path.dirname(__file__), "04_simulator.py"))
    spec = importlib.util.spec_from_loader("sim", loader)
    sim  = importlib.util.module_from_spec(spec)
    loader.exec_module(sim)
    stations  = sim.load_environment_data()
    env_train = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_TRAIN))
    env_eval  = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_EVAL))
    return env_train, env_eval, sim


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=N_TRAIN)
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Caminho para dqn_weights.npz — pula treino")
    parser.add_argument("--no-save",   action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env_train, env_eval, sim = load_env()

    # Baselines
    print("=== BASELINES (100 episódios) ===")
    baselines = {}
    for name, pol in [("Aleatoria", sim.RandomPolicy(np.random.default_rng(0))),
                      ("Gulosa",    sim.GreedyPolicy())]:
        rewards, unmets = [], []
        for _ in range(100):
            res = sim.run_episode(env_eval, pol)
            rewards.append(res["total_reward"])
            unmets.append(res["unmet_dep"] + res["unmet_arr"])
        baselines[name] = {"reward": float(np.mean(rewards)),
                           "unmet":  float(np.mean(unmets)),
                           "std":    float(np.std(rewards))}
        print(f"  {name:<12}: reward={np.mean(rewards):7.1f} (+/-{np.std(rewards):.1f})"
              f"  unmet={np.mean(unmets):.1f}/ep")

    # Treino ou carga
    if args.eval_only:
        data = np.load(args.eval_only)
        W = {k: data[k] for k in data.files}
    else:
        W, history = train(env_train, env_eval, n_episodes=args.episodes)

    # Avaliação final
    print("\n=== AVALIAÇÃO FINAL DQN (100 episódios, ε=0) ===")
    ql_r, ql_u = evaluate(env_eval, W, 100,
                          rng=np.random.default_rng(SEED_EVAL))
    ql_std = 0.0
    rewards_final = []
    for _ in range(100):
        obs, state = env_eval.reset()
        feat = get_features(env_eval)
        ep_r = 0.0
        while not state.done:
            mask = get_valid_mask(env_eval)
            q, _, _ = forward(feat[np.newaxis], W)
            q[0][~mask] = -np.inf
            a_idx  = int(np.argmax(q[0]))
            obs, r, done, info = env_eval.step(ALL_ACTIONS[a_idx])
            feat  = get_features(env_eval)
            state = env_eval._state
            ep_r += r
        rewards_final.append(ep_r)
    ql_r   = float(np.mean(rewards_final))
    ql_std = float(np.std(rewards_final))

    base_r = baselines["Aleatoria"]["reward"]
    melhoria = (ql_r - base_r) / abs(base_r) * 100

    print(f"  DQN          : reward={ql_r:7.1f} (+/-{ql_std:.1f})  unmet={ql_u:.1f}/ep")
    print(f"\n  Melhoria vs Aleatória: {melhoria:+.1f}%")

    print("\n=== TABELA COMPARATIVA ===")
    print(f"  {'Política':<14} {'Reward':>10} {'Unmet/ep':>10} {'vs Aleat.':>10}")
    print(f"  {'-'*48}")
    for name, d in {**baselines,
                    "DQN": {"reward": ql_r, "unmet": ql_u}}.items():
        imp = (d["reward"] - base_r) / abs(base_r) * 100
        print(f"  {name:<14} {d['reward']:>10.1f} {d['unmet']:>10.1f} {imp:>+9.1f}%")

    # Salvar
    if not args.no_save and not args.eval_only:
        path = os.path.join(RESULTS_DIR, "dqn_weights.npz")
        np.savez(path, **W)
        print(f"\n  Pesos salvos: {path}")

        summary = {
            "episodes": args.episodes,
            "baselines": baselines,
            "dqn": {"reward": ql_r, "unmet": ql_u, "std": ql_std},
            "melhoria_pct": melhoria,
            "history": {k: (v if isinstance(v, list) else v.tolist())
                        for k, v in history.items()
                        if k != "losses"},
        }
        with open(os.path.join(RESULTS_DIR, "dqn_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Resumo salvo: {RESULTS_DIR}/dqn_summary.json")


if __name__ == "__main__":
    main()
