"""
06_improved_qlearning.py
Q-Learning tabular com melhorias sobre 05_qlearning.py:

  Melhoria 1 — Espaco de acoes fixo (fix aliasing):
    Todas as 60 acoes (12 destinos × 5 deltas) representadas sempre com
    indices globais fixos. Acoes invalidas sao mascaradas com -inf no argmax.
    Elimina bug onde obs_keys identicas tinham action-lists diferentes.

  Melhoria 2 — Q(λ) com eligibility traces (Peng's Q(λ)):
    Propagacao de recompensa ao longo dos 32 passos do episodio via traces
    decaindo com fator γ·λ. Lambda configuravel; λ=0 recupera Q(0) padrao.

  Melhoria 3 — Reward shaping por desequilibrio de clusters:
    Bonus intermediario proporcional a reducao do desequilibrio entre clusters
    a cada passo. Coeficiente η configuravel; η=0 desativa o shaping.

Uso:
    python scripts/06_improved_qlearning.py                     # todos os melhoramentos, L=2
    python scripts/06_improved_qlearning.py --occ-levels 4      # L=4
    python scripts/06_improved_qlearning.py --lam 0.0 --eta 0.0 # apenas fix aliasing
    python scripts/06_improved_qlearning.py --episodes 20000
"""

import os, sys, time, pickle, argparse, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARAMETROS
# ─────────────────────────────────────────────────────────────────────────────
ALPHA        = 0.1
GAMMA        = 1.0
LAMBDA       = 0.9    # eligibility trace decay (0 = Q-Learning padrao)
ETA          = 0.3    # reward shaping weight (0 = sem shaping)
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY_EP = 0.80
N_TRAIN      = 10_000
EVAL_EVERY   = 1000
EVAL_EPISODES= 30
SEED_TRAIN   = 42
SEED_EVAL    = 123
N_STATIONS   = 12
N_DELTAS     = 5
DELTAS       = [-10, -5, 0, 5, 10]

# Espaco de acoes FIXO: indice global para (dest, delta)
ALL_ACTIONS  = [(dest, delta) for dest in range(N_STATIONS) for delta in DELTAS]
N_ACTIONS    = len(ALL_ACTIONS)   # 60
ACTION_INDEX = {a: i for i, a in enumerate(ALL_ACTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE com espaco fixo de 60 acoes
# ─────────────────────────────────────────────────────────────────────────────
class QTable:
    def __init__(self, seed: int = 0):
        self._table: dict[tuple, np.ndarray] = {}
        self.updates = 0
        self._rng = np.random.default_rng(seed)

    def get_q(self, key: tuple) -> np.ndarray:
        if key not in self._table:
            # Ruido uniforme pequeno para quebrar empates no argmax
            self._table[key] = self._rng.uniform(-0.01, 0.01, N_ACTIONS)
        return self._table[key]

    def update(self, key: tuple, action_idx: int, delta_td: float, alpha: float):
        """Atualiza Q(s,a) += alpha * delta_td."""
        q = self.get_q(key)
        q[action_idx] += alpha * delta_td
        self.updates += 1

    def update_trace(self, key: tuple, trace: np.ndarray, delta_td: float, alpha: float):
        """Atualiza todos os Q(s,a) proporcional ao trace de elegibilidade."""
        q = self.get_q(key)
        q += alpha * delta_td * trace
        self.updates += 1

    def best_action(self, key: tuple, valid_mask: np.ndarray) -> tuple[int, int]:
        """Retorna (action_idx, best action) usando mascara de acoes validas."""
        q = self.get_q(key).copy()
        q[~valid_mask] = -np.inf
        idx = int(np.argmax(q))
        return idx, ALL_ACTIONS[idx]

    @property
    def n_states_visited(self) -> int:
        return len(self._table)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"table": dict(self._table), "updates": self.updates}, f)
        print(f"  Q-table salva: {path} ({self.n_states_visited} estados, {self.updates} updates)")

    @classmethod
    def load(cls, path: str) -> "QTable":
        with open(path, "rb") as f:
            data = pickle.load(f)
        qt = cls()
        qt._table  = data["table"]
        qt.updates = data.get("updates", 0)
        print(f"  Q-table carregada: {path} ({qt.n_states_visited} estados)")
        return qt


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: mascara de acoes validas com indices globais fixos
# ─────────────────────────────────────────────────────────────────────────────
def get_valid_mask(env) -> np.ndarray:
    """Retorna array booleano [N_ACTIONS] com True nas acoes validas."""
    valid_set = set(map(tuple, env.valid_actions()))
    return np.array([a in valid_set for a in ALL_ACTIONS])


# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING: desequilibrio de clusters
# ─────────────────────────────────────────────────────────────────────────────
def cluster_imbalance(env) -> float:
    """
    Soma de |occ_ratio_cluster - 0.5| para cada cluster.
    Valor 0 = balanceamento perfeito; maximo = 0.5 * n_clusters.
    """
    cluster_occ = defaultdict(float)
    cluster_cap = defaultdict(float)
    occ = env._state.occupancy   # array[n_stations]
    for i, s in enumerate(env.stations):
        cluster_occ[s.cluster] += occ[i]
        cluster_cap[s.cluster] += s.cap
    imbal = 0.0
    for c in cluster_occ:
        if cluster_cap[c] > 0:
            ratio = cluster_occ[c] / cluster_cap[c]
            imbal += abs(ratio - 0.5)
    return imbal


# ─────────────────────────────────────────────────────────────────────────────
# POLITICA epsilon-greedy com espaco fixo
# ─────────────────────────────────────────────────────────────────────────────
class QLearningPolicy:
    def __init__(self, qtable: QTable, epsilon: float = 0.0,
                 rng: np.random.Generator = None):
        self.qtable  = qtable
        self.epsilon = epsilon
        self.rng     = rng or np.random.default_rng(0)

    def select_action(self, env, obs: np.ndarray) -> tuple[tuple, int, np.ndarray]:
        """Retorna (action, action_idx, valid_mask)."""
        key        = env.obs_to_key(obs)
        valid_mask = get_valid_mask(env)

        if self.rng.random() < self.epsilon:
            valid_idxs = np.where(valid_mask)[0]
            idx = int(self.rng.choice(valid_idxs))
        else:
            idx, _ = self.qtable.best_action(key, valid_mask)

        return ALL_ACTIONS[idx], idx, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# AVALIACAO
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(env, policy: QLearningPolicy, n_episodes: int) -> tuple[float, float]:
    old_eps      = policy.epsilon
    policy.epsilon = 0.0
    rewards, unmets = [], []
    for _ in range(n_episodes):
        obs, state = env.reset()
        ep_r, ep_u = 0.0, 0
        while not state.done:
            action, _, _ = policy.select_action(env, obs)
            obs, r, done, info = env.step(action)
            state  = env._state
            ep_r  += r
            ep_u  += info.unmet_departures + info.unmet_arrivals
        rewards.append(ep_r)
        unmets.append(ep_u)
    policy.epsilon = old_eps
    return float(np.mean(rewards)), float(np.mean(unmets))


# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO com Q(λ) e reward shaping
# ─────────────────────────────────────────────────────────────────────────────
def train(env, n_episodes: int = N_TRAIN, lam: float = LAMBDA,
          eta: float = ETA, verbose: bool = True,
          qt_init: QTable = None, eps_init: float = None,
          ep_offset: int = 0, decay_total_ep: int = None) -> tuple[QTable, dict]:

    qt     = qt_init if qt_init is not None else QTable()
    rng    = np.random.default_rng(SEED_TRAIN)
    eps0   = eps_init if eps_init is not None else EPS_START
    policy = QLearningPolicy(qt, epsilon=eps0, rng=rng)

    total_for_decay  = decay_total_ep if decay_total_ep is not None else n_episodes
    global_decay_end = int(total_for_decay * EPS_DECAY_EP)
    remaining_decay  = max(0, global_decay_end - ep_offset)
    decay_end        = min(n_episodes, remaining_decay)
    eps_delta        = (eps0 - EPS_END) / max(decay_end, 1)

    history = {"train_rewards": [], "eval_rewards": [], "eval_unmet": [],
               "eval_episodes": [], "epsilon_log": [], "states_visited": []}

    t_start = time.time()
    if verbose:
        print(f"\nTreinando {n_episodes:,} episodios | λ={lam} η={eta} alpha={ALPHA}")
        print(f"  Epsilon: {eps0:.3f} -> {EPS_END} em {decay_end:,} eps\n")

    for ep in range(n_episodes):
        obs, state = env.reset()
        ep_reward  = 0.0

        # Eligibility traces para este episodio: {key: array[N_ACTIONS]}
        traces: dict[tuple, np.ndarray] = {}

        while not state.done:
            key        = env.obs_to_key(obs)
            action, act_idx, valid_mask = policy.select_action(env, obs)

            # Imbalance antes (para reward shaping)
            imbal_before = cluster_imbalance(env) if eta > 0 else 0.0

            obs_next, reward, done, _ = env.step(action)
            state = env._state

            # Reward shaping
            if eta > 0:
                imbal_after = cluster_imbalance(env)
                reward = reward + eta * (imbal_before - imbal_after)

            ep_reward += reward

            # Q-Learning update com eligibility traces (Peng's Q(λ))
            key_next    = env.obs_to_key(obs_next)
            mask_next   = get_valid_mask(env) if not done else np.zeros(N_ACTIONS, dtype=bool)
            q_next_max  = 0.0
            if not done:
                q_next_max = np.max(qt.get_q(key_next)[mask_next]) if mask_next.any() else 0.0

            td_error = reward + GAMMA * q_next_max - qt.get_q(key)[act_idx]

            if lam > 0:
                # Acumular trace para (key, act_idx)
                if key not in traces:
                    traces[key] = np.zeros(N_ACTIONS)
                traces[key][act_idx] += 1.0

                # Atualizar todos os estados com trace acumulado
                for k, e in traces.items():
                    qt.get_q(k)[:] += ALPHA * td_error * e
                    qt.updates += 1

                # Decair traces
                decay_factor = GAMMA * lam
                for k in traces:
                    traces[k] *= decay_factor
                # Podar traces negligenciaveis para eficiencia
                traces = {k: e for k, e in traces.items() if np.max(e) > 1e-6}
            else:
                # Q(0): atualizacao padrao
                qt.update(key, act_idx, td_error, ALPHA)

            obs = obs_next

        history["train_rewards"].append(ep_reward)

        # Decaimento de epsilon
        if ep < decay_end:
            policy.epsilon = max(EPS_END, policy.epsilon - eps_delta)
        else:
            policy.epsilon = EPS_END

        # Avaliacao periodica
        if (ep + 1) % EVAL_EVERY == 0:
            eval_r, eval_u = evaluate(env, policy, EVAL_EPISODES)
            history["eval_rewards"].append(eval_r)
            history["eval_unmet"].append(eval_u)
            history["eval_episodes"].append(ep_offset + ep + 1)
            history["epsilon_log"].append(policy.epsilon)
            history["states_visited"].append(qt.n_states_visited)
            if verbose:
                elapsed = time.time() - t_start
                pct = (ep + 1) / n_episodes * 100
                print(f"  ep {ep+1:>6,}/{n_episodes:,} ({pct:4.0f}%)  "
                      f"eps={policy.epsilon:.3f}  "
                      f"train_r={np.mean(history['train_rewards'][-EVAL_EVERY:]):6.1f}  "
                      f"eval_r={eval_r:6.1f}  unmet={eval_u:.1f}  "
                      f"states={qt.n_states_visited:,}  t={elapsed:.0f}s")

    return qt, history


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: carrega simulador com N_OCC_LEVELS configuravel
# ─────────────────────────────────────────────────────────────────────────────
def load_env(occ_levels: int = 2):
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader(
        "sim", os.path.join(os.path.dirname(__file__), "04_simulator.py"))
    spec = importlib.util.spec_from_loader("sim", loader)
    sim  = importlib.util.module_from_spec(spec)
    loader.exec_module(sim)
    sim.N_OCC_LEVELS = occ_levels
    stations  = sim.load_environment_data()
    env_train = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_TRAIN))
    env_eval  = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_EVAL))
    return env_train, env_eval, sim


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,   default=N_TRAIN)
    parser.add_argument("--lam",        type=float, default=LAMBDA,
                        help="Eligibility trace decay (0=Q-learning padrao)")
    parser.add_argument("--eta",        type=float, default=ETA,
                        help="Reward shaping weight (0=sem shaping)")
    parser.add_argument("--occ-levels", type=int,   default=2,
                        help="Niveis de ocupacao por cluster (2 ou 4)")
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--ep-done",    type=int,   default=0)
    parser.add_argument("--no-save",    action="store_true")
    parser.add_argument("--tag",        type=str,   default=None,
                        help="Sufixo para o nome do arquivo de saida")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env_train, env_eval, sim = load_env(args.occ_levels)

    state_space = args.occ_levels**4 * N_STATIONS * 4 * 32
    print(f"Espaco de estados (L={args.occ_levels}): {state_space:,}")
    print(f"Acoes fixas: {N_ACTIONS} | λ={args.lam} | η={args.eta}")

    # Baselines
    print("\n=== BASELINES (50 episodios) ===")
    baselines = {}
    for name, pol in [("Aleatoria", sim.RandomPolicy(np.random.default_rng(0))),
                      ("Gulosa",    sim.GreedyPolicy())]:
        rewards, unmets = [], []
        for _ in range(50):
            res = sim.run_episode(env_eval, pol)
            rewards.append(res["total_reward"])
            unmets.append(res["unmet_dep"] + res["unmet_arr"])
        baselines[name] = {"reward": float(np.mean(rewards)),
                           "unmet":  float(np.mean(unmets)),
                           "std":    float(np.std(rewards))}
        print(f"  {name:<12}: reward={np.mean(rewards):7.1f} (+/-{np.std(rewards):.1f})"
              f"  unmet={np.mean(unmets):.1f}/ep")

    # Treino
    qt_init = QTable.load(args.resume) if args.resume else None
    qt, history = train(env_train, n_episodes=args.episodes,
                        lam=args.lam, eta=args.eta, verbose=True,
                        qt_init=qt_init, ep_offset=args.ep_done,
                        decay_total_ep=args.ep_done + args.episodes)

    # Avaliacao final
    print("\n=== AVALIACAO FINAL (100 episodios, ε=0) ===")
    ql_pol = QLearningPolicy(qt, epsilon=0.0, rng=np.random.default_rng(SEED_EVAL))
    rewards, unmets = [], []
    for _ in range(100):
        obs, state = env_eval.reset()
        ep_r, ep_u = 0.0, 0
        while not state.done:
            action, _, _ = ql_pol.select_action(env_eval, obs)
            obs, r, done, info = env_eval.step(action)
            state  = env_eval._state
            ep_r  += r
            ep_u  += info.unmet_departures + info.unmet_arrivals
        rewards.append(ep_r); unmets.append(ep_u)

    ql_r, ql_std, ql_u = np.mean(rewards), np.std(rewards), np.mean(unmets)
    base_r = baselines["Aleatoria"]["reward"]
    melhoria = (ql_r - base_r) / abs(base_r) * 100

    print(f"  Q-Learning    : reward={ql_r:7.1f} (+/-{ql_std:.1f})  unmet={ql_u:.1f}/ep")
    print(f"\n  Melhoria vs Aleatoria: {melhoria:+.1f}%")
    print(f"  Estados visitados: {qt.n_states_visited:,}/{state_space:,} ({qt.n_states_visited/state_space*100:.1f}%)")

    print("\n=== TABELA COMPARATIVA ===")
    print(f"  {'Politica':<14} {'Reward':>10} {'Unmet/ep':>10} {'vs Aleat.':>10}")
    print(f"  {'-'*48}")
    base_r_val = baselines['Aleatoria']['reward']
    for name, d in {**baselines, "Q-Learning": {"reward": ql_r, "unmet": ql_u}}.items():
        imp = (d['reward'] - base_r_val) / abs(base_r_val) * 100
        print(f"  {name:<14} {d['reward']:>10.1f} {d['unmet']:>10.1f} {imp:>+9.1f}%")

    # Salvar
    if not args.no_save:
        tag  = args.tag or f"L{args.occ_levels}_lam{args.lam}_eta{args.eta}_{args.episodes//1000}k"
        path = os.path.join(RESULTS_DIR, f"qtable_improved_{tag}.pkl")
        qt.save(path)
        summary = {
            "tag": tag, "occ_levels": args.occ_levels,
            "lam": args.lam, "eta": args.eta,
            "episodes": args.episodes,
            "baselines": baselines,
            "qlearning": {"reward": ql_r, "unmet": ql_u, "std": ql_std},
            "states_visited": qt.n_states_visited,
            "state_space": state_space,
            "coverage_pct": qt.n_states_visited / state_space * 100,
            "history": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                        for k, v in history.items()},
        }
        hist_path = path.replace(".pkl", "_summary.json")
        with open(hist_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Resultados salvos em: {path}")


if __name__ == "__main__":
    main()
