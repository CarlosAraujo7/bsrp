"""
05_qlearning.py
Q-Learning tabular com epsilon-greedy para o BSRP.

Treina um agente Q-Learning no ambiente BSRPEnv e compara com as politicas
baseline (aleatoria e gulosa).

Uso:
    python scripts/05_qlearning.py                  # treina e avalia (10k episodios)
    python scripts/05_qlearning.py --episodes 20000 # treino mais longo
    python scripts/05_qlearning.py --eval-only results/qtable.pkl  # so avalia
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

# Caminho base
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARAMETROS
# ─────────────────────────────────────────────────────────────────────────────
ALPHA        = 0.1     # taxa de aprendizado
GAMMA        = 1.0     # desconto (episodio finito, sem desconto)
EPS_START    = 1.0     # exploracao inicial
EPS_END      = 0.05    # exploracao minima
EPS_DECAY_EP = 0.80    # fracao dos episodios para decair epsilon (0 -> 1)
N_TRAIN      = 10_000  # episodios de treino
EVAL_EVERY   = 200     # avaliar a cada N episodios de treino
EVAL_EPISODES= 30      # episodios por avaliacao
SEED_TRAIN   = 42
SEED_EVAL    = 123


# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE
# ─────────────────────────────────────────────────────────────────────────────
class QTable:
    """
    Q-table esparsa: dicionario de (state_key -> array[n_actions]).
    Acoes sao indexadas pela lista retornada por env.valid_actions() no momento
    da consulta — o indice e determinístico dado o estado.
    """
    def __init__(self, n_actions_estimate: int = 36):
        self._table: dict[tuple, np.ndarray] = {}
        self._n_act = n_actions_estimate
        self.updates = 0

    def get_q(self, key: tuple, n_actions: int) -> np.ndarray:
        if key not in self._table:
            self._table[key] = np.zeros(n_actions)
        elif len(self._table[key]) != n_actions:
            old = self._table[key]
            new = np.zeros(n_actions)
            copy_len = min(len(old), n_actions)
            new[:copy_len] = old[:copy_len]
            self._table[key] = new
        return self._table[key]

    def update(self, key: tuple, action_idx: int, target: float, alpha: float,
               n_actions: int = 36):
        q = self.get_q(key, n_actions)   # garante inicializacao
        q[action_idx] += alpha * (target - q[action_idx])
        self.updates += 1

    def best_action_idx(self, key: tuple, n_actions: int) -> int:
        return int(np.argmax(self.get_q(key, n_actions)))

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
        qt.updates = data["updates"]
        print(f"  Q-table carregada: {path} ({qt.n_states_visited} estados)")
        return qt


# ─────────────────────────────────────────────────────────────────────────────
# POLITICA Q-LEARNING (epsilon-greedy)
# ─────────────────────────────────────────────────────────────────────────────
class QLearningPolicy:
    def __init__(self, qtable: QTable, epsilon: float = 0.0,
                 rng: np.random.Generator = None):
        self.qtable  = qtable
        self.epsilon = epsilon
        self.rng     = rng or np.random.default_rng(0)

    def select_action(self, env, obs: np.ndarray) -> tuple[int, int]:
        actions = env.valid_actions()
        key     = env.obs_to_key(obs)

        if self.rng.random() < self.epsilon:
            idx = self.rng.integers(len(actions))
        else:
            idx = self.qtable.best_action_idx(key, len(actions))
            idx = min(idx, len(actions) - 1)

        return actions[idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DE TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def train(env, n_episodes: int = N_TRAIN, verbose: bool = True,
          qt_init: QTable = None, eps_init: float = None,
          ep_offset: int = 0, decay_total_ep: int = None) -> tuple[QTable, dict]:
    """
    ep_offset:      episodios ja treinados antes desta chamada.
    eps_init:       epsilon inicial (None = EPS_START).
    qt_init:        Q-table pre-treinada para continuar treinamento.
    decay_total_ep: total de episodios planejados para o decay global.
                    Se None, usa n_episodes como base.
    """
    qt     = qt_init if qt_init is not None else QTable()
    rng    = np.random.default_rng(SEED_TRAIN)
    eps0   = eps_init if eps_init is not None else EPS_START
    policy = QLearningPolicy(qt, epsilon=eps0, rng=rng)

    # decay: quantos passos restam ate o fim do decay global
    total_for_decay  = decay_total_ep if decay_total_ep is not None else n_episodes
    global_decay_end = int(total_for_decay * EPS_DECAY_EP)
    remaining_decay  = max(0, global_decay_end - ep_offset)
    decay_end   = min(n_episodes, remaining_decay)
    eps_delta   = (eps0 - EPS_END) / max(decay_end, 1)

    history = {
        "train_rewards":   [],
        "eval_rewards":    [],
        "eval_unmet":      [],
        "eval_episodes":   [],
        "epsilon_log":     [],
        "states_visited":  [],
    }

    t_start = time.time()
    if verbose:
        print(f"\nTreinando por {n_episodes:,} episodios  (alpha={ALPHA}, gamma={GAMMA})")
        print(f"  Epsilon: {EPS_START} -> {EPS_END} em {decay_end:,} eps")
        print(f"  Avaliacao a cada {EVAL_EVERY} eps ({EVAL_EPISODES} ep/avaliacao)\n")

    for ep in range(n_episodes):
        obs, state = env.reset()
        ep_reward  = 0.0

        while not state.done:
            actions        = env.valid_actions()
            key            = env.obs_to_key(obs)
            action, act_idx = policy.select_action(env, obs)

            obs_next, reward, done, _ = env.step(action)
            state = env._state
            ep_reward += reward

            # Q-Learning update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s') - Q(s,a))
            key_next   = env.obs_to_key(obs_next)
            actions_next = env.valid_actions()
            q_next_max = np.max(qt.get_q(key_next, len(actions_next))) if not done else 0.0
            target     = reward + GAMMA * q_next_max
            qt.update(key, act_idx, target, ALPHA, n_actions=len(actions))

            obs = obs_next

        history["train_rewards"].append(ep_reward)

        # Decaimento de epsilon
        if ep < decay_end:
            policy.epsilon = max(EPS_END, policy.epsilon - eps_delta)
        else:
            policy.epsilon = EPS_END

        # Avaliacao periodica
        if (ep + 1) % EVAL_EVERY == 0:
            eval_r, eval_u = evaluate_policy(env, policy, EVAL_EPISODES, SEED_EVAL)
            history["eval_rewards"].append(eval_r)
            history["eval_unmet"].append(eval_u)
            history["eval_episodes"].append(ep + 1)
            history["epsilon_log"].append(policy.epsilon)
            history["states_visited"].append(qt.n_states_visited)

            if verbose:
                elapsed = time.time() - t_start
                pct     = (ep + 1) / n_episodes * 100
                print(f"  ep {ep+1:>6,}/{n_episodes:,} ({pct:4.0f}%)  "
                      f"eps={policy.epsilon:.3f}  "
                      f"train_r={np.mean(history['train_rewards'][-EVAL_EVERY:]):6.1f}  "
                      f"eval_r={eval_r:6.1f}  unmet={eval_u:.1f}  "
                      f"states={qt.n_states_visited:,}  "
                      f"t={elapsed:.0f}s")

    return qt, history


# ─────────────────────────────────────────────────────────────────────────────
# AVALIACAO
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(env, policy, n_episodes: int, seed: int = SEED_EVAL) -> tuple[float, float]:
    """Avalia politica com epsilon=0 (greedy puro). Retorna (reward_medio, unmet_medio)."""
    old_eps       = getattr(policy, "epsilon", 0)
    if hasattr(policy, "epsilon"):
        policy.epsilon = 0.0

    rewards, unmets = [], []
    for _ in range(n_episodes):
        obs, state = env.reset()
        ep_r = 0.0
        ep_u = 0
        while not state.done:
            action = policy.select_action(env, obs)
            if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], int) and not isinstance(action[0], tuple):
                # Politica retorna (action, idx) — pegar so a action
                pass
            obs, r, done, info = env.step(action[0] if isinstance(action[0], tuple) else action)
            state   = env._state
            ep_r   += r
            ep_u   += info.unmet_departures + info.unmet_arrivals
        rewards.append(ep_r)
        unmets.append(ep_u)

    if hasattr(policy, "epsilon"):
        policy.epsilon = old_eps

    return float(np.mean(rewards)), float(np.mean(unmets))


def run_baseline_eval(env, n_episodes: int = 100) -> dict:
    """Avalia politicas baseline para comparacao."""
    # import aqui para evitar circular
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "sim", os.path.join(os.path.dirname(__file__), "04_simulator.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    results = {}
    for name, policy in [("Aleatoria", sim.RandomPolicy(np.random.default_rng(0))),
                         ("Gulosa",    sim.GreedyPolicy())]:
        rewards, unmets = [], []
        for _ in range(n_episodes):
            res = sim.run_episode(env, policy)
            rewards.append(res["total_reward"])
            unmets.append(res["unmet_dep"] + res["unmet_arr"])
        results[name] = {
            "reward": float(np.mean(rewards)),
            "unmet":  float(np.mean(unmets)),
            "std":    float(np.std(rewards)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: envs separados para treino e avaliacao
# ─────────────────────────────────────────────────────────────────────────────
def load_env():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sim", os.path.join(os.path.dirname(__file__), "04_simulator.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    stations = sim.load_environment_data()
    env_train = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_TRAIN))
    env_eval  = sim.BSRPEnv(stations, rng=np.random.default_rng(SEED_EVAL))
    return env_train, env_eval, sim


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=N_TRAIN)
    parser.add_argument("--eval-only", type=str,  default=None,
                        help="Caminho para qtable.pkl — pula treino e so avalia")
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Caminho para qtable.pkl — continua treino a partir deste checkpoint")
    parser.add_argument("--ep-done",   type=int,  default=0,
                        help="Episodios ja treinados (para calculo correto de epsilon ao resumir)")
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Pula avaliacao de baselines (usa valores salvos se disponiveis)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env_train, env_eval, sim = load_env()

    # ── Baselines ──────────────────────────────────────────────────────────
    import json as _json
    _baseline_cache = os.path.join(RESULTS_DIR, "baselines.json")
    baselines = {}
    if args.no_baselines and os.path.exists(_baseline_cache):
        with open(_baseline_cache) as f:
            baselines = _json.load(f)
        print("=== BASELINES (carregados do cache) ===")
        for name, d in baselines.items():
            print(f"  {name:<12}: reward={d['reward']:7.1f} (+/-{d['std']:.1f})  unmet={d['unmet']:.1f}/ep")
    else:
        print("=== AVALIACAO BASELINES (100 episodios) ===")
        for name, policy in [("Aleatoria", sim.RandomPolicy(np.random.default_rng(0))),
                             ("Gulosa",    sim.GreedyPolicy())]:
            rewards, unmets = [], []
            for _ in range(100):
                res = sim.run_episode(env_eval, policy)
                rewards.append(res["total_reward"])
                unmets.append(res["unmet_dep"] + res["unmet_arr"])
            baselines[name] = {"reward": float(np.mean(rewards)),
                               "unmet":  float(np.mean(unmets)),
                               "std":    float(np.std(rewards))}
            print(f"  {name:<12}: reward={np.mean(rewards):7.1f} (+/-{np.std(rewards):.1f})  "
                  f"unmet={np.mean(unmets):.1f}/ep")
        # salvar cache
        with open(_baseline_cache, "w") as f:
            _json.dump(baselines, f, indent=2)
        print(f"  (baselines salvos em {_baseline_cache})")

    # ── Treino / carga ─────────────────────────────────────────────────────
    if args.eval_only:
        qt = QTable.load(args.eval_only)
        history = None
    elif args.resume:
        qt_loaded = QTable.load(args.resume)
        ep_done   = args.ep_done
        total_ep  = ep_done + args.episodes
        # Calcula epsilon no ponto de retomada (assumindo decay sobre total_ep)
        decay_end = int(total_ep * EPS_DECAY_EP)
        eps_delta = (EPS_START - EPS_END) / max(decay_end, 1)
        eps_at_resume = max(EPS_END, EPS_START - ep_done * eps_delta)
        print(f"\nRetomando treino: {ep_done:,} eps ja concluidos, "
              f"epsilon={eps_at_resume:.3f}, +{args.episodes:,} eps")
        qt, history = train(env_train, n_episodes=args.episodes,
                            qt_init=qt_loaded, eps_init=eps_at_resume,
                            ep_offset=ep_done)
        if not args.no_save:
            qt.save(os.path.join(RESULTS_DIR, "qtable.pkl"))
            import json
            hist_path = os.path.join(RESULTS_DIR, "history_resume.json")
            with open(hist_path, "w") as f:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in history.items()}, f)
            print(f"  Historico salvo: {hist_path}")
    else:
        qt, history = train(env_train, n_episodes=args.episodes)
        if not args.no_save:
            qt.save(os.path.join(RESULTS_DIR, "qtable.pkl"))
            np.save(os.path.join(RESULTS_DIR, "history.npy"), history)

    # ── Avaliacao final do agente ──────────────────────────────────────────
    print("\n=== AVALIACAO FINAL DO AGENTE Q-LEARNING (100 episodios, epsilon=0) ===")
    rng_eval = np.random.default_rng(SEED_EVAL)
    ql_policy = QLearningPolicy(qt, epsilon=0.0, rng=rng_eval)
    rewards, unmets = [], []
    for _ in range(100):
        obs, state = env_eval.reset()
        ep_r, ep_u = 0.0, 0
        while not state.done:
            action, _ = ql_policy.select_action(env_eval, obs)
            obs, r, done, info = env_eval.step(action)
            state  = env_eval._state
            ep_r  += r
            ep_u  += info.unmet_departures + info.unmet_arrivals
        rewards.append(ep_r)
        unmets.append(ep_u)

    ql_reward = np.mean(rewards)
    ql_unmet  = np.mean(unmets)
    ql_std    = np.std(rewards)
    print(f"  Q-Learning   : reward={ql_reward:7.1f} (+/-{ql_std:.1f})  unmet={ql_unmet:.1f}/ep")

    # ── Tabela comparativa ─────────────────────────────────────────────────
    print("\n=== COMPARACAO FINAL ===\n")
    print(f"  {'Politica':<14} {'Reward medio':>14} {'Unmet/ep':>10} {'Melhoria vs Aleat.':>20}")
    print(f"  {'-'*62}")
    base_r = baselines['Aleatoria']['reward']
    for name, d in {**baselines, "Q-Learning": {"reward": ql_reward, "unmet": ql_unmet}}.items():
        improv = (d['reward'] - base_r) / abs(base_r) * 100 if base_r != 0 else 0
        print(f"  {name:<14} {d['reward']:>14.1f} {d['unmet']:>10.1f} {improv:>19.1f}%")

    # ── Salvar resultados para o paper ─────────────────────────────────────
    if not args.no_save and history is not None:
        results_summary = {
            "baselines":    baselines,
            "qlearning":    {"reward": ql_reward, "unmet": ql_unmet, "std": ql_std},
            "hyperparams":  {"alpha": ALPHA, "gamma": GAMMA, "eps_start": EPS_START,
                             "eps_end": EPS_END, "n_train": args.episodes},
            "qtable_info":  {"n_states": qt.n_states_visited, "n_updates": qt.updates},
            "history":      history,
        }
        import json
        # history tem arrays numpy — converter para listas
        def to_serializable(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32, np.int64)): return int(o)
            return o

        with open(os.path.join(RESULTS_DIR, "results_summary.json"), "w") as f:
            json.dump({k: (to_serializable(v) if not isinstance(v, dict)
                           else {kk: to_serializable(vv) for kk, vv in v.items()})
                       for k, v in results_summary.items()}, f, indent=2, default=str)
        print(f"\n  Resultados salvos em: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
