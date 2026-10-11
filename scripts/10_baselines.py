"""
10_baselines.py  —  Baselines da literatura para o BSRP
========================================================
Políticas:
  1. Threshold Policy  — escolhe destino e delta com base em desvio de uma meta
     de ocupação por estação (target = 50% da capacidade).
     Referência conceptual: políticas de rebalanceamento por threshold descritas em
     Raviv & Kolka (2013) e Dell'Amico et al. (2014).

  2. Rollout Greedy (1-step lookahead)  — para cada ação válida, simula o resultado
     imediato + 1 passo de política gulosa e escolhe a ação de maior valor.
     Referência conceptual: Bertsekas (2020) "Rollout and Policy Improvement".

Formato de ação do simulador: (dest, delta_at_current)
  - dest  : estação destino para o qual o veículo vai ao final do passo
  - delta : bikes depositadas (+) ou retiradas (-) NA ESTAÇÃO ATUAL antes de partir
"""
import os, sys, json, copy, argparse, time
import numpy as np
import importlib.machinery, importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)

loader = importlib.machinery.SourceFileLoader(
    "sim", os.path.join(BASE, "scripts", "04_simulator.py"))
spec = importlib.util.spec_from_loader("sim", loader)
sim  = importlib.util.module_from_spec(spec)
loader.exec_module(sim)

EVAL_SEEDS  = [123, 456, 789]
DELTAS      = [-10, -5, -2, 0, 2, 5, 10, 20]
VEHICLE_CAP = 20

# ── Threshold Policy ──────────────────────────────────────────────────────────
class ThresholdPolicy:
    """
    Meta de ocupação: target_ratio × capacity.
    Destino: estação mais desviada da meta que se beneficia de uma visita.
    Delta na estação atual: corrige a estação atual em direção à meta.
    """
    def __init__(self, target_ratio=0.5):
        self.target = target_ratio

    def select_action(self, env, obs):
        s    = env._state
        n    = len(env.stations)
        caps = np.array([env.stations[i].cap for i in range(n)], dtype=float)
        occ  = s.occupancy.astype(float)
        tgt  = caps * self.target
        dev  = occ - tgt          # positivo = excesso; negativo = déficit

        load      = s.vehicle_load
        load_rat  = load / VEHICLE_CAP
        pos       = s.vehicle_pos

        # ── delta na estação atual ────────────────────────────────────────
        cur_ratio = occ[pos] / max(caps[pos], 1)
        if cur_ratio < 0.35 and load > 0:
            # estação atual quase vazia e veículo tem bikes → deposita
            delta = min(10, load, int(caps[pos] - occ[pos]))
        elif cur_ratio > 0.75 and load < VEHICLE_CAP:
            # estação atual quase cheia e veículo tem espaço → recolhe
            delta = -min(10, int(occ[pos]), VEHICLE_CAP - load)
        else:
            delta = 0

        # ── escolha do destino ────────────────────────────────────────────
        best_dest  = (pos + 1) % n
        best_score = -1e9
        for i in range(n):
            if i == pos:
                continue
            ratio = occ[i] / max(caps[i], 1)
            if load_rat > 0.5:
                # veículo carregado → prefere estações deficitárias (para depositar)
                score = -dev[i]   # quanto mais negativo dev, maior score
            else:
                # veículo vazio → prefere estações com excesso (para recolher)
                score = dev[i]
            # bônus por magnitude de desequilíbrio
            score += abs(dev[i]) * 0.1
            if score > best_score:
                best_score = score
                best_dest  = i

        # garante que (best_dest, delta) é ação válida; se não, faz fallback
        valid_set = set(map(tuple, env.valid_actions()))
        if (best_dest, delta) in valid_set:
            return (best_dest, delta)

        # tenta manter o destino e ajustar delta
        for d in DELTAS:
            if (best_dest, d) in valid_set:
                return (best_dest, d)

        # fallback: qualquer ação válida
        va = env.valid_actions()
        return tuple(va[0]) if len(va) > 0 else (0, 0)


# ── Rollout Greedy (1-step lookahead) ────────────────────────────────────────
class RolloutGreedyPolicy:
    """
    Valor de cada ação a_0: r(a_0) + γ × r(greedy(s_1))
    Usa cópia rasa do estado para simulação (rápido).
    """
    def __init__(self, gamma=1.0):
        self.gamma   = gamma
        self._greedy = sim.GreedyPolicy()

    def _simulate_step(self, env, action):
        """Cria cópia leve do env, executa action, retorna (r, new_env)."""
        env2 = copy.copy(env)
        env2._state = copy.copy(env._state)
        env2._state.occupancy = env._state.occupancy.copy()
        obs2, r, done, _ = env2.step(tuple(action))
        return r, done, env2, obs2

    def select_action(self, env, obs):
        valid = env.valid_actions()
        if len(valid) == 0:
            return (0, 0)

        best_a     = tuple(valid[0])
        best_value = -1e9

        for a in valid:
            r0, done, env2, obs2 = self._simulate_step(env, a)
            r1 = 0.0
            if not done:
                a1 = self._greedy.select_action(env2, obs2)
                if a1 is not None:
                    _, r1, _, _ = env2.step(tuple(a1))
            value = r0 + self.gamma * r1
            if value > best_value:
                best_value = value
                best_a     = tuple(a)

        return best_a


# ── avaliação multi-seed ──────────────────────────────────────────────────────
def eval_policy_multiseed(policy_obj, n_stations, n_ep=100):
    results = {}
    for seed in EVAL_SEEDS:
        profiles     = os.path.join(BASE, f"demand_profiles_{n_stations}stations_normalized.csv")
        stations_csv = os.path.join(BASE, f"station_selected_{n_stations}.csv")
        stations = sim.load_environment_data(profiles_path=profiles,
                                              stations_path=stations_csv)
        env = sim.BSRPEnv(stations, rng=np.random.default_rng(seed))
        ep_rewards = []
        for _ in range(n_ep):
            obs, state = env.reset()
            ep_r = 0.0
            while not state.done:
                action = policy_obj.select_action(env, obs)
                obs, r, done, _ = env.step(action)
                ep_r += r
                state = env._state
            ep_rewards.append(ep_r)
        arr = np.array(ep_rewards)
        results[seed] = {"reward": float(arr.mean()), "std": float(arr.std())}
    return results


def run_scale(n_stations):
    all_actions = [(d, delta) for d in range(n_stations) for delta in DELTAS]

    policies = {
        "threshold": ThresholdPolicy(target_ratio=0.5),
        "rollout":   RolloutGreedyPolicy(gamma=1.0),
    }

    results = {}
    for name, pol in policies.items():
        print(f"  {name} ({n_stations} est.)...", end="", flush=True)
        t0 = time.time()
        ms = eval_policy_multiseed(pol, n_stations, n_ep=100)
        mean_r = np.mean([v["reward"] for v in ms.values()])
        std_r  = np.mean([v["std"]    for v in ms.values()])
        elapsed = time.time() - t0
        print(f"  {mean_r:.2f} ±{std_r:.2f}  ({elapsed:.0f}s)")
        results[name] = {"mean": mean_r, "std": std_r,
                         "multiseed": {str(k): v for k, v in ms.items()}}

    out_path = os.path.join(RESULTS_DIR, f"baselines_{n_stations}st.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=int, default=None)
    parser.add_argument("--all",      action="store_true")
    args = parser.parse_args()
    scales = [4, 8, 12] if args.all else [args.stations]
    for n in scales:
        print(f"\n=== {n} estações ===")
        run_scale(n)
