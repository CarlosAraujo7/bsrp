"""
04_simulator.py
Simulador de demanda para o BSRP (Bike Sharing Rebalancing Problem).

Modela um dia operacional (06h-22h, 32 intervalos de 30 min) como MDP:
  - Estado:   ocupação agregada por cluster + posição e carga do veículo + intervalo
  - Ação:     (próxima estação a visitar, Δbikes a depositar/remover na estação atual)
  - Transição: demanda estocástica via Poisson calibrado nos perfis históricos
  - Recompensa: negativa proporcional às insatisfações (demanda não atendida + devolução bloqueada)

Uso standalone:
    python scripts/04_simulator.py --demo          # roda 5 episódios com política aleatória
    python scripts/04_simulator.py --validate      # testes de sanidade do ambiente
"""

import numpy as np
import pandas as pd
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..")
PROFILES_CSV = os.path.join(DATA_DIR, "demand_profiles_12stations_normalized.csv")
STATIONS_CSV = os.path.join(DATA_DIR, "station_selected_12.csv")

INTERVAL_MIN   = 30          # minutos por intervalo de decisão
DAY_START_H    = 6           # hora de início do episódio
DAY_END_H      = 22          # hora de fim
N_INTERVALS    = (DAY_END_H - DAY_START_H) * 60 // INTERVAL_MIN  # 32
VEHICLE_CAP    = 20          # capacidade do veículo redistribuidor (bikes)
SPEED_KMH      = 20.0        # velocidade média do veículo (km/h, inclui trânsito)
N_OCC_LEVELS   = 4           # discretização de ocupação por cluster: 0,1,2,3
SEED_DEFAULT   = 42
DEMAND_SCALE   = 3.5         # fator de escala para calibrar demanda (ajustado vs dados reais)

# ──────────────────────────────────────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Station:
    idx:     int
    name:    str
    cluster: int
    cap:     int
    lat:     float
    lon:     float
    dep_avg: np.ndarray   # shape (N_INTERVALS,) — partidas médias por intervalo
    arr_avg: np.ndarray   # shape (N_INTERVALS,)

@dataclass
class SimState:
    """Estado completo (não discretizado) do simulador a cada passo."""
    occupancy:   np.ndarray   # (N_STATIONS,) — bikes em cada estação
    vehicle_pos: int          # índice da estação atual do veículo
    vehicle_load: int         # bikes no veículo
    t:           int          # intervalo atual (0..N_INTERVALS-1)
    done:        bool = False

@dataclass
class StepInfo:
    """Informações extras retornadas a cada step (para logging e análise)."""
    unmet_departures: int     # usuários que não encontraram bike
    unmet_arrivals:   int     # usuários que não encontraram vaga
    total_departures: int
    total_arrivals:   int
    travel_intervals: int     # intervalos gastos em trânsito neste step

# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────────────────────────────────────
def load_environment_data(profiles_path: str = PROFILES_CSV,
                          stations_path: str = STATIONS_CSV):
    """Carrega perfis de demanda e metadados das estações."""
    profiles = pd.read_csv(profiles_path)
    stations_df = pd.read_csv(stations_path)

    # Alinhar índice de intervalo (profiles usa 0..47 para 24h;
    # o simulador usa 0..31 para 06h-22h)
    # Intervalo 0 do sim = intervalo 12 dos profiles (06h / 30min = índice 12)
    offset = DAY_START_H * 60 // INTERVAL_MIN   # = 12

    stations = []
    for _, row in stations_df.iterrows():
        pf = profiles[profiles['station'] == row['name']]
        dep = np.zeros(N_INTERVALS)
        arr = np.zeros(N_INTERVALS)
        for i in range(N_INTERVALS):
            profile_interval = offset + i
            dep_rows = pf[pf['interval'] == profile_interval]['departures_avg']
            arr_rows = pf[pf['interval'] == profile_interval]['arrivals_avg']
            dep[i] = float(dep_rows.iloc[0]) if len(dep_rows) > 0 else 0.0
            arr[i] = float(arr_rows.iloc[0]) if len(arr_rows) > 0 else 0.0

        stations.append(Station(
            idx     = len(stations),
            name    = row['name'],
            cluster = int(row['cluster']),
            cap     = int(row['capacity']),
            lat     = float(row['lat']),
            lon     = float(row['lon']),
            dep_avg = dep,
            arr_avg = arr,
        ))
    return stations


def compute_travel_matrix(stations: list[Station]) -> np.ndarray:
    """
    Matriz de tempos de viagem (em intervalos de 30 min) entre estações.
    Usa distância euclidiana em graus → km (aprox. 111 km/grau) + velocidade.
    """
    n = len(stations)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                T[i, j] = 0.0
            else:
                dlat = (stations[i].lat - stations[j].lat) * 111.0
                dlon = (stations[i].lon - stations[j].lon) * 111.0 * np.cos(np.radians(stations[i].lat))
                dist_km = np.sqrt(dlat**2 + dlon**2)
                hours = dist_km / SPEED_KMH
                T[i, j] = hours * 60 / INTERVAL_MIN  # em intervalos
    return T


# ──────────────────────────────────────────────────────────────────────────────
# AMBIENTE MDP
# ──────────────────────────────────────────────────────────────────────────────
class BSRPEnv:
    """
    Ambiente MDP para o Bike Sharing Rebalancing Problem.

    Episódio: um dia operacional (06h-22h), 32 intervalos de 30 min.
    A cada passo, o agente decide:
      - Para qual estação mover o veículo a seguir (dest)
      - Quantas bikes depositar (>0) ou remover (<0) na estação atual antes de partir

    O veículo leva ceil(T[pos][dest]) intervalos em trânsito.
    Durante o trânsito, a demanda continua ocorrendo nas estações.
    O agente só age novamente ao chegar na próxima estação.
    """

    def __init__(self, stations: list[Station], rng: np.random.Generator = None):
        self.stations = stations
        self.n        = len(stations)
        self.travel_T = compute_travel_matrix(stations)
        self.rng      = rng or np.random.default_rng(SEED_DEFAULT)

        # Mapa de cluster → índices das estações
        self.cluster_stations: dict[int, list[int]] = {}
        for s in stations:
            self.cluster_stations.setdefault(s.cluster, []).append(s.idx)
        self.n_clusters = max(s.cluster for s in stations) + 1

    # ──────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────
    def reset(self, initial_occupancy: Optional[np.ndarray] = None) -> tuple[np.ndarray, SimState]:
        """
        Inicia novo episódio.
        Returns: (obs, state)
          obs: vetor de observação discretizado (para Q-table)
          state: estado completo (para logging)
        """
        if initial_occupancy is not None:
            occ = initial_occupancy.copy()
        else:
            # Ocupação inicial: ~50% da capacidade com leve ruído
            occ = np.array([
                int(np.clip(s.cap * 0.5 + self.rng.integers(-2, 3), 0, s.cap))
                for s in self.stations
            ], dtype=int)

        self._state = SimState(
            occupancy    = occ,
            vehicle_pos  = 0,       # veículo começa na estação 0
            vehicle_load = 0,
            t            = 0,
            done         = False,
        )
        self._cumulative_reward   = 0.0
        self._total_unmet_dep     = 0
        self._total_unmet_arr     = 0
        return self._get_obs(), self._state

    # ──────────────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────────────
    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, StepInfo]:
        """
        Executa uma ação.

        Args:
            action: (dest, delta_bikes)
              dest:        índice da próxima estação a visitar
              delta_bikes: bikes a depositar na estação atual (>0) ou remover (<0)

        Returns:
            (obs, reward, done, info)
        """
        assert not self._state.done, "Episódio encerrado — chame reset()."
        dest, delta = action
        s = self._state

        # 1. EXECUTAR OPERAÇÃO NA ESTAÇÃO ATUAL (antes de partir)
        pos = s.vehicle_pos
        cap_pos = self.stations[pos].cap

        # Clipping: não depositar mais do que a estação suporta, não remover mais do que há
        if delta > 0:
            delta = min(delta, s.vehicle_load, cap_pos - s.occupancy[pos])
        elif delta < 0:
            delta = -min(-delta, s.occupancy[pos], VEHICLE_CAP - s.vehicle_load)

        s.occupancy[pos]  = np.clip(s.occupancy[pos] + delta, 0, cap_pos)
        s.vehicle_load   -= delta   # positivo = depositar → reduz carga do veículo

        # 2. CALCULAR TEMPO DE VIAGEM (intervalos inteiros, mínimo 1 se dest != pos)
        travel_intervals = max(1, int(np.ceil(self.travel_T[pos][dest]))) if dest != pos else 0

        # 3. SIMULAR DEMANDA NOS INTERVALOS ATÉ CHEGAR NO DESTINO
        total_reward      = 0.0
        total_unmet_dep   = 0
        total_unmet_arr   = 0
        total_dep         = 0
        total_arr         = 0

        n_steps = max(travel_intervals, 1)   # pelo menos 1 intervalo de demanda
        for _ in range(n_steps):
            if s.t >= N_INTERVALS:
                s.done = True
                break
            r, ud, ua, td, ta = self._simulate_demand_step(s)
            total_reward    += r
            total_unmet_dep += ud
            total_unmet_arr += ua
            total_dep       += td
            total_arr       += ta
            s.t += 1

        # 4. ATUALIZAR POSIÇÃO DO VEÍCULO
        s.vehicle_pos = dest

        self._cumulative_reward += total_reward
        self._total_unmet_dep   += total_unmet_dep
        self._total_unmet_arr   += total_unmet_arr

        if s.t >= N_INTERVALS:
            s.done = True

        info = StepInfo(
            unmet_departures = total_unmet_dep,
            unmet_arrivals   = total_unmet_arr,
            total_departures = total_dep,
            total_arrivals   = total_arr,
            travel_intervals = travel_intervals,
        )
        return self._get_obs(), total_reward, s.done, info

    # ──────────────────────────────────────────────────────────────────────
    # DEMANDA ESTOCÁSTICA (Poisson)
    # ──────────────────────────────────────────────────────────────────────
    def _simulate_demand_step(self, s: SimState) -> tuple[float, int, int, int, int]:
        """
        Simula um intervalo de demanda: amostra chegadas e partidas via Poisson
        para cada estação usando as médias históricas. Atualiza ocupação.

        Returns: (reward, unmet_dep, unmet_arr, total_dep, total_arr)
        """
        unmet_dep = 0
        unmet_arr = 0
        total_dep = 0
        total_arr = 0
        t = s.t

        for st in self.stations:
            i = st.idx
            lam_dep = max(0.0, st.dep_avg[t] * DEMAND_SCALE)
            lam_arr = max(0.0, st.arr_avg[t] * DEMAND_SCALE)

            n_dep = int(self.rng.poisson(lam_dep))
            n_arr = int(self.rng.poisson(lam_arr))

            # Partidas: limitadas pela ocupação atual
            actual_dep  = min(n_dep, s.occupancy[i])
            blocked_dep = n_dep - actual_dep

            # Chegadas: limitadas pela capacidade disponível
            space       = self.stations[i].cap - (s.occupancy[i] - actual_dep)
            actual_arr  = min(n_arr, space)
            blocked_arr = n_arr - actual_arr

            s.occupancy[i] = s.occupancy[i] - actual_dep + actual_arr
            s.occupancy[i] = int(np.clip(s.occupancy[i], 0, self.stations[i].cap))

            unmet_dep += blocked_dep
            unmet_arr += blocked_arr
            total_dep += n_dep
            total_arr += n_arr

        reward = -(unmet_dep + unmet_arr)
        return reward, unmet_dep, unmet_arr, total_dep, total_arr

    # ──────────────────────────────────────────────────────────────────────
    # OBSERVAÇÃO DISCRETIZADA (para Q-table)
    # ──────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        Converte estado contínuo para vetor discretizado usado como chave na Q-table.
        Componentes:
          [0..n_clusters-1] : nível de ocupação do cluster (0..N_OCC_LEVELS-1)
          [n_clusters]      : posição do veículo (0..n_stations-1)
          [n_clusters+1]    : carga do veículo discretizada em 4 faixas
          [n_clusters+2]    : intervalo de tempo (0..N_INTERVALS-1)
        """
        s   = self._state
        obs = np.zeros(self.n_clusters + 3, dtype=np.int32)

        for c in range(self.n_clusters):
            idxs     = self.cluster_stations[c]
            total_occ = sum(s.occupancy[i] for i in idxs)
            total_cap = sum(self.stations[i].cap for i in idxs)
            ratio     = total_occ / max(total_cap, 1)
            obs[c]    = int(np.floor(ratio * N_OCC_LEVELS))
            obs[c]    = min(obs[c], N_OCC_LEVELS - 1)

        obs[self.n_clusters]     = s.vehicle_pos
        # Carga do veículo: 0-4, 5-9, 10-14, 15-20 → 0,1,2,3
        obs[self.n_clusters + 1] = min(3, s.vehicle_load // 5)
        obs[self.n_clusters + 2] = s.t

        return obs

    def obs_to_key(self, obs: np.ndarray) -> tuple:
        """Converte vetor de observação em chave hashável para Q-table."""
        return tuple(obs.tolist())

    # ──────────────────────────────────────────────────────────────────────
    # ESPAÇO DE AÇÃO
    # ──────────────────────────────────────────────────────────────────────
    def valid_actions(self) -> list[tuple[int, int]]:
        """
        Retorna lista de ações válidas no estado atual.
        Ação = (dest, delta_bikes) onde delta_bikes ∈ {-5, 0, 5, 10} bikes
        e dest ∈ {0..n-1}.
        Simplificação: granularidade de 5 bikes reduz o espaço de ação.
        """
        s      = self._state
        pos    = s.vehicle_pos
        cap_pos = self.stations[pos].cap

        actions = []
        for dest in range(self.n):
            for delta in [-10, -5, 0, 5, 10]:
                # Verificar viabilidade
                if delta > 0:
                    if s.vehicle_load < delta:
                        continue
                    if s.occupancy[pos] + delta > cap_pos:
                        continue
                elif delta < 0:
                    if s.occupancy[pos] < -delta:
                        continue
                    if s.vehicle_load - delta > VEHICLE_CAP:
                        continue
                actions.append((dest, delta))

        return actions if actions else [(pos, 0)]

    # ──────────────────────────────────────────────────────────────────────
    # PROPRIEDADES DE DIAGNÓSTICO
    # ──────────────────────────────────────────────────────────────────────
    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def service_rate(self) -> float:
        """Fração de demanda atendida ao longo do episódio."""
        total = self._total_unmet_dep + self._total_unmet_arr
        return 1.0 if total == 0 else 1.0 - total / max(1, total + 100)

    def render_state(self):
        """Imprime estado atual de forma legível."""
        s = self._state
        hour = DAY_START_H + s.t * INTERVAL_MIN // 60
        minute = (s.t * INTERVAL_MIN) % 60
        print(f"\n[t={s.t:02d} | {hour:02d}:{minute:02d}]  "
              f"Veículo: {self.stations[s.vehicle_pos].name[:25]:<25}  "
              f"Carga: {s.vehicle_load:>2} bikes")
        for st in self.stations:
            bar_fill = int(s.occupancy[st.idx] / st.cap * 20)
            bar_empty = 20 - bar_fill
            pct = s.occupancy[st.idx] / st.cap * 100
            marker = "◄" if st.idx == s.vehicle_pos else " "
            print(f"  {marker} [{st.cluster}] {st.name[:35]:<35} "
                  f"{'█'*bar_fill}{'░'*bar_empty} {s.occupancy[st.idx]:>3}/{st.cap:<3} ({pct:4.0f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# POLÍTICAS BASELINE
# ──────────────────────────────────────────────────────────────────────────────
class RandomPolicy:
    """Seleciona ação aleatória entre as válidas."""
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(SEED_DEFAULT)

    def select_action(self, env: BSRPEnv, obs: np.ndarray) -> tuple[int, int]:
        actions = env.valid_actions()
        return actions[self.rng.integers(len(actions))]


class GreedyPolicy:
    """
    Política gulosa melhorada: considera a carga do veículo para decidir destino.
    - Veículo cheio  → prioriza estações VAZIAS para depositar
    - Veículo vazio  → prioriza estações CHEIAS para recolher
    - Caso misto     → vai para a estação mais desequilibrada onde pode agir
    Nunca fica parado na estação atual (sempre explora).
    """
    def select_action(self, env: BSRPEnv, obs: np.ndarray) -> tuple[int, int]:
        s   = env._state
        load_ratio = s.vehicle_load / VEHICLE_CAP   # 0..1

        best_dest  = (s.vehicle_pos + 1) % env.n    # fallback: próxima em round-robin
        best_score = -1.0

        for i, st in enumerate(env.stations):
            if i == s.vehicle_pos:
                continue
            ratio = s.occupancy[i] / max(st.cap, 1)

            if load_ratio > 0.5:
                # Veículo com carga: prefere estações com espaço para depositar
                score = (1.0 - ratio) * load_ratio + abs(ratio - 0.5) * (1 - load_ratio)
            else:
                # Veículo leve: prefere estações cheias para recolher
                score = ratio * (1 - load_ratio) + abs(ratio - 0.5) * load_ratio

            if score > best_score:
                best_score = score
                best_dest  = i

        dest    = best_dest
        dest_st = env.stations[dest]
        ratio   = s.occupancy[dest] / max(dest_st.cap, 1)

        # Opera também na estação ATUAL antes de partir
        cur_st  = env.stations[s.vehicle_pos]
        cur_ratio = s.occupancy[s.vehicle_pos] / max(cur_st.cap, 1)
        if cur_ratio < 0.35 and s.vehicle_load > 0:
            delta = min(10, s.vehicle_load, cur_st.cap - s.occupancy[s.vehicle_pos])
        elif cur_ratio > 0.75 and s.vehicle_load < VEHICLE_CAP:
            delta = -min(10, s.occupancy[s.vehicle_pos], VEHICLE_CAP - s.vehicle_load)
        else:
            delta = 0

        return (dest, delta)


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER DE EPISÓDIO
# ──────────────────────────────────────────────────────────────────────────────
def run_episode(env: BSRPEnv, policy, verbose: bool = False) -> dict:
    obs, state = env.reset()
    total_reward     = 0.0
    total_unmet_dep  = 0
    total_unmet_arr  = 0
    total_dep        = 0
    total_arr        = 0
    n_steps          = 0

    if verbose:
        env.render_state()

    while not state.done:
        action = policy.select_action(env, obs)
        obs, reward, done, info = env.step(action)
        state = env._state
        total_reward    += reward
        total_unmet_dep += info.unmet_departures
        total_unmet_arr += info.unmet_arrivals
        total_dep       += info.total_departures
        total_arr       += info.total_arrivals
        n_steps         += 1

        if verbose and n_steps % 4 == 0:
            env.render_state()

    satisfaction = 1.0 - (total_unmet_dep + total_unmet_arr) / max(1, total_dep + total_arr)
    return {
        "total_reward":    total_reward,
        "unmet_dep":       total_unmet_dep,
        "unmet_arr":       total_unmet_arr,
        "total_dep":       total_dep,
        "total_arr":       total_arr,
        "satisfaction":    satisfaction,
        "n_steps":         n_steps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO E DEMO
# ──────────────────────────────────────────────────────────────────────────────
def validate(env: BSRPEnv):
    """Testes de sanidade do ambiente."""
    print("=== VALIDAÇÃO DO AMBIENTE ===\n")
    errors = 0

    # Teste 1: reset retorna obs e estado válidos
    obs, state = env.reset()
    assert obs.shape == (env.n_clusters + 3,), f"Shape obs incorreto: {obs.shape}"
    assert not state.done
    assert 0 <= state.vehicle_pos < env.n
    assert state.vehicle_load == 0
    assert state.t == 0
    print("  ✓ reset() retorna estado inicial válido")

    # Teste 2: ocupação sempre dentro de [0, cap]
    rng_pol = RandomPolicy(np.random.default_rng(0))
    for ep in range(10):
        obs, state = env.reset()
        while not state.done:
            action = rng_pol.select_action(env, obs)
            obs, r, done, info = env.step(action)
            state = env._state
            for i, st in enumerate(env.stations):
                assert 0 <= state.occupancy[i] <= st.cap, \
                    f"Ocupação fora dos limites na estação {st.name}: {state.occupancy[i]}"
    print("  ✓ Ocupação sempre dentro de [0, cap] em 10 episódios aleatórios")

    # Teste 3: carga do veículo sempre em [0, VEHICLE_CAP]
    obs, state = env.reset()
    while not state.done:
        action = rng_pol.select_action(env, obs)
        obs, r, done, info = env.step(action)
        state = env._state
        assert 0 <= state.vehicle_load <= VEHICLE_CAP, \
            f"Carga do veículo inválida: {state.vehicle_load}"
    print(f"  ✓ Carga do veículo sempre em [0, {VEHICLE_CAP}]")

    # Teste 4: recompensa sempre ≤ 0
    for ep in range(5):
        result = run_episode(env, rng_pol)
        assert result["total_reward"] <= 0, f"Recompensa positiva inesperada: {result['total_reward']}"
    print("  ✓ Recompensa sempre ≤ 0")

    # Teste 5: espaço de estados
    obs_dims = [N_OCC_LEVELS] * env.n_clusters + [env.n, 4, N_INTERVALS]
    total_states = 1
    for d in obs_dims: total_states *= d
    print(f"\n  Espaço de estados: {' × '.join(str(d) for d in obs_dims)} = {total_states:,}")
    print(f"  Espaço de ações por estado: ~{len(env.valid_actions())} ações")

    if errors == 0:
        print("\n✓ Todos os testes passaram.")
    else:
        print(f"\n✗ {errors} erro(s) encontrado(s).")


def demo(n_episodes: int = 5):
    """Demonstração: compara política aleatória vs. gulosa."""
    print("Carregando dados...")
    stations = load_environment_data()
    print(f"  {len(stations)} estações carregadas.")

    rng = np.random.default_rng(SEED_DEFAULT)
    env = BSRPEnv(stations, rng=rng)

    print("\n=== VALIDAÇÃO ===")
    validate(env)

    print("\n=== COMPARAÇÃO DE POLÍTICAS ===\n")
    policies = {
        "Aleatória": RandomPolicy(np.random.default_rng(0)),
        "Gulosa":    GreedyPolicy(),
    }

    for name, policy in policies.items():
        results = [run_episode(env, policy) for _ in range(n_episodes)]
        avg_reward = np.mean([r["total_reward"]  for r in results])
        avg_sat    = np.mean([r["satisfaction"]  for r in results]) * 100
        avg_unmet  = np.mean([r["unmet_dep"] + r["unmet_arr"] for r in results])
        print(f"  {name:<12}: reward_avg={avg_reward:7.1f} | unmet={avg_unmet:5.1f}/ep | service={avg_sat:.1f}%")

    print("\n=== EPISODIO EXEMPLO (politica gulosa, verbose) ===")
    env2 = BSRPEnv(stations, rng=np.random.default_rng(99))
    run_episode(env2, GreedyPolicy(), verbose=True)
    print(f"\n  Recompensa total: {env2.cumulative_reward:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",     action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    stations = load_environment_data()
    env = BSRPEnv(stations)

    if args.validate:
        validate(env)
    else:
        demo(args.episodes)
