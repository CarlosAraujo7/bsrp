"""
bsrp/env.py — ambiente BSRP com API de cenários (PROTOCOLO_RIGOR.md §2).

Mesma dinâmica de scripts/04_simulator.py (verificada por tests/test_regression_legacy.py), mas:
  * a demanda de um episódio é um CENÁRIO pré-amostrado (`Scenario`), não consumo de um gerador:
    todos os métodos avaliados no mesmo cenário veem exatamente a mesma demanda (common random numbers);
  * demanda não atendida é contabilizada (o campo `unmet` do legado era sempre 0,0);
  * ações: índice a em [0, 5n): destino = a // 5, delta = DELTAS[a % 5] (mesmo ordenamento do legado).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_INTERVALS = 32            # 06h-22h em passos de 30 min
INTERVAL_MIN = 30
VEHICLE_CAP = 20
SPEED_KMH = 20.0
DELTAS = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
N_DELTAS = len(DELTAS)
DAY_START_H = 6
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, eq=False)
class Instance:
    name: str
    cap: np.ndarray        # (n,) int
    cluster: np.ndarray    # (n,) int  (só para a representação "por cluster")
    lat: np.ndarray
    lon: np.ndarray
    lam_dep: np.ndarray    # (32, n) partidas esperadas por intervalo (já com qualquer escala aplicada)
    lam_arr: np.ndarray    # (32, n)
    station_names: tuple

    @property
    def n(self) -> int:
        return len(self.cap)

    @property
    def n_actions(self) -> int:
        return self.n * N_DELTAS

    @property
    def n_clusters(self) -> int:
        return int(self.cluster.max()) + 1

    def travel_int(self) -> np.ndarray:
        """Intervalos inteiros de deslocamento: 0 na diagonal, senão max(1, ceil(T)) (como no legado)."""
        dlat = (self.lat[:, None] - self.lat[None, :]) * 111.0
        dlon = (self.lon[:, None] - self.lon[None, :]) * 111.0 * np.cos(np.radians(self.lat[:, None]))
        T = np.sqrt(dlat ** 2 + dlon ** 2) / SPEED_KMH * 60.0 / INTERVAL_MIN
        out = np.maximum(1, np.ceil(T)).astype(np.int64)
        np.fill_diagonal(out, 0)
        return out

    def with_scale(self, k: float, name: str | None = None) -> "Instance":
        return Instance(name or f"{self.name}x{k:g}", self.cap, self.cluster, self.lat, self.lon,
                        self.lam_dep * k, self.lam_arr * k, self.station_names)


def legacy_instance(n_stations: int, demand_scale: float = 3.5) -> Instance:
    """Instância do legado a partir dos CSVs de perfis (idêntica ao load_environment_data de 04_simulator.py)."""
    pf = ROOT / f"demand_profiles_{n_stations}stations_normalized.csv"
    sf = ROOT / f"station_selected_{n_stations}.csv"
    prof = {}
    with open(pf, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            prof.setdefault((r["station"], int(r["interval"])), (float(r["departures_avg"]), float(r["arrivals_avg"])))
    names, cap, cl, lat, lon, dep, arr = [], [], [], [], [], [], []
    with open(sf, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            names.append(r["name"]); cap.append(int(r["capacity"])); cl.append(int(r["cluster"]))
            lat.append(float(r["lat"])); lon.append(float(r["lon"]))
            offset = DAY_START_H * 60 // INTERVAL_MIN
            d = [prof.get((r["name"], offset + i), (0.0, 0.0)) for i in range(N_INTERVALS)]
            dep.append([x[0] for x in d]); arr.append([x[1] for x in d])
    lam_dep = np.maximum(0.0, np.array(dep).T * demand_scale)   # (32, n)
    lam_arr = np.maximum(0.0, np.array(arr).T * demand_scale)
    return Instance(f"legacy{n_stations}", np.array(cap, np.int64), np.array(cl, np.int64), np.array(lat), np.array(lon),
                    lam_dep, lam_arr, tuple(names))


# ──────────────────────────────────────────────────────────────────────────────
# CENÁRIOS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    init_occ: np.ndarray   # (n,) int
    dep: np.ndarray        # (32, n) int
    arr: np.ndarray        # (32, n) int


class ScenarioSet:
    """E cenários empilhados; salvável em .npz com hash de conteúdo."""

    def __init__(self, init_occ, dep, arr, meta=None):
        self.init_occ, self.dep, self.arr = init_occ, dep, arr
        self.meta = meta or {}

    def __len__(self):
        return len(self.init_occ)

    def __getitem__(self, k) -> Scenario:
        return Scenario(self.init_occ[k], self.dep[k], self.arr[k])

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.init_occ, self.dep, self.arr):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def save(self, path):
        np.savez_compressed(path, init_occ=self.init_occ, dep=self.dep, arr=self.arr)

    @staticmethod
    def load(path):
        z = np.load(path)
        return ScenarioSet(z["init_occ"], z["dep"], z["arr"])


def sample_scenarios(inst: Instance, n_episodes: int, rng: np.random.Generator, jitter: int = 2) -> ScenarioSet:
    """Amostra E cenários do modelo de demanda de `inst` (Poisson independente por estação, como no legado)."""
    n = inst.n
    base = inst.cap * 0.5
    init = np.clip(base[None, :] + rng.integers(-jitter, jitter + 1, size=(n_episodes, n)), 0, inst.cap[None, :]).astype(np.int64)
    dep = rng.poisson(inst.lam_dep[None], size=(n_episodes, N_INTERVALS, n)).astype(np.int16)
    arr = rng.poisson(inst.lam_arr[None], size=(n_episodes, N_INTERVALS, n)).astype(np.int16)
    return ScenarioSet(init, dep, arr)


def legacy_scenarios(inst: Instance, seed: int, n_episodes: int) -> ScenarioSet:
    """Reproduz EXATAMENTE a sequência de números aleatórios do simulador legado para `default_rng(seed)`:
    por episódio, n sorteios escalares integers(-2,3) (ocupação inicial) e depois, por intervalo e estação,
    poisson(partidas) seguido de poisson(chegadas). Só para o teste de regressão."""
    rng = np.random.default_rng(seed)
    n = inst.n
    lam = np.stack([inst.lam_dep, inst.lam_arr], axis=-1)      # (32, n, 2) -> ordem C: t, estação, (dep, arr)
    init = np.zeros((n_episodes, n), np.int64)
    dep = np.zeros((n_episodes, N_INTERVALS, n), np.int16)
    arr = np.zeros((n_episodes, N_INTERVALS, n), np.int16)
    for e in range(n_episodes):
        for i in range(n):
            init[e, i] = int(np.clip(inst.cap[i] * 0.5 + rng.integers(-2, 3), 0, inst.cap[i]))
        d = rng.poisson(lam)
        dep[e], arr[e] = d[..., 0], d[..., 1]
    return ScenarioSet(init, dep, arr)


# ──────────────────────────────────────────────────────────────────────────────
# AMBIENTE
# ──────────────────────────────────────────────────────────────────────────────
class BSRPEnv:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.n = inst.n
        self.cap = inst.cap
        self.travel = inst.travel_int()
        self._k_valid = np.arange(N_DELTAS)

    # -- ciclo de vida ---------------------------------------------------------
    def reset(self, sc: Scenario):
        self.sc = sc
        self.occ = sc.init_occ.astype(np.int64).copy()
        self.pos = 0
        self.load = 0
        self.t = 0
        self.done = False
        self.unmet_dep = 0
        self.unmet_arr = 0
        self.total_dep = 0
        self.total_arr = 0
        return self

    def valid_delta(self) -> np.ndarray:
        """(5,) bool: quais deltas são viáveis na estação atual (independe do destino)."""
        occ_p, cap_p, load = int(self.occ[self.pos]), int(self.cap[self.pos]), self.load
        return np.array([occ_p >= 10 and load + 10 <= VEHICLE_CAP,
                         occ_p >= 5 and load + 5 <= VEHICLE_CAP,
                         True,
                         load >= 5 and occ_p + 5 <= cap_p,
                         load >= 10 and occ_p + 10 <= cap_p])

    def valid_mask(self) -> np.ndarray:
        return np.tile(self.valid_delta(), self.n)

    def step(self, a: int):
        dest, k = divmod(int(a), N_DELTAS)
        return self.step_raw(dest, int(DELTAS[k]))

    def step_raw(self, dest: int, delta: int):
        """Executa (dest, delta) com a mesma semântica do legado; delta arbitrário é recortado."""
        pos, cap_p = self.pos, int(self.cap[self.pos])
        occ_p = int(self.occ[pos])
        if delta > 0:
            delta = min(delta, self.load, cap_p - occ_p)
        elif delta < 0:
            delta = -min(-delta, occ_p, VEHICLE_CAP - self.load)
        self.occ[pos] = min(max(occ_p + delta, 0), cap_p)
        self.load -= delta
        travel = int(self.travel[pos, dest])            # 0 se dest == pos
        n_steps = max(travel, 1)
        reward = 0
        for _ in range(n_steps):
            if self.t >= N_INTERVALS:
                self.done = True
                break
            reward += self._demand_interval()
            self.t += 1
        self.pos = dest
        if self.t >= N_INTERVALS:
            self.done = True
        return float(reward), self.done

    def _demand_interval(self) -> int:
        t = self.t
        nd = self.sc.dep[t].astype(np.int64)
        na = self.sc.arr[t].astype(np.int64)
        act_dep = np.minimum(nd, self.occ)
        space = self.cap - (self.occ - act_dep)
        act_arr = np.minimum(na, space)
        ud = int((nd - act_dep).sum())
        ua = int((na - act_arr).sum())
        self.occ = np.clip(self.occ - act_dep + act_arr, 0, self.cap)
        self.unmet_dep += ud
        self.unmet_arr += ua
        self.total_dep += int(nd.sum())
        self.total_arr += int(na.sum())
        return -(ud + ua)

    # -- observações -----------------------------------------------------------
    def features(self) -> np.ndarray:
        """Features do DQN legado: ocupação/cap (n), one-hot da posição (n), carga/20, t/32 -> (2n+2,) float32."""
        f = np.zeros(2 * self.n + 2, np.float32)
        f[: self.n] = self.occ / np.maximum(self.cap, 1)
        f[self.n + self.pos] = 1.0
        f[-2] = self.load / 20.0
        f[-1] = self.t / 32.0
        return f

    def clone(self) -> "BSRPEnv":
        c = object.__new__(BSRPEnv)
        c.__dict__.update(self.__dict__)
        c.occ = self.occ.copy()
        return c

    def unmet(self) -> int:
        return self.unmet_dep + self.unmet_arr


def run_episode(env: BSRPEnv, policy, sc: Scenario) -> dict:
    """Roda um episódio com `policy.act(env) -> índice de ação` ou `(dest, delta)`."""
    env.reset(sc)
    total = 0.0
    steps = 0
    while not env.done:
        a = policy.act(env)
        if isinstance(a, tuple):
            r, _ = env.step_raw(*a)
        else:
            r, _ = env.step(a)
        total += r
        steps += 1
    return {"reward": total, "unmet": env.unmet(), "steps": steps}


def evaluate(env: BSRPEnv, policy, scenarios: ScenarioSet) -> np.ndarray:
    """Reward por cenário (E,)."""
    return np.array([run_episode(env, policy, scenarios[k])["reward"] for k in range(len(scenarios))])
