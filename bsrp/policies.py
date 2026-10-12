"""
bsrp/policies.py — referências sem aprendizado (PROTOCOLO_RIGOR.md §5).

* Stay            : veículo parado (piso real; não existia no legado)
* RandomPolicy    : ação válida uniforme
* LegacyGreedy    : porte exato da GreedyPolicy de scripts/04_simulator.py (regressão)
* LegacyThreshold : porte exato da ThresholdPolicy de scripts/10_baselines.py, com a peculiaridade de
                    escolher o 1º delta viável da lista [-10,-5,-2,0,2,5,10,20] (regressão)
* GreedyFamily    : mesma pontuação da gulosa, com limiares e passo calibráveis
* ThresholdFamily : política por desvio de meta, calibrável
* Rollout         : lookahead por simulação de futuros AMOSTRADOS do modelo (não espia a demanda real)
Toda política expõe act(env) -> índice de ação (ou (dest, delta) nas Legacy*).
"""
from __future__ import annotations

import itertools

import numpy as np

from .env import (DELTAS, N_DELTAS, N_INTERVALS, VEHICLE_CAP, BSRPEnv, Scenario, ScenarioSet, evaluate)

_LEG_T_DELTAS = [-10, -5, -2, 0, 2, 5, 10, 20]


class Stay:
    name = "stay"

    def act(self, env: BSRPEnv):
        return env.pos * N_DELTAS + 2        # dest = pos, delta = 0


class RandomPolicy:
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, env: BSRPEnv):
        valid = np.flatnonzero(env.valid_mask())
        return int(valid[self.rng.integers(len(valid))])


def _greedy_scores(env: BSRPEnv) -> np.ndarray:
    ratio = env.occ / np.maximum(env.cap, 1)
    lr = env.load / VEHICLE_CAP
    if lr > 0.5:
        sc = (1.0 - ratio) * lr + np.abs(ratio - 0.5) * (1 - lr)
    else:
        sc = ratio * (1 - lr) + np.abs(ratio - 0.5) * lr
    sc = sc.copy()
    sc[env.pos] = -np.inf
    return sc


def _best_dest(env: BSRPEnv, score: np.ndarray) -> int:
    if not np.isfinite(score).any():            # n == 1
        return env.pos
    return int(np.argmax(score))                 # 1º máximo, como o '>' estrito do legado


class LegacyGreedy:
    name = "legacy_greedy"

    def act(self, env: BSRPEnv):
        dest = _best_dest(env, _greedy_scores(env))
        pos = env.pos
        cur = env.occ[pos] / max(env.cap[pos], 1)
        if cur < 0.35 and env.load > 0:
            delta = min(10, env.load, int(env.cap[pos] - env.occ[pos]))
        elif cur > 0.75 and env.load < VEHICLE_CAP:
            delta = -min(10, int(env.occ[pos]), VEHICLE_CAP - env.load)
        else:
            delta = 0
        return (dest, int(delta))


class LegacyThreshold:
    name = "legacy_threshold"

    def __init__(self, target_ratio: float = 0.5):
        self.target = target_ratio

    def act(self, env: BSRPEnv):
        n, pos, load = env.n, env.pos, env.load
        caps = env.cap.astype(float)
        occ = env.occ.astype(float)
        dev = occ - caps * self.target
        cur = occ[pos] / max(caps[pos], 1)
        if cur < 0.35 and load > 0:
            delta = min(10, load, int(caps[pos] - occ[pos]))
        elif cur > 0.75 and load < VEHICLE_CAP:
            delta = -min(10, int(occ[pos]), VEHICLE_CAP - load)
        else:
            delta = 0
        best, best_s = (pos + 1) % n, -1e9
        for i in range(n):
            if i == pos:
                continue
            s = (-dev[i] if load / VEHICLE_CAP > 0.5 else dev[i]) + abs(dev[i]) * 0.1
            if s > best_s:
                best_s, best = s, i
        vd = env.valid_delta()
        ok = lambda d: d in (-10, -5, 0, 5, 10) and vd[list(DELTAS).index(d)]
        if ok(delta):
            return (best, int(delta))
        for d in _LEG_T_DELTAS:
            if ok(d):
                return (best, d)
        return (0, 0)


def _pick_delta(env: BSRPEnv, want: int) -> int:
    """Índice de delta viável de maior magnitude com sinal de `want`, módulo máx. |want|; 0 se nenhum."""
    vd = env.valid_delta()
    if want > 0:
        for k in (4, 3):                         # +10, +5
            if vd[k] and DELTAS[k] <= want:
                return k
    elif want < 0:
        for k in (0, 1):                         # -10, -5
            if vd[k] and -DELTAS[k] <= -want:
                return k
    return 2


class GreedyFamily:
    """Pontuação da gulosa v1; limiares `lo`/`hi` da estação atual e passo máximo calibráveis."""

    def __init__(self, lo=0.35, hi=0.75, max_delta=10):
        self.lo, self.hi, self.max_delta = lo, hi, max_delta
        self.name = f"greedy(lo={lo},hi={hi},d={max_delta})"

    def act(self, env: BSRPEnv):
        dest = _best_dest(env, _greedy_scores(env))
        pos = env.pos
        cur = env.occ[pos] / max(env.cap[pos], 1)
        want = 0
        if cur < self.lo and env.load > 0:
            want = min(self.max_delta, env.load, int(env.cap[pos] - env.occ[pos]))
        elif cur > self.hi and env.load < VEHICLE_CAP:
            want = -min(self.max_delta, int(env.occ[pos]), VEHICLE_CAP - env.load)
        return dest * N_DELTAS + _pick_delta(env, want)


class ThresholdFamily:
    """Meta de ocupação `target`; destino pelo desvio (em bicicletas) com bônus `w_abs`·|desvio|."""

    def __init__(self, target=0.5, lo=0.35, hi=0.75, max_delta=10, w_abs=0.1):
        self.p = (target, lo, hi, max_delta, w_abs)
        self.name = f"threshold(t={target},lo={lo},hi={hi},d={max_delta},w={w_abs})"

    def act(self, env: BSRPEnv):
        target, lo, hi, max_delta, w_abs = self.p
        pos, load = env.pos, env.load
        dev = env.occ - env.cap * target
        score = (-dev if load / VEHICLE_CAP > 0.5 else dev) + np.abs(dev) * w_abs
        score = score.astype(float).copy()
        score[pos] = -np.inf
        dest = _best_dest(env, score)
        cur = env.occ[pos] / max(env.cap[pos], 1)
        want = 0
        if cur < lo and load > 0:
            want = min(max_delta, load, int(env.cap[pos] - env.occ[pos]))
        elif cur > hi and load < VEHICLE_CAP:
            want = -min(max_delta, int(env.occ[pos]), VEHICLE_CAP - load)
        return dest * N_DELTAS + _pick_delta(env, want)


def greedy_grid():
    return [GreedyFamily(lo, hi, d) for lo, hi, d in itertools.product((0.25, 0.35, 0.45), (0.6, 0.75, 0.9), (5, 10))]


def threshold_grid():
    return [ThresholdFamily(t, lo, hi, d, w) for t, lo, hi, d, w in
            itertools.product((0.4, 0.5, 0.6), (0.25, 0.35, 0.45), (0.6, 0.75, 0.9), (5, 10), (0.0, 0.1, 0.3))]


def calibrate(env: BSRPEnv, candidates, val: ScenarioSet):
    """Escolhe, em VALIDAÇÃO, o candidato de maior reward médio. Retorna (melhor, tabela [(nome, média)])."""
    table = [(c.name, float(evaluate(env, c, val).mean())) for c in candidates]
    best = max(range(len(candidates)), key=lambda i: table[i][1])
    return candidates[best], table


class Rollout:
    """Para cada ação válida: r(a) + retorno de `base` por `h` decisões adicionais, em K futuros amostrados
    do MODELO (mesmos K futuros para todas as ações = common random numbers). Não vê a demanda real."""
    name = "rollout"

    def __init__(self, base, h=1, K=4, seed=0):
        self.base, self.h, self.K = base, h, K
        self.rng = np.random.default_rng(seed)
        self.name = f"rollout(h={h},K={K},base={getattr(base, 'name', '?')})"

    def _futures(self, env: BSRPEnv):
        inst, t = env.inst, env.t
        L = N_INTERVALS - t
        dep = np.zeros((self.K, N_INTERVALS, env.n), np.int16)
        arr = np.zeros((self.K, N_INTERVALS, env.n), np.int16)
        dep[:, t:] = self.rng.poisson(inst.lam_dep[t:][None], size=(self.K, L, env.n))
        arr[:, t:] = self.rng.poisson(inst.lam_arr[t:][None], size=(self.K, L, env.n))
        return dep, arr

    def act(self, env: BSRPEnv):
        dep, arr = self._futures(env)
        valid = np.flatnonzero(env.valid_mask())
        best_a, best_v = int(valid[0]), -np.inf
        for a in valid:
            tot = 0.0
            for k in range(self.K):
                c = env.clone()
                c.sc = Scenario(env.sc.init_occ, dep[k], arr[k])
                r, _ = c.step(int(a))
                for _ in range(self.h):
                    if c.done:
                        break
                    b = self.base.act(c)
                    r2, _ = c.step_raw(*b) if isinstance(b, tuple) else c.step(b)
                    r += r2
                tot += r
            if tot > best_v:
                best_v, best_a = tot, int(a)
        return best_a
