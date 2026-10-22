"""
bsrp/evo.py — métodos EVOLUTIVOS do estudo (DESVIOS.md 2026-10-22, P12), numpy puro.

* CMAES     : (μ/μ_w, λ)-CMA-ES (Hansen & Ostermeier) com controle de passo CSA, atualização de posto 1 e posto μ. Minimiza f.
* ESAgent   : NEUROEVOLUÇÃO por estratégia evolutiva ao estilo OpenAI-ES (Salimans et al., 2017) sobre a MESMA rede do DQN (MLP 64–64 → pontuação por ação;
              ação = argmax sobre as ações válidas). Amostragem antitética, fitness por postos centrados (comum a toda a população) e Adam sobre o gradiente
              estimado. Todos os membros da população de uma geração veem os MESMOS `bs` cenários (números aleatórios comuns). O orçamento é contado em PASSOS DE
              AMBIENTE (decisões), igual ao dos aprendizes por reforço; a avaliação periódica em validação não conta.
Interface de agente igual à do DQN: act(env) / select(env, eps, rng); `train()` de bsrp/train.py chama `fit(...)` quando o agente é um ESAgent.
"""
from __future__ import annotations

import time

import numpy as np

from .agents import Net, _GreedyAct
from .env import N_DELTAS, BSRPEnv, evaluate, run_episode, sample_scenarios


# ──────────────────────────────────────────────────────────────────────────────
# CMA-ES
# ──────────────────────────────────────────────────────────────────────────────
class CMAES:
    """CMA-ES padrão. ask() devolve (λ, n); tell(X, f) atualiza com os valores de f (MENOR é melhor)."""

    def __init__(self, x0, sigma0=1.0, popsize=None, rng=None):
        self.n = n = len(x0)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.C = np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.ps, self.pc = np.zeros(n), np.zeros(n)
        self.gen = 0
        self._eig_gen = 0

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = (z * self.D) @ self.B.T                 # y = B D z
        return self.mean + self.sigma * self._y

    def tell(self, X, f):
        order = np.argsort(f)[: self.mu]
        ysel = (np.asarray(X)[order] - self.mean) / self.sigma
        old = self.mean.copy()
        self.mean = old + self.sigma * (self.w @ ysel)
        yw = (self.mean - old) / self.sigma
        Cinvsqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinvsqrt @ yw)
        hsig = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (ysel.T * self.w) @ ysel
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1
        if self.gen - self._eig_gen >= 1 / (self.c1 + self.cmu) / self.n / 10:               # decomposição espectral esporádica
            self._eig_gen = self.gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d2, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(d2, 1e-20))


# ──────────────────────────────────────────────────────────────────────────────
# NEUROEVOLUÇÃO (OpenAI-ES)
# ──────────────────────────────────────────────────────────────────────────────
def _flat_params(net: Net) -> np.ndarray:
    return np.concatenate([w.ravel() for w in net.W] + [b.ravel() for b in net.b]).astype(np.float32)


def _set_params(net: Net, theta: np.ndarray) -> None:
    i = 0
    for k in range(len(net.W)):
        n = net.W[k].size
        net.W[k] = theta[i:i + n].reshape(net.W[k].shape)
        i += n
    for k in range(len(net.b)):
        n = net.b[k].size
        net.b[k] = theta[i:i + n]
        i += n


def centered_ranks(f: np.ndarray) -> np.ndarray:
    """Postos centrados em [−0,5; 0,5] (empates recebem postos distintos pela ordem estável): fitness invariante a transformações monótonas."""
    r = np.empty(len(f), np.float64)
    r[np.argsort(f, kind="stable")] = np.arange(len(f))
    return r / (len(f) - 1) - 0.5


class ESAgent(_GreedyAct):
    def __init__(self, inst, hidden=(64, 64), pop=64, sigma=0.05, lr=0.02, bs=4, rng_init=None, rng_sample=None):
        assert pop % 2 == 0 and pop >= 2, "população par (amostragem antitética)"
        self.n_in, self.n_act = 2 * inst.n + 2, inst.n_actions
        self.net = Net(self.n_in, hidden, self.n_act, rng_init)
        self.theta = _flat_params(self.net)
        _set_params(self.net, self.theta)
        self.pop, self.sigma, self.lr, self.bs = int(pop), float(sigma), float(lr), int(bs)
        self.name = f"es{list(hidden)}"

    def select(self, env, eps, rng):
        q = self.net.forward(env.features()[None])[0].copy()
        q[~env.valid_mask()] = -np.inf
        return int(np.argmax(q))

    def _fitness(self, env, theta, scen):
        _set_params(self.net, theta)
        tot, steps = 0.0, 0
        for k in range(len(scen)):
            r = run_episode(env, self, scen[k])
            tot += r["reward"]
            steps += r["steps"]
        return tot / len(scen), steps

    def fit(self, inst, r_noise, r_env, budget_steps, val, eval_every):
        env, env_val = BSRPEnv(inst), BSRPEnv(inst)
        d = len(self.theta)
        half = self.pop // 2
        m, v, t = np.zeros(d), np.zeros(d), 0
        steps, episodes, next_eval, gen = 0, 0, eval_every, 0
        curve, t0 = [], time.time()
        theta = self.theta.astype(np.float64)
        while steps < budget_steps:
            scen = sample_scenarios(inst, self.bs, r_env)                     # mesmos cenários para toda a população (números aleatórios comuns)
            eps = r_noise.standard_normal((half, d))
            fp, fm = np.empty(half), np.empty(half)
            for i in range(half):
                fp[i], s1 = self._fitness(env, (theta + self.sigma * eps[i]).astype(np.float32), scen)
                fm[i], s2 = self._fitness(env, (theta - self.sigma * eps[i]).astype(np.float32), scen)
                steps += s1 + s2
            episodes += 2 * half * self.bs
            u = centered_ranks(np.concatenate([fp, fm]))
            g = ((u[:half] - u[half:]) @ eps) / (self.pop * self.sigma)       # gradiente (ASCENDENTE) estimado
            t += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta = theta + self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            gen += 1
            if steps >= next_eval:
                _set_params(self.net, theta.astype(np.float32))
                curve.append({"steps": steps, "episodes": episodes, "val": float(evaluate(env_val, self, val).mean())})
                next_eval += eval_every
        self.theta = theta.astype(np.float32)
        _set_params(self.net, self.theta)                                       # checkpoint final = último (sem seleção)
        return {"curve": curve, "episodes": episodes, "train_seconds": round(time.time() - t0, 1), "n_states": None, "generations": gen}
