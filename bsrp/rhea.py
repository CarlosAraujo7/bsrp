"""
bsrp/rhea.py — Rolling Horizon Evolutionary Algorithm (RHEA): planejador EVOLUTIVO ONLINE com o modelo da demanda (DESVIOS.md 2026-10-22, P12c).

A cada decisão evolui-se uma população de SEQUÊNCIAS de L ações planas (destino × delta), avaliadas em K futuros AMOSTRADOS DO MODELO de Poisson (números aleatórios
comuns para toda a população naquela decisão; nunca a demanda real). A primeira ação do melhor plano é executada; o plano restante, deslocado de uma posição, entra na
população da decisão seguinte (shift buffer). Operadores: seleção por torneio, cruzamento uniforme, mutação gene a gene, elitismo. O primeiro gene é sempre uma ação
VÁLIDA no estado real; os demais genes são recortados pelo simulador (como em `BSRPEnv.step`). É o análogo evolutivo do `policies.Rollout` (Monte Carlo + heurística-base).
"""
from __future__ import annotations

import numpy as np

from .env import N_INTERVALS, BSRPEnv, Scenario


class RHEA:
    def __init__(self, L=4, pop=16, gens=10, K=8, elite=2, mut=None, tourn=3, seed=0):
        self.L, self.pop, self.gens, self.K, self.elite, self.tourn = int(L), int(pop), int(gens), int(K), int(elite), int(tourn)
        self.mut = mut if mut is not None else 1.0 / max(1, int(L))
        self.rng = np.random.default_rng(seed)
        self._plan = None
        self._t_prev = -1
        self.name = f"rhea(L={L},P={pop},G={gens},K={K})"

    # -- futuros do modelo -------------------------------------------------------
    def _futures(self, env: BSRPEnv):
        inst, t = env.inst, env.t
        span = min(self.L + 1, N_INTERVALS - t)                      # cada ação consome ≥ 1 intervalo
        dep = np.zeros((self.K, N_INTERVALS, env.n), np.int16)
        arr = np.zeros((self.K, N_INTERVALS, env.n), np.int16)
        dep[:, t:t + span] = self.rng.poisson(inst.lam_dep[t:t + span][None], size=(self.K, span, env.n))
        arr[:, t:t + span] = self.rng.poisson(inst.lam_arr[t:t + span][None], size=(self.K, span, env.n))
        return dep, arr

    def _fitness(self, env, genome, dep, arr):
        tot = 0.0
        for k in range(self.K):
            c = env.clone()
            c.sc = Scenario(env.sc.init_occ, dep[k], arr[k])
            for a in genome:
                if c.done:
                    break
                r, _ = c.step(int(a))
                tot += r
        return tot / self.K

    # -- evolução ------------------------------------------------------------------
    def _random_genome(self, valid0, n_act):
        g = self.rng.integers(0, n_act, size=self.L)
        g[0] = self.rng.choice(valid0)
        return g

    def act(self, env: BSRPEnv):
        n_act = env.n * 5
        valid0 = np.flatnonzero(env.valid_mask())
        if env.t == 0 or env.t < self._t_prev:                       # novo episódio
            self._plan = None
        self._t_prev = env.t
        dep, arr = self._futures(env)
        popu = []
        if self._plan is not None:                                    # shift buffer: plano anterior sem a ação já executada + gene novo
            g = np.concatenate([self._plan, self.rng.integers(0, n_act, size=1)])[: self.L]
            if g[0] not in set(valid0.tolist()):
                g[0] = self.rng.choice(valid0)
            popu.append(g)
        while len(popu) < self.pop:
            popu.append(self._random_genome(valid0, n_act))
        fit = np.array([self._fitness(env, g, dep, arr) for g in popu])
        for _ in range(self.gens - 1):
            order = np.argsort(-fit)
            nxt = [popu[i].copy() for i in order[: self.elite]]
            nfit = list(fit[order[: self.elite]])
            while len(nxt) < self.pop:
                pa = popu[max(self.rng.integers(0, len(popu), size=self.tourn), key=lambda i: fit[i])]
                pb = popu[max(self.rng.integers(0, len(popu), size=self.tourn), key=lambda i: fit[i])]
                child = np.where(self.rng.random(self.L) < 0.5, pa, pb)                    # cruzamento uniforme (gene 0 herda de um pai válido)
                m = self.rng.random(self.L) < self.mut
                child = np.where(m, self.rng.integers(0, n_act, size=self.L), child)
                if m[0]:
                    child[0] = self.rng.choice(valid0)
                nxt.append(child)
                nfit.append(self._fitness(env, child, dep, arr))
            popu, fit = nxt, np.array(nfit)
        best = popu[int(np.argmax(fit))]
        self._plan = best[1:].copy()
        return int(best[0])
