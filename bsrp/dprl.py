"""
bsrp/dprl.py — ADAPTAÇÃO do DPRL (Dual-Policy RL) de Liang et al. (2024) ao nosso simulador de 1 veículo (DESVIOS.md 2026-10-20, P11).

No artigo original, duas DQN compartilham a representação de estado: a política de INVENTÁRIO decide, quando o veículo chega a uma estação, quantas bicicletas
coletar/deixar (3 níveis de preenchimento), e a política de ROTEAMENTO decide, antes de partir, a próxima estação. Aqui cada decisão do simulador (destino, delta)
vira DUAS sub-decisões em sequência: (1) inventário: escolhe delta ∈ {−10, −5, 0, 5, 10} (válido na estação atual), efeito imediato e sem passagem de tempo;
(2) roteamento: escolhe o destino (n estações) no estado pós-inventário; só então o tempo passa e a recompensa (demanda perdida) é recebida. O estado é o do DQN
(`env.features()`) mais um bit de estágio (0 = inventário, 1 = roteamento). As cabeças têm n_ações = 5 e n, em vez de 5n.

É uma ADAPTAÇÃO, não uma reprodução: 1 veículo, simulador e instâncias próprios, cinco níveis de delta (os do nosso problema) em vez de três níveis de preenchimento,
sem inicialização heurística do roteamento, rede 64–64 e os hiperparâmetros do DQN plano (não reajustados). Mesma regra de aprendizado do DQN (Double DQN, replay,
rede-alvo); o valor do estado seguinte de cada cabeça vem da OUTRA cabeça (inventário → roteamento → inventário), com γ = 1 e episódio de horizonte fixo.
"""
from __future__ import annotations

import numpy as np

from .agents import Adam, Net, Replay, _GreedyAct
from .env import DELTAS, N_DELTAS, VEHICLE_CAP, BSRPEnv


def apply_inventory(env: BSRPEnv, delta: int) -> None:
    """Primeira metade de `BSRPEnv.step_raw`: carga/descarga na estação atual, sem passagem de tempo (mesma semântica e mesmo recorte)."""
    pos, cap_p = env.pos, int(env.cap[env.pos])
    occ_p = int(env.occ[pos])
    if delta > 0:
        delta = min(delta, env.load, cap_p - occ_p)
    elif delta < 0:
        delta = -min(-delta, occ_p, VEHICLE_CAP - env.load)
    env.occ[pos] = min(max(occ_p + delta, 0), cap_p)
    env.load -= delta


class _Head:
    """Uma DQN (rede, rede-alvo, Adam, replay) com n_ações próprio; o alvo usa os valores do estado seguinte da OUTRA cabeça."""

    def __init__(self, n_in, n_act, n_next, hidden, lr, buffer, rng_init):
        self.n_act = n_act
        self.net, self.tgt = Net(n_in, hidden, n_act, rng_init), Net(n_in, hidden, n_act, rng_init)
        self.tgt.copy_from(self.net)
        self.opt = Adam(self.net, lr)
        self.buf = Replay(buffer, n_in, n_next)                     # a máscara guardada é a do ESTADO SEGUINTE (ações da outra cabeça)


class DualDQN(_GreedyAct):
    def __init__(self, inst, hidden=(64, 64), lr=3e-4, gamma=1.0, buffer=10_000, batch=64, train_every=4, target_sync=1000,
                 rng_init=None, rng_sample=None):
        self.n = inst.n
        self.n_in = 2 * inst.n + 3                                  # features do DQN + bit de estágio
        # a cabeça de inventário olha para o roteamento (n destinos, todos válidos) e vice-versa (5 deltas, máscara do estado seguinte)
        self.inv = _Head(self.n_in, N_DELTAS, inst.n, hidden, lr, buffer, rng_init)
        self.route = _Head(self.n_in, inst.n, N_DELTAS, hidden, lr, buffer, rng_init)
        self.gamma, self.batch, self.train_every, self.target_sync = gamma, batch, train_every, target_sync
        self.rng_sample = rng_sample
        self.steps = 0
        self._pend = None
        self.name = f"dprl{list(hidden)}"

    # -- seleção: inventário, depois roteamento no estado pós-inventário --------------
    def _pick(self, head, s, mask, eps, rng):
        if rng is not None and rng.random() < eps:
            return int(rng.choice(np.flatnonzero(mask)))
        q = head.net.forward(s[None])[0].copy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def select(self, env, eps, rng):
        s_inv = np.append(env.features(), np.float32(0.0)).astype(np.float32)
        k = self._pick(self.inv, s_inv, env.valid_delta(), eps, rng)
        c = env.clone()
        apply_inventory(c, int(DELTAS[k]))
        s_route = np.append(c.features(), np.float32(1.0)).astype(np.float32)
        dest = self._pick(self.route, s_route, np.ones(self.n, bool), eps, rng)
        self._pend = (s_inv, k, s_route, dest)
        return dest * N_DELTAS + k                                  # mesmo índice plano do simulador: destino × 5 + delta

    def observe(self, s, a, r, s2, done, m2):
        s_inv, k, s_route, dest = self._pend
        assert a == dest * N_DELTAS + k, "observe deve seguir o select da mesma decisão"
        s_next = np.append(s2, np.float32(0.0)).astype(np.float32)   # próximo estado de inventário
        self.inv.buf.push(s_inv, k, 0.0, s_route, 0.0, np.ones(self.n, bool))          # inventário: sem recompensa, segue para o roteamento
        self.route.buf.push(s_route, dest, r, s_next, float(done), m2[:N_DELTAS])      # roteamento: recebe a demanda perdida; m2[:5] = deltas válidos no próximo estado
        self.steps += 1
        if self.steps % self.train_every == 0:
            for head, other in ((self.inv, self.route), (self.route, self.inv)):
                if head.buf.n >= self.batch:
                    self._update(head, other, *head.buf.sample(self.batch, self.rng_sample))
        if self.steps % self.target_sync == 0:
            self.inv.tgt.copy_from(self.inv.net)
            self.route.tgt.copy_from(self.route.net)

    def _update(self, head, other, s, a, r, s2, d, m2):
        q, acts = head.net.forward(s, keep=True)
        q2t = other.tgt.forward(s2)                                 # valor do estado seguinte vem da outra cabeça (Double DQN)
        q2n = np.where(m2, other.net.forward(s2), -np.inf)
        a2 = q2n.argmax(1)
        nxt = q2t[np.arange(len(a2)), a2]
        td = r + self.gamma * nxt * (1 - d)
        tgt = q.copy()
        tgt[np.arange(len(a)), a] = td
        gW, gb = head.net.backward(acts, (q - tgt) / len(a))
        head.opt.step(gW, gb)
