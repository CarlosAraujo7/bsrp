"""
bsrp/seo.py — adaptação do A2C de Seo et al. (2022), "Rebalancing Docked Bicycle Sharing System with Approximate Dynamic
Programming and Reinforcement Learning", J. Adv. Transp. 2022:2780711 (DESVIOS.md 2026-10-14, P4b).

É uma ADAPTAÇÃO, não uma reprodução: o simulador, o passo de decisão, o estado (aqui as features do DQN em vez do índice binário
de preenchimento, que depende de um buffer β não informado no texto), a ação (5n com máscara) e a recompensa diferem do artigo.
O que segue o artigo: ator-crítico com vantagem TD(0), atualização ONLINE a cada transição (Algoritmo 2), rede de 3 camadas
ocultas × 24 unidades, taxa de aprendizado 5e-4 no ator e no crítico, fator de desconto γ = 0,9; sem bônus de entropia.
`entropy` (β) e `reward_scale` (c) têm padrão 0 e 1 (= como publicado) e só mudam na variante AJUSTADA (DESVIOS.md 2026-10-15).
"""
from __future__ import annotations

import numpy as np

from .agents import Adam, Net, _GreedyAct


class A2C(_GreedyAct):
    def __init__(self, inst, hidden=(24, 24, 24), lr_actor=5e-4, lr_critic=5e-4, gamma=0.9, entropy=0.0, reward_scale=1.0,
                 rng_init=None, **_unused):
        self.n_in, self.n_act = 2 * inst.n + 2, inst.n_actions
        self.actor = Net(self.n_in, hidden, self.n_act, rng_init)
        self.critic = Net(self.n_in, hidden, 1, rng_init)
        self.opt_a, self.opt_c = Adam(self.actor, lr_actor), Adam(self.critic, lr_critic)
        self.gamma = gamma
        self.entropy, self.reward_scale = float(entropy), float(reward_scale)    # 0 e 1 = como publicado (DESVIOS.md 2026-10-15)
        self._m = None
        self.name = f"a2c_seo(gamma={gamma:g})"

    def _probs(self, x, mask):
        z = self.actor.forward(x)
        z = np.where(mask, z, -1e9)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z) * mask
        return e / e.sum(-1, keepdims=True)

    def select(self, env, eps, rng):
        """Amostra da política (explora pela própria política; `eps` é ignorado). Sem `rng` (avaliação): ação mais provável."""
        mask = env.valid_mask()
        self._m = mask
        p = self._probs(env.features()[None], mask)[0]
        if rng is None:
            return int(np.argmax(p))
        return int(rng.choice(self.n_act, p=p))

    def observe(self, s, a, r, s2, done, m2):
        s1, s21 = s[None], s2[None]
        v, acts_c = self.critic.forward(s1, keep=True)
        v2 = self.critic.forward(s21)
        td = float(self.reward_scale * r + self.gamma * v2[0, 0] * (1.0 - float(done)) - v[0, 0])    # δ = c·r + γ V(s') − V(s)
        gW, gb = self.critic.backward(acts_c, np.array([[-td]], np.float32))           # semi-gradiente de ½ δ² em V
        self.opt_c.step(gW, gb)
        z, acts_a = self.actor.forward(s1, keep=True)
        m = self._m[None]
        zm = np.where(m, z, -1e9)
        zm = zm - zm.max(1, keepdims=True)
        p = np.exp(zm) * m
        p /= p.sum(1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[0, a] = 1.0
        dz = -td * (onehot - p)                                                        # gradiente de −δ·log π(a|s) nos logits
        if self.entropy > 0.0:                                                         # bônus de entropia (só na variante ajustada)
            logp = np.log(np.where(m, p, 1.0) + 1e-12)
            H = -(p * logp).sum(1, keepdims=True)
            dz = dz + self.entropy * p * (logp + H)                                    # gradiente de −β·H nos logits
        gW, gb = self.actor.backward(acts_a, dz * m)
        self.opt_a.step(gW, gb)
