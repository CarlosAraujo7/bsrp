"""
bsrp/hybrid.py — valor terminal aprendido para o RHEA (DESVIOS.md 2026-10-22, P13).

O RHEA-V avalia cada sequência de ações como [soma das recompensas dos L passos simulados no MODELO] + V(estado final), com V(s) = max sobre as ações VÁLIDAS de
Q(s, ·) da rede de um DQN já treinado e CONGELADO (gamma = 1: o valor está na mesma unidade da recompensa). Aqui ficam o salvamento/carregamento dos pesos do DQN
e o `QValue` (valor em lote e política gulosa da rede); a busca evolutiva é `bsrp/rhea.py` (parâmetro `value`).
"""
from __future__ import annotations

import numpy as np


def save_dqn(agent, path) -> None:
    """Grava os pesos da rede do DQN (W0, b0, W1, b1, ...) em .npz."""
    arrs = {}
    for k, (W, b) in enumerate(zip(agent.net.W, agent.net.b)):
        arrs[f"W{k}"], arrs[f"b{k}"] = W, b
    np.savez(path, **arrs)


class QValue:
    """Rede Q congelada: `__call__(envs)` devolve V(s) = max_a Q(s, a) sobre ações válidas (0 se o episódio acabou); `act(env)` é a política gulosa do DQN."""
    name = "dqn_frozen"

    def __init__(self, W, b):
        self.W = [np.asarray(w, np.float32) for w in W]
        self.b = [np.asarray(x, np.float32) for x in b]

    @staticmethod
    def from_npz(path) -> "QValue":
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        return QValue([z[f"W{k}"] for k in range(n)], [z[f"b{k}"] for k in range(n)])

    def q(self, x):
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.maximum(0, x)
        return x

    def __call__(self, envs) -> np.ndarray:
        feats = np.stack([e.features() for e in envs])
        q = self.q(feats)
        q[~np.stack([e.valid_mask() for e in envs])] = -np.inf
        v = q.max(axis=1)
        v[np.array([e.done for e in envs])] = 0.0
        return v

    def act(self, env) -> int:
        q = self.q(env.features()[None])[0].copy()
        q[~env.valid_mask()] = -np.inf
        return int(np.argmax(q))
