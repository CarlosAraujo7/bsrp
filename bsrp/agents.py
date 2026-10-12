"""
bsrp/agents.py — aprendizes do protocolo (PROTOCOLO_RIGOR.md §5), numpy puro:

  R0  QTabular(repr="cluster")   Q-learning tabular, ocupação por cluster (4 níveis) + posição + carga + t
  R1  QTabular(repr="station")   Q-learning tabular, ocupação binária por estação  + posição + carga + t
  R2  DQN(hidden=())             mesmo laço DQN, aproximador LINEAR nas features do DQN legado
  R3  DQN(hidden=(64, 64))       rede do DQN legado (Double DQN, replay, rede-alvo)

Todos usam espaço de ações FIXO (5n) com máscara de ações válidas (inclusive no alvo), e geradores aleatórios
independentes passados explicitamente (nada de np.random global). Também contém o porte da política DQN legada
(`LegacyDQNPolicy`, só para regressão).
"""
from __future__ import annotations

import numpy as np

from .env import DELTAS, N_DELTAS, VEHICLE_CAP, BSRPEnv


def _relu(x):
    return np.maximum(0, x)


class LegacyDQNPolicy:
    """Rede 26→64→64→5n do legado: argmax de Q sobre ações válidas (ε = 0)."""
    name = "legacy_dqn"

    def __init__(self, W: dict):
        self.W = {k: np.asarray(v, np.float32) for k, v in W.items()}

    @staticmethod
    def from_npz(path):
        z = np.load(path)
        return LegacyDQNPolicy({k: z[k] for k in z.files})

    def q(self, x):
        W = self.W
        h1 = _relu(x @ W["W1"] + W["b1"])
        h2 = _relu(h1 @ W["W2"] + W["b2"])
        return h2 @ W["W3"] + W["b3"]

    def act(self, env: BSRPEnv):
        q = self.q(env.features()[None])[0].copy()
        q[~env.valid_mask()] = -np.inf
        return int(np.argmax(q))


# ──────────────────────────────────────────────────────────────────────────────
# REDES (MLP genérica; hidden=() => linear)
# ──────────────────────────────────────────────────────────────────────────────
class Net:
    def __init__(self, n_in, hidden, n_out, rng: np.random.Generator):
        sizes = [n_in, *hidden, n_out]
        self.W = []
        self.b = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.b.append(np.zeros(b, np.float32))

    def forward(self, x, keep=False):
        acts = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.maximum(0, x)
            acts.append(x)
        return (x, acts) if keep else x

    def backward(self, acts, dq):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dq
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = acts[k].T @ d
            gb[k] = d.sum(0)
            if k > 0:
                d = (d @ self.W[k].T) * (acts[k] > 0)
        return gW, gb

    def copy_from(self, other: "Net"):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


class Adam:
    def __init__(self, net: Net, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps, self.t = net, lr, b1, b2, eps, 0
        self.mW = [np.zeros_like(w) for w in net.W]; self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]; self.vb = [np.zeros_like(b) for b in net.b]

    def step(self, gW, gb):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in range(len(self.net.W)):
            for p, g, m, v in ((self.net.W[k], gW[k], self.mW[k], self.vW[k]), (self.net.b[k], gb[k], self.mb[k], self.vb[k])):
                m *= self.b1; m += (1 - self.b1) * g
                v *= self.b2; v += (1 - self.b2) * g * g
                p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class Replay:
    """Buffer circular em arrays; amostragem com o gerador passado (reprodutível)."""

    def __init__(self, cap, n_in, n_act):
        self.s = np.zeros((cap, n_in), np.float32); self.s2 = np.zeros((cap, n_in), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_act), bool)
        self.cap, self.n, self.i = cap, 0, 0

    def push(self, s, a, r, s2, d, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.m2[i] = s, a, r, s2, d, m2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k, rng):
        idx = rng.integers(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx]


# ──────────────────────────────────────────────────────────────────────────────
# AGENTES  (interface: select(env, eps, rng) / observe(...) / act(env) [ε = 0])
# ──────────────────────────────────────────────────────────────────────────────
class _GreedyAct:
    def act(self, env: BSRPEnv):
        return self.select(env, 0.0, None)


class DQN(_GreedyAct):
    def __init__(self, inst, hidden=(64, 64), lr=3e-4, gamma=1.0, buffer=10_000, batch=64, train_every=4,
                 target_sync=1000, double=True, use_replay=True, use_target=True,
                 rng_init=None, rng_sample=None):
        self.n_in, self.n_act = 2 * inst.n + 2, inst.n_actions
        self.net = Net(self.n_in, hidden, self.n_act, rng_init)
        self.tgt = Net(self.n_in, hidden, self.n_act, rng_init)
        self.tgt.copy_from(self.net)
        self.opt = Adam(self.net, lr)
        self.gamma, self.batch, self.train_every = gamma, batch, train_every
        self.target_sync, self.double, self.use_replay, self.use_target = target_sync, double, use_replay, use_target
        self.buf = Replay(buffer, self.n_in, self.n_act)
        self.rng_sample = rng_sample
        self.steps = 0
        self.name = f"dqn{list(hidden)}" if hidden else "linear"

    def select(self, env, eps, rng):
        mask = env.valid_mask()
        if rng is not None and rng.random() < eps:
            return int(rng.choice(np.flatnonzero(mask)))
        q = self.net.forward(env.features()[None])[0].copy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def observe(self, s, a, r, s2, done, m2):
        self.steps += 1
        if self.use_replay:
            self.buf.push(s, a, r, s2, float(done), m2)
            if self.buf.n >= self.batch and self.steps % self.train_every == 0:
                self._update(*self.buf.sample(self.batch, self.rng_sample))
        else:
            self._update(s[None], np.array([a]), np.array([r], np.float32), s2[None], np.array([float(done)], np.float32), m2[None])
        if self.use_target and self.steps % self.target_sync == 0:
            self.tgt.copy_from(self.net)

    def _update(self, s, a, r, s2, d, m2):
        q, acts = self.net.forward(s, keep=True)
        tnet = self.tgt if self.use_target else self.net
        q2t = tnet.forward(s2)
        if self.double:
            q2n = self.net.forward(s2)
            q2n = np.where(m2, q2n, -np.inf)
            a2 = q2n.argmax(1)
            nxt = q2t[np.arange(len(a2)), a2]
        else:
            nxt = np.where(m2, q2t, -np.inf).max(1)
        td = r + self.gamma * nxt * (1 - d)
        tgt = q.copy()
        tgt[np.arange(len(a)), a] = td
        gW, gb = self.net.backward(acts, (q - tgt) / len(a))
        self.opt.step(gW, gb)


class QTabular(_GreedyAct):
    def __init__(self, inst, repr="cluster", alpha=0.1, gamma=1.0):
        assert repr in ("cluster", "station")
        self.repr, self.alpha, self.gamma, self.n_act = repr, alpha, gamma, inst.n_actions
        self.cl = inst.cluster
        self.ncl = inst.n_clusters
        self.cl_cap = np.array([inst.cap[inst.cluster == c].sum() for c in range(self.ncl)], float)
        self.cap = inst.cap
        self.Q: dict = {}
        self.steps = 0
        self.name = f"tab_{repr}"

    def key(self, env):
        if self.repr == "cluster":
            occ_c = np.bincount(self.cl, weights=env.occ, minlength=self.ncl)
            lv = np.minimum((occ_c / np.maximum(self.cl_cap, 1) * 4).astype(np.int64), 3)
        else:
            lv = (env.occ * 2 >= self.cap).astype(np.int64)
        return (*lv.tolist(), env.pos, min(3, env.load // 5), env.t)

    def _q(self, k):
        q = self.Q.get(k)
        if q is None:
            q = self.Q[k] = np.zeros(self.n_act, np.float32)
        return q

    def select(self, env, eps, rng):
        mask = env.valid_mask()
        if rng is not None and rng.random() < eps:
            return int(rng.choice(np.flatnonzero(mask)))
        q = np.where(mask, self._q(self.key(env)), -np.inf)
        return int(np.argmax(q))

    def update(self, k, a, r, k2, done, m2):
        self.steps += 1
        q = self._q(k)
        nxt = 0.0 if done else float(np.where(m2, self._q(k2), -np.inf).max())
        q[a] += self.alpha * (r + self.gamma * nxt - q[a])

    @property
    def n_states(self):
        return len(self.Q)


class PolicyGradient(_GreedyAct):
    """REINFORCE com linha de base aprendida (critic MC), softmax mascarado e bônus de entropia.
    Atualiza a cada `episodes_per_update` episódios; ignora ε (explora pela própria política)."""

    def __init__(self, inst, hidden=(64, 64), lr_actor=3e-4, lr_critic=1e-3, gamma=1.0, entropy=0.01,
                 episodes_per_update=8, rng_init=None, rng_sample=None):
        self.n_in, self.n_act = 2 * inst.n + 2, inst.n_actions
        self.actor = Net(self.n_in, hidden, self.n_act, rng_init)
        self.critic = Net(self.n_in, hidden, 1, rng_init)
        self.opt_a, self.opt_c = Adam(self.actor, lr_actor), Adam(self.critic, lr_critic)
        self.gamma, self.beta, self.K = gamma, entropy, episodes_per_update
        self.traj, self.batch = [], []
        self._m = None
        self.name = "pg[64, 64]"

    def _probs(self, x, mask):
        z = self.actor.forward(x)
        z = np.where(mask, z, -1e9)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z) * mask
        return e / e.sum(-1, keepdims=True)

    def select(self, env, eps, rng):
        mask = env.valid_mask()
        self._m = mask
        if rng is None:                                   # avaliação: ação mais provável
            return int(np.argmax(self._probs(env.features()[None], mask)[0]))
        p = self._probs(env.features()[None], mask)[0]
        return int(rng.choice(self.n_act, p=p))

    def observe(self, s, a, r, s2, done, m2):
        self.traj.append((s, a, r, self._m))

    def end_episode(self):
        T = len(self.traj)
        if T == 0:
            return
        rets, g = np.zeros(T, np.float32), 0.0
        for t in range(T - 1, -1, -1):
            g = self.traj[t][2] + self.gamma * g
            rets[t] = g
        s = np.stack([x[0] for x in self.traj]); a = np.array([x[1] for x in self.traj]); m = np.stack([x[3] for x in self.traj])
        self.batch.append((s, a, m, rets))
        self.traj = []
        if len(self.batch) >= self.K:
            self._update()

    def _update(self):
        s = np.concatenate([b[0] for b in self.batch]); a = np.concatenate([b[1] for b in self.batch])
        m = np.concatenate([b[2] for b in self.batch]); G = np.concatenate([b[3] for b in self.batch])
        self.batch = []
        v, acts_c = self.critic.forward(s, keep=True)
        adv = G - v[:, 0]
        gW, gb = self.critic.backward(acts_c, (v - G[:, None]) / len(G))
        self.opt_c.step(gW, gb)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        z, acts_a = self.actor.forward(s, keep=True)
        zm = np.where(m, z, -1e9); zm = zm - zm.max(1, keepdims=True)
        p = np.exp(zm) * m; p /= p.sum(1, keepdims=True)
        onehot = np.zeros_like(p); onehot[np.arange(len(a)), a] = 1.0
        logp = np.log(np.where(m, p, 1.0) + 1e-12)
        H = -(p * logp).sum(1, keepdims=True)
        dz = -adv[:, None] * (onehot - p) + self.beta * p * (logp + H)
        gW, gb = self.actor.backward(acts_a, (dz * m) / len(a))
        self.opt_a.step(gW, gb)
