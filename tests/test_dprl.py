"""Adaptação do DPRL (bsrp/dprl.py): inventário + roteamento equivalem à ação plana do simulador; regra de aprendizado igual à do DQN; reprodutível."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import agents as AG                      # noqa: E402
from bsrp import dprl as DP                        # noqa: E402
from bsrp import env as E                          # noqa: E402
from bsrp import train as T                        # noqa: E402

INST = E.legacy_instance(4)


def scenario(seed=0):
    return E.sample_scenarios(INST, 1, np.random.default_rng(seed))[0]


class TestInventoryHalf(unittest.TestCase):
    def test_inventory_then_travel_equals_the_flat_step(self):
        rng = np.random.default_rng(1)
        for trial in range(60):
            a, b = E.BSRPEnv(INST), E.BSRPEnv(INST)
            sc = scenario(trial % 5)
            a.reset(sc)
            b.reset(sc)
            for _ in range(8):                                       # estados variados: avança com ações válidas aleatórias
                act = int(rng.choice(np.flatnonzero(a.valid_mask())))
                a.step(act); b.step(act)
            dest, k = int(rng.integers(0, INST.n)), int(rng.integers(0, E.N_DELTAS))
            ra, _ = a.step_raw(dest, int(E.DELTAS[k]))
            DP.apply_inventory(b, int(E.DELTAS[k]))
            rb, _ = b.step_raw(dest, 0)
            self.assertEqual(ra, rb)
            np.testing.assert_array_equal(a.occ, b.occ)
            self.assertEqual((a.load, a.pos, a.t), (b.load, b.pos, b.t))

    def test_apply_inventory_does_not_advance_time_or_move(self):
        env = E.BSRPEnv(INST).reset(scenario())
        t0, pos0 = env.t, env.pos
        DP.apply_inventory(env, 10)
        self.assertEqual((env.t, env.pos), (t0, pos0))


class TestDualDQN(unittest.TestCase):
    def make(self, seed=0, **kw):
        from bsrp.train import streams
        r_init, _, _, r_samp = streams(seed)
        return DP.DualDQN(INST, rng_init=r_init, rng_sample=r_samp, **kw)

    def test_heads_have_the_factored_action_spaces_and_a_stage_bit_in_the_input(self):
        ag = self.make()
        self.assertEqual(ag.n_in, 2 * INST.n + 3)
        self.assertEqual(ag.inv.net.W[-1].shape[1], E.N_DELTAS)
        self.assertEqual(ag.route.net.W[-1].shape[1], INST.n)

    def test_select_returns_valid_flat_actions_and_observe_fills_both_buffers(self):
        ag = self.make(buffer=50, batch=8, train_every=1, target_sync=10)
        env = E.BSRPEnv(INST).reset(scenario())
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = env.features()
            a = ag.select(env, 0.5, rng)
            self.assertTrue(env.valid_mask()[a])                      # ação plana válida (delta válido na estação atual)
            r, done = env.step(a)
            ag.observe(s, a, r, env.features(), done, env.valid_mask())
            if done:
                break
        self.assertEqual(ag.inv.buf.n, ag.route.buf.n)
        self.assertGreater(ag.inv.buf.n, 0)

    def test_update_rule_matches_the_flat_dqn_when_the_other_head_is_the_same_network(self):
        """Com uma cabeça apontando para si mesma, `_update` deve coincidir com `DQN._update` (mesmo Double DQN)."""
        from bsrp.train import streams
        r_init, _, _, r_samp = streams(3)
        flat = AG.DQN(INST, hidden=(16, 16), lr=1e-3, rng_init=r_init, rng_sample=r_samp)
        # cabeça com o mesmo tamanho de saída do DQN plano: reusa a mesma rede
        ag = self.make(seed=3, hidden=(16, 16), lr=1e-3)
        head = DP._Head(flat.n_in, flat.n_act, flat.n_act, (16, 16), 1e-3, 10, np.random.default_rng(0))
        head.net.copy_from(flat.net); head.tgt.copy_from(flat.tgt)
        head.opt = AG.Adam(head.net, 1e-3)
        rng = np.random.default_rng(5)
        B = 6
        s, s2 = rng.normal(size=(B, flat.n_in)).astype(np.float32), rng.normal(size=(B, flat.n_in)).astype(np.float32)
        a = rng.integers(0, flat.n_act, B)
        r = rng.normal(size=B).astype(np.float32)
        d = np.zeros(B, np.float32)
        m2 = np.ones((B, flat.n_act), bool)
        flat._update(s, a, r, s2, d, m2)
        ag._update(head, head, s, a, r, s2, d, m2)
        for w1, w2 in zip(flat.net.W, head.net.W):
            np.testing.assert_allclose(w1, w2, rtol=1e-5, atol=1e-7)

    def test_same_seed_gives_identical_training(self):
        sc = E.sample_scenarios(INST, 4, np.random.default_rng(0))
        out = []
        for _ in range(2):
            ag = T.make_agent("dprl", INST, 1, buffer=200, batch=16, train_every=2, target_sync=50)
            info = T.train(INST, ag, 1, 600, sc, 300)
            out.append([c["val"] for c in info["curve"]])
        self.assertEqual(out[0], out[1])

    def test_greedy_act_is_deterministic_and_valid(self):
        ag = T.make_agent("dprl", INST, 0)
        env = E.BSRPEnv(INST).reset(scenario())
        a1, a2 = ag.act(env), ag.act(env)
        self.assertEqual(a1, a2)
        self.assertTrue(env.valid_mask()[a1])


if __name__ == "__main__":
    unittest.main()
