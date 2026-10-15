"""A2C adaptado de Seo et al. (2022) (bsrp/seo.py, scripts/24_seo_a2c.py)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp import train as T                                  # noqa: E402
from bsrp.seo import A2C                                     # noqa: E402

_spec = importlib.util.spec_from_file_location("seo24", ROOT / "scripts" / "24_seo_a2c.py")
S24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S24)

INST = E.legacy_instance(4)
VAL = E.sample_scenarios(INST, 8, np.random.default_rng(1))
TEST = E.sample_scenarios(INST, 12, np.random.default_rng(2))


def agent(seed=3, **kw):
    return A2C(INST, rng_init=T.streams(seed)[0], **kw)


def run(seed=3, steps=3000, **kw):
    ag = agent(seed, **kw)
    info = T.train(INST, ag, seed, steps, VAL, steps // 2)
    return ag, info, T.final_eval(INST, ag, TEST)


class TestPublishedSettings(unittest.TestCase):
    def test_architecture_and_hyperparameters_as_published(self):
        ag = agent()
        self.assertEqual([w.shape for w in ag.actor.W], [(10, 24), (24, 24), (24, 24), (24, INST.n_actions)])   # 3 camadas ocultas × 24
        self.assertEqual([w.shape for w in ag.critic.W][-1], (24, 1))
        self.assertEqual((ag.opt_a.lr, ag.opt_c.lr, ag.gamma), (5e-4, 5e-4, 0.9))
        self.assertEqual(S24.PUBLISHED, {"hidden": [24, 24, 24], "lr_actor": 5e-4, "lr_critic": 5e-4})
        self.assertEqual(S24.VARIANTS["published"]["gamma"], 0.9)
        self.assertEqual(S24.VARIANTS["gamma1"]["gamma"], 1.0)


class TestBehaviour(unittest.TestCase):
    def test_only_valid_actions_are_sampled_and_chosen(self):
        ag, env, rng = agent(), E.BSRPEnv(INST), np.random.default_rng(0)
        for k in range(3):
            env.reset(VAL[k])
            while not env.done:
                a = ag.select(env, 1.0, rng)
                self.assertTrue(env.valid_mask()[a])
                self.assertTrue(env.valid_mask()[ag.act(env)])
                env.step(a)

    def test_positive_td_error_raises_probability_of_taken_action(self):
        env = E.BSRPEnv(INST).reset(VAL[0])
        for sign, expect_up in ((+1.0, True), (-1.0, False)):
            ag = agent(lr_actor=1e-2, lr_critic=1e-2)
            s = env.features()
            a = ag.select(env, 0.0, np.random.default_rng(0))
            p0 = ag._probs(s[None], ag._m)[0, a]
            # r muito maior (ou menor) que V(s) − γV(s') -> δ com o sinal pedido
            ag.observe(s, a, 5.0 * sign, s, True, ag._m)
            p1 = ag._probs(s[None], ag._m)[0, a]
            self.assertEqual(p1 > p0, expect_up, (sign, p0, p1))

    def test_critic_moves_toward_the_td_target(self):
        env = E.BSRPEnv(INST).reset(VAL[0])
        ag = agent(lr_critic=1e-2)
        s = env.features()
        ag.select(env, 0.0, np.random.default_rng(0))
        v0 = float(ag.critic.forward(s[None])[0, 0])
        for _ in range(40):
            ag.observe(s, 0, -3.0, s, True, ag._m)        # alvo terminal = −3
        v1 = float(ag.critic.forward(s[None])[0, 0])
        self.assertLess(abs(v1 + 3.0), abs(v0 + 3.0))

    def test_actor_untouched_by_probabilities_of_invalid_actions(self):
        ag, env = agent(), E.BSRPEnv(INST).reset(VAL[0])
        ag.select(env, 0.0, np.random.default_rng(0))
        p = ag._probs(env.features()[None], ag._m)[0]
        self.assertTrue(np.all(p[~ag._m] == 0.0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_identical_and_different_seed_differs(self):
        (_, i1, f1), (_, i2, f2), (_, _, f3) = run(3), run(3), run(4)
        self.assertEqual([c["val"] for c in i1["curve"]], [c["val"] for c in i2["curve"]])
        np.testing.assert_array_equal(f1["test_rewards"], f2["test_rewards"])
        self.assertFalse(np.array_equal(f1["test_rewards"], f3["test_rewards"]))

    def test_global_numpy_rng_untouched(self):
        np.random.seed(999)
        before = np.random.get_state()[1].copy()
        run(5, steps=1500)
        np.testing.assert_array_equal(before, np.random.get_state()[1])


if __name__ == "__main__":
    unittest.main()
