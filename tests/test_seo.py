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


class TestTunedVariantKnobs(unittest.TestCase):
    """entropy e reward_scale: padrão = como publicado; ativados mudam o treino do jeito esperado."""

    def _observe_once(self, **kw):
        env = E.BSRPEnv(INST).reset(VAL[0])
        ag = agent(lr_actor=1e-2, lr_critic=1e-2, **kw)
        s = env.features()
        a = ag.select(env, 0.0, np.random.default_rng(0))
        ag.observe(s, a, -2.0, s, True, ag._m)
        return ag, s

    def test_defaults_equal_published_behaviour(self):
        a1, s = self._observe_once()
        a2, _ = self._observe_once(entropy=0.0, reward_scale=1.0)
        for x, y in zip(a1.actor.W, a2.actor.W):
            np.testing.assert_array_equal(x, y)
        self.assertEqual((a1.entropy, a1.reward_scale), (0.0, 1.0))

    def test_entropy_bonus_changes_the_actor_update_and_pushes_toward_uniform(self):
        env = E.BSRPEnv(INST).reset(VAL[0])
        ents = {}
        for beta in (0.0, 1.0):
            ag = agent(lr_actor=5e-2, lr_critic=1e-2, entropy=beta)
            s = env.features()
            ag.select(env, 0.0, np.random.default_rng(0))
            p0 = ag._probs(s[None], ag._m)[0]
            for _ in range(30):                                        # td ≈ 0: só o bônus de entropia move o ator
                ag.observe(s, int(np.argmax(p0)), 0.0, s, False, ag._m)
            p = ag._probs(s[None], ag._m)[0]
            ents[beta] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        self.assertGreater(ents[1.0], ents[0.0])                       # com bônus a política fica mais espalhada

    def test_reward_scale_only_matters_relative_to_the_entropy_bonus(self):
        """Com Adam, escalar a recompensa não muda o 1º passo quando não há entropia; com entropia, muda o equilíbrio."""
        def actor_after(scale, beta):
            env = E.BSRPEnv(INST).reset(VAL[0])
            ag = agent(lr_actor=1e-2, lr_critic=1e-2, entropy=beta, reward_scale=scale)
            s = env.features()
            ag.select(env, 0.0, np.random.default_rng(0))
            ag.observe(s, 0, -10.0, s, True, ag._m)
            return ag.actor.W
        a, b = actor_after(1.0, 0.0), actor_after(0.1, 0.0)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-5)               # sem entropia: invariante à escala (Adam)
        c, d = actor_after(1.0, 5.0), actor_after(0.1, 5.0)
        self.assertTrue(any(not np.allclose(x, y, atol=1e-5) for x, y in zip(c, d)))     # com entropia: a escala pesa

class TestTuningPool(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("tune26", ROOT / "scripts" / "26_tune_a2c.py")
        self.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.M)

    def test_pool_has_16_configs_cfg0_is_published_and_is_deterministic(self):
        c = self.M.config_list()
        self.assertEqual(len(c), 16)
        self.assertEqual(c[0], {"lr_actor": 5e-4, "lr_critic": 5e-4, "gamma": 0.9, "entropy": 0.0, "reward_scale": 1.0})
        self.assertEqual(c, self.M.config_list())

    def test_pool_respects_the_registered_ranges_and_covers_every_discrete_value(self):
        c = self.M.config_list()[1:]
        for x in c:
            self.assertTrue(1e-4 <= x["lr_actor"] <= 10 ** -2.5 and 1e-4 <= x["lr_critic"] <= 10 ** -2.5)
            self.assertIn(x["gamma"], (0.9, 0.95, 0.99, 1.0))
            self.assertIn(x["entropy"], (0.0, 0.01, 0.03, 0.1, 0.3))
            self.assertIn(x["reward_scale"], (1.0, 0.3, 0.1))
        for k, vals in (("gamma", {0.9, 0.95, 0.99, 1.0}), ("entropy", {0.0, 0.01, 0.03, 0.1, 0.3}), ("reward_scale", {1.0, 0.3, 0.1})):
            self.assertEqual({x[k] for x in c}, vals)                   # nenhum valor do espaço fica sem teste


if __name__ == "__main__":
    unittest.main()
