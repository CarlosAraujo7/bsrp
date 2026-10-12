"""
Portão G0 (continuação): reprodutibilidade dos aprendizes e sanidade das políticas.

  python -m unittest tests.test_learners -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp import policies as P                               # noqa: E402
from bsrp import train as T                                  # noqa: E402

INST = E.legacy_instance(4)
VAL = E.sample_scenarios(INST, 8, np.random.default_rng(1))
TEST = E.sample_scenarios(INST, 12, np.random.default_rng(2))


def run(method, seed, steps=3000):
    ag = T.make_agent(method, INST, seed)
    info = T.train(INST, ag, seed, steps, VAL, steps // 2)
    return info, T.final_eval(INST, ag, TEST)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_identical_all_methods(self):
        for m in ("tab_cluster", "tab_station", "linear", "dqn"):
            (i1, f1), (i2, f2) = run(m, 3), run(m, 3)
            self.assertEqual([c["val"] for c in i1["curve"]], [c["val"] for c in i2["curve"]], m)
            np.testing.assert_array_equal(f1["test_rewards"], f2["test_rewards"], err_msg=m)

    def test_different_seeds_differ(self):
        for m in ("tab_cluster", "dqn"):
            (_, f1), (_, f2) = run(m, 1), run(m, 2)
            self.assertFalse(np.array_equal(f1["test_rewards"], f2["test_rewards"]), m)

    def test_global_numpy_rng_untouched(self):
        np.random.seed(999)
        before = np.random.get_state()[1].copy()
        run("dqn", 5)
        np.testing.assert_array_equal(before, np.random.get_state()[1])

    def test_streams_are_independent(self):
        a, b, c, d = T.streams(7)
        xs = [g.random(5).tolist() for g in (a, b, c, d)]
        self.assertEqual(len({tuple(x) for x in xs}), 4)


class TestPolicies(unittest.TestCase):
    def test_policies_only_emit_valid_actions(self):
        env = E.BSRPEnv(INST)
        pols = [P.Stay(), P.RandomPolicy(np.random.default_rng(0)), P.GreedyFamily(), P.ThresholdFamily(),
                P.Rollout(P.GreedyFamily(), h=1, K=2, seed=0)]
        for pol in pols:
            for k in range(3):
                env.reset(VAL[k])
                while not env.done:
                    a = pol.act(env)
                    self.assertTrue(env.valid_mask()[a], f"{pol.name} emitiu ação inválida {a}")
                    env.step(a)

    def test_calibrated_beats_stay_on_val(self):
        env = E.BSRPEnv(INST)
        stay = E.evaluate(env, P.Stay(), VAL).mean()
        best, _ = P.calibrate(env, P.threshold_grid()[:12], VAL)
        self.assertGreater(E.evaluate(env, best, VAL).mean(), stay)

    def test_rollout_does_not_peek_at_real_future(self):
        """Mudar a demanda REAL futura não pode mudar a decisão do rollout (ele só vê futuros amostrados do modelo)."""
        env = E.BSRPEnv(INST)
        sc = VAL[0]
        env.reset(sc)
        a1 = P.Rollout(P.GreedyFamily(), 1, 3, seed=9).act(env)
        sc2 = E.Scenario(sc.init_occ, sc.dep + 5, sc.arr + 9)       # demanda real totalmente diferente
        env.reset(sc2)
        a2 = P.Rollout(P.GreedyFamily(), 1, 3, seed=9).act(env)
        self.assertEqual(a1, a2)


if __name__ == "__main__":
    unittest.main()
