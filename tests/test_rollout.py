"""Executor do rollout (scripts/23_rollout.py): leitura de políticas, escolha da referência por VALIDAÇÃO, reprodutibilidade por episódio."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp import policies as P                               # noqa: E402

_spec = importlib.util.spec_from_file_location("rollout23", ROOT / "scripts" / "23_rollout.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

INST = E.legacy_instance(4)
SCEN = E.sample_scenarios(INST, 6, np.random.default_rng(5))


class TestParse(unittest.TestCase):
    def test_greedy_roundtrip(self):
        for g in P.greedy_grid()[:5] + P.greedy_grid3()[:5]:
            p = R.parse_policy(g.name)
            self.assertEqual((p.lo, p.hi, p.max_delta), (g.lo, g.hi, g.max_delta))
            self.assertEqual(p.name, g.name)

    def test_threshold_grid2_extends_hi_below_grid1(self):
        g1, g2 = P.threshold_grid(), P.threshold_grid2()
        self.assertTrue(min(c.p[2] for c in g1) == 0.6 and max(c.p[2] for c in g2) < 0.6)   # grade 2 só cobre hi < 0,6
        self.assertTrue(all(c.p[2] > c.p[1] for c in g2))                                    # hi > lo
        self.assertEqual(len({c.name for c in g1} & {c.name for c in g2}), 0)                 # sem sobreposição com a grade 1
        for c in g2[:6]:
            self.assertEqual(R.parse_policy(c.name).p, c.p)

    def test_threshold_roundtrip(self):
        for g in P.threshold_grid()[:6]:
            p = R.parse_policy(g.name)
            self.assertEqual(p.p, g.p)
            self.assertEqual(p.name, g.name)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            R.parse_policy("stay")


class TestBestReference(unittest.TestCase):
    def test_chosen_by_validation_not_by_test(self):
        rows = [{"instance": "x", "method": "greedy_cal", "policy": "greedy(lo=0.25,hi=0.55,d=10)", "test_mean": -1.0, "val_table": {"a": -5.0}},
                {"instance": "x", "method": "threshold_cal", "policy": "threshold(t=0.5,lo=0.35,hi=0.75,d=10,w=0.1)", "test_mean": -9.0, "val_table": {"b": -3.0}},
                {"instance": "y", "method": "greedy_cal2", "policy": "greedy(lo=0.1,hi=0.5,d=5)", "test_mean": 0.0, "val_table": {"c": 0.0}}]
        fam, row = R.best_reference(rows, "x")
        self.assertEqual(fam, "threshold_cal")                    # pior no teste, melhor em val -> vence
        with self.assertRaises(SystemExit):
            R.best_reference(rows, "z")


class TestEpisodeSeeding(unittest.TestCase):
    def test_result_independent_of_episode_order(self):
        env = E.BSRPEnv(INST)
        ro = P.Rollout(P.GreedyFamily(0.25, 0.55, 10), h=1, K=2, seed=0)
        full = R.eval_set(env, ro, SCEN, "test", 1, 2)
        only3 = R.eval_set(env, P.Rollout(P.GreedyFamily(0.25, 0.55, 10), h=1, K=2, seed=0), SCEN, "test", 1, 2)
        np.testing.assert_array_equal(full, only3)
        # episódio 3 isolado = episódio 3 do conjunto completo (a semente deriva do índice, não do histórico)
        ro3 = P.Rollout(P.GreedyFamily(0.25, 0.55, 10), h=1, K=2, seed=0)
        ro3.rng = np.random.default_rng(np.random.SeedSequence([R.SALT, R.SETS["test"], 3, 1, 2]))
        self.assertEqual(E.run_episode(env, ro3, SCEN[3])["reward"], full[3])

    def test_different_h_or_k_use_different_streams(self):
        env = E.BSRPEnv(INST)
        a = R.eval_set(env, P.Rollout(P.GreedyFamily(), h=1, K=2), SCEN, "test", 1, 2)
        b = R.eval_set(env, P.Rollout(P.GreedyFamily(), h=1, K=4), SCEN, "test", 1, 4)
        self.assertEqual(len(a), len(b))


if __name__ == "__main__":
    unittest.main()
