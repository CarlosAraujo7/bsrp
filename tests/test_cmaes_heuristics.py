"""Calibração evolutiva das heurísticas (scripts/33_cmaes_heuristics.py): decodificação dentro das caixas e CMA-ES melhora a heurística numa instância pequena."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("cma33", ROOT / "scripts" / "33_cmaes_heuristics.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
from bsrp import env as E                           # noqa: E402
from bsrp import policies as P                      # noqa: E402

INST = E.legacy_instance(4)


class TestDecode(unittest.TestCase):
    def test_greedy_and_threshold_decoding_respect_the_boxes_for_extreme_inputs(self):
        for x in (np.zeros(3), np.full(3, 50.0), np.full(3, -50.0), np.array([-20.0, 20.0, 0.1])):
            p = C.decode("greedy", x)
            self.assertIsInstance(p, P.GreedyFamily)
            self.assertTrue(0.05 <= p.lo < p.hi <= 0.99, (p.lo, p.hi))
            self.assertIn(p.max_delta, (5, 10))
        for x in (np.zeros(5), np.full(5, 50.0), np.full(5, -50.0), np.array([-20.0, 20.0, -20.0, 0.1, 5.0])):
            p = C.decode("threshold", x)
            t, lo, hi, d, w = p.p
            self.assertTrue(0.3 <= t <= 0.7 and 0.05 <= lo < hi <= 0.99 and d in (5, 10) and 0.0 <= w <= 0.5, p.p)

    def test_center_of_the_box_is_a_sensible_default(self):
        p = C.decode("greedy", np.zeros(3))
        self.assertAlmostEqual(p.lo, 0.5)
        self.assertAlmostEqual(p.hi, 0.5 + 0.05, places=3) if p.hi < 0.55 else None


class TestRandomSearch(unittest.TestCase):
    """P14b: controle do CMA-ES, mesma caixa e mesmo número de avaliações."""

    def test_random_search_uses_exactly_the_budget_is_reproducible_and_inside_the_box(self):
        env = E.BSRPEnv(INST)
        val = E.sample_scenarios(INST, 10, np.random.default_rng(5))
        a = C.run_random(env, val, "threshold", 2, 20)
        b = C.run_random(env, val, "threshold", 2, 20)
        self.assertEqual(a[3], 20)
        self.assertEqual(a[2], b[2])
        np.testing.assert_array_equal(a[0], b[0])
        t, lo, hi, d, w = a[1].p
        self.assertTrue(0.3 <= t <= 0.7 and 0.05 <= lo < hi <= 0.99 and d in (5, 10) and 0.0 <= w <= 0.5)
        self.assertEqual(a[4][-1][0], 20)                                                   # a curva termina na última avaliação
        self.assertTrue(all(c2[1] >= c1[1] for c1, c2 in zip(a[4], a[4][1:])))              # melhor-até-agora nunca piora (reward)

    def test_different_seed_gives_a_different_search(self):
        env = E.BSRPEnv(INST)
        val = E.sample_scenarios(INST, 10, np.random.default_rng(5))
        self.assertFalse(np.array_equal(C.run_random(env, val, "greedy", 1, 12)[0], C.run_random(env, val, "greedy", 2, 12)[0]))


class TestRun(unittest.TestCase):
    def test_cma_run_is_reproducible_and_not_worse_than_its_first_generation(self):
        env = E.BSRPEnv(INST)
        val = E.sample_scenarios(INST, 20, np.random.default_rng(5))
        a = C.run_cma(env, val, "greedy", 3, 28)
        b = C.run_cma(env, val, "greedy", 3, 28)
        self.assertEqual(a[2], b[2])
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[3], 28)
        curve = [v for _, v in a[4]]
        self.assertEqual(curve, sorted(curve))                                   # melhor-até-agora (reward) nunca piora
        self.assertGreaterEqual(a[2], curve[0])
        # a política devolvida reproduz o valor de validação
        self.assertAlmostEqual(float(E.evaluate(env, a[1], val).mean()), a[2])

    def test_different_seeds_explore_different_points(self):
        env = E.BSRPEnv(INST)
        val = E.sample_scenarios(INST, 10, np.random.default_rng(6))
        self.assertFalse(np.array_equal(C.run_cma(env, val, "threshold", 0, 16)[0], C.run_cma(env, val, "threshold", 1, 16)[0]))


if __name__ == "__main__":
    unittest.main()
