"""Métodos evolutivos (bsrp/evo.py): CMA-ES converge em funções-teste; OpenAI-ES sobe um fitness quadrático e respeita o orçamento em passos; reprodutível."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp import evo as EV                        # noqa: E402
from bsrp import train as T                       # noqa: E402

INST = E.legacy_instance(4)


def run_cma(f, x0, sigma0, evals, seed=0, popsize=None):
    cma = EV.CMAES(x0, sigma0, popsize, np.random.default_rng(seed))
    best, n = np.inf, 0
    while n < evals:
        X = cma.ask()
        fx = np.array([f(x) for x in X])
        cma.tell(X, fx)
        best = min(best, fx.min())
        n += len(fx)
    return best, cma


class TestCMAES(unittest.TestCase):
    def test_sphere_and_ill_conditioned_ellipsoid(self):
        best, _ = run_cma(lambda x: float(x @ x), np.ones(6) * 3, 1.0, 3000)
        self.assertLess(best, 1e-8)
        scale = 10.0 ** (3 * np.arange(6) / 5)                                      # condição 1e6 na matriz de Hessiana
        best, _ = run_cma(lambda x: float(((scale * x) ** 2).sum()), np.ones(6), 1.0, 12000)
        self.assertLess(best, 1e-6)

    def test_rosenbrock_2d(self):
        rosen = lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
        best, cma = run_cma(rosen, np.array([-1.0, 1.0]), 0.5, 4000)
        self.assertLess(best, 1e-8)
        np.testing.assert_allclose(cma.mean, [1.0, 1.0], atol=1e-3)

    def test_same_seed_same_trajectory_and_different_seed_differs(self):
        f = lambda x: float(((x - 1) ** 2).sum())
        b1, c1 = run_cma(f, np.zeros(4), 1.0, 400, seed=3)
        b2, c2 = run_cma(f, np.zeros(4), 1.0, 400, seed=3)
        b3, c3 = run_cma(f, np.zeros(4), 1.0, 400, seed=4)
        self.assertEqual(b1, b2)
        np.testing.assert_array_equal(c1.mean, c2.mean)
        self.assertNotEqual(b1, b3)

    def test_default_population_follows_the_standard_rule(self):
        self.assertEqual(EV.CMAES(np.zeros(3)).lam, 4 + int(3 * np.log(3)))
        self.assertEqual(EV.CMAES(np.zeros(5), popsize=12).lam, 12)


def run_sep(f, x0, sigma0, evals, seed=0, popsize=None):
    es = EV.SepCMAES(x0, sigma0, popsize, np.random.default_rng(seed))
    best, n = np.inf, 0
    while n < evals:
        X = es.ask()
        fx = np.array([f(x) for x in X])
        es.tell(X, fx)
        best = min(best, fx.min())
        n += len(fx)
    return best, es


class TestSepCMAES(unittest.TestCase):
    """P15: sep-CMA-ES converge em funções separáveis (inclusive mal condicionadas), é reprodutível e usa C diagonal."""

    def test_sphere_and_separable_ill_conditioned_ellipsoid(self):
        best, _ = run_sep(lambda x: float(x @ x), np.ones(20) * 3, 1.0, 12000)
        self.assertLess(best, 1e-8)
        scale = 10.0 ** (4 * np.arange(20) / 19)                                    # condição 1e8 na Hessiana, eixos alinhados
        best, es = run_sep(lambda x: float(((scale * x) ** 2).sum()), np.ones(20), 1.0, 60000)
        self.assertLess(best, 1e-6)
        self.assertGreater(float(np.corrcoef(np.log(es.C), -np.log(scale))[0, 1]), 0.3)  # a diagonal aprendeu (parte d)a escala dos eixos: C ∝ 1/escala²

    def test_reproducible_and_default_population(self):
        f = lambda x: float(((x - 1) ** 2).sum())
        b1, e1 = run_sep(f, np.zeros(5), 1.0, 500, seed=3)
        b2, e2 = run_sep(f, np.zeros(5), 1.0, 500, seed=3)
        self.assertEqual(b1, b2)
        np.testing.assert_array_equal(e1.mean, e2.mean)
        self.assertEqual(EV.SepCMAES(np.zeros(5)).lam, 4 + int(3 * np.log(5)))
        es = EV.SepCMAES(np.zeros(100))
        self.assertLessEqual(es.c1 + es.cmu, 1.0)

    def test_agent_dispatch_budget_reproducibility_and_valid_actions(self):
        val = E.sample_scenarios(INST, 4, np.random.default_rng(5))
        out = []
        for _ in range(2):
            ag = T.make_agent("sepcma", INST, 7, hidden=[8, 8], pop=8, bs=2, sigma=0.05)
            info = T.train(INST, ag, 7, 3000, val, 1000)
            out.append(([c["val"] for c in info["curve"]], ag.theta.copy()))
            gen_steps = 8 * 2 * E.N_INTERVALS
            self.assertLess(info["episodes"] * E.N_INTERVALS, 3000 + gen_steps)
            self.assertGreaterEqual(info["episodes"] * E.N_INTERVALS, 3000)
        self.assertEqual(out[0][0], out[1][0])
        np.testing.assert_array_equal(out[0][1], out[1][1])
        env = E.BSRPEnv(INST).reset(E.sample_scenarios(INST, 1, np.random.default_rng(0))[0])
        while not env.done:
            a = ag.act(env)
            self.assertTrue(env.valid_mask()[a])
            env.step(a)

    def test_agent_climbs_a_quadratic_fitness(self):
        class Quad(EV.SepCMAAgent):
            target = None

            def _fitness(self, env, theta, scen):
                return -float(((theta - self.target) ** 2).sum()), 32

        ag = Quad(INST, hidden=(4,), pop=16, sigma=0.3, bs=1, rng_init=np.random.default_rng(0))
        ag.target = ag.theta + 0.5
        d0 = float(np.linalg.norm(ag.theta - ag.target))
        info = ag.fit(INST, np.random.default_rng(1), np.random.default_rng(2), 32 * 16 * 150, E.sample_scenarios(INST, 2, np.random.default_rng(3)), 10 ** 9)
        self.assertLess(float(np.linalg.norm(ag.theta - ag.target)), 0.2 * d0)
        self.assertEqual(info["generations"], 150)


class TestES(unittest.TestCase):
    def test_centered_ranks_are_invariant_to_monotone_transforms_and_centered(self):
        f = np.array([3.0, -1.0, 7.0, 0.5, 2.0])
        u = EV.centered_ranks(f)
        np.testing.assert_allclose(u, EV.centered_ranks(np.exp(f)))
        self.assertAlmostEqual(float(u.sum()), 0.0)
        self.assertEqual((u.min(), u.max()), (-0.5, 0.5))

    def test_params_roundtrip(self):
        ag = T.make_agent("es", INST, 0, hidden=[8, 8])
        theta = EV._flat_params(ag.net)
        ag2 = T.make_agent("es", INST, 1, hidden=[8, 8])
        EV._set_params(ag2.net, theta)
        x = np.random.default_rng(0).normal(size=(3, ag.n_in)).astype(np.float32)
        np.testing.assert_array_equal(ag.net.forward(x), ag2.net.forward(x))

    def test_es_update_climbs_a_quadratic_fitness_and_counts_steps(self):
        class Quad(EV.ESAgent):
            target = None

            def _fitness(self, env, theta, scen):
                return -float(((theta - self.target) ** 2).sum()), 32          # 32 passos por avaliação

        ag = Quad(INST, hidden=(4,), pop=32, sigma=0.1, lr=0.05, bs=1, rng_init=np.random.default_rng(0))
        ag.target = ag.theta + 0.5
        d0 = float(np.linalg.norm(ag.theta - ag.target))
        budget = 32 * 32 * 200                                                  # 200 gerações
        info = ag.fit(INST, np.random.default_rng(1), np.random.default_rng(2), budget, E.sample_scenarios(INST, 2, np.random.default_rng(3)), 10 ** 9)
        d1 = float(np.linalg.norm(ag.theta - ag.target))
        self.assertLess(d1, 0.2 * d0)
        self.assertEqual(info["generations"], 200)
        self.assertEqual(info["episodes"], 200 * 32)

    def test_train_dispatches_to_fit_respects_the_budget_and_is_reproducible(self):
        val = E.sample_scenarios(INST, 4, np.random.default_rng(5))
        out = []
        for _ in range(2):
            ag = T.make_agent("es", INST, 7, hidden=[8, 8], pop=8, bs=2, sigma=0.05, lr=0.02)
            info = T.train(INST, ag, 7, 3000, val, 1000)
            out.append(([c["val"] for c in info["curve"]], ag.theta.copy()))
            gen_steps = 8 * 2 * E.N_INTERVALS                                    # pop × bs episódios × 32 passos
            self.assertLess(info["episodes"] * E.N_INTERVALS, 3000 + gen_steps)  # para na primeira geração que ultrapassa o orçamento
            self.assertGreaterEqual(info["episodes"] * E.N_INTERVALS, 3000)
        self.assertEqual(out[0][0], out[1][0])
        np.testing.assert_array_equal(out[0][1], out[1][1])

    def test_policy_only_emits_valid_actions(self):
        ag = T.make_agent("es", INST, 0)
        env = E.BSRPEnv(INST).reset(E.sample_scenarios(INST, 1, np.random.default_rng(0))[0])
        while not env.done:
            a = ag.act(env)
            self.assertTrue(env.valid_mask()[a])
            env.step(a)


if __name__ == "__main__":
    unittest.main()
