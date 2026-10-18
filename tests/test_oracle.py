"""Planejador clarividente (bsrp/oracle.py): igual à programação dinâmica exaustiva sobre o próprio BSRPEnv numa instância minúscula;
com largura finita é viável (≤ ótimo) e nunca pior que 'parar'; a largura maior não piora o valor no caso de teste."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp import oracle as O                      # noqa: E402


def tiny_instance(cap=(10, 10), lat=(40.00, 40.02), lon=(-74.00, -74.00), lam=0.6):
    n = len(cap)
    return E.Instance("tiny", np.array(cap, np.int64), np.zeros(n, np.int64), np.array(lat), np.array(lon),
                      np.full((E.N_INTERVALS, n), lam), np.full((E.N_INTERVALS, n), lam), tuple(f"s{i}" for i in range(n)))


def exact_value(inst, sc):
    """Programação dinâmica exaustiva com memória sobre BSRPEnv (independente do código do oráculo)."""
    env = E.BSRPEnv(inst)
    env.reset(sc)
    memo = {}

    def rec(e):
        if e.done:
            return 0.0
        key = (e.t, e.pos, e.load, e.occ.tobytes())
        if key in memo:
            return memo[key]
        best = -np.inf
        mask = e.valid_mask()
        for a in np.nonzero(mask)[0]:
            c = e.clone()
            r, _ = c.step(int(a))
            best = max(best, r + rec(c))
        memo[key] = best
        return best

    return rec(env)


class TestOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = tiny_instance()
        cls.sc = [E.sample_scenarios(cls.inst, 1, np.random.default_rng(s))[0] for s in (3, 4)]
        cls.exact = [exact_value(cls.inst, sc) for sc in cls.sc]

    def test_unpruned_search_equals_exhaustive_dp_on_the_env(self):
        for sc, ex in zip(self.sc, self.exact):
            self.assertAlmostEqual(O.plan_value(self.inst, sc, width=None), ex)

    def test_wide_beam_reaches_the_exact_value_on_the_tiny_instance(self):
        for sc, ex in zip(self.sc, self.exact):
            self.assertAlmostEqual(O.plan_value(self.inst, sc, width=100_000), ex)

    def test_finite_beam_is_feasible_and_not_worse_than_staying(self):
        env = E.BSRPEnv(self.inst)

        class Stay:
            def act(self, e):
                return (e.pos, 0)

        for sc, ex in zip(self.sc, self.exact):
            v = O.plan_value(self.inst, sc, width=5)
            stay = E.run_episode(env, Stay(), sc)["reward"]
            self.assertLessEqual(v, ex + 1e-9)                        # limite inferior do ótimo clarividente
            self.assertGreaterEqual(v, stay - 1e-9)                   # parar é uma das sequências viáveis, mas o feixe só enxerga o potencial 'parar' no fim:
                                                                      # com o potencial correto, o feixe com largura ≥ 1 nunca fica abaixo dela

    def test_travel_longer_than_one_interval_is_handled(self):
        inst = tiny_instance(lat=(40.00, 40.30))                      # ~33 km: vários intervalos de viagem
        self.assertGreater(int(inst.travel_int().max()), 1)
        sc = E.sample_scenarios(inst, 1, np.random.default_rng(9))[0]
        self.assertAlmostEqual(O.plan_value(inst, sc, width=None), exact_value(inst, sc))


if __name__ == "__main__":
    unittest.main()
