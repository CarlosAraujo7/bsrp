"""scripts/35_cost_steps.py: a contagem de passos de modelo por decisão é exata (um passo por chamada a step_raw; o passo real da decisão é descontado)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp.rhea import RHEA                        # noqa: E402

_spec = importlib.util.spec_from_file_location("cost35", ROOT / "scripts" / "35_cost_steps.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

INST = E.legacy_instance(4, 8.0)
SC = E.sample_scenarios(INST, 2, np.random.default_rng(0))


class TestCounter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._raw = C.install_counter()

    @classmethod
    def tearDownClass(cls):
        E.BSRPEnv.step_raw = cls._raw

    def test_step_counts_once(self):
        env = E.BSRPEnv(INST).reset(SC[0])
        C.COUNT["n"] = 0
        env.step(int(np.flatnonzero(env.valid_mask())[0]))
        self.assertEqual(C.COUNT["n"], 1)                                   # step chama step_raw: sem contagem dupla

    def test_rhea_steps_per_decision_matches_the_evaluation_budget(self):
        """RHEA(L, P, G, K): (P + (G − 1)(P − elite)) avaliações × K futuros × L passos por decisão (no máximo; o fim do dia corta), mais o passo real."""
        pol = RHEA(L=3, pop=6, gens=3, K=2, seed=1)
        C.COUNT["n"] = 0
        r = E.run_episode(E.BSRPEnv(INST), pol, SC[0])
        per = (C.COUNT["n"] - r["steps"]) / r["steps"]
        full = (6 + 2 * (6 - 2)) * 2 * 3
        self.assertLessEqual(per, full)
        self.assertGreater(per, 0.5 * full)                                  # só as últimas decisões do dia têm menos de L passos


if __name__ == "__main__":
    unittest.main()
