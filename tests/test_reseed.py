"""Reavaliação das melhores configs com mais sementes (scripts/29_reseed_top.py; DESVIOS.md 2026-10-18, P8c)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reseed29", ROOT / "scripts" / "29_reseed_top.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

B = 200_000


def row(ci, seed, v, budget=B):
    return {"method": "dqn", "cfg_id": ci, "cfg": {"c": ci}, "seed": seed, "budget": budget, "val_final": v}


class TestReseed(unittest.TestCase):
    def test_top_configs_use_only_base_seeds_full_pairs_and_the_right_budget(self):
        rows = [row(0, 1000, -10), row(0, 1001, -12),                        # média −11
                row(1, 1000, -5), row(1, 1001, -7),                          # média −6  (melhor)
                row(2, 1000, -8), row(2, 1001, -9),                          # média −8,5
                row(3, 1000, -1),                                            # só uma semente: fora
                row(4, 1000, 0, budget=100_000), row(4, 1001, 0, budget=100_000),   # outro orçamento: fora
                row(5, 1000, -30), row(5, 1001, -30), row(6, 1002, 0)]       # semente extra não entra no ranking-base
        top = R.top_configs(rows, B, k=2)
        self.assertEqual(list(top), [1, 2])
        self.assertAlmostEqual(top[1][1], -6.0)

    def test_repeated_runs_count_once_by_last_row(self):
        rows = [row(0, 1000, -50), row(0, 1000, -10), row(0, 1001, -12)]
        self.assertAlmostEqual(R.top_configs(rows, B, k=1)[0][1], -11.0)

    def test_choose_waits_for_all_seeds_and_averages_all_of_them(self):
        base = [row(1, 1000, -5), row(1, 1001, -7), row(2, 1000, -8), row(2, 1001, -9)]
        top = R.top_configs(base, B, k=2)
        extra = [row(1, 1002, -20), row(2, 1002, -6), row(2, 1003, -7)]
        self.assertIsNone(R.choose(top, extra, base, B, {1000, 1001, 1002, 1003}))        # falta (1, 1003)
        extra.append(row(1, 1003, -18))
        sel, means = R.choose(top, extra, base, B, {1000, 1001, 1002, 1003})
        self.assertEqual(sel, 2)                                                           # com 2 sementes a 1 ganhava (−6 × −8,5)
        self.assertAlmostEqual(means[1], (-5 - 7 - 20 - 18) / 4)
        self.assertAlmostEqual(means[2], (-8 - 9 - 6 - 7) / 4)


if __name__ == "__main__":
    unittest.main()
