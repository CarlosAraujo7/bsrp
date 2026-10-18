"""Regra pré-registrada da verificação do teto de batch (scripts/27_batch512_check.py; DESVIOS.md 2026-10-18)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("b512", ROOT / "scripts" / "27_batch512_check.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


class TestDecisionRule(unittest.TestCase):
    def test_requires_both_seeds_better_and_mean_gain_at_least_one(self):
        self.assertTrue(B.decide([-34.0, -33.0], [-32.0, -31.5]))            # ambas melhoram e +1,75 em média
        self.assertFalse(B.decide([-34.0, -33.0], [-32.0, -33.5]))           # uma semente piora
        self.assertFalse(B.decide([-34.0, -33.0], [-33.6, -32.7]))           # ambas melhoram, mas só +0,35 (< 1,0)
        self.assertFalse(B.decide([-34.0, -33.0], [-34.0, -33.0]))           # empate não conta
        self.assertTrue(B.decide([-34.0, -33.0], [-33.0, -32.0], min_gain=1.0))   # exatamente 1,0: conta

    def test_rule_constants_match_the_registered_text(self):
        self.assertEqual(B.MIN_GAIN, 1.0)


if __name__ == "__main__":
    unittest.main()
