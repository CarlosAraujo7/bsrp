"""Funções auxiliares do gerador de relatório (scripts/25_make_report.py)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("report25", ROOT / "scripts" / "25_make_report.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)


class TestHelpers(unittest.TestCase):
    def test_decimal_comma(self):
        self.assertEqual(RP.pt(9.84), "9,8")
        self.assertEqual(RP.pt(-0.466, 2), "-0,47")

    def test_tex_minus_only_on_numbers(self):
        self.assertEqual(RP.tex_num("-2,6"), "$-$2,6")
        self.assertEqual(RP.tex_num("a-b"), "a-b")

    def test_every_method_has_label_and_color(self):
        for m in RP.ORDER:
            self.assertIn(m, RP.LABEL)
            self.assertIn(m, RP.COL)
        self.assertEqual(len(set(RP.COL.values())), len(RP.COL))               # uma cor por entidade


if __name__ == "__main__":
    unittest.main()
