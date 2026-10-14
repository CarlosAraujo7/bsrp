"""Pool de configurações do ajuste (16_tune.py): o pool estendido (DESVIOS.md 2026-10-14) contém o original como prefixo."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "16_tune.py")
TUNE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TUNE)

METHODS = list(TUNE.DEFAULTS)


class TestConfigPool(unittest.TestCase):
    def test_ext_prefix_is_original_pool(self):
        for m in METHODS:
            self.assertEqual(TUNE.config_list(m, 8, "orig"), TUNE.config_list(m, 16, "ext")[:8], m)

    def test_deterministic(self):
        for m in METHODS:
            self.assertEqual(TUNE.config_list(m, 16, "ext"), TUNE.config_list(m, 16, "ext"), m)

    def test_pool_size_and_keys(self):
        for m in METHODS:
            cfgs = TUNE.config_list(m, 16, "ext")
            self.assertEqual(len(cfgs), 16)
            for c in cfgs:
                self.assertEqual(set(c), set(TUNE.DEFAULTS[m]), m)

    def test_ext_respects_caps(self):
        for m in ("dqn", "linear"):
            for c in TUNE.config_list(m, 16, "ext")[TUNE.N_ORIG:]:
                self.assertLessEqual(c["batch"], 256)
                self.assertLessEqual(c["target_sync"], 16_000)
                self.assertLessEqual(c["buffer"], 200_000)
        for c in TUNE.config_list("dqn", 16, "ext")[TUNE.N_ORIG:]:
            self.assertIn(c["train_every"], (1, 2))
            self.assertGreaterEqual(c["target_sync"], 4000)

    def test_original_pool_reproduces_logged_winners(self):
        """A config escolhida e gravada em runs/tuned_<inst>.json é reproduzida exatamente pelo gerador."""
        for inst in ("cb4wd", "cb8wd", "cb12wd"):
            p = ROOT / "runs" / f"tuned_{inst}.json"
            if not p.exists():
                self.skipTest(f"{p.name} ausente")
            with open(p, encoding="utf-8") as f:
                tuned = json.load(f)
            for m, t in tuned.items():
                self.assertEqual(TUNE.config_list(m, t["n_configs"], "orig")[t["cfg_id"]], t["cfg"], (inst, m))


if __name__ == "__main__":
    unittest.main()
