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


class TestSharding(unittest.TestCase):
    """Fatias paralelas do ajuste (--shard i/N): partição completa, disjunta, equilibrada e determinística."""

    @staticmethod
    def jobs():
        return [(TUNE.est_cost(m, c), m, ci, s) for m in METHODS for ci, c in enumerate(TUNE.config_list(m, 16, "ext")) for s in (1000, 1001)]

    def test_partition_complete_and_disjoint(self):
        jobs = self.jobs()
        for n in (1, 3, 6, 8):
            owner = TUNE.shard_jobs(jobs, n)
            self.assertEqual(set(owner), {j[1:] for j in jobs})
            self.assertEqual(set(owner.values()), set(range(n)))

    def test_balanced_within_one_largest_job(self):
        jobs = self.jobs()
        cost = {j[1:]: j[0] for j in jobs}
        for n in (3, 6, 8):
            loads = [0.0] * n
            for k, i in TUNE.shard_jobs(jobs, n).items():
                loads[i] += cost[k]
            self.assertLessEqual(max(loads) - min(loads), max(cost.values()))

    def test_deterministic(self):
        jobs = self.jobs()
        self.assertEqual(TUNE.shard_jobs(jobs, 6), TUNE.shard_jobs(list(reversed(jobs)), 6))

    def test_cost_model_orders_dqn_configs(self):
        cheap = TUNE.est_cost("dqn", {"batch": 128, "train_every": 2})
        heavy = TUNE.est_cost("dqn", {"batch": 256, "train_every": 1})
        self.assertGreater(heavy, 2 * cheap * 0.9)
        self.assertLess(TUNE.est_cost("pg", {}), cheap)


if __name__ == "__main__":
    unittest.main()
