"""Rollout nas escalas maiores (scripts/32_rollout_scale.py): fatiamento completo e sem sobreposição; retomada pula o que já foi feito."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rs32", ROOT / "scripts" / "32_rollout_scale.py")
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)


class TestSharding(unittest.TestCase):
    def test_jobs_cover_test_then_real_days(self):
        jobs = RS.episode_jobs(5, 3)
        self.assertEqual(jobs, [("test", k) for k in range(5)] + [("real_test", k) for k in range(3)])

    def test_shards_partition_the_jobs_without_overlap(self):
        jobs = RS.episode_jobs(30, 24)
        parts = [RS.my_jobs(jobs, i, 8) for i in range(8)]
        flat = [j for p in parts for j in p]
        self.assertEqual(sorted(flat), sorted(jobs))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_resume_skips_done_without_changing_the_assignment(self):
        jobs = RS.episode_jobs(10, 4)
        full = RS.my_jobs(jobs, 2, 4)
        done = {full[0], full[1]}
        self.assertEqual(RS.my_jobs(jobs, 2, 4, done), full[2:])


if __name__ == "__main__":
    unittest.main()
