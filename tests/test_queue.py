"""Fila de execuções (scripts/22_queue.py): pendências derivadas dos resultados, ordem, um escritor, retentativa e STOP."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("queue22", ROOT / "scripts" / "22_queue.py")
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)

CFG_DQN = {"lr": 3e-4, "target_sync": 4000, "train_every": 1, "batch": 256, "buffer": 100_000, "eps_decay_frac": 0.5}
CFG_TAB = {"alpha": 0.5, "eps_decay_frac": 0.5, "eps_end": 0.1}
WRITE = "import sys, json; open(sys.argv[1], 'w').write(json.dumps({'k': int(sys.argv[2])}) + '\\n')"


def dump(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestBuildJobs(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        runs = self.root / "runs"
        runs.mkdir()
        orig = {"dqn": {"cfg_id": 1, "cfg": CFG_DQN}, "linear": {"cfg_id": 5, "cfg": CFG_DQN}, "tab_station": {"cfg_id": 5, "cfg": CFG_TAB}}
        ext = {"dqn": {"cfg_id": 9, "cfg": CFG_DQN}, "linear": {"cfg_id": 5, "cfg": CFG_DQN}, "tab_station": {"cfg_id": 11, "cfg": CFG_TAB}}
        (runs / "tuned_cb12wd.json").write_text(json.dumps(orig))
        (runs / "tuned_ext_cb12wd.json").write_text(json.dumps(ext))
        dump(runs / "final_ext_cb12wd.jsonl", [{"method": "dqn", "seed": 0}, {"method": "dqn", "seed": 1}])
        dump(runs / "ablation_cb12wd.jsonl", [{"variant": "noreplay", "seed": 0}])

    def tearDown(self):
        self._td.cleanup()

    def test_finals_only_for_changed_methods_and_skip_done(self):
        jobs = [j for j in Q.build_jobs(self.root, ("cb12wd",), ("finals",))]
        got = {(j.key[2], j.key[3]) for j in jobs}
        self.assertEqual(got, {("dqn", s) for s in range(2, 10)} | {("tab_station", s) for s in range(10)})   # linear não mudou; dqn 0,1 feitos
        self.assertFalse(any(j.key[2] == "pg" for j in jobs))                                                    # ainda sem ajuste

    def test_ablation_jobs(self):
        jobs = Q.build_jobs(self.root, ("cb12wd",), ("ablation",))
        self.assertEqual(len(jobs), 4 * 10 - 1)
        self.assertNotIn(("cb12wd", "ablation", "noreplay", 0), {j.key for j in jobs})

    def test_sens_jobs_need_scenarios_and_use_base_config(self):
        self.assertEqual(Q.build_jobs(self.root, ("cb12wd",), ("sens",)), [])                      # sem cenários gerados: nada
        sc = self.root / "scenarios"
        sc.mkdir()
        for k in ("val", "test"):
            (sc / f"{k}_cb12wdx0.5.npz").write_bytes(b"")
        (sc / "val_cb12we.npz").write_bytes(b"")                                                     # só val: ainda não entra
        dump(self.root / "runs" / "sens_cb12wdx0.5.jsonl", [{"method": "dqn", "seed": 0}])
        jobs = Q.build_jobs(self.root, ("cb12wd",), ("sens",))
        self.assertEqual({j.key[0] for j in jobs}, {"cb12wdx0.5"})
        self.assertEqual(len(jobs), 3 * 5 - 1)                                                       # 3 métodos no fixture × 5 sementes − 1 feita
        self.assertTrue(all(j.kind == "sens" and j.target.name == "sens_cb12wdx0.5.jsonl" for j in jobs))
        j = jobs[0]
        self.assertEqual(j.cmd[j.cmd.index("--instances") + 1], "cb12wdx0.5")
        self.assertTrue(j.cmd[j.cmd.index("--tuned-file") + 1].endswith("tuned_ext_cb12wd.json"))   # config da escala-base
        self.assertIn("--tuned", j.cmd)

    def test_seo_jobs_cover_variants_and_seeds_and_skip_done(self):
        dump(self.root / "runs" / "seo_a2c_cb12wd.jsonl", [{"variant": "published", "seed": 0}, {"variant": "gamma1", "seed": 3}])
        jobs = Q.build_jobs(self.root, ("cb12wd",), ("seo",))
        self.assertEqual(len(jobs), 2 * 10 - 2)
        self.assertEqual({j.key[2] for j in jobs}, {"published", "gamma1"})
        self.assertTrue(all(j.kind == "seo" and j.target.name == "seo_a2c_cb12wd.jsonl" for j in jobs))
        j = jobs[0]
        self.assertIn("24_seo_a2c.py", j.cmd[2])
        self.assertEqual(Path(j.cmd[j.cmd.index("--out") + 1]), j.tmp)

    def test_seo_tuned_variant_appears_only_when_the_tuned_file_exists(self):
        before = {j.key[2] for j in Q.build_jobs(self.root, ("cb12wd",), ("seo",))}
        self.assertEqual(before, {"published", "gamma1"})
        (self.root / "runs" / "tuned_a2c_cb12wd.json").write_text(json.dumps({"cfg": {"gamma": 0.99}}))
        after = Q.build_jobs(self.root, ("cb12wd",), ("seo",))
        self.assertEqual({j.key[2] for j in after}, {"published", "gamma1", "tuned"})
        self.assertEqual(len(after), 3 * 10)

    def test_sensrt_jobs_use_the_variants_own_tuned_file_and_wait_for_it(self):
        self.assertEqual(Q.build_jobs(self.root, ("cb12wd",), ("sensrt",)), [])                    # nenhuma variante ajustada ainda
        (self.root / "runs" / "tuned_ext_cb12wdx0.5.json").write_text(json.dumps({"dqn": {"cfg_id": 12, "cfg": CFG_DQN}}))
        (self.root / "runs" / "tuned_ext_cb12we.json").write_text(json.dumps({"dqn": {"cfg_id": 9, "cfg": CFG_DQN}}))     # = config da escala-base: não reroda
        dump(self.root / "runs" / "sens_retuned_cb12wdx0.5.jsonl", [{"method": "dqn", "seed": 0}, {"method": "dqn", "seed": 4}])
        jobs = Q.build_jobs(self.root, ("cb12wd",), ("sensrt",))
        self.assertEqual({j.key for j in jobs}, {("cb12wdx0.5", "sensrt", "dqn", s) for s in (1, 2, 3)})
        j = jobs[0]
        self.assertEqual(j.kind, "sensrt")
        self.assertEqual(j.target.name, "sens_retuned_cb12wdx0.5.jsonl")
        self.assertEqual(j.cmd[j.cmd.index("--instances") + 1], "cb12wdx0.5")
        self.assertTrue(j.cmd[j.cmd.index("--tuned-file") + 1].endswith("tuned_ext_cb12wdx0.5.json"))   # config da PRÓPRIA variante
        self.assertEqual(j.cmd[j.cmd.index("--methods") + 1], "dqn")
        self.assertIn("--tuned", j.cmd)

    def test_kind_rank_puts_finals_before_sens_before_sensrt_before_seo_before_ablation(self):
        self.assertEqual(sorted(Q.KIND_RANK, key=Q.KIND_RANK.get), ["finals", "sens", "sensrt", "seo", "ablation"])

    def test_sens_names_match_instance_names(self):
        names = [n for lst in Q.SENS.values() for _, n in lst]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(("4", "legacy4"), Q.SENS["cb4wd"])                                              # legado a 3,5 = spec "4", nome legacy4
        self.assertTrue(all(len(Q.SENS_SEEDS) == 5 for _ in names))

    def test_instance_without_tuning_files_is_ignored(self):
        self.assertEqual(Q.build_jobs(self.root, ("cb4wd", "cb8wd"), ("finals", "ablation")), [])

    def test_order_finals_first_then_most_expensive(self):
        jobs = Q.order(Q.build_jobs(self.root, ("cb12wd",), ("finals", "ablation")))
        kinds = [j.kind for j in jobs]
        self.assertEqual(kinds, sorted(kinds, key=lambda k: k != "finals"))
        self.assertEqual(jobs[0].key[2], "dqn")                                  # dqn (caro) antes de tab_station
        fin = [j for j in jobs if j.kind == "finals"]
        self.assertEqual([j.cost for j in fin], sorted((j.cost for j in fin), reverse=True))

    def test_jobs_write_to_temp_files_never_to_targets(self):
        for j in Q.build_jobs(self.root, ("cb12wd",), ("finals", "ablation")):
            self.assertIn("--out", j.cmd)
            self.assertEqual(Path(j.cmd[j.cmd.index("--out") + 1]), j.tmp)
            self.assertNotEqual(j.tmp, j.target)
            self.assertEqual(j.tmp.parent.name, "_q")


class TestRunQueue(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self.target = self.dir / "result.jsonl"

    def tearDown(self):
        self._td.cleanup()

    def jobs(self, n=5, fail=()):
        def get_jobs():
            done = {r["k"] for r in Q.read_rows(self.target)}
            out = []
            for i in range(n):
                if i in done:
                    continue
                tmp = self.dir / "_q" / f"j{i}.jsonl"
                cmd = [sys.executable, "-c", "import sys; sys.exit(1)"] if i in fail else [sys.executable, "-c", WRITE, str(tmp), str(i)]
                out.append(Q.Job(("t", i), "finals", float(i), cmd, tmp, self.target))
            return out
        return get_jobs

    def test_all_jobs_run_once_and_are_merged(self):
        logs = []
        dead = Q.run_queue(self.jobs(5), lambda: 2, self.dir / "_q", poll=0.05, log=logs.append)
        self.assertEqual(dead, set())
        self.assertEqual(sorted(r["k"] for r in Q.read_rows(self.target)), [0, 1, 2, 3, 4])
        self.assertEqual(list((self.dir / "_q").glob("j*.jsonl")), [])            # temporários removidos
        self.assertEqual(sum("início" in l for l in logs), 5)

    def test_failed_job_is_retried_then_abandoned_and_queue_continues(self):
        logs = []
        dead = Q.run_queue(self.jobs(4, fail=(2,)), lambda: 2, self.dir / "_q", poll=0.05, retries=2, log=logs.append)
        self.assertEqual(dead, {("t", 2)})
        self.assertEqual(sorted(r["k"] for r in Q.read_rows(self.target)), [0, 1, 3])
        self.assertEqual(sum("FALHA" in l for l in logs), 2)

    def test_stop_file_starts_nothing(self):
        (self.dir / "_q").mkdir()
        (self.dir / "_q" / "STOP").write_text("")
        Q.run_queue(self.jobs(3), lambda: 2, self.dir / "_q", poll=0.05, log=lambda m: None)
        self.assertEqual(Q.read_rows(self.target), [])

    def test_workers_file_overrides_default(self):
        (self.dir / "_q").mkdir()
        fn = Q.workers_from_file(self.dir / "_q", 4)
        self.assertEqual(fn(), 4)
        (self.dir / "_q" / "WORKERS").write_text("7\n")
        self.assertEqual(fn(), 7)
        (self.dir / "_q" / "WORKERS").write_text("lixo")
        self.assertEqual(fn(), 4)


if __name__ == "__main__":
    unittest.main()
