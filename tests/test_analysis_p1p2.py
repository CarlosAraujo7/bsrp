"""Análise do P1 e da ablação (scripts/21_analyze_p1_p2.py) sobre dados sintéticos de efeito conhecido."""
import importlib.util
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analysis21", ROOT / "scripts" / "21_analyze_p1_p2.py")
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

INST = "cb12wd"
CFG_ORIG = {"lr": 4e-4, "batch": 128, "buffer": 50_000}
CFG_NEW = {"lr": 5e-4, "batch": 256, "buffer": 100_000}


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(tmp):
    rng = np.random.default_rng(0)
    seeds = list(range(10))
    (tmp / f"tuned_{INST}.json").write_text(json.dumps({"dqn": {"cfg_id": 1, "cfg": CFG_ORIG, "val_mean": -80.0, "n_configs": 8, "seeds": [1000, 1001]}}))
    (tmp / f"tuned_ext_{INST}.json").write_text(json.dumps({"dqn": {"cfg_id": 9, "cfg": CFG_NEW, "val_mean": -70.0, "n_configs": 16, "seeds": [1000, 1001]}}))
    pool = []
    for i in range(16):
        cfg = CFG_NEW if i == 9 else (CFG_ORIG if i == 1 else {"lr": 1e-4 * (i + 1), "batch": 64, "buffer": 10_000 * (1 + i % 5)})
        for s in (1000, 1001):
            pool.append({"method": "dqn", "cfg_id": i, "cfg": cfg, "seed": s, "val_final": -100.0 + i * 3.0})
    dump(tmp / f"tune_{INST}.jsonl", pool)
    full = -60.0 + rng.normal(0, 1.0, 10)
    dump(tmp / f"final_{INST}.jsonl", [{"method": "dqn", "seed": s, "cfg": CFG_ORIG, "test_mean": -64.0 + rng.normal(0, 1.0), "real_test_mean": -45.0} for s in seeds])
    dump(tmp / f"final_ext_{INST}.jsonl", [{"method": "dqn", "seed": s, "cfg": CFG_NEW, "cfg_id": 9, "test_mean": float(full[s]), "real_test_mean": float(full[s]) + 15} for s in seeds])
    variants = {"noreplay": full - 5.0 + rng.normal(0, 0.5, 10),         # completo melhor por ~5
                "notarget": full + rng.normal(0, 0.1, 10),               # equivalente
                "nodouble": full + 3.0 + rng.normal(0, 0.5, 10),         # variante melhor por ~3
                "vanilla": full - 8.0 + rng.normal(0, 0.5, 10)}
    dump(tmp / f"ablation_{INST}.jsonl", [{"variant": v, "seed": s, "cfg": CFG_NEW, "cfg_id": 9, "test_mean": float(x[s]), "real_test_mean": float(x[s]) + 15}
                                          for v, x in variants.items() for s in seeds])


class TestAnalysisP1P2(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        build(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def test_p1_report(self):
        txt = "\n".join(AN.p1_report(self.tmp))
        self.assertIn("| cb12wd | dqn | 1 (-80,00) | 9 (-70,00) | 10,00 | sim |", txt)
        self.assertIn("batch=256 (máx.)", txt)                    # a escolhida está no máximo explorado do pool
        self.assertNotIn("pool incompleto", txt)
        self.assertIn("Reruns", txt)                              # houve mudança -> compara teste original × estendido

    def test_p1_unchanged_choice_is_not_compared(self):
        ext = json.loads((self.tmp / f"tuned_ext_{INST}.json").read_text())
        ext["dqn"].update({"cfg_id": 1, "cfg": CFG_ORIG, "val_mean": -80.0})
        (self.tmp / f"tuned_ext_{INST}.json").write_text(json.dumps(ext))
        txt = "\n".join(AN.p1_report(self.tmp))
        self.assertIn("não (original mantida)", txt)
        self.assertNotIn("Reruns", txt)

    def test_p2_report_readings(self):
        lines = AN.p2_report(self.tmp, instances=(INST,))
        txt = "\n".join(lines)
        row = {l.split("|")[1].strip(): l for l in lines if l.startswith("| ") and l.count("|") > 11}
        self.assertIn("necessária", row["noreplay"])
        self.assertIn("equivalente", row["notarget"])
        self.assertIn("prejudica", row["nodouble"])
        self.assertIn("exploratória", row["vanilla"])
        self.assertNotIn("ATENÇÃO", txt)
        self.assertIn("final_ext_cb12wd.jsonl", txt)              # referência = rerun estendido

    def test_p2_flags_config_mismatch(self):
        rows = [json.loads(l) for l in (self.tmp / f"ablation_{INST}.jsonl").read_text().splitlines()]
        for r in rows:
            r["cfg"] = CFG_ORIG
        dump(self.tmp / f"ablation_{INST}.jsonl", rows)
        self.assertIn("ATENÇÃO", "\n".join(AN.p2_report(self.tmp, instances=(INST,))))

    def test_primary_rows_use_ext_only_where_config_changed(self):
        orig = json.loads((self.tmp / f"tuned_{INST}.json").read_text())
        ext = json.loads((self.tmp / f"tuned_ext_{INST}.json").read_text())
        orig["linear"] = {"cfg_id": 5, "cfg": CFG_ORIG, "val_mean": -90.0, "n_configs": 8, "seeds": [1000, 1001]}
        ext["linear"] = {"cfg_id": 5, "cfg": CFG_ORIG, "val_mean": -90.0, "n_configs": 16, "seeds": [1000, 1001]}
        (self.tmp / f"tuned_{INST}.json").write_text(json.dumps(orig))
        (self.tmp / f"tuned_ext_{INST}.json").write_text(json.dumps(ext))
        rows = [json.loads(l) for l in (self.tmp / f"final_{INST}.jsonl").read_text().splitlines()]
        rows += [{"method": "linear", "seed": s, "cfg": CFG_ORIG, "test_mean": -130.0} for s in range(10)]
        dump(self.tmp / f"final_{INST}.jsonl", rows)
        prim = AN.primary_rows(self.tmp, INST)
        by = defaultdict(list)
        for r in prim:
            by[(r["method"], r["p1_source"])].append(r["seed"])
        self.assertEqual(set(by), {("dqn", "ext"), ("linear", "orig")})        # dqn mudou -> ext; linear não -> original
        self.assertEqual(sorted(by[("dqn", "ext")]), list(range(10)))
        self.assertEqual(sorted(by[("linear", "orig")]), list(range(10)))
        self.assertTrue(all(r["cfg_id"] == 9 for r in prim if r["method"] == "dqn"))   # nenhuma linha da config original do dqn

    def test_tost_and_welch_sanity(self):
        d = np.random.default_rng(1).normal(0.0, 0.1, 10)
        self.assertLess(AN.tost_paired(d), 0.05)                   # diferença ~0 e pequena -> equivalente com δ = 0,5
        self.assertGreater(AN.tost_paired(d + 1.0), 0.5)           # diferença de 1 -> não equivalente
        diff, (lo, hi), p = AN.welch(np.arange(10.0) + 5, np.arange(10.0))
        self.assertAlmostEqual(diff, 5.0)
        self.assertTrue(lo < 5.0 < hi)
        self.assertLess(p, 0.01)


if __name__ == "__main__":
    unittest.main()
