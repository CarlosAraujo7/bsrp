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


class TestAnalysisP3(unittest.TestCase):
    """Sensibilidade (P3): a tabela marca onde a conclusão muda."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        rng = np.random.default_rng(3)

        def heur(inst, stay, h):
            return [{"instance": inst, "method": "stay", "test_mean": stay},
                    {"instance": inst, "method": "greedy_cal", "test_mean": h, "val_table": {"cfg": h}},
                    {"instance": inst, "method": "greedy_cal3", "test_mean": h - 5, "val_table": {"cfg": h - 6}}]   # a melhor por VAL é greedy_cal

        def learners(inst, dqn, lin, ts, tc):
            out = []
            for m, mu in (("dqn", dqn), ("linear", lin), ("tab_station", ts), ("tab_cluster", tc)):
                out += [{"instance": inst, "method": m, "seed": s, "test_mean": float(mu + rng.normal(0, 0.2))} for s in range(5)]
            return out

        # variante A: DQN claramente melhor; tabulares/linear piores que a heurística (−8)
        # variante B: DQN ≈ heurística (C1 não significativo) e tab_cluster melhor que a heurística
        dump(self.tmp / "heuristics_sens_a.jsonl", heur("cb4wdx0.5", -10.0, -8.0) + heur("cb4wdx1.5", -10.0, -8.0))
        dump(self.tmp / "sens_cb4wdx0.5.jsonl", learners("cb4wdx0.5", -3.0, -12.0, -9.0, -9.5))
        dump(self.tmp / "sens_cb4wdx1.5.jsonl", learners("cb4wdx1.5", -8.0, -12.0, -9.0, -7.0))

    def tearDown(self):
        self._td.cleanup()

    def test_p3_flags_where_conclusion_changes(self):
        lines = AN.p3_report(self.tmp, bases=("cb4wd",))
        txt = "\n".join(lines)
        row_a = next(l for l in lines if "`cb4wdx0.5`" in l)
        row_b = next(l for l in lines if "`cb4wdx1.5`" in l)
        self.assertNotIn("⚠", row_a)                                           # conclusão mantida
        self.assertIn("greedy_cal", row_a)                                     # melhor heurística escolhida por val, não por teste
        self.assertIn("⚠", row_b)
        self.assertIn("cb4wdx1.5: C1 NÃO é significativo", txt)
        self.assertIn("cb4wdx1.5: tab_cluster supera(m) a melhor heurística", txt)
        self.assertNotIn("cb4wdx0.5:", txt.split("Onde a conclusão muda")[1])
        self.assertIn("(pendente)", txt)                                       # variantes ainda sem execuções

    def test_p3_distinguishes_worse_from_not_significant(self):
        dump(self.tmp / "heuristics_sens_a.jsonl",
             [{"instance": "cb4wdx0.75", "method": "stay", "test_mean": -10.0},
              {"instance": "cb4wdx0.75", "method": "greedy_cal", "test_mean": -4.0, "val_table": {"c": -4.0}}])
        dump(self.tmp / "sens_cb4wdx0.75.jsonl", [{"instance": "cb4wdx0.75", "method": "dqn", "seed": s, "test_mean": -5.0 + 0.05 * s} for s in range(5)])
        txt = "\n".join(AN.p3_report(self.tmp, bases=("cb4wd",)))
        self.assertIn("cb4wdx0.75: o DQN é SIGNIFICATIVAMENTE PIOR", txt)
        self.assertNotIn("cb4wdx0.75: C1 NÃO é significativo", txt)

    def test_p3_reduction_is_not_reported_when_heuristic_is_near_zero(self):
        dump(self.tmp / "heuristics_sens_a.jsonl",
             [{"instance": "legacy4x1.5", "method": "stay", "test_mean": -0.4},
              {"instance": "legacy4x1.5", "method": "greedy_cal", "test_mean": -0.1, "val_table": {"c": -0.1}}])
        dump(self.tmp / "sens_legacy4x1.5.jsonl", [{"instance": "legacy4x1.5", "method": "dqn", "seed": s, "test_mean": -0.2 + 0.01 * s} for s in range(5)])
        row = next(l for l in AN.p3_report(self.tmp, bases=("cb4wd",)) if "`legacy4x1.5`" in l)
        self.assertIn("n/d (heurística ≈ 0)", row)
        self.assertNotIn("%", row)

    def test_p3_reduction_and_c1_values(self):
        row = next(l for l in AN.p3_report(self.tmp, bases=("cb4wd",)) if "`cb4wdx0.5`" in l)
        cells = [c.strip() for c in row.split("|")]
        self.assertIn(cells[7], ("62%", "63%"))                                # (8 − ~3) / 8 ≈ 62,5 %
        c1 = float(cells[6].split(" ")[0].replace(",", "."))
        self.assertAlmostEqual(c1, 5.0, delta=0.5)                             # DQN (−3) − heurística (−8)

class TestAnalysisP4(unittest.TestCase):
    """P4: referência por VALIDAÇÃO entre famílias; rollout de cabeçalho por val; contrastes com DQN e A2C."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        rng = np.random.default_rng(7)
        inst = "cb4wd"
        # a threshold é a melhor em VAL (−20 contra −22) apesar de pior no teste: a regra por val a escolhe como referência
        dump(self.tmp / "heuristics_cb_wd.jsonl",
             [{"instance": inst, "method": "stay", "test_mean": -47.0},
              {"instance": inst, "method": "greedy_cal2", "policy": "greedy(lo=0.25,hi=0.55,d=10)", "test_mean": -24.0, "val_table": {"g": -22.0}}])
        dump(self.tmp / "heuristics_cb_wd_threshold.jsonl",
             [{"instance": inst, "method": "threshold_cal", "policy": "threshold(t=0.5,lo=0.35,hi=0.6,d=10,w=0.0)", "test_mean": -26.0, "val_table": {"t": -20.0}}])
        dump(self.tmp / f"rollout_{inst}.jsonl",
             [{"instance": inst, "h": 1, "K": 4, "val_mean": -18.0, "test_mean": -20.0, "real_test_mean": -21.0},
              {"instance": inst, "h": 2, "K": 16, "val_mean": -12.0, "test_mean": -14.0, "real_test_mean": -15.0}])   # melhor em val
        dump(self.tmp / f"seo_a2c_{inst}.jsonl",
             [{"instance": inst, "variant": "published", "seed": s, "test_mean": float(-30 + rng.normal(0, 1)), "curve": [{"val": -29.0}],
               "real_test_mean": -28.0} for s in range(10)]
             + [{"instance": inst, "variant": "gamma1", "seed": s, "test_mean": float(-18 + rng.normal(0, 1)), "curve": [{"val": -17.0}],
                 "real_test_mean": -16.0} for s in range(10)])
        dump(self.tmp / "final_primary.jsonl",
             [{"instance": inst, "method": "dqn", "seed": s, "test_mean": float(-10 + rng.normal(0, 0.5))} for s in range(10)])

    def tearDown(self):
        self._td.cleanup()

    def test_reference_is_chosen_by_validation(self):
        txt = "\n".join(AN.p4_report(self.tmp, bases=("cb4wd",)))
        self.assertIn("Referência por val: **threshold_cal**", txt)
        self.assertIn("(teste -26,00;", txt)

    def test_headline_rollout_is_best_in_validation_and_contrasts_present(self):
        lines = AN.p4_report(self.tmp, bases=("cb4wd",))
        txt = "\n".join(lines)
        head = next(l for l in lines if "cabeçalho" in l and l.startswith("| rollout"))
        self.assertIn("h=2, K=16", head)
        self.assertIn("DQN − rollout de cabeçalho (h=2, K=16)", txt)
        self.assertIn("10/10 sementes melhores: sim", txt)                 # DQN (−10) > rollout (−14) em todas as sementes
        self.assertIn("DQN − A2C Seo (publicado)", txt)
        self.assertIn("A2C Seo (γ = 1,0), n = 10", txt)

    def test_reduction_is_relative_to_the_validation_reference(self):
        row = next(l for l in AN.p4_report(self.tmp, bases=("cb4wd",)) if "h=2, K=16" in l and l.startswith("| rollout"))
        self.assertIn("46%", row)                                          # (26 − 14) / 26 = 46 %


if __name__ == "__main__":
    unittest.main()
