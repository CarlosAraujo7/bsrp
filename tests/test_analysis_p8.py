"""Análise do P8 (scripts/30_analyze_p8.py): fração do ganho contra o clarividente, regra de 'limite ≈ zero', leitura do orçamento 4×,
reavaliação das top-4 e P3c."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("an30", ROOT / "scripts" / "30_analyze_p8.py")
P8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P8)


def dump(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def rewards(mu, n=1000, seed=0, sd=1.0):
    return list(np.random.default_rng(seed).normal(mu, sd, n))


class TestOracleSection(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb4wd"
        # parada −40; heurística −20; rollout −10; DQN médio −10 (3 sementes: −9,5, −10, −10,5); clarividente −3 (não ≈ 0: −5% de 40 = −2)
        dump(t / "heuristics_cb_wd.jsonl", [
            {"instance": inst, "method": "stay", "test_rewards": [-40.0] * 1000, "real_test_rewards": [-40.0] * 24, "test_mean": -40.0},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -20.0}, "test_rewards": [-20.0] * 1000,
             "real_test_rewards": [-20.0] * 24, "test_mean": -20.0}])
        dump(t / "rollout_cb4wd.jsonl", [{"instance": inst, "h": 3, "K": 64, "val_mean": -10.0, "test_mean": -10.0,
                                          "test_rewards": [-10.0] * 1000, "real_test_rewards": [-10.0] * 24}])
        dump(t / "final_primary.jsonl", [{"instance": inst, "method": "dqn", "seed": s, "test_mean": -10.5,
                                          "test_rewards": [-10.0 - 0.5 * (s - 1)] * 1000, "real_test_rewards": [-10.5] * 24} for s in range(3)])
        for W, v_test in ((300, -4.0), (1000, -3.0)):
            rows = [{"instance": inst, "set": "test", "idx": k, "width": W, "value": v_test} for k in range(5)]
            rows += [{"instance": inst, "set": "real", "idx": k, "width": W, "value": v_test} for k in range(24)]
            dump(t / f"oracle_{inst}_w{W}.jsonl", rows)

    def tearDown(self):
        self._td.cleanup()

    def test_fraction_of_gain_is_against_the_best_width_and_flags_a_loose_bound(self):
        txt = "\n".join(P8.oracle_section(self.tmp, bases=("cb4wd",)))
        row = next(l for l in txt.splitlines() if l.startswith("| teste sintético"))
        cells = [c.strip() for c in row.split("|")]
        # melhor valor por cenário = max(−4, −3) = −3 → denominador 37: heurística 20/37 = 54 %; rollout 30/37 = 81 %; DQN (−10 + 40)/37 = 81 %
        self.assertEqual(cells[10], "54%")
        self.assertEqual(cells[11], "81%")
        self.assertTrue(cells[12].startswith("81%"))
        self.assertTrue(cells[13].startswith("não"))                 # −3 < −5% × 40 = −2: limite não é ≈ zero
        self.assertIn("Leitura (fixada antes de ver os dados)", txt)

    def test_near_zero_clairvoyant_is_declared_loose(self):
        for W in (300, 1000):
            rows = [{"instance": "cb4wd", "set": "test", "idx": k, "width": W, "value": -1.0} for k in range(5)]
            rows += [{"instance": "cb4wd", "set": "real", "idx": k, "width": W, "value": -1.0} for k in range(24)]
            dump(self.tmp / f"oracle_cb4wd_w{W}.jsonl", rows)
        txt = "\n".join(P8.oracle_section(self.tmp, bases=("cb4wd",)))
        row = next(l for l in txt.splitlines() if l.startswith("| teste sintético"))
        self.assertIn("sim: limite ≈ nenhuma demanda perdida", row)

    def test_incomplete_real_days_are_not_reported(self):
        dump(self.tmp / "oracle_cb4wd_w300.jsonl", [{"instance": "cb4wd", "set": "test", "idx": k, "width": 300, "value": -3.0} for k in range(5)]
             + [{"instance": "cb4wd", "set": "real", "idx": k, "width": 300, "value": -3.0} for k in range(10)])
        dump(self.tmp / "oracle_cb4wd_w1000.jsonl", [])
        txt = "\n".join(P8.oracle_section(self.tmp, bases=("cb4wd",)))
        self.assertIn("| teste sintético", txt)
        self.assertNotIn("| dias reais", txt)

    def test_pending_when_there_is_no_oracle_file(self):
        (self.tmp / "oracle_cb4wd_w300.jsonl").unlink()
        (self.tmp / "oracle_cb4wd_w1000.jsonl").unlink()
        self.assertIn("(pendente)", "\n".join(P8.oracle_section(self.tmp, bases=("cb4wd",))))


class TestBudgetSection(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb12wd"
        dump(t / "heuristics_cb_wd.jsonl", [{"instance": inst, "method": "stay", "test_rewards": [-90.0] * 1000, "real_test_rewards": [-90.0] * 24, "test_mean": -90.0},
                                           {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -80.0}, "test_rewards": [-80.0] * 1000,
                                            "real_test_rewards": [-80.0] * 24, "test_mean": -80.0}])
        dump(t / "rollout_cb12wd.jsonl", [{"instance": inst, "h": 3, "K": 64, "val_mean": -50.0, "test_mean": -50.0, "test_rewards": [-50.0] * 1000,
                                           "real_test_rewards": [-50.0] * 24}])
        rng = np.random.default_rng(1)
        dump(t / "final_primary.jsonl", [{"instance": inst, "method": "dqn", "seed": s, "test_mean": float(-56 + rng.normal(0, 1)), "test_rewards": [-56.0] * 1000,
                                          "real_test_rewards": [-56.0] * 24} for s in range(10)])
        self.rng = rng

    def tearDown(self):
        self._td.cleanup()

    def test_reading_when_more_budget_helps_and_dqn_overtakes_the_rollout(self):
        dump(self.tmp / "budget4x_cb12wd.jsonl", [{"instance": "cb12wd", "method": "dqn", "seed": s, "test_mean": float(-45 + self.rng.normal(0, 0.5))} for s in range(5)])
        txt = "\n".join(P8.budget_section(self.tmp, bases=("cb12wd",)))
        self.assertIn("o orçamento de 500 mil passos limita o DQN", txt)
        self.assertIn("DQN(2 M) supera o rollout", txt)

    def test_reading_when_more_budget_does_not_help_and_rollout_stays_better(self):
        dump(self.tmp / "budget4x_cb12wd.jsonl", [{"instance": "cb12wd", "method": "dqn", "seed": s, "test_mean": float(-56 + self.rng.normal(0, 1))} for s in range(5)])
        txt = "\n".join(P8.budget_section(self.tmp, bases=("cb12wd",)))
        self.assertIn("sem ganho detectável com 4× o orçamento", txt)
        self.assertIn("o rollout continua melhor", txt)

    def test_pending_with_fewer_than_two_seeds(self):
        dump(self.tmp / "budget4x_cb12wd.jsonl", [{"instance": "cb12wd", "method": "dqn", "seed": 0, "test_mean": -45.0}])
        self.assertIn("(pendente)", "\n".join(P8.budget_section(self.tmp, bases=("cb12wd",))))


class TestReseedAndP3c(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_reseed_confirmed_changed_and_pending(self):
        t = self.tmp
        dump(t / "final_primary.jsonl", [{"instance": "cb8wd", "method": "dqn", "seed": s, "test_mean": -30.0 + 0.1 * s} for s in range(10)])
        (t / "reseed_cb4wd_decision.json").write_text(json.dumps({"means_all_seeds": {"11": -7.0, "3": -7.5}, "p1_cfg_id": 11, "selected_cfg_id": 11, "changed": False}))
        (t / "reseed_cb8wd_decision.json").write_text(json.dumps({"means_all_seeds": {"9": -33.0, "5": -31.0}, "p1_cfg_id": 9, "selected_cfg_id": 5, "changed": True}))
        dump(t / "final_reseed_cb8wd.jsonl", [{"instance": "cb8wd", "method": "dqn", "seed": s, "test_mean": -29.0 + 0.1 * s} for s in range(10)])
        txt = "\n".join(P8.reseed_section(t, bases=("cb4wd", "cb8wd", "cb12wd")))
        self.assertIn("seleção confirmada", txt)
        self.assertIn("mudou; finais (n = 10)", txt)
        self.assertTrue(any(l.startswith("| cb12wd") and "(pendente)" in l for l in txt.splitlines()))

    def test_p3c_uses_up_to_ten_seeds_and_the_retuned_file(self):
        t = self.tmp
        cfg = {"lr": 3e-4}
        mk = lambda cid, v: {"dqn": {"cfg_id": cid, "cfg": cfg, "val_mean": v, "n_configs": 16, "seeds": [1000, 1001], "budget": 200000}}
        (t / "tuned_ext_cb12wd.json").write_text(json.dumps(mk(9, -69.0)))
        (t / "tuned_ext_cb12wdx0.5.json").write_text(json.dumps(mk(14, -7.3)))
        (t / "tuned_ext_cb12we.json").write_text(json.dumps(mk(14, -18.6)))
        dump(t / "heuristics_sens_a.jsonl", [{"instance": n, "method": m, "test_mean": v, "val_table": {"c": v}} for n in ("cb12wdx0.5", "cb12we")
                                            for m, v in (("stay", -20.0), ("greedy_cal", -10.0))])
        rng = np.random.default_rng(5)
        for n in ("cb12wdx0.5", "cb12we"):
            dump(t / f"sens_{n}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-11 + rng.normal(0, 0.3))} for s in range(10)]
                 + [{"method": "linear", "seed": s, "test_mean": -30.0} for s in range(5)])
            dump(t / f"sens_retuned_{n}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-9 + rng.normal(0, 0.3))} for s in range(10)])
        txt = "\n".join(P8.reseed_section(t, bases=()))
        row = next(l for l in txt.splitlines() if l.startswith("| K = 0,5 (`cb12wdx0.5`)"))
        cells = [c.strip() for c in row.split("|")]
        num = lambda c: float(c.split(" ")[0].replace(",", "."))
        self.assertEqual(cells[2], "10")                                   # 10 sementes sem reajuste (o linear não entra)
        self.assertEqual(cells[4], "10")
        self.assertAlmostEqual(num(cells[3]), -1.0, delta=0.3)             # C1 sem reajuste ≈ −11 − (−10)
        self.assertAlmostEqual(num(cells[5]), 1.0, delta=0.3)              # C1 reajustado ≈ −9 − (−10)


class TestScaleData(unittest.TestCase):
    """P9: DQN × heurística de 4 a 30 estações; 20/30 vêm de scale_/scale2m_ e das heurísticas em heuristics_scale_<inst>.jsonl."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb20wd"
        dump(t / f"heuristics_scale_{inst}.jsonl", [
            {"instance": inst, "method": "stay", "test_mean": -360.0, "real_test_mean": -310.0},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -280.0}, "test_mean": -278.0, "real_test_mean": -265.0},
            {"instance": inst, "method": "greedy_cal2", "policy": "g2", "val_table": {"c": -281.0}, "test_mean": -270.0, "real_test_mean": -260.0}])   # melhor em TESTE, não em val
        rng = np.random.default_rng(2)
        dump(t / f"scale_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-250 + rng.normal(0, 2)), "real_test_mean": -240.0} for s in range(5)])
        dump(t / f"scale2m_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-200 + rng.normal(0, 2)), "real_test_mean": -190.0} for s in range(5)])

    def tearDown(self):
        self._td.cleanup()

    def test_reference_is_chosen_by_validation_and_reduction_is_relative_to_it(self):
        d = P8.scale_data(self.tmp, scales=("cb20wd",))["cb20wd"]
        self.assertEqual(d["heur_family"], "greedy_cal")
        self.assertAlmostEqual(d["heur"], 278.0)
        self.assertAlmostEqual(d["500k"]["red"], 100 * (278 - 250) / 278, delta=1.0)
        self.assertAlmostEqual(d["2m"]["red"], 100 * (278 - 200) / 278, delta=1.0)
        self.assertGreater(d["2m"]["c1"], d["500k"]["c1"])                       # C1 em reward: positivo favorece o DQN
        self.assertTrue(d["500k"]["red_lo"] <= d["500k"]["red"] <= d["500k"]["red_hi"])
        self.assertNotIn("rollout", d)                                           # não rodado em 20/30

    def test_scales_without_enough_seeds_or_reference_are_skipped(self):
        dump(self.tmp / "scale2m_cb20wd.jsonl", [{"method": "dqn", "seed": 0, "test_mean": -200.0}])
        d = P8.scale_data(self.tmp, scales=("cb20wd", "cb30wd"))
        self.assertNotIn("2m", d["cb20wd"])
        self.assertNotIn("cb30wd", d)

    def test_section_lists_pending_and_the_no_rollout_note(self):
        txt = "\n".join(P8.scale_section(self.tmp))
        self.assertIn("O rollout em 20/30 estações usa (h, K) = (3, 64) sem ajuste", txt)
        self.assertIn("não rodado", txt)                                         # sem episódios de rollout: coluna marcada


class TestScaleRollout(unittest.TestCase):
    """P9c: rollout por episódio em 20/30 estações, comparado à heurística e ao DQN NOS MESMOS n primeiros cenários."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb20wd"
        rng = np.random.default_rng(4)
        dump(t / f"heuristics_scale_{inst}.jsonl", [
            {"instance": inst, "method": "stay", "test_mean": -360.0, "test_rewards": [-360.0] * 1000},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -280.0}, "test_mean": -278.0, "test_rewards": [-250.0] * 5 + [-278.0] * 995}])
        # DQN: nos 5 primeiros cenários é pior que no resto (a comparação por subconjunto deve usar só os 5 primeiros)
        dump(t / f"scale_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": -220.0, "test_rewards": [-240.0] * 5 + [-219.0] * 995} for s in range(3)])
        dump(t / f"scale2m_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": -200.0, "test_rewards": [-200.0 + float(rng.normal(0, 0.5))] * 1000} for s in range(3)])
        rows = [{"instance": inst, "set": "test", "idx": k, "reward": -210.0} for k in range(5)] + \
               [{"instance": inst, "set": "real_test", "idx": k, "reward": -190.0} for k in range(24)]
        dump(t / f"rollout_scale_{inst}.jsonl", rows)

    def tearDown(self):
        self._td.cleanup()

    def test_rollout_is_compared_on_the_same_first_scenarios(self):
        d = P8.scale_data(self.tmp, scales=("cb20wd",))["cb20wd"]
        ro = d["rollout"]
        self.assertEqual(ro["n"], 5)
        self.assertAlmostEqual(ro["heur_sub"], 250.0)                              # heurística nos 5 primeiros, não nos 1000
        self.assertAlmostEqual(ro["red"], 100 * (250 - 210) / 250)
        self.assertAlmostEqual(ro["500k"]["dqn_red"], 100 * (250 - 240) / 250)     # DQN também nos 5 primeiros
        self.assertAlmostEqual(ro["500k"]["c"], -240.0 - (-210.0))                 # DQN − rollout em reward: negativo = rollout melhor
        self.assertGreater(ro["2m"]["c"], 0)

    def test_incomplete_episodes_do_not_produce_a_rollout_entry(self):
        rows = [{"instance": "cb20wd", "set": "test", "idx": k, "reward": -210.0} for k in range(5)] + \
               [{"instance": "cb20wd", "set": "real_test", "idx": k, "reward": -190.0} for k in range(10)]        # faltam dias reais
        dump(self.tmp / "rollout_scale_cb20wd.jsonl", rows)
        self.assertNotIn("rollout", P8.scale_data(self.tmp, scales=("cb20wd",))["cb20wd"])


class TestArchData(unittest.TestCase):
    """P10: DQN com rede larga × 64–64 (Welch) e × rollout (subconjunto de cenários em 30 est.)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb30wd"
        rng = np.random.default_rng(8)
        dump(t / f"heuristics_scale_{inst}.jsonl", [
            {"instance": inst, "method": "stay", "test_mean": -470.0, "test_rewards": [-470.0] * 1000},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -380.0}, "test_mean": -380.0, "test_rewards": [-380.0] * 1000}])
        dump(t / f"scale_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-340 + rng.normal(0, 1)), "test_rewards": [-340.0] * 1000} for s in range(5)])
        dump(t / f"rollout_scale_{inst}.jsonl", [{"instance": inst, "set": "test", "idx": k, "reward": -300.0} for k in range(300)]
             + [{"instance": inst, "set": "real_test", "idx": k, "reward": -290.0} for k in range(24)])
        dump(t / f"arch_h128_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-330 + rng.normal(0, 1)), "test_rewards": [-330.0] * 1000, "train_seconds": 100.0} for s in range(5)])
        dump(t / f"arch_h256_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-345 + rng.normal(0, 1)), "test_rewards": [-345.0] * 1000, "train_seconds": 200.0} for s in range(5)])

    def tearDown(self):
        self._td.cleanup()

    def test_delta_vs_64_64_and_vs_rollout_on_the_subset_and_the_reading(self):
        d = P8.arch_data(self.tmp, insts=("cb30wd",))["cb30wd"]
        wide, wider = d["Wide"], d["Wider"]
        self.assertAlmostEqual(wide["delta"], 10.0, delta=2.0)                  # 128–128 melhora ~10 de demanda perdida vs 64–64 (reward: positivo)
        self.assertGreater(wide["delta_lo"], 0)
        self.assertLess(wider["delta_hi"], 0)                                   # 256–256 piora
        self.assertAlmostEqual(wide["vs_roll"], -330.0 - (-300.0), delta=2.0)   # DQN − rollout em reward: negativo = rollout melhor
        self.assertEqual(d["rollout_subset"], 300)
        txt = "\n".join(P8.arch_section(self.tmp))
        self.assertIn("a capacidade da rede limita o DQN", txt)
        self.assertIn("a rede maior piora", txt)

    def test_pending_without_arch_rows(self):
        (self.tmp / "arch_h256_cb30wd.jsonl").unlink()
        d = P8.arch_data(self.tmp, insts=("cb30wd",))["cb30wd"]
        self.assertIn("Wide", d)
        self.assertNotIn("Wider", d)
        self.assertIn("(pendente)", "\n".join(P8.arch_section(self.tmp)))


class TestDprlData(unittest.TestCase):
    """P11: adaptação do DPRL × DQN plano (Welch), × heurística e × rollout (subconjunto em 30 est.)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb30wd"
        rng = np.random.default_rng(9)
        dump(t / f"heuristics_scale_{inst}.jsonl", [
            {"instance": inst, "method": "stay", "test_mean": -470.0, "test_rewards": [-470.0] * 1000},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -380.0}, "test_mean": -380.0, "test_rewards": [-380.0] * 1000}])
        dump(t / f"scale_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": float(-340 + rng.normal(0, 1)), "test_rewards": [-340.0] * 1000} for s in range(5)])
        dump(t / f"rollout_scale_{inst}.jsonl", [{"instance": inst, "set": "test", "idx": k, "reward": -300.0} for k in range(300)]
             + [{"instance": inst, "set": "real_test", "idx": k, "reward": -290.0} for k in range(24)])
        dump(t / f"dprl_{inst}.jsonl", [{"method": "dprl", "seed": s, "test_mean": float(-320 + rng.normal(0, 1)), "test_rewards": [-320.0] * 1000, "train_seconds": 90.0} for s in range(5)])

    def tearDown(self):
        self._td.cleanup()

    def test_dual_vs_flat_welch_reduction_and_vs_rollout_on_the_subset(self):
        d = P8.dprl_data(self.tmp, scales=("cb30wd",))["cb30wd"]
        self.assertAlmostEqual(d["delta"], 20.0, delta=2.0)                     # dual melhora ~20 de demanda perdida vs plano (reward positivo)
        self.assertGreater(d["delta_lo"], 0)
        self.assertAlmostEqual(d["red"], 100 * (380 - 320) / 380, delta=0.5)
        self.assertAlmostEqual(d["flat_red"], 100 * (380 - 340) / 380, delta=0.5)
        self.assertAlmostEqual(d["vs_roll"], -320.0 - (-300.0), delta=1.0)      # dual − rollout em reward: negativo = rollout melhor
        txt = "\n".join(P8.dprl_section(self.tmp))
        self.assertIn("a fatoração em duas políticas melhora o DQN", txt)

    def test_worse_and_tie_readings_and_pending(self):
        dump(self.tmp / "dprl_cb30wd.jsonl", [{"method": "dprl", "seed": s, "test_mean": -360.0 + 0.1 * s, "test_rewards": [-360.0] * 1000} for s in range(5)])
        self.assertIn("piora o DQN", "\n".join(P8.dprl_section(self.tmp)))
        dump(self.tmp / "dprl_cb30wd.jsonl", [{"method": "dprl", "seed": 0, "test_mean": -340.0, "test_rewards": [-340.0] * 1000}])
        self.assertEqual(P8.dprl_data(self.tmp, scales=("cb30wd",)), {})                 # menos de 2 sementes
        self.assertIn("(pendente)", "\n".join(P8.dprl_section(self.tmp)))


class TestCecData(unittest.TestCase):
    """Tabela única do CEC: todos os métodos nos mesmos n primeiros cenários de teste."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        t = self.tmp
        inst = "cb30wd"
        rng = np.random.default_rng(12)
        dump(t / f"heuristics_scale_{inst}.jsonl", [
            {"instance": inst, "method": "stay", "test_mean": -470.0, "test_rewards": [-470.0] * 1000, "real_test_mean": -450.0},
            {"instance": inst, "method": "greedy_cal", "policy": "g", "val_table": {"c": -380.0}, "test_mean": -380.0, "test_rewards": [-400.0] * 5 + [-380.0] * 995, "real_test_mean": -390.0}])
        dump(t / f"scale_{inst}.jsonl", [{"method": "dqn", "seed": s, "test_mean": -340.0, "test_rewards": [-340.0 + float(rng.normal(0, 1))] * 1000, "real_test_mean": -350.0} for s in range(4)])
        dump(t / f"es_{inst}.jsonl", [{"method": "es", "seed": s, "test_mean": -400.0, "test_rewards": [-400.0 + float(rng.normal(0, 1))] * 1000, "real_test_mean": -410.0} for s in range(4)])
        dump(t / f"rollout_scale_{inst}.jsonl", [{"instance": inst, "set": "test", "idx": k, "reward": -300.0} for k in range(300)]
             + [{"instance": inst, "set": "real_test", "idx": k, "reward": -290.0} for k in range(24)])
        dump(t / f"rhea_{inst}.jsonl", [{"instance": inst, "config": "S", "set": "test", "idx": k, "reward": -310.0} for k in range(300)]
             + [{"instance": inst, "config": "S", "set": "real_test", "idx": k, "reward": -295.0} for k in range(24)]
             + [{"instance": inst, "config": "L", "set": "test", "idx": k, "reward": -305.0} for k in range(10)])             # L incompleto: não entra
        rows = []
        for sd in range(4):
            for fam, val, test in (("greedy", -375.0, -378.0), ("threshold", -377.0 - sd, -379.0)):
                rows.append({"instance": inst, "family": fam, "seed": sd, "val_best": val, "test_mean": test, "test_rewards": [test] * 1000, "real_test_mean": -388.0})
        dump(t / f"cmaes_{inst}.jsonl", rows)

    def tearDown(self):
        self._td.cleanup()

    def test_subset_reduction_and_availability(self):
        d = P8.cec_data(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertAlmostEqual(d["grid"]["mean"], (5 * 400 + 295 * 380) / 300, places=6)                 # grade no subconjunto
        self.assertEqual(d["heur"]["src"], "cma")                                                        # val do CMA (−375) > val da grade (−380): referência = execução do CMA-ES
        self.assertAlmostEqual(d["heur"]["mean"], 378.0)
        self.assertAlmostEqual(d["rollout"]["mean"], 300.0)
        self.assertIn("rhea_s", d)
        self.assertNotIn("rhea_l", d)                                                                    # incompleto
        self.assertEqual(d["dqn"]["n"], 4)
        self.assertEqual(d["cma"]["n"], 4)                                                               # uma linha por execução, família de melhor validação
        self.assertAlmostEqual(d["cma"]["mean"], 378.0)                                                  # greedy tem a melhor validação em todas as execuções
        self.assertGreater(d["dqn"]["red"], d["es"]["red"])

    def test_contrasts_welch_and_vs_deterministic(self):
        c = P8.cec_contrasts(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertAlmostEqual(c["es_vs_dqn"]["delta"], -60.0, delta=2.0)                               # ES − DQN em reward: negativo = DQN melhor
        self.assertLess(c["es_vs_dqn"]["hi"], 0)
        self.assertAlmostEqual(c["dqn_vs_rollout"]["delta"], -340.0 - (-300.0), delta=2.0)
        self.assertIn("es_vs_rhea_s", c)


    def test_reference_stays_grid_when_cma_is_worse_in_validation(self):
        dump(self.tmp / "cmaes_cb30wd.jsonl", [{"instance": "cb30wd", "family": f, "seed": s, "val_best": -390.0, "test_mean": -370.0, "test_rewards": [-370.0] * 1000,
                                                "real_test_mean": -1.0} for s in range(3) for f in ("greedy", "threshold")])
        d = P8.cec_data(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertEqual(d["heur"]["src"], "grid")                                                       # a escolha é por validação: o teste (−370, melhor) não entra
        self.assertAlmostEqual(d["heur"]["mean"], (5 * 400 + 295 * 380) / 300, places=6)

    def test_calibration_verdicts(self):
        c = P8.cec_calibration(self.tmp, scales=("cb30wd",))["cb30wd"]
        self.assertAlmostEqual(c["grid_test"], 380.0)
        self.assertAlmostEqual(c["cma_test"], 378.0)
        self.assertAlmostEqual(c["delta"], 2.0)                                                          # reward: CMA melhor que a grade em 2 (positivo)
        self.assertEqual(c["n"], 4)
        self.assertEqual(c["verdict"], "inconclusive" if c["lo"] <= 0 else "better")                    # IC degenerado (variância zero): lo = hi = 2 → "better"
        self.assertEqual(c["verdict"], "better")

    def test_calibration_equivalent_requires_tost(self):
        rows = [{"instance": "cb30wd", "family": "greedy", "seed": s, "val_best": -375.0, "test_mean": -380.0 + e, "test_rewards": [-380.0] * 1000, "real_test_mean": -1.0}
                for s, e in enumerate((-0.1, 0.1, 0.05, -0.05, 0.0, 0.02))]
        rows += [{**r, "family": "threshold", "val_best": -400.0} for r in rows]
        dump(self.tmp / "cmaes_cb30wd.jsonl", rows)
        c = P8.cec_calibration(self.tmp, scales=("cb30wd",))["cb30wd"]
        self.assertEqual(c["verdict"], "equivalent")                                                     # IC dentro de ±0,5 e TOST rejeita
        self.assertLess(c["p_tost"], 0.05)

    def test_es_budget_and_reached(self):
        t = self.tmp
        dump(t / "scale_cb30wd.jsonl", [{"method": "dqn", "seed": s, "test_mean": -340.0 + (s - 1.5)} for s in range(4)])
        dump(t / "es_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -400.0 + (s - 1.5)} for s in range(4)])
        dump(t / "es2m_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -390.0 + s} for s in range(5)])
        dump(t / "es8m_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -341.0 + s} for s in range(3)])
        dump(t / "es32m_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -338.0 + 0.5 * s} for s in range(3)])
        b = P8.cec_es_budget(t, scales=("cb30wd",))["cb30wd"]
        self.assertEqual(sorted(b["budgets"]), ["2M", "32M", "500k", "8M"])
        self.assertAlmostEqual(b["dqn"], 340.0, delta=1.0)
        self.assertLess(b["budgets"]["500k"]["vs_dqn_hi"], 0)                                           # ES 500k (−400) pior que o DQN (−340)
        self.assertLess(b["budgets"]["500k"]["vs_heur_hi"], 0)                                          # ES 500k (−400) pior que a referência (−378)
        self.assertGreater(b["budgets"]["32M"]["vs_heur"], 0)                                            # ES 32M (−338) melhor que a referência (−378)
        self.assertEqual(b["reached"], "8M")                                                             # primeiro cujo IC inclui 0: 8M (−340 ± ...)

    def test_curves_are_positive_unmet_with_ci_over_seeds(self):
        t = self.tmp
        dump(t / "scale_cb30wd.jsonl", [{"method": "dqn", "seed": s, "test_mean": -340.0, "train_seconds": 100.0 + s,
                                         "curve": [{"steps": 50000, "val": -400.0 - s}, {"steps": 100000, "val": -380.0 + s}]} for s in range(3)])
        dump(t / "es_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -420.0, "train_seconds": 40.0,
                                      "curve": [{"steps": 51000, "val": -450.0}, {"steps": 102000, "val": -440.0 + 2 * s}]} for s in range(3)])
        c = P8.cec_curves(t, scales=("cb30wd",))["cb30wd"]
        self.assertEqual(c["dqn"]["n"], 3)
        self.assertAlmostEqual(c["dqn"]["mean"][0], 401.0)                                               # val −400 −s → demanda 400 + s, média 401
        self.assertAlmostEqual(c["dqn"]["mean"][1], 379.0)
        self.assertTrue((c["dqn"]["lo"] < c["dqn"]["mean"]).all() and (c["dqn"]["hi"] > c["dqn"]["mean"]).all())
        self.assertEqual(list(c["es"]), ["500k"])                                                        # só o orçamento com arquivo
        self.assertAlmostEqual(c["es"]["500k"]["steps"][1], 102000.0)

    def test_sep_budget_uses_its_own_files_and_leaves_the_es_curve_alone(self):
        t = self.tmp
        dump(t / "scale_cb30wd.jsonl", [{"method": "dqn", "seed": s, "test_mean": -340.0 + (s - 1.5), "train_seconds": 1000.0} for s in range(4)])
        dump(t / "es_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -400.0 + (s - 1.5), "train_seconds": 40.0} for s in range(4)])
        dump(t / "sepcma_cb30wd.jsonl", [{"method": "sepcma", "seed": s, "test_mean": -390.0 + (s - 1.5), "train_seconds": 50.0} for s in range(4)])
        dump(t / "sepcma32m_cb30wd.jsonl", [{"method": "sepcma", "seed": s, "test_mean": -360.0 + s, "train_seconds": 2400.0} for s in range(3)])
        sb = P8.cec_sep_budget(t, scales=("cb30wd",))["cb30wd"]
        self.assertEqual(sorted(sb["budgets"]), ["32M", "500k"])
        self.assertAlmostEqual(sb["budgets"]["500k"]["mean"], 390.0)
        self.assertLess(sb["budgets"]["500k"]["vs_dqn_hi"], 0)
        self.assertAlmostEqual(sb["budgets"]["32M"]["secs"], 2400.0)
        eb = P8.cec_es_budget(t, scales=("cb30wd",))["cb30wd"]
        self.assertEqual(sorted(eb["budgets"]), ["500k"])                                      # o ES não enxerga os arquivos do sep-CMA-ES

    def test_es_budget_training_time_and_config(self):
        t = self.tmp
        dump(t / "scale_cb30wd.jsonl", [{"method": "dqn", "seed": s, "test_mean": -340.0 + (s - 1.5), "train_seconds": 1000.0} for s in range(4)])
        dump(t / "es_cb30wd.jsonl", [{"method": "es", "seed": s, "test_mean": -400.0 + (s - 1.5), "train_seconds": 40.0} for s in range(4)])
        b = P8.cec_es_budget(t, scales=("cb30wd",))["cb30wd"]
        self.assertAlmostEqual(b["dqn_secs"], 1000.0)
        self.assertAlmostEqual(b["budgets"]["500k"]["secs"], 40.0)
        dump(t / "tuned_es_cb30wd.json", [])
        (t / "tuned_es_cb30wd.json").write_text(json.dumps({"es": {"cfg_id": 8, "cfg": {"pop": 64, "sigma": 0.13, "lr": 0.14, "bs": 4}, "val_mean": -401.5}}), encoding="utf-8")
        c = P8.cec_es_config(t, scales=("cb30wd",))["cb30wd"]
        self.assertEqual((c["pop"], c["bs"], c["cfg_id"]), (64, 4, 8))
        self.assertAlmostEqual(c["val"], 401.5)

    def _write_hybrid(self, n_seeds=3, cfg="S", full=True):
        t = self.tmp
        rows = []
        for s in range(n_seeds):
            n_test = 300 if (full or s == 0) else 5                                        # semente incompleta não entra
            rows += [{"instance": "cb30wd", "config": cfg, "dqn_seed": s, "set": "test", "idx": k, "reward": -330.0 - s} for k in range(n_test)]
            rows += [{"instance": "cb30wd", "config": cfg, "dqn_seed": s, "set": "real_test", "idx": k, "reward": -320.0} for k in range(24)]
        dump(t / "rheav_cb30wd.jsonl", rows)
        dump(t / "agents" / "dqn_cb30wd.jsonl", [{"method": "dqn", "seed": s, "test_mean": -345.0 - s, "test_rewards": [-345.0 - s] * 1000, "matches_original": True} for s in range(n_seeds)])

    def test_hybrid_rows_table_and_pairing_with_the_same_dqn(self):
        self._write_hybrid(3)
        d = P8.cec_data(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertEqual(d["rhea_s_v"]["n"], 3)
        self.assertAlmostEqual(d["rhea_s_v"]["mean"], 331.0)                                  # sementes −330, −331, −332
        self.assertAlmostEqual(d["rhea_s_v"]["real"], 320.0)
        self.assertNotIn("rhea_l_v", d)                                                        # sem linhas L
        c = P8.cec_contrasts(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertAlmostEqual(c["rhea_s_v_vs_dqnsame"]["delta"], 15.0)                        # −330−s contra −345−s: +15 em todas as sementes (pareado: IC degenerado)
        self.assertAlmostEqual(c["rhea_s_v_vs_dqnsame"]["dqn"], 346.0)
        self.assertAlmostEqual(c["rhea_s_v_vs_rhea_s"]["delta"], -331.0 - (-310.0), delta=1e-6)  # contra o RHEA-S sem valor (−310): o híbrido sintético é pior em 21
        self.assertAlmostEqual(c["rhea_s_v_vs_rollout"]["delta"], -331.0 - (-300.0), delta=1e-6)

    def test_hybrid_incomplete_seed_is_dropped_and_agent_check(self):
        self._write_hybrid(3, full=False)
        self.assertEqual([x["seed"] for x in P8.cec_hybrid_rows(self.tmp, "cb30wd", "S")], [0])
        self.assertNotIn("rhea_s_v", P8.cec_data(self.tmp, scales=("cb30wd",), n=300)["cb30wd"])   # < 2 sementes completas
        self.assertEqual(P8.cec_agent_check(self.tmp, scales=("cb30wd",))["cb30wd"], {"n": 3, "n_match": 3, "n_orig": 3})

    def test_controls_rhea_vs_random_shooting_and_cma_vs_random_search(self):
        t = self.tmp
        inst = "cb30wd"
        rows = [{"instance": inst, "config": "S", "set": "test", "idx": k, "reward": -310.0} for k in range(300)]
        rows += [{"instance": inst, "config": "RS", "set": "test", "idx": k, "reward": -316.0 + (k % 3)} for k in range(300)]      # controle pior em ~5
        dump(t / f"rhea_{inst}.jsonl", rows)
        rs, cm = [], []
        for sd in range(4):
            for fam in ("greedy", "threshold"):
                cm.append({"instance": inst, "family": fam, "seed": sd, "val_best": -375.0 - (fam == "threshold"), "test_mean": -378.0 + 0.1 * sd})
                rs.append({"instance": inst, "family": fam, "seed": sd, "val_best": -380.0, "test_mean": -383.0 + 0.1 * sd})
        dump(t / f"cmaes_{inst}.jsonl", cm)
        dump(t / f"rsearch_{inst}.jsonl", rs)
        c = P8.cec_controls(t, scales=(inst,), n=300)[inst]
        self.assertAlmostEqual(c["rhea_vs_rs_s"]["delta"], 5.0, delta=1.5)                                   # RHEA melhor que o tiro aleatório em ~5
        self.assertGreater(c["rhea_vs_rs_s"]["lo"], 0)
        self.assertNotIn("rhea_vs_rs_l", c)                                                                  # sem linhas L/RL
        self.assertAlmostEqual(c["cma_vs_random"]["delta"], 5.0, places=6)
        self.assertEqual(c["cma_vs_random"]["verdict"], "better")
        self.assertEqual((c["cma_vs_random"]["n_cma"], c["cma_vs_random"]["n_random"]), (4, 4))

    def test_sepcma_rows_and_contrasts_against_es_and_dqn(self):
        t = self.tmp
        rng = np.random.default_rng(4)
        dump(t / "sepcma_cb30wd.jsonl", [{"method": "sepcma", "seed": s, "test_mean": -380.0, "test_rewards": [-380.0 + float(rng.normal(0, 1))] * 1000, "real_test_mean": -385.0} for s in range(4)])
        d = P8.cec_data(t, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertEqual(d["sepcma"]["n"], 4)
        self.assertNotIn("sepcma2m", d)
        c = P8.cec_contrasts(t, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertAlmostEqual(c["sepcma_vs_es"]["delta"], 20.0, delta=2.0)                  # sep-CMA −380 contra ES −400: +20 (positivo favorece o sep-CMA-ES)
        self.assertGreater(c["sepcma_vs_es"]["lo"], 0)
        self.assertLess(c["sepcma_vs_dqn"]["hi"], 0)                                          # DQN (−340) ainda melhor
        self.assertIn("sepcma_vs_rollout", c)

    def test_rhea_vs_rollout_paired_bootstrap(self):
        r = P8.cec_rhea_vs_rollout(self.tmp, scales=("cb30wd",), n=300)["cb30wd"]
        self.assertAlmostEqual(r["rollout"], 300.0)
        self.assertAlmostEqual(r["S"]["delta"], -10.0)                                                   # RHEA-S −310 contra rollout −300: pior em 10
        self.assertNotIn("L", r)                                                                         # L incompleto


if __name__ == "__main__":
    unittest.main()
