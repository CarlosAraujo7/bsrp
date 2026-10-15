"""bsrp/tembici.py com dados sintéticos de resposta conhecida (DESVIOS.md 2026-10-15, P5)."""
import csv
import gzip
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp import tembici as TB                               # noqa: E402

DAYS = ("2026-10-13", "2026-10-14", "2026-10-15")           # ter, qua, qui
# estação: (amplitude, lat, lon, capacidade)
STATIONS = {"S1": (2, -22.900, -43.200, 20), "S2": (1, -22.902, -43.202, 16), "S3": (1, -22.905, -43.204, 14),
            "S4": (1, -22.600, -43.200, 18)}               # S4 a ~33 km das demais


def local_midnight_utc(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=TB.LOCAL).timestamp())


def build_root(tmp: Path, days=DAYS, jump_day=None, gap_day=None, short_day=None, city="rio"):
    """Cada estação: a cada meia hora operacional (06–22h) os 6 incrementos por coleta são
    [+a,+a,+a,−a,0,0] (intervalo par: chegadas 3a, partidas a) ou [−a,−a,−a,+a,0,0] (ímpar: chegadas a, partidas 3a)."""
    rows = []
    for day in days:
        t0 = local_midnight_utc(day)
        for sid, (a, _, _, cap) in STATIONS.items():
            veh = 8
            for p in range(288):
                k, j = divmod(p, 6)
                if gap_day == day and 100 <= p <= 103 and sid == "S1":
                    continue
                if short_day == day and p >= 150:               # dia truncado: só coletas até 12:30 local
                    continue
                delta = 0
                if 12 <= k < 44:
                    pat = [1, 1, 1, -1, 0, 0] if k % 2 == 0 else [-1, -1, -1, 1, 0, 0]
                    delta = a * pat[j]
                if jump_day == day and sid == "S1":
                    delta += {76: 6, 244: -6}.get(p, 0)           # reposição operacional (+6 depois −6) em coletas sem incremento de usuário
                if p > 0 or True:
                    veh += delta
                ts = t0 + 300 * p
                rows.append((ts, sid, veh, cap))
    d = tmp / "data" / "raw" / "tembici" / city
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "dump.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "feed_ts", "station_id", "veh", "veh_dis", "docks", "docks_dis", "inst", "rent", "ret", "last_rep", "types"])
        for ts, sid, veh, cap in rows:
            w.writerow([ts, ts, sid, veh, 0, cap - veh, 0, 1, 1, 1, ts, "FIT"])
    si = tmp / "data" / "raw" / "tembici" / "station_information" / city
    si.mkdir(parents=True, exist_ok=True)
    stations = [{"station_id": sid, "name": [{"text": f"Estação {sid}", "language": "pt"}], "lat": lat, "lon": lon, "capacity": cap}
                for sid, (_, lat, lon, cap) in STATIONS.items()]
    with gzip.open(si / "20261015T000000Z.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"data": {"stations": stations}}, f)
    return tmp


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def flows(self, **kw):
        build_root(self.tmp, **kw)
        return TB.compute_flows(TB.read_snapshots("rio", self.tmp)), TB.read_station_info("rio", self.tmp)


class TestFlows(Base):
    def test_counts_match_pattern_in_local_time(self):
        cf, _ = self.flows()
        s1, s2 = cf.station_ids.index("S1"), cf.station_ids.index("S2")
        for d in range(len(DAYS)):
            for k in range(TB.OFFSET, TB.OFFSET + TB.N_INTERVALS):
                even = (k % 2 == 0)
                self.assertEqual(cf.arr[s1, d, k], 6 if even else 2, (d, k))
                self.assertEqual(cf.dep[s1, d, k], 2 if even else 6, (d, k))
                self.assertEqual(cf.arr[s2, d, k], 3 if even else 1)
            self.assertEqual(float(cf.arr[s1, d, :TB.OFFSET].sum() + cf.arr[s1, d, TB.OFFSET + TB.N_INTERVALS:].sum()), 0.0)
        self.assertTrue(np.all(cf.cover[s1, :, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS] >= TB.MIN_COVER))

    def test_operational_jumps_are_excluded_and_counted(self):
        cf, _ = self.flows(jump_day=DAYS[1])
        s1, d = cf.station_ids.index("S1"), 1
        self.assertEqual(int(cf.ops[s1, d].sum()), 2)
        self.assertEqual(int(cf.ops[s1, 0].sum()), 0)
        k = 12                                                  # intervalo do salto +6 (coleta p=76 -> intervalo 12)
        self.assertEqual(cf.arr[s1, d, k], 6)                   # inalterado: o +6 não virou demanda de usuário

    def test_jump_threshold_is_a_parameter_and_excluded_volume_is_reported(self):
        build_root(self.tmp, jump_day=DAYS[1])
        snap = TB.read_snapshots("rio", self.tmp)
        a, b = TB.compute_flows(snap), TB.compute_flows(snap, big_jump=8)
        s1 = a.station_ids.index("S1")
        self.assertEqual((int(a.ops[s1, 1].sum()), float(a.ops_vol[s1, 1].sum())), (2, 12.0))     # +6 e −6 excluídos
        self.assertEqual(int(b.ops[s1, 1].sum()), 0)                                              # |Δ| = 6 < 8: vira demanda
        self.assertEqual(float(b.arr[s1, 1, 12]), 6 + 6)                                          # padrão 6 + salto +6
        self.assertEqual(float(b.dep[s1, 1, 40]), 2 + 6)                                          # padrão 2 + salto −6

    def test_gap_invalidates_only_affected_intervals(self):
        cf, _ = self.flows(gap_day=DAYS[2])
        s1, d = cf.station_ids.index("S1"), 2
        bad = np.flatnonzero(cf.cover[s1, d] < TB.MIN_COVER)
        self.assertTrue(set([16, 17]).issubset(set(bad.tolist())))
        self.assertTrue(set(bad.tolist()).issubset(set(range(14, 20))))

    def test_stock_at_6h_and_usable_days(self):
        cf, _ = self.flows(short_day=DAYS[1])
        s1 = cf.station_ids.index("S1")
        self.assertEqual(float(cf.stock06[s1, 0]), 10.0)           # 8 + o primeiro incremento (+2) da própria coleta das 06:00
        us = TB.usable_days(cf)
        self.assertEqual(us.tolist(), [True, False, True])

    def test_dates_and_day_type(self):
        cf, _ = self.flows()
        self.assertEqual(cf.dates, list(DAYS))
        self.assertEqual(TB.day_type("2026-10-14"), 0)           # quarta
        self.assertEqual(TB.day_type("2026-10-17"), 1)           # sábado
        self.assertEqual(TB.day_type("2026-10-12"), 1)           # feriado (segunda)
        self.assertEqual(TB.day_type("2026-11-02"), 1)


class TestSplit(unittest.TestCase):
    @staticmethod
    def span(first, n):
        d0 = date.fromisoformat(first)
        return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]

    def test_no_holdout_before_enough_weeks(self):
        dates = self.span("2026-10-11", 12)                       # dom 11/10 .. qui 22/10: só 1 semana completa
        us = np.array([d != "2026-10-11" for d in dates])
        out = TB.assign_split(dates, us)
        self.assertFalse(set(out) & {"val", "test", "train"})
        self.assertEqual(int((out == "dev").sum()), 11)
        self.assertEqual(out[0], "unused")

    def test_last_complete_week_is_test_previous_is_val(self):
        dates = self.span("2026-10-11", 32)                       # 11/10 .. 11/11: semanas 12–18, 19–25, 26/10–1/11, 2–8/11 completas
        us = np.ones(len(dates), bool)
        us[0] = False                                             # dom 11/10 parcial
        out = TB.assign_split(dates, us)
        by = {d: s for d, s in zip(dates, out)}
        self.assertEqual({by[d] for d in self.span("2026-11-02", 7)}, {"test"})
        self.assertEqual({by[d] for d in self.span("2026-10-26", 7)}, {"val"})
        self.assertEqual({by[d] for d in self.span("2026-10-12", 14)}, {"train"})
        self.assertEqual({by[d] for d in self.span("2026-11-09", 3)}, {"unused"})      # depois do teste

    def test_week_with_an_unusable_day_is_not_complete(self):
        dates = self.span("2026-10-12", 28)                       # 4 semanas completas
        us = np.ones(len(dates), bool)
        us[dates.index("2026-10-30")] = False                     # quebra a semana 26/10–1/11
        out = TB.assign_split(dates, us)
        self.assertEqual(int(sum(1 for s in out if s in ("val", "test", "train"))), 0)  # sobram 3 completas -> sem hold-out


class TestInstance(Base):
    def test_selection_by_volume_within_radius_and_lambda(self):
        cf, info = self.flows()
        days = np.array([0, 1, 2])
        vol, frac = TB.station_volume(cf, days)
        s = {sid: i for i, sid in enumerate(cf.station_ids)}
        self.assertGreater(vol[s["S1"]], vol[s["S2"]])
        self.assertAlmostEqual(vol[s["S2"]], 32 * 4.0)           # chegadas+partidas por dia = 32 × (3+1)
        idx = TB.select_stations(cf, info, days, n=3, radius_km=3.0)
        self.assertEqual([cf.station_ids[i] for i in idx][0], "S1")
        self.assertNotIn(s["S4"], idx)                            # fora do raio, apesar de volume igual ao de S2
        inst = TB.build_instance(cf, info, idx, days, "tmb_test", n_clusters=2)
        self.assertEqual(inst.lam_dep.shape, (32, len(idx)))
        j = [cf.station_ids[i] for i in idx].index("S1")
        self.assertAlmostEqual(float(inst.lam_arr[0, j]), 6.0)    # intervalo 0 do episódio = 06h = k 12 (par)
        self.assertAlmostEqual(float(inst.lam_dep[1, j]), 6.0)    # intervalo ímpar: partidas 3a
        self.assertEqual(int(inst.cap[j]), 20)
        self.assertEqual(inst.station_names[j], "Estação S1")
        self.assertEqual(inst.cluster.shape, (len(idx),))

    def test_real_scenarios_use_observed_flows_and_drop_bad_days(self):
        cf, info = self.flows(gap_day=DAYS[2])
        days = np.array([0, 1, 2])
        idx = TB.select_stations(cf, info, np.array([0, 1]), n=3, radius_km=3.0)
        inst = TB.build_instance(cf, info, idx, np.array([0, 1]), "tmb_test", n_clusters=2)
        rs = TB.real_scenarios(cf, inst, idx, days, max_invalid=0.01)
        j = [cf.station_ids[i] for i in idx].index("S1")
        self.assertEqual(rs.meta["days"], [DAYS[0], DAYS[1]])
        self.assertEqual(rs.meta["dropped"], [DAYS[2]])           # 2 de 96 células inválidas (2,1%) > 1%
        self.assertEqual(int(rs.dep[0, 1, j]), 6)                 # intervalo 1 (ímpar): partidas 3a = 6
        self.assertEqual(int(rs.init_occ[0, j]), 10)              # estoque observado às 06h (8 + o 1º incremento da coleta das 06:00)
        rs2 = TB.real_scenarios(cf, inst, idx, days, max_invalid=0.03)
        self.assertEqual(len(rs2), 3)                             # com tolerância maior o dia entra, com λ nas células inválidas


class TestKMeans(unittest.TestCase):
    def test_deterministic_and_separates_groups(self):
        lat = np.array([0.0, 0.001, 0.002, 0.30, 0.301, 0.302])
        lon = np.zeros(6)
        a, b = TB.kmeans_labels(lat, lon, 2), TB.kmeans_labels(lat, lon, 2)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a[:3])), 1)
        self.assertEqual(len(set(a[3:])), 1)
        self.assertNotEqual(a[0], a[3])


class TestG3(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 4
        k = np.arange(32)[:, None]
        base = 2.0 + 1.5 * np.sin(k / 5.0) + np.arange(n)[None] * 0.5
        self.inst = E.Instance("g3", np.full(n, 12), np.arange(n), np.zeros(n), np.zeros(n), np.maximum(base - 0.4 * np.cos(k / 3.0), 0.1),
                               np.maximum(base + 0.4 * np.cos(k / 3.0), 0.1), tuple("abcd"))
        self.good = E.sample_scenarios(self.inst, 40, rng)

    def test_perfect_model_passes_all_three_tolerances(self):
        m = TB.g3_metrics(self.inst, self.good)
        self.assertTrue(m["G3_passed"], m)
        self.assertGreater(m["corr_net"], 0.8)
        self.assertLess(m["volume_rel_err"], 0.1)
        self.assertTrue(0.5 <= m["unmet_ratio"] <= 2.0)

    def test_mismatched_model_fails(self):
        wrong = E.Instance("w", self.inst.cap, self.inst.cluster, self.inst.lat, self.inst.lon, self.inst.lam_dep * 3.0,
                           self.inst.lam_arr * 3.0, self.inst.station_names)
        m = TB.g3_metrics(wrong, self.good)
        self.assertFalse(m["G3_passed"])
        self.assertFalse(m["pass"]["volume"])


if __name__ == "__main__":
    unittest.main()
