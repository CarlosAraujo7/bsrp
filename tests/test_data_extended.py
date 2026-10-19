"""Instâncias maiores (P9; bsrp/data.py: extended_base): as 12 legadas + as de maior volume, determinísticas, sem mexer nas escalas 4/8/12."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import data as D                          # noqa: E402
from bsrp import env as E                           # noqa: E402

FL = D.Flows()


class TestExtendedBase(unittest.TestCase):
    def test_extension_contains_the_legacy_12_in_order_and_adds_distinct_high_volume_stations(self):
        base12 = E.legacy_instance(12)
        for n in (20, 30):
            ext = D.extended_base(n, FL)
            self.assertEqual(ext.n, n)
            self.assertEqual(list(ext.station_names[:12]), list(base12.station_names))
            self.assertEqual(len(set(ext.station_names)), n)
            self.assertTrue((ext.cap >= 20).all())
            np.testing.assert_array_equal(ext.cap[:12], base12.cap)
            days = FL.days("train", 0)
            vol = FL.dep[:, days, :].sum(axis=(1, 2)) + FL.arr[:, days, :].sum(axis=(1, 2))
            v_new = [vol[r] for r in FL.rows_for(list(ext.station_names[12:]))]
            self.assertEqual(v_new, sorted(v_new, reverse=True))               # as novas entram em ordem de volume

    def test_30_extends_20_with_the_same_first_20(self):
        e20, e30 = D.extended_base(20, FL), D.extended_base(30, FL)
        self.assertEqual(list(e30.station_names[:20]), list(e20.station_names))

    def test_calibrated_instance_shapes_and_legacy_scales_are_unchanged(self):
        inst = D.calibrated_instance(20, FL, 0, name="cb20wd")
        self.assertEqual(inst.lam_dep.shape, (E.N_INTERVALS, 20))
        self.assertTrue((inst.lam_dep >= 0).all() and inst.lam_dep.sum() > 0)
        i12, i20 = D.calibrated_instance(12, FL, 0), inst
        np.testing.assert_allclose(i20.lam_dep[:, :12], i12.lam_dep)               # as 12 primeiras têm exatamente o mesmo λ
        self.assertEqual(D.calibrated_instance(4, FL, 0).n, 4)

    def test_real_scenarios_for_extended_instance(self):
        rs = D.real_scenarios(20, FL, "test", 0)
        self.assertEqual(rs.dep.shape[1:], (E.N_INTERVALS, 20))
        self.assertEqual(len(rs), 24)
        np.testing.assert_array_equal(rs.dep[:, :, :12], D.real_scenarios(12, FL, "test", 0).dep)


if __name__ == "__main__":
    unittest.main()
