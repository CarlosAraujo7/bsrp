"""Variantes da ablação do DQN (scripts/20_ablation.py; DESVIOS.md 2026-10-14): as flags fazem o que o texto afirma."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp import train as T                                  # noqa: E402

_spec = importlib.util.spec_from_file_location("ablation", ROOT / "scripts" / "20_ablation.py")
ABL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ABL)

INST = E.legacy_instance(4)
VAL = E.sample_scenarios(INST, 8, np.random.default_rng(1))
TEST = E.sample_scenarios(INST, 12, np.random.default_rng(2))
BASE = dict(buffer=2000, batch=32, train_every=2, target_sync=200)


def run(flags, seed=3, steps=2000):
    ag = T.make_agent("dqn", INST, seed, **BASE, **flags)
    T.train(INST, ag, seed, steps, VAL, steps)
    return ag, T.final_eval(INST, ag, TEST)["test_rewards"]


def weights(ag):
    return [w.copy() for w in ag.net.W]


class TestAblationVariants(unittest.TestCase):
    def test_variants_are_accepted_by_make_agent(self):
        for name, flags in ABL.VARIANTS.items():
            T.make_agent("dqn", INST, 0, **BASE, **flags)

    def test_noreplay_never_fills_buffer_but_learns(self):
        ag, _ = run(ABL.VARIANTS["noreplay"])
        fresh = T.make_agent("dqn", INST, 3, **BASE, **ABL.VARIANTS["noreplay"])
        self.assertEqual(ag.buf.n, 0)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(weights(ag), weights(fresh))))

    def test_replay_fills_buffer(self):
        ag, _ = run({})
        self.assertEqual(ag.buf.n, BASE["buffer"])

    def test_notarget_target_net_never_synced(self):
        ag, _ = run(ABL.VARIANTS["notarget"])
        fresh = T.make_agent("dqn", INST, 3, **BASE, **ABL.VARIANTS["notarget"])
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(ag.tgt.W, fresh.tgt.W)))

    def test_double_is_noop_without_target(self):
        """DESVIOS (i): sem rede-alvo, double=True e double=False treinam exatamente igual."""
        (a1, r1), (a2, r2) = run({"use_target": False, "double": True}), run({"use_target": False, "double": False})
        for x, y in zip(weights(a1), weights(a2)):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(r1, r2)

    def test_each_component_changes_the_result(self):
        _, full = run({})
        for name in ("noreplay", "notarget", "nodouble", "vanilla"):
            _, r = run(ABL.VARIANTS[name])
            self.assertFalse(np.array_equal(full, r), name)

    def test_variant_reproducible_per_seed(self):
        (_, r1), (_, r2) = run(ABL.VARIANTS["vanilla"]), run(ABL.VARIANTS["vanilla"])
        np.testing.assert_array_equal(r1, r2)


if __name__ == "__main__":
    unittest.main()
