"""
Portão G0 (PROTOCOLO_RIGOR.md §7): o pacote `bsrp` reproduz o simulador legado e os resultados salvos.

  python -m unittest tests.test_regression_legacy -v
"""
import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import env as E                                    # noqa: E402
from bsrp.agents import LegacyDQNPolicy                      # noqa: E402
from bsrp.policies import LegacyGreedy, LegacyThreshold, RandomPolicy, Stay   # noqa: E402


def _load(name, rel):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / rel))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


SIM = _load("legacy_sim", "scripts/04_simulator.py")
BASE = _load("legacy_base", "scripts/10_baselines.py")


def legacy_env(n, seed):
    st = SIM.load_environment_data(profiles_path=str(ROOT / f"demand_profiles_{n}stations_normalized.csv"),
                                   stations_path=str(ROOT / f"station_selected_{n}.csv"))
    return SIM.BSRPEnv(st, rng=np.random.default_rng(seed))


def legacy_rewards(n, seed, policy, episodes):
    env = legacy_env(n, seed)
    out = []
    for _ in range(episodes):
        out.append(SIM.run_episode(env, policy)["total_reward"])
    return np.array(out)


def new_rewards(n, seed, policy, episodes):
    inst = E.legacy_instance(n)
    sc = E.legacy_scenarios(inst, seed, episodes)
    env = E.BSRPEnv(inst)
    return E.evaluate(env, policy, sc)


class TestScenariosAndInstance(unittest.TestCase):
    def test_instance_matches_legacy_tables(self):
        for n in (4, 8, 12):
            inst = E.legacy_instance(n)
            env = legacy_env(n, 1)
            self.assertEqual(inst.n, len(env.stations))
            np.testing.assert_allclose(inst.lam_dep[:, 0], env.stations[0].dep_avg * SIM.DEMAND_SCALE)
            np.testing.assert_array_equal(inst.travel_int(),
                                          np.where(np.eye(n) == 1, 0, np.maximum(1, np.ceil(env.travel_T))).astype(int))

    def test_crn_same_demand_for_every_policy(self):
        inst = E.legacy_instance(4)
        sc = E.sample_scenarios(inst, 5, np.random.default_rng(0))
        env = E.BSRPEnv(inst)
        for k in range(5):
            a = E.run_episode(env, Stay(), sc[k]); da, aa = env.total_dep, env.total_arr
            b = E.run_episode(env, LegacyGreedy(), sc[k]); db, ab = env.total_dep, env.total_arr
            self.assertEqual((da, aa), (db, ab))

    def test_digest_is_stable(self):
        inst = E.legacy_instance(4)
        a = E.sample_scenarios(inst, 10, np.random.default_rng(7)).digest()
        b = E.sample_scenarios(inst, 10, np.random.default_rng(7)).digest()
        c = E.sample_scenarios(inst, 10, np.random.default_rng(8)).digest()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestInvariants(unittest.TestCase):
    def test_invariants_and_unmet_accounting(self):
        inst = E.legacy_instance(12)
        env = E.BSRPEnv(inst)
        sc = E.sample_scenarios(inst, 8, np.random.default_rng(1))
        pol = RandomPolicy(np.random.default_rng(2))
        for k in range(8):
            env.reset(sc[k]); tot = 0.0
            while not env.done:
                r, _ = env.step(pol.act(env)); tot += r
                self.assertTrue((env.occ >= 0).all() and (env.occ <= env.cap).all())
                self.assertTrue(0 <= env.load <= E.VEHICLE_CAP)
                self.assertLessEqual(r, 0)
            self.assertEqual(-tot, env.unmet())          # o campo unmet do legado era sempre 0


class TestAgainstLegacySimulator(unittest.TestCase):
    """Episódio a episódio: mesmo cenário (sequência de aleatórios do legado) => mesmo reward."""

    def test_greedy_exact(self):
        for n in (4, 8, 12):
            np.testing.assert_array_equal(legacy_rewards(n, 123, SIM.GreedyPolicy(), 12),
                                          new_rewards(n, 123, LegacyGreedy(), 12), err_msg=f"n={n}")

    def test_threshold_exact(self):
        for n in (4, 8, 12):
            np.testing.assert_array_equal(legacy_rewards(n, 456, BASE.ThresholdPolicy(0.5), 12),
                                          new_rewards(n, 456, LegacyThreshold(0.5), 12), err_msg=f"n={n}")


class TestAgainstStoredResults(unittest.TestCase):
    """Pesos DQN legados avaliados no pacote novo reproduzem os JSONs salvos (100 ep por semente)."""

    def _check(self, n, npz, js):
        d = json.load(open(ROOT / "results" / "scaling" / js, encoding="utf-8"))["multiseed"]
        pol = LegacyDQNPolicy.from_npz(ROOT / "results" / "scaling" / npz)
        for seed, rec in d.items():
            got = new_rewards(n, int(seed), pol, 100)
            self.assertAlmostEqual(got.mean(), rec["reward"], places=2, msg=f"{npz} seed {seed}")
            self.assertAlmostEqual(got.std(), rec["std"], places=1, msg=f"{npz} seed {seed} std")

    def test_dqn_4st(self):
        self._check(4, "dqn_4st_ep5000.npz", "dqn_4st.json")

    def test_dqn_8st_ep13500(self):
        self._check(8, "dqn_8st_ep13500.npz", "dqn_8st_ep13500.json")

    def test_threshold_4st_stored(self):
        d = json.load(open(ROOT / "results" / "baselines" / "baselines_4st.json", encoding="utf-8"))["threshold"]["multiseed"]
        for seed, rec in d.items():
            got = new_rewards(4, int(seed), LegacyThreshold(0.5), 100)
            self.assertAlmostEqual(got.mean(), rec["reward"], places=2)


if __name__ == "__main__":
    unittest.main()
