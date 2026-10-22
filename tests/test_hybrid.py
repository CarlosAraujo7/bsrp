"""RHEA com valor terminal aprendido (bsrp/hybrid.py, bsrp/rhea.py `value`, scripts/37_rheav.py): gancho neutro com valor zero, efeito de um valor forte, QValue em lote ×
rede do agente, ida e volta dos pesos e partição das fatias."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp import train as T                       # noqa: E402
from bsrp.hybrid import QValue, save_dqn          # noqa: E402
from bsrp.rhea import RHEA                        # noqa: E402

INST = E.legacy_instance(4, 8.0)
SC = E.sample_scenarios(INST, 4, np.random.default_rng(0))


def actions(pol, sc):
    env = E.BSRPEnv(INST).reset(sc)
    acts = []
    while not env.done:
        a = pol.act(env)
        acts.append(a)
        env.step(a)
    return acts


class TestRHEAValueHook(unittest.TestCase):
    def test_zero_value_is_neutral(self):
        a0 = actions(RHEA(L=3, pop=6, gens=3, K=2, seed=3), SC[0])
        a1 = actions(RHEA(L=3, pop=6, gens=3, K=2, seed=3, value=lambda envs: np.zeros(len(envs))), SC[0])
        self.assertEqual(a0, a1)

    def test_strong_value_changes_the_plan(self):
        base = actions(RHEA(L=3, pop=8, gens=4, K=2, seed=3), SC[1])
        end_at_station_0 = lambda envs: 1e4 * np.array([float(e.pos == 0) for e in envs])      # noqa: E731
        hyb = actions(RHEA(L=3, pop=8, gens=4, K=2, seed=3, value=end_at_station_0), SC[1])
        self.assertNotEqual(base, hyb)

    def test_value_receives_the_k_terminal_envs_of_each_genome(self):
        calls = []
        pol = RHEA(L=2, pop=4, gens=2, K=3, seed=1, value=lambda envs: (calls.append(len(envs)), np.zeros(len(envs)))[1])
        env = E.BSRPEnv(INST).reset(SC[2])
        pol.act(env)
        self.assertTrue(calls and all(n == 3 for n in calls))                   # um lote de K = 3 estados finais por genoma
        self.assertEqual(len(calls), 4 + (2 - 1) * (4 - 2))                     # P + (G − 1)(P − elite) avaliações


class TestQValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = T.make_agent("dqn", INST, seed=5, lr=1e-3, target_sync=100, train_every=1, batch=32, buffer=2000)
        sc = E.sample_scenarios(INST, 1, np.random.default_rng(1))[0]
        env = E.BSRPEnv(INST).reset(sc)
        rng = np.random.default_rng(2)
        for _ in range(200):                                                   # treina um pouco para os pesos não serem os iniciais
            if env.done:
                env.reset(E.sample_scenarios(INST, 1, rng)[0])
            s = env.features()
            a = cls.agent.select(env, 1.0, rng)
            r, d = env.step(a)
            cls.agent.observe(s, a, r, env.features(), d, env.valid_mask())

    def test_roundtrip_and_value_equals_masked_max(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.npz"
            save_dqn(self.agent, p)
            qv = QValue.from_npz(p)
        env = E.BSRPEnv(INST).reset(SC[3])
        for _ in range(10):
            q = self.agent.net.forward(env.features()[None])[0].copy()
            q[~env.valid_mask()] = -np.inf
            self.assertAlmostEqual(float(qv([env])[0]), float(q.max()), places=4)
            self.assertEqual(qv.act(env), self.agent.act(env))                  # mesma política gulosa do agente original
            env.step(qv.act(env))

    def test_batch_matches_single_and_done_is_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.npz"
            save_dqn(self.agent, p)
            qv = QValue.from_npz(p)
        e1, e2 = E.BSRPEnv(INST).reset(SC[0]), E.BSRPEnv(INST).reset(SC[1])
        e2.step(int(np.flatnonzero(e2.valid_mask())[3]))
        v = qv([e1, e2])
        self.assertAlmostEqual(float(v[0]), float(qv([e1])[0]), places=4)
        self.assertAlmostEqual(float(v[1]), float(qv([e2])[0]), places=4)
        while not e1.done:
            e1.step(qv.act(e1))
        self.assertEqual(float(qv([e1])[0]), 0.0)


class TestRheavScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rheav37", ROOT / "scripts" / "37_rheav.py")
        cls.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.M)

    def test_shards_partition_the_jobs(self):
        jobs = self.M.episode_jobs([0, 1], 5, 3)
        self.assertEqual(len(jobs), 2 * (5 + 3))
        parts = [self.M.my_jobs(jobs, i, 3) for i in range(3)]
        self.assertEqual(sorted(sum(parts, [])), sorted(jobs))
        self.assertEqual(self.M.my_jobs(jobs, 0, 3, done={jobs[0]}), [j for j in parts[0] if j != jobs[0]])

    def test_seed_depends_on_set_index_config_and_dqn_seed(self):
        s = self.M.rhea_seed
        self.assertEqual(s("test", 3, "S", 0), s("test", 3, "S", 0))
        self.assertEqual(len({s("test", 3, "S", 0), s("real_test", 3, "S", 0), s("test", 4, "S", 0), s("test", 3, "L", 0), s("test", 3, "S", 1)}), 5)


if __name__ == "__main__":
    unittest.main()
