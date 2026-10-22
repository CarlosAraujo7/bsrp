"""RHEA (bsrp/rhea.py): só ações válidas, reprodutível, usa só o modelo (não espia a demanda real) e melhora a parada numa instância pequena."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp import policies as P                    # noqa: E402
from bsrp.rhea import RHEA                        # noqa: E402

INST = E.legacy_instance(4, 8.0)                  # demanda alta: parar perde bastante
SC = E.sample_scenarios(INST, 6, np.random.default_rng(0))


def episode_actions(pol, sc):
    env = E.BSRPEnv(INST).reset(sc)
    acts = []
    while not env.done:
        a = pol.act(env)
        assert env.valid_mask()[a], "ação inválida"
        acts.append(a)
        env.step(a)
    return acts, env


class TestRHEA(unittest.TestCase):
    def test_only_valid_actions_and_full_episode(self):
        acts, env = episode_actions(RHEA(L=3, pop=6, gens=3, K=2, seed=1), SC[0])
        self.assertEqual(len(acts), E.N_INTERVALS)
        self.assertTrue(env.done)

    def test_same_seed_same_actions_different_seed_differs(self):
        a1, _ = episode_actions(RHEA(L=3, pop=6, gens=3, K=2, seed=4), SC[1])
        a2, _ = episode_actions(RHEA(L=3, pop=6, gens=3, K=2, seed=4), SC[1])
        a3, _ = episode_actions(RHEA(L=3, pop=6, gens=3, K=2, seed=5), SC[1])
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, a3)

    def test_does_not_peek_at_the_real_future(self):
        """Trocar a demanda real FUTURA (de t = 1 em diante) não muda a primeira ação: o plano só usa futuros do modelo."""
        pol1, pol2 = RHEA(L=3, pop=6, gens=3, K=3, seed=7), RHEA(L=3, pop=6, gens=3, K=3, seed=7)
        env1 = E.BSRPEnv(INST).reset(SC[2])
        sc2 = E.Scenario(SC[2].init_occ.copy(), SC[2].dep.copy(), SC[2].arr.copy())
        sc2.dep[1:] = 50
        sc2.arr[1:] = 0
        env2 = E.BSRPEnv(INST).reset(sc2)
        self.assertEqual(pol1.act(env1), pol2.act(env2))

    def test_shift_buffer_reuses_the_tail_of_the_previous_plan(self):
        pol = RHEA(L=3, pop=6, gens=2, K=2, seed=2)
        env = E.BSRPEnv(INST).reset(SC[3])
        a = pol.act(env)
        plan = pol._plan.copy()
        self.assertEqual(len(plan), 2)                                  # L − 1 genes restantes
        env.step(a)
        pol.act(env)                                                    # não deve reiniciar o plano no meio do episódio
        self.assertIsNotNone(pol._plan)

    def test_beats_staying_on_a_small_high_demand_instance(self):
        env = E.BSRPEnv(INST)
        stay = np.mean([E.run_episode(env, P.Stay(), SC[k])["reward"] for k in range(4)])
        rh = np.mean([E.run_episode(env, RHEA(L=3, pop=10, gens=5, K=4, seed=k), SC[k])["reward"] for k in range(4)])
        self.assertGreater(rh, stay)


if __name__ == "__main__":
    unittest.main()
