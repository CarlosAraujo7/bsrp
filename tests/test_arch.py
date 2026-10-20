"""P10: largura/profundidade da rede do DQN via `hidden` (bsrp/train.py: make_agent); o padrão 64-64 e a reprodutibilidade não mudam."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                         # noqa: E402
from bsrp import train as T                       # noqa: E402

INST = E.legacy_instance(4)


def n_params(agent):
    return sum(w.size for w in agent.net.W) + sum(b.size for b in agent.net.b)


class TestHidden(unittest.TestCase):
    def test_default_is_64_64_and_list_from_json_is_accepted(self):
        self.assertEqual(T.make_agent("dqn", INST, 0).name, "dqn[64, 64]")
        self.assertEqual(T.make_agent("dqn", INST, 0, hidden=[64, 64]).name, "dqn[64, 64]")
        self.assertEqual(T.make_agent("dqn", INST, 0, hidden=[128, 128, 128]).name, "dqn[128, 128, 128]")

    def test_passing_the_default_explicitly_gives_the_same_network(self):
        a0, a1 = T.make_agent("dqn", INST, 3), T.make_agent("dqn", INST, 3, hidden=(64, 64))
        x = np.random.default_rng(0).normal(size=(5, a0.n_in)).astype(np.float32)
        np.testing.assert_array_equal(a0.net.forward(x), a1.net.forward(x))

    def test_wider_network_has_more_parameters_and_the_shapes_follow_hidden(self):
        small, big = T.make_agent("dqn", INST, 0, hidden=(16, 16)), T.make_agent("dqn", INST, 0, hidden=(128, 128))
        self.assertGreater(n_params(big), 10 * n_params(small))
        self.assertEqual([w.shape for w in big.net.W], [(big.n_in, 128), (128, 128), (128, big.n_act)])

    def test_other_methods_are_unaffected(self):
        self.assertEqual(T.make_agent("linear", INST, 0).name, "linear")


if __name__ == "__main__":
    unittest.main()
