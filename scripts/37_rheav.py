"""
37_rheav.py — RHEA com VALOR TERMINAL aprendido (RHEA-V; bsrp/rhea.py com `value=QValue`) nos primeiros cenários de teste e nos dias reais (DESVIOS.md 2026-10-22, P13).

  python scripts/37_rheav.py --instance cb12wd --config S --dqn-seeds 0 --n-test 300 --shard 0/1

Usa os pesos do DQN congelado de runs/agents/dqn_<inst>_s<seed>.npz (scripts/36_train_dqn_agents.py; espera o arquivo aparecer, no máximo `--wait-min` minutos). Configurações FIXAS e sem ajuste,
as mesmas do P12c (S: L = 4, P = 16, G = 10, K = 8; L: L = 4, P = 24, G = 15, K = 16). Um episódio por linha em runs/rheav_<inst>.jsonl (retomável; fatias em paralelo); a semente do
RHEA de cada episódio deriva de (conjunto, índice, configuração, semente do DQN), então o resultado não depende do fatiamento.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                                                    # noqa: E402
from bsrp.hybrid import QValue                                               # noqa: E402
from bsrp.rhea import RHEA                                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

CONFIGS = {"S": dict(L=4, pop=16, gens=10, K=8), "L": dict(L=4, pop=24, gens=15, K=16)}
CODE = {"S": 0, "L": 1}
SALT = 20261023
SETS = {"test": 2, "real_test": 3}


def episode_jobs(dqn_seeds, n_test: int, n_real: int):
    return [(s, w, k) for s in dqn_seeds for w, n in (("test", n_test), ("real_test", n_real)) for k in range(n)]


def my_jobs(jobs, shard_i: int, shard_n: int, done=frozenset()):
    """Fatia i de N (índice mod N) dos jobs ainda não feitos; atribuição fixa, independente de quais já foram feitos."""
    return [j for i, j in enumerate(jobs) if i % shard_n == shard_i and j not in done]


def done_set(path, tag):
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["dqn_seed"], r["set"], r["idx"]) for r in map(json.loads, (l for l in f if l.strip())) if r["config"] == tag}


def rhea_seed(which: str, k: int, config: str, dqn_seed: int) -> int:
    return int(np.random.SeedSequence([SALT, SETS[which], k, CODE[config], dqn_seed]).generate_state(1)[0])


def wait_for(path: Path, minutes: float):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > 60 * minutes:
            raise TimeoutError(f"{path} não apareceu em {minutes} min")
        time.sleep(20)
    time.sleep(2)                                                           # o arquivo pode estar sendo fechado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--dqn-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--wait-min", type=float, default=240.0)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    env = E.BSRPEnv(inst)
    out = ROOT / "runs" / f"rheav_{inst.name}.jsonl"
    git, dirty = RP.git_state()
    cfg = CONFIGS[a.config]
    jobs = my_jobs(episode_jobs(a.dqn_seeds, a.n_test, len(sc["real_test"])), si, sn, done_set(out, a.config))
    print(f"[{inst.name}] RHEA-V {a.config} {cfg}; DQN {a.dqn_seeds}; fatia {si}/{sn}: {len(jobs)} episódios", flush=True)
    qv, qv_seed = None, None
    for s, which, k in jobs:
        if s != qv_seed:
            p = ROOT / "runs" / "agents" / f"dqn_{inst.name}_s{s}.npz"
            wait_for(p, a.wait_min)
            qv, qv_seed = QValue.from_npz(p), s
        pol = RHEA(seed=rhea_seed(which, k, a.config, s), value=qv, **cfg)
        t0 = time.time()
        r = E.run_episode(env, pol, sc[which][k])["reward"]
        RP.log(out, {"instance": inst.name, "config": a.config, **cfg, "dqn_seed": s, "set": which, "idx": k, "reward": float(r), "seconds": round(time.time() - t0, 1),
                     "commit": git, "dirty": dirty, "digest": sc[which].digest()})
    print(f"[{inst.name}] FIM fatia {si}/{sn}", flush=True)


if __name__ == "__main__":
    main()
