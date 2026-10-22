"""
34_rhea.py — RHEA (planejador evolutivo online; bsrp/rhea.py) nos primeiros cenários de teste e nos dias reais (DESVIOS.md 2026-10-22, P12c).

  python scripts/34_rhea.py --instance cb12wd --config S --n-test 300 --shard 0/8

Duas configurações FIXAS e sem ajuste: S (L = 4, P = 16, G = 10, K = 8; custo por decisão ≈ o do rollout (3, 64)) e L (L = 4, P = 24, G = 15, K = 16; ≈ 4,5× o custo de S).
Cada episódio é um job (conjunto, índice) com semente derivada de (conjunto, índice, config), então o resultado não depende do fatiamento. Uma linha por episódio em
runs/rhea_<inst>.jsonl (retomável; fatias em paralelo).
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
from bsrp.rhea import RHEA                                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

CONFIGS = {"S": dict(L=4, pop=16, gens=10, K=8), "L": dict(L=4, pop=24, gens=15, K=16)}
SALT = 20261022
SETS = {"test": 2, "real_test": 3}


def episode_jobs(n_test: int, n_real: int):
    return [("test", k) for k in range(n_test)] + [("real_test", k) for k in range(n_real)]


def my_jobs(jobs, shard_i: int, shard_n: int, done=frozenset()):
    return [j for i, j in enumerate(jobs) if i % shard_n == shard_i and j not in done]


def done_set(path, tag):
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["set"], r["idx"]) for r in map(json.loads, (l for l in f if l.strip())) if r["config"] == tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    env = E.BSRPEnv(inst)
    out = ROOT / "runs" / f"rhea_{inst.name}.jsonl"
    git, dirty = RP.git_state()
    cfg = CONFIGS[a.config]
    jobs = my_jobs(episode_jobs(a.n_test, len(sc["real_test"])), si, sn, done_set(out, a.config))
    print(f"[{inst.name}] RHEA {a.config} {cfg}; fatia {si}/{sn}: {len(jobs)} episódios", flush=True)
    for which, k in jobs:
        pol = RHEA(seed=int(np.random.SeedSequence([SALT, SETS[which], k, sum(map(ord, a.config))]).generate_state(1)[0]), **cfg)
        t0 = time.time()
        r = E.run_episode(env, pol, sc[which][k])["reward"]
        RP.log(out, {"instance": inst.name, "config": a.config, **cfg, "set": which, "idx": k, "reward": float(r), "seconds": round(time.time() - t0, 1),
                     "commit": git, "dirty": dirty, "digest": sc[which].digest()})
    print(f"[{inst.name}] FIM fatia {si}/{sn}", flush=True)


if __name__ == "__main__":
    main()
