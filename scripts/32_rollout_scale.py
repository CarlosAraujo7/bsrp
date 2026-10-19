"""
32_rollout_scale.py — rollout (h, K fixos) nas escalas maiores (P9c; DESVIOS.md 2026-10-19).

  python scripts/32_rollout_scale.py --instance cb20wd --n-test 300 --h 3 --K 64 --shard 0/8

Base do rollout = a heurística calibrada de maior reward em VALIDAÇÃO (runs/heuristics_scale_<inst>.jsonl); futuros amostrados do MODELO de Poisson. Cada
episódio é um job (conjunto, índice) com semente derivada de (conjunto, índice, h, K), como em 23_rollout.py, então o resultado não depende do fatiamento.
Uma linha por episódio em runs/rollout_scale_<inst>.jsonl (retomável; várias fatias em paralelo: as linhas são curtas).
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
from bsrp import policies as P                                               # noqa: E402

_s = importlib.util.spec_from_file_location("rollout23", ROOT / "scripts" / "23_rollout.py")
RO = importlib.util.module_from_spec(_s)
_s.loader.exec_module(RO)
RP = RO.RP


def episode_jobs(n_test: int, n_real: int):
    return [("test", k) for k in range(n_test)] + [("real_test", k) for k in range(n_real)]


def my_jobs(jobs, shard_i: int, shard_n: int, done=frozenset()):
    """Fatia i de N (índice mod N) dos jobs ainda não feitos; atribuição fixa, independente de quais já foram feitos."""
    return [j for i, j in enumerate(jobs) if i % shard_n == shard_i and j not in done]


def done_set(path):
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["set"], r["idx"]) for r in map(json.loads, (l for l in f if l.strip()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    env = E.BSRPEnv(inst)
    hrows = [json.loads(l) for l in open(ROOT / "runs" / f"heuristics_scale_{inst.name}.jsonl", encoding="utf-8") if l.strip()]
    fam, ref = RO.best_reference(hrows, inst.name)
    base = RO.parse_policy(ref["policy"])
    ro = P.Rollout(base, h=a.h, K=a.K, seed=0)
    out = ROOT / "runs" / f"rollout_scale_{inst.name}.jsonl"
    git, dirty = RP.git_state()
    jobs = my_jobs(episode_jobs(a.n_test, len(sc["real_test"])), si, sn, done_set(out))
    print(f"[{inst.name}] rollout h={a.h} K={a.K}, base {fam} = {ref['policy']}; fatia {si}/{sn}: {len(jobs)} episódios", flush=True)
    for which, k in jobs:
        ro.rng = np.random.default_rng(np.random.SeedSequence([RO.SALT, RO.SETS[which], k, a.h, a.K]))
        t0 = time.time()
        r = E.run_episode(env, ro, sc[which][k])["reward"]
        RP.log(out, {"instance": inst.name, "set": which, "idx": k, "h": a.h, "K": a.K, "base_family": fam, "base_policy": ref["policy"],
                     "reward": float(r), "seconds": round(time.time() - t0, 1), "commit": git, "dirty": dirty,
                     "digest": sc[which].digest()})
    print(f"[{inst.name}] FIM fatia {si}/{sn}", flush=True)


if __name__ == "__main__":
    main()
