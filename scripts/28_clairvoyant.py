"""
28_clairvoyant.py — valor do planejador CLARIVIDENTE (bsrp/oracle.py) nos primeiros cenários de teste e nos dias reais (DESVIOS.md 2026-10-18, P8a).

  python scripts/28_clairvoyant.py --instance cb12wd --widths 300 1000 --n-test 200 --shard 0/3

Grava uma linha por (conjunto, índice, largura) em runs/oracle_<inst>_w<W>.jsonl (retomável; várias fatias podem rodar em paralelo: as linhas são curtas).
O valor é o reward (≤ 0) da melhor sequência de ações encontrada conhecendo a demanda do cenário: limite INFERIOR do ótimo clarividente.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import oracle as O                                                 # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)


def done_keys(path):
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["set"], r["idx"]) for r in map(json.loads, (l for l in f if l.strip()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=[300, 1000])
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    sets = {"test": sc["test"], "real": sc.get("real_test")}
    for W in a.widths:
        out = ROOT / "runs" / f"oracle_{inst.name}_w{W}.jsonl"
        done = done_keys(out)
        jobs = [(name, k) for name, S in sets.items() if S is not None for k in range(a.n_test if name == "test" else len(S))]
        mine = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
        print(f"[{inst.name}] W={W} fatia {si}/{sn}: {len(mine)} execuções", flush=True)
        for name, k in mine:
            t0 = time.time()
            v = O.plan_value(inst, sets[name][k], W)
            RP.log(out, {"instance": inst.name, "set": name, "idx": k, "width": W, "value": v, "seconds": round(time.time() - t0, 2),
                         "digest": (sc["test"] if name == "test" else sets[name]).digest()})
        print(f"[{inst.name}] W={W} FIM fatia {si}/{sn}", flush=True)


if __name__ == "__main__":
    main()
