"""
23_rollout.py — rollout sobre a melhor heurística calibrada (P4a; especificado em DESVIOS.md 2026-10-14).

Para cada instância: base = a referência de heurística de maior reward em VALIDAÇÃO entre greedy_cal / greedy_cal2 / greedy_cal3 /
threshold_cal / threshold_cal2; Rollout(base, h, K) com h ∈ {1, 2}, K ∈ {4, 16}; futuros amostrados do MODELO de Poisson (nunca da demanda real).
A semente do rollout deriva de (conjunto, índice do episódio, h, K): o resultado não depende da ordem dos episódios. Avalia em
val (200), teste (1000) e dias reais; uma linha por (instância, h, K) em runs/rollout_<out>.jsonl; retomável.

  python scripts/23_rollout.py --instances cb4wd cb8wd cb12wd --out runs/rollout_cb_wd.jsonl
  python scripts/23_rollout.py --instances cb12wd --out runs/rollout_cb12wd.jsonl --grid 3:16 2:64 3:64
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                                                    # noqa: E402
from bsrp import policies as P                                               # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

SALT = 20261014
SETS = {"val": 1, "test": 2, "real_test": 3}
HEUR_FILES = ("heuristics_cb_wd.jsonl", "heuristics_cb_wd_g3.jsonl", "heuristics_cb_wd_threshold.jsonl", "heuristics_cb_wd_threshold2.jsonl")
FAMILIES = ("greedy_cal", "greedy_cal2", "greedy_cal3", "threshold_cal", "threshold_cal2")
GRID = ((1, 4), (1, 16), (2, 4), (2, 16))                                     # (h, K)


def parse_policy(name: str):
    """'greedy(lo=0.25,hi=0.55,d=10)' / 'threshold(t=0.5,lo=0.35,hi=0.75,d=10,w=0.1)' -> política."""
    kv = dict(re.findall(r"(\w+)=([\d.]+)", name))
    if name.startswith("greedy("):
        return P.GreedyFamily(float(kv["lo"]), float(kv["hi"]), int(float(kv["d"])))
    if name.startswith("threshold("):
        return P.ThresholdFamily(float(kv["t"]), float(kv["lo"]), float(kv["hi"]), int(float(kv["d"])), float(kv["w"]))
    raise ValueError(name)


def best_reference(rows, inst):
    """(família, linha) da heurística calibrada de maior reward em VALIDAÇÃO para a instância."""
    cal = {}
    for r in rows:
        if r["instance"] == inst and r["method"] in FAMILIES and r.get("val_table"):
            cal[r["method"]] = r
    if not cal:
        raise SystemExit(f"sem heurísticas calibradas para {inst}")
    fam = max(cal, key=lambda k: max(cal[k]["val_table"].values()))
    return fam, cal[fam]


def eval_set(env, pol, scenarios, which, h, K):
    out = []
    for k in range(len(scenarios)):
        pol.rng = np.random.default_rng(np.random.SeedSequence([SALT, SETS[which], k, h, K]))   # independe da ordem/fatiamento
        out.append(E.run_episode(env, pol, scenarios[k])["reward"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heuristics", nargs="+", default=[str(ROOT / "runs" / f) for f in HEUR_FILES])
    ap.add_argument("--grid", nargs="+", default=None, help="configs h:K (padrão: 1:4 1:16 2:4 2:16); extensão registrada em DESVIOS.md 2026-10-15: 3:16 2:64 3:64")
    a = ap.parse_args()
    grid = GRID if a.grid is None else tuple(tuple(int(x) for x in g.split(":")) for g in a.grid)
    hrows = [json.loads(l) for f in a.heuristics if Path(f).exists() for l in open(f, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["instance"], r["h"], r["K"]) for r in (json.loads(l) for l in open(out, encoding="utf-8") if l.strip())}
    for spec in a.instances:
        inst, sc = RP.get_bundle(spec)
        env = E.BSRPEnv(inst)
        fam, ref = best_reference(hrows, inst.name)
        base = parse_policy(ref["policy"])
        print(f"[{inst.name}] referência (por val): {fam} = {ref['policy']}", flush=True)
        for h, K in grid:
            if (inst.name, h, K) in done:
                continue
            ro = P.Rollout(base, h=h, K=K, seed=0)
            t0 = time.time()
            res = {w: eval_set(env, ro, sc[w], w, h, K) for w in ("val", "test") + (("real_test",) if "real_test" in sc else ())}
            rec = RP.base_rec(inst, sc, "rollout")
            rec.update({"method": f"rollout_h{h}_K{K}", "h": h, "K": K, "base_family": fam, "base_policy": ref["policy"],
                        "val_mean": float(res["val"].mean()), "test_mean": float(res["test"].mean()),
                        "test_std_scen": float(res["test"].std(ddof=1)), "test_rewards": res["test"].round(1).tolist(),
                        "seconds": round(time.time() - t0, 1)})
            if "real_test" in res:
                rec.update({"real_test_mean": float(res["real_test"].mean()), "real_test_rewards": res["real_test"].round(1).tolist()})
            RP.log(out, rec)
            print(f"[{inst.name}] rollout h={h} K={K:<2}: val {rec['val_mean']:8.3f} | test {rec['test_mean']:8.3f}"
                  + (f" | dias reais {rec['real_test_mean']:8.3f}" if "real_test_mean" in rec else "") + f"  ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
