"""
29_reseed_top.py — reavaliação das melhores configs do DQN com mais sementes de ajuste (DESVIOS.md 2026-10-18, P8c).

As `--top` configs de maior val médio (sementes 1000 e 1001, runs/tune_<inst>.jsonl) recebem as sementes extras (1002, 1003 por padrão), 200 mil
passos; a escolha passa a ser pela média de TODAS as sementes de ajuste entre essas configs. Regra pré-registrada: se a escolhida diferir da do
P1 (runs/tuned_ext_<inst>.json), grava runs/tuned_reseed_<inst>.json (a fila `reseedfinals` roda então 10 sementes × 500 mil passos); se for a
mesma, a seleção é confirmada.

  python scripts/29_reseed_top.py --instance cb12wd --shard 0/3
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

_tu = importlib.util.spec_from_file_location("tune16", ROOT / "scripts" / "16_tune.py")
TU = importlib.util.module_from_spec(_tu)
_tu.loader.exec_module(TU)
RP = TU.RP
BASE_SEEDS = (1000, 1001)


def read_rows(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def top_configs(rows, budget, k=4, seeds=BASE_SEEDS):
    """cfg_id -> (cfg, val médio nas sementes-base) das k melhores; execuções repetidas valem pela última linha."""
    last, cfgs = {}, {}
    for r in rows:
        if r["method"] == "dqn" and r["budget"] == budget and r["seed"] in seeds:
            last[(r["cfg_id"], r["seed"])] = r["val_final"]
            cfgs[r["cfg_id"]] = r["cfg"]
    means = {}
    for (ci, _), v in last.items():
        means.setdefault(ci, []).append(v)
    means = {ci: float(np.mean(v)) for ci, v in means.items() if len(v) == len(seeds)}
    best = sorted(means, key=lambda ci: (-means[ci], ci))[:k]
    return {ci: (cfgs[ci], means[ci]) for ci in best}


def choose(top, extra_rows, base_rows, budget, all_seeds):
    """Média de todas as sementes (base + extras) por config entre as `top`; retorna (escolhida, {cfg_id: média}) ou None se faltar semente."""
    vals = {ci: {} for ci in top}
    for r in list(base_rows) + list(extra_rows):
        if r["method"] == "dqn" and r["budget"] == budget and r["cfg_id"] in vals and r["seed"] in all_seeds:
            vals[r["cfg_id"]][r["seed"]] = r["val_final"]
    if any(len(v) < len(all_seeds) for v in vals.values()):
        return None
    means = {ci: float(np.mean(list(v.values()))) for ci, v in vals.items()}
    return max(means, key=lambda ci: (means[ci], -ci)), means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1002, 1003])
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    base_rows = read_rows(ROOT / "runs" / f"tune_{inst.name}.jsonl")
    top = top_configs(base_rows, a.budget, a.top)
    p1 = json.load(open(ROOT / "runs" / f"tuned_ext_{inst.name}.json", encoding="utf-8"))["dqn"]
    assert p1["cfg_id"] == max(top, key=lambda ci: top[ci][1]), "a config do P1 deveria ser a de maior val médio do pool"
    out = ROOT / "runs" / f"tune_reseed_{inst.name}.jsonl"
    done = {(r["cfg_id"], r["seed"]) for r in read_rows(out)}
    jobs = [(ci, s) for ci in sorted(top) for s in a.seeds]
    mine = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
    print(f"[{inst.name}] top-{a.top}: {sorted(top)}; fatia {si}/{sn}: {len(mine)} execuções", flush=True)
    for ci, s in mine:
        t0 = time.time()
        info = TU.run_one(inst, sc, "dqn", top[ci][0], s, a.budget, a.eval_every)
        vf = float(np.mean([c["val"] for c in info["curve"][-2:]]))
        RP.log(out, {"instance": inst.name, "method": "dqn", "cfg_id": ci, "cfg": top[ci][0], "seed": s, "budget": a.budget, "val_final": vf,
                     "curve": info["curve"], "seconds": round(time.time() - t0, 1), "val_digest": sc["val"].digest()})
        print(f"[{inst.name}] cfg {ci} seed {s}: val {vf:8.3f} ({time.time() - t0:.0f}s)", flush=True)
    res = choose(top, read_rows(out), base_rows, a.budget, set(BASE_SEEDS) | set(a.seeds))
    if res is None:
        print(f"[{inst.name}] FIM fatia {si}/{sn} (decisão pendente: faltam execuções de outras fatias)", flush=True)
        return
    sel, means = res
    changed = sel != p1["cfg_id"]
    rec = {"instance": inst.name, "top": {str(ci): top[ci][1] for ci in top}, "means_all_seeds": {str(ci): m for ci, m in means.items()},
           "p1_cfg_id": p1["cfg_id"], "selected_cfg_id": sel, "changed": changed, "seeds": sorted(set(BASE_SEEDS) | set(a.seeds))}
    (ROOT / "runs" / f"reseed_{inst.name}_decision.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if changed:
        (ROOT / "runs" / f"tuned_reseed_{inst.name}.json").write_text(
            json.dumps({"dqn": {"cfg_id": sel, "cfg": top[sel][0], "val_mean": means[sel], "n_configs": a.top, "seeds": rec["seeds"],
                                "budget": a.budget, "space": "reseed"}}, indent=1), encoding="utf-8")
    print(f"[{inst.name}] FIM fatia {si}/{sn}: médias {np.round(list(means.values()), 2).tolist()} (configs {sorted(top)}) -> "
          f"{'MUDOU: nova config ' + str(sel) + ' (P1: ' + str(p1['cfg_id']) + ')' if changed else 'seleção confirmada (cfg ' + str(sel) + ')'}", flush=True)


if __name__ == "__main__":
    main()
