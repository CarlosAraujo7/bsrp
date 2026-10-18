"""
27_batch512_check.py — verificação do teto de batch do DQN (DESVIOS.md 2026-10-18).

A config escolhida do DQN (runs/tuned_ext_<inst>.json) com batch = 512 e o resto igual, 2 sementes de ajuste × 200.000 passos,
comparada em VALIDAÇÃO com as linhas existentes da config original em runs/tune_<inst>.jsonl. Regra pré-registrada: rerodar os
finais com batch 512 só se val(512) > val(original) nas DUAS sementes E a média melhora ≥ 1,0 unidade de reward.

  python scripts/27_batch512_check.py --instance cb12wd
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

MIN_GAIN = 1.0


def decide(base, new, min_gain=MIN_GAIN) -> bool:
    """val(batch 512) > val(original) nas DUAS sementes e média ≥ min_gain melhor."""
    base, new = np.asarray(base, float), np.asarray(new, float)
    return bool(np.all(new > base) and (new.mean() - base.mean()) >= min_gain)


def read_rows(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    a = ap.parse_args()
    inst, sc = RP.get_bundle(a.instance)
    with open(ROOT / "runs" / f"tuned_ext_{inst.name}.json", encoding="utf-8") as f:
        t = json.load(f)["dqn"]
    cfg = {**t["cfg"], "batch": 512}
    out = ROOT / "runs" / f"tune_b512_{inst.name}.jsonl"
    done = {r["seed"]: r for r in read_rows(out)}
    for seed in a.seeds:
        if seed in done:
            continue
        t0 = time.time()
        info = TU.run_one(inst, sc, "dqn", cfg, seed, a.budget, a.eval_every)
        vf = float(np.mean([c["val"] for c in info["curve"][-2:]]))
        RP.log(out, {"instance": inst.name, "method": "dqn", "variant": "batch512", "base_cfg_id": t["cfg_id"], "cfg": cfg, "seed": seed,
                     "budget": a.budget, "val_final": vf, "curve": info["curve"], "seconds": round(time.time() - t0, 1),
                     "val_digest": sc["val"].digest()})
        done[seed] = {"val_final": vf}
        print(f"[{inst.name}] dqn batch512 seed {seed}: val {vf:8.3f} ({time.time() - t0:.0f}s)", flush=True)
    base = {r["seed"]: r["val_final"] for r in read_rows(ROOT / "runs" / f"tune_{inst.name}.jsonl")
            if r["method"] == "dqn" and r["cfg_id"] == t["cfg_id"] and r["budget"] == a.budget}
    pairs = [(base[s], done[s]["val_final"]) for s in a.seeds if s in base]
    if len(pairs) < len(a.seeds):
        raise SystemExit(f"[{inst.name}] faltam linhas da config original para as sementes {a.seeds}")
    b, n = zip(*pairs)
    go = decide(b, n)
    res = {"instance": inst.name, "base_cfg_id": t["cfg_id"], "seeds": a.seeds, "val_base": list(b), "val_b512": list(n),
           "mean_gain": float(np.mean(n) - np.mean(b)), "rerun_finals": go}
    (ROOT / "runs" / f"b512_{inst.name}_decision.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[{inst.name}] val original {np.round(b, 2).tolist()} | batch 512 {np.round(n, 2).tolist()} | ganho médio {res['mean_gain']:+.2f} "
          f"-> {'RERODAR os finais com batch 512' if go else 'sem ganho com batch 512; mantém a configuração'}", flush=True)


if __name__ == "__main__":
    main()
