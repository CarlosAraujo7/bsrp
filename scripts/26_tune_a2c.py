"""
26_tune_a2c.py — ajuste do A2C adaptado de Seo et al. (2022) com o MESMO orçamento dos demais aprendizes
(DESVIOS.md 2026-10-15): 16 configurações (a nº 0 = como publicado) × 2 sementes de ajuste (1000, 1001) × 200.000 passos,
escolha pelo val médio dos 2 últimos pontos da curva, por escala. Retomável. Saídas:
  runs/tune_a2c_<inst>.jsonl     uma linha por (configuração, semente)
  runs/tuned_a2c_<inst>.json     a configuração escolhida

  python scripts/26_tune_a2c.py --instance cb4wd
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
from bsrp import train as T                                                  # noqa: E402
from bsrp.seo import A2C                                                     # noqa: E402

_rp = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_rp)
_rp.loader.exec_module(RP)
_tu = importlib.util.spec_from_file_location("tune16", ROOT / "scripts" / "16_tune.py")
TU = importlib.util.module_from_spec(_tu)
_tu.loader.exec_module(TU)

PUBLISHED = {"lr_actor": 5e-4, "lr_critic": 5e-4, "gamma": 0.9, "entropy": 0.0, "reward_scale": 1.0}
N_CFG = 16


def config_list(n_cfg: int = N_CFG) -> list:
    """cfg 0 = como publicado; as demais em desenho balanceado (hipercubo latino discreto), semente fixa."""
    rng = np.random.default_rng(np.random.SeedSequence([4242, 20261015]))
    n = n_cfg - 1
    cols = {"lr_actor": TU._bal_log(rng, -4.0, -2.5, n), "lr_critic": TU._bal_log(rng, -4.0, -2.5, n),
            "gamma": TU._bal(rng, [0.9, 0.95, 0.99, 1.0], n), "entropy": TU._bal(rng, [0.0, 0.01, 0.03, 0.1, 0.3], n),
            "reward_scale": TU._bal(rng, [1.0, 0.3, 0.1], n)}
    return [dict(PUBLISHED)] + [{k: (float(v[i])) for k, v in cols.items()} for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    a = ap.parse_args()
    inst, sc = RP.get_bundle(a.instance)
    out = ROOT / "runs" / f"tune_a2c_{inst.name}.jsonl"
    best_path = ROOT / "runs" / f"tuned_a2c_{inst.name}.json"
    done = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    done[(r["cfg_id"], r["seed"])] = r
    cfgs = config_list()
    scores = []
    for ci, cfg in enumerate(cfgs):
        vals = []
        for seed in a.seeds:
            if (ci, seed) in done:
                r = done[(ci, seed)]
                if r["cfg"] != cfg or r["budget"] != a.budget or r["val_digest"] != sc["val"].digest():
                    raise SystemExit(f"[{inst.name}] cfg {ci} seed {seed}: linha existente não corresponde à config/orçamento/cenários atuais")
                vals.append(r["val_final"])
                continue
            t0 = time.time()
            agent = A2C(inst, rng_init=T.streams(seed)[0], **cfg)
            info = T.train(inst, agent, seed, a.budget, sc["val"], a.eval_every)
            vf = float(np.mean([c["val"] for c in info["curve"][-2:]]))
            RP.log(out, {"instance": inst.name, "method": "a2c_seo", "cfg_id": ci, "cfg": cfg, "seed": seed, "budget": a.budget,
                         "val_final": vf, "curve": info["curve"], "seconds": round(time.time() - t0, 1), "val_digest": sc["val"].digest()})
            vals.append(vf)
            print(f"[{inst.name}] a2c cfg {ci:2d} seed {seed}: val {vf:8.3f} ({time.time() - t0:.0f}s)", flush=True)
        scores.append(float(np.mean(vals)))
    b = int(np.argmax(scores))
    json.dump({"cfg_id": b, "cfg": cfgs[b], "val_mean": scores[b], "all_val_means": scores, "published_val_mean": scores[0],
               "n_configs": len(cfgs), "seeds": a.seeds, "budget": a.budget}, open(best_path, "w"), indent=1)
    print(f"[{inst.name}] a2c: melhor cfg {b} (val {scores[b]:.3f}; publicado {scores[0]:.3f}) -> {cfgs[b]}", flush=True)


if __name__ == "__main__":
    main()
