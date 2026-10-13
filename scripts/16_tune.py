"""
16_tune.py — ajuste de hiperparâmetros (PROTOCOLO_RIGOR.md §3, especificado em DESVIOS.md).

Mesmo número de configurações para TODOS os aprendizes (a configuração padrão é a nº 0 de cada lista), mesmas
sementes de ajuste (1000, 1001; disjuntas das sementes finais 0–19), mesmo orçamento reduzido, escolha por reward
médio em VALIDAÇÃO (último ponto da curva; nunca teste). Saídas:
  runs/tune_<inst>.jsonl     uma linha por (método, configuração, semente)
  runs/tuned_<inst>.json     melhor configuração por método

  python scripts/16_tune.py --instance cb4wd --methods dqn linear tab_station tab_cluster pg --configs 8 --seeds 1000 1001 --budget 100000
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # BLAS de 1 thread: 2,5x mais rapido sob disputa de CPU
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

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

DEFAULTS = {
    "dqn": {"lr": 3e-4, "target_sync": 1000, "train_every": 4, "batch": 64, "buffer": 10_000, "eps_decay_frac": 0.7},
    "linear": {"lr": 3e-4, "target_sync": 1000, "train_every": 4, "batch": 64, "buffer": 10_000, "eps_decay_frac": 0.7},
    "tab_cluster": {"alpha": 0.1, "eps_decay_frac": 0.7, "eps_end": 0.05},
    "tab_station": {"alpha": 0.1, "eps_decay_frac": 0.7, "eps_end": 0.05},
    "pg": {"lr_actor": 3e-4, "lr_critic": 1e-3, "entropy": 0.01, "episodes_per_update": 8},
}
TRAIN_KEYS = {"eps_decay_frac", "eps_end"}


def sample_cfg(method, rng):
    if method in ("dqn", "linear"):
        return {"lr": float(10 ** rng.uniform(-4, -2.5)), "target_sync": int(rng.choice([250, 500, 1000, 2000, 4000])),
                "train_every": int(rng.choice([1, 2, 4])), "batch": int(rng.choice([32, 64, 128])),
                "buffer": int(rng.choice([10_000, 50_000])), "eps_decay_frac": float(rng.choice([0.3, 0.5, 0.7, 0.9]))}
    if method.startswith("tab_"):
        return {"alpha": float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.5])), "eps_decay_frac": float(rng.choice([0.3, 0.5, 0.7, 0.9])),
                "eps_end": float(rng.choice([0.01, 0.05, 0.1]))}
    if method == "pg":
        return {"lr_actor": float(10 ** rng.uniform(-4, -2.5)), "lr_critic": float(10 ** rng.uniform(-3.5, -2)),
                "entropy": float(rng.choice([0.003, 0.01, 0.03, 0.1])), "episodes_per_update": int(rng.choice([4, 8, 16, 32]))}
    raise ValueError(method)


def config_list(method, n_cfg):
    rng = np.random.default_rng(np.random.SeedSequence([4242, sum(map(ord, method))]))   # fixa por método
    return [DEFAULTS[method]] + [sample_cfg(method, rng) for _ in range(n_cfg - 1)]


def run_one(inst, sc, method, cfg, seed, budget, eval_every):
    agent_kw = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    train_kw = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    ag = T.make_agent(method, inst, seed, **agent_kw)
    info = T.train(inst, ag, seed, budget, sc["val"], eval_every, **train_kw)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULTS))
    ap.add_argument("--configs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    a = ap.parse_args()
    inst, sc = RP.get_bundle(a.instance)
    out = ROOT / "runs" / f"tune_{inst.name}.jsonl"
    best_path = ROOT / "runs" / f"tuned_{inst.name}.json"
    best = json.load(open(best_path)) if best_path.exists() else {}
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            r = json.loads(l); done.add((r["method"], r["cfg_id"], r["seed"]))
    for method in a.methods:
        cfgs = config_list(method, a.configs)
        scores = []
        for ci, cfg in enumerate(cfgs):
            vals = []
            for seed in a.seeds:
                if (method, ci, seed) in done:
                    vals.append([json.loads(l) for l in open(out, encoding="utf-8")
                                 if json.loads(l)["method"] == method and json.loads(l)["cfg_id"] == ci and json.loads(l)["seed"] == seed][0]["val_final"])
                    continue
                t0 = time.time()
                info = run_one(inst, sc, method, cfg, seed, a.budget, a.eval_every)
                vf = float(np.mean([c["val"] for c in info["curve"][-2:]]))       # média dos 2 últimos pontos (reduz ruído)
                RP.log(out, {"instance": inst.name, "method": method, "cfg_id": ci, "cfg": cfg, "seed": seed, "budget": a.budget,
                             "val_final": vf, "curve": info["curve"], "seconds": round(time.time() - t0, 1),
                             "val_digest": sc["val"].digest()})
                vals.append(vf)
                print(f"[{inst.name}] {method:<12} cfg {ci} seed {seed}: val {vf:8.3f} ({time.time()-t0:.0f}s)", flush=True)
            scores.append(float(np.mean(vals)))
        b = int(np.argmax(scores))
        best[method] = {"cfg_id": b, "cfg": cfgs[b], "val_mean": scores[b], "all_val_means": scores, "n_configs": len(cfgs),
                        "seeds": a.seeds, "budget": a.budget}
        json.dump(best, open(best_path, "w"), indent=1)
        print(f"[{inst.name}] {method}: melhor cfg {b} (val {scores[b]:.3f}; padrão {scores[0]:.3f}) -> {cfgs[b]}", flush=True)


if __name__ == "__main__":
    main()
