"""
24_seo_a2c.py — A2C adaptado de Seo et al. (2022) (P4b; especificado em DESVIOS.md 2026-10-14).

Hiperparâmetros COMO PUBLICADOS (rede 3×24, lr 5e-4, γ = 0,9), sem ajuste; variante `gamma1` com γ = 1,0 (nossa métrica é a
demanda perdida total do dia). Mesmo orçamento (500.000 passos) e sementes (0–9) dos demais aprendizes. Uma linha por
(instância, variante, semente) em --out; retomável.

  python scripts/24_seo_a2c.py --instances cb4wd cb8wd cb12wd --variants published gamma1 --out runs/seo_a2c.jsonl
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                                                    # noqa: E402
from bsrp import train as T                                                  # noqa: E402
from bsrp.seo import A2C                                                     # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

VARIANTS = {"published": {"gamma": 0.9}, "gamma1": {"gamma": 1.0}}
PUBLISHED = {"hidden": [24, 24, 24], "lr_actor": 5e-4, "lr_critic": 5e-4}


def done_pairs(path):
    s = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    s.add((r["instance"], r["variant"], r["seed"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for spec in a.instances:
        inst, sc = RP.get_bundle(spec)
        env = E.BSRPEnv(inst)
        for seed in a.seeds:
            for v in a.variants:
                if (inst.name, v, seed) in done_pairs(a.out):
                    continue
                r_init = T.streams(seed)[0]
                agent = A2C(inst, hidden=tuple(PUBLISHED["hidden"]), lr_actor=PUBLISHED["lr_actor"], lr_critic=PUBLISHED["lr_critic"],
                            rng_init=r_init, **VARIANTS[v])
                info = T.train(inst, agent, seed, a.budget, sc["val"], a.eval_every)
                fe = T.final_eval(inst, agent, sc["test"])
                rec = RP.base_rec(inst, sc, "learner")
                rec.update({"method": f"a2c_seo_{v}", "variant": v, "seed": seed, "budget_steps": a.budget, "eval_every": a.eval_every,
                            "cfg": {**PUBLISHED, **VARIANTS[v]}, **info, **{k: x for k, x in fe.items() if k != "test_rewards"},
                            "test_rewards": fe["test_rewards"].round(1).tolist(), **RP.real_eval(env, agent, sc)})
                RP.log(Path(a.out), rec)
                print(f"[{inst.name}] a2c_seo_{v:<9} seed {seed}: test {fe['test_mean']:8.3f}  val(last) {info['curve'][-1]['val']:8.3f}"
                      + (f"  real-test {rec['real_test_mean']:8.3f}" if "real_test_mean" in rec else "") + f"  {info['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
