"""
20_ablation.py — ablação do DQN (protocolo C7–C9; especificada em DESVIOS.md 2026-10-14, P2).

Variantes do DQN (rede 64-64) sobre a configuração escolhida em validação no P1 (runs/tuned_ext_<inst>.json), com o
mesmo orçamento e o mesmo cronograma de ε do DQN completo, sementes 0–9 (init, exploração e ambiente de treino derivam
da semente -> pareamento por semente com o DQN completo). Uma linha por (variante, semente) em runs/ablation_<inst>.jsonl;
retomável (pula pares já gravados).

  noreplay   use_replay=False                                  (1 atualização por transição, lote 1)
  notarget   use_target=False                                  (alvo calculado com a rede online; double vira no-op)
  nodouble   double=False
  vanilla    as três

  python scripts/20_ablation.py --instance cb12wd
  python scripts/20_ablation.py --instance cb4wd --variants noreplay vanilla --seeds 0 1 2
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                                                    # noqa: E402
from bsrp import train as T                                                  # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

VARIANTS = {
    "noreplay": {"use_replay": False},
    "notarget": {"use_target": False},
    "nodouble": {"double": False},
    "vanilla": {"use_replay": False, "use_target": False, "double": False},
}
TRAIN_KEYS = ("eps_decay_frac", "eps_end")


def done_pairs(path):
    s = set()
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                s.add((r["variant"], r["seed"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--tuned-file", default=None, help="padrão: runs/tuned_ext_<inst>.json (config do DQN escolhida no P1)")
    ap.add_argument("--out", default=None, help="padrão: runs/ablation_<inst>.jsonl")
    a = ap.parse_args()
    inst, sc = RP.get_bundle(a.instance)
    env = E.BSRPEnv(inst)
    tuned_path = Path(a.tuned_file) if a.tuned_file else ROOT / "runs" / f"tuned_ext_{inst.name}.json"
    with open(tuned_path, encoding="utf-8") as f:
        t = json.load(f)["dqn"]
    cfg = t["cfg"]
    agent_kw = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    train_kw = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    out = Path(a.out) if a.out else ROOT / "runs" / f"ablation_{inst.name}.jsonl"
    print(f"[{inst.name}] base: dqn cfg {t['cfg_id']} {cfg}", flush=True)
    for seed in a.seeds:                       # semente no laço externo: interrompido, todas as variantes têm sementes iguais
        for v in a.variants:
            if (v, seed) in done_pairs(out):
                continue
            agent = T.make_agent("dqn", inst, seed, **agent_kw, **VARIANTS[v])
            info = T.train(inst, agent, seed, a.budget, sc["val"], a.eval_every, **train_kw)
            fe = T.final_eval(inst, agent, sc["test"])
            rec = RP.base_rec(inst, sc, "ablation")
            rec.update({"method": f"dqn_{v}", "variant": v, "flags": VARIANTS[v], "seed": seed, "budget_steps": a.budget,
                        "cfg": cfg, "cfg_id": t["cfg_id"], "eval_every": a.eval_every, **info,
                        **{k: x for k, x in fe.items() if k != "test_rewards"},
                        "test_rewards": fe["test_rewards"].round(1).tolist(), **RP.real_eval(env, agent, sc)})
            RP.log(out, rec)
            print(f"[{inst.name}] {v:<9} seed {seed}: test {fe['test_mean']:7.3f}  val(last) {info['curve'][-1]['val']:7.3f}"
                  + (f"  real-test {rec['real_test_mean']:7.3f}" if "real_test_mean" in rec else "")
                  + f"  {info['train_seconds']:.0f}s", flush=True)
    print(f"[{inst.name}] FIM", flush=True)


if __name__ == "__main__":
    main()
