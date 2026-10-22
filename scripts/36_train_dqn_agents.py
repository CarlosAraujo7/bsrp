"""
36_train_dqn_agents.py — retreina os DQN da tabela principal e SALVA os pesos (DESVIOS.md 2026-10-22, P13).

  python -X utf8 scripts/36_train_dqn_agents.py --instance cb12wd --seeds 0 --tuned-file runs/tuned_ext_cb12wd.json

Mesmo código, config e orçamento do `13_run_protocol.py learners` (500 mil passos, eval_every 50 mil, escala 3,5, config escolhida em validação): se o treino for determinístico, o agente
retreinado é o MESMO da tabela principal. Grava runs/agents/dqn_<inst>_s<seed>.npz e uma linha em runs/agents/dqn_<inst>.jsonl com o teste (1000 cenários) e `matches_original`
(teste idêntico à linha original em final_primary.jsonl / scale_<inst>.jsonl; None se a linha original não existir).
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
from bsrp.hybrid import save_dqn                                             # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

BUDGET, EVAL_EVERY, SCALE = 500_000, 50_000, 3.5


def original_rows(inst_name: str) -> dict:
    """seed -> linha do DQN da tabela principal (final_primary.jsonl para 4/8/12 est.; scale_<inst>.jsonl para 20/30)."""
    p = ROOT / "runs" / ("final_primary.jsonl" if inst_name in ("cb4wd", "cb8wd", "cb12wd") else f"scale_{inst_name}.jsonl")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {r["seed"]: r for r in rows if r["method"] == "dqn" and r.get("instance", inst_name) == inst_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tuned-file", required=True, help="config do DQN da escala (runs/tuned_ext_<inst>.json; 20/30 est. usam a de cb12wd, como na tabela principal)")
    a = ap.parse_args()
    inst, sc = RP.get_bundle(a.instance, SCALE)
    cfg = json.load(open(a.tuned_file, encoding="utf-8"))["dqn"]["cfg"]
    train_kw = {k: v for k, v in cfg.items() if k in ("eps_decay_frac", "eps_end")}
    agent_kw = {k: v for k, v in cfg.items() if k not in ("eps_decay_frac", "eps_end")}
    orig = original_rows(inst.name)
    out_dir = ROOT / "runs" / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        w = out_dir / f"dqn_{inst.name}_s{seed}.npz"
        agent = T.make_agent("dqn", inst, seed, **agent_kw)
        info = T.train(inst, agent, seed, BUDGET, sc["val"], EVAL_EVERY, **train_kw)
        fe = T.final_eval(inst, agent, sc["test"])
        save_dqn(agent, w)
        o = orig.get(seed)
        rec = RP.base_rec(inst, sc, "agent")
        rec.update({"method": "dqn", "seed": seed, "budget_steps": BUDGET, "cfg": cfg, "weights": w.name, "eval_every": EVAL_EVERY, **info,
                    "test_mean": fe["test_mean"], "test_rewards": fe["test_rewards"].round(1).tolist(),
                    "original_test_mean": o["test_mean"] if o else None,
                    "matches_original": (o["test_rewards"] == fe["test_rewards"].round(1).tolist()) if o else None})
        RP.log(out_dir / f"dqn_{inst.name}.jsonl", rec)
        print(f"[{inst.name}] dqn seed {seed}: test {fe['test_mean']:.3f} (original {o['test_mean'] if o else None})  igual = {rec['matches_original']}  {info['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
