"""
35_cost_steps.py — custo por DECISÃO do rollout (3, 64), do RHEA-S e do RHEA-L em PASSOS DE SIMULADOR (contagem exata de chamadas a `BSRPEnv.step_raw`) e em ms (DESVIOS.md 2026-10-22, P12c).

  python -X utf8 scripts/35_cost_steps.py --note "máquina compartilhada"

Para cada escala, roda o cenário de teste 0 com cada planejador (base do rollout = heurística gravada em runs/rollout_<inst>_h3K64.jsonl ou rollout_scale_<inst>.jsonl) e conta os passos de
MODELO por decisão (descontado o passo real de cada decisão). A contagem não depende da máquina; os ms dependem da CPU e da carga (anotada em --note) e servem só para a ordem de grandeza.
Grava runs/cost_steps.json.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import env as E                                                    # noqa: E402
from bsrp import policies as P                                               # noqa: E402
from bsrp.rhea import RHEA                                                   # noqa: E402

_s = importlib.util.spec_from_file_location("rollout23", ROOT / "scripts" / "23_rollout.py")
RO = importlib.util.module_from_spec(_s)
_s.loader.exec_module(RO)
RP = RO.RP

SCALES = ("cb4wd", "cb8wd", "cb12wd", "cb20wd", "cb30wd")
PLANNERS = {"rollout": lambda base: P.Rollout(base, h=3, K=64, seed=0),
            "rhea_s": lambda base: RHEA(L=4, pop=16, gens=10, K=8, seed=0),
            "rhea_l": lambda base: RHEA(L=4, pop=24, gens=15, K=16, seed=0)}
COUNT = {"n": 0}


def install_counter():
    """Conta chamadas a step_raw (step chama step_raw: contar só o nível mais baixo evita contagem dupla). Retorna a função original."""
    raw = E.BSRPEnv.step_raw

    def counted(self, *a, **k):
        COUNT["n"] += 1
        return raw(self, *a, **k)

    E.BSRPEnv.step_raw = counted
    return raw


def measure(name: str) -> dict:
    inst, sc = RP.get_bundle(name)
    env = E.BSRPEnv(inst)
    p = ROOT / "runs" / (f"rollout_{name}_h3K64.jsonl" if name in ("cb4wd", "cb8wd", "cb12wd") else f"rollout_scale_{name}.jsonl")
    with open(p, encoding="utf-8") as f:
        row = json.loads(f.readline())
    base = RO.parse_policy(row["base_policy"])
    rec = {}
    for tag, make in PLANNERS.items():
        COUNT["n"] = 0
        t0 = time.perf_counter()
        r = E.run_episode(env, make(base), sc["test"][0])
        dt = time.perf_counter() - t0
        rec[tag] = {"decisions": r["steps"], "model_steps_per_decision": (COUNT["n"] - r["steps"]) / r["steps"], "ms_per_decision": 1e3 * dt / r["steps"]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="condição da máquina na hora da medida")
    a = ap.parse_args()
    install_counter()
    git, dirty = RP.git_state()
    out = {"others_busy_note": a.note, "commit": git, "dirty": dirty}
    for name in SCALES:
        out[name] = measure(name)
        print(name, {k: (round(v["model_steps_per_decision"]), round(v["ms_per_decision"], 1)) for k, v in out[name].items()}, flush=True)
    (ROOT / "runs" / "cost_steps.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
