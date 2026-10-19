"""
31_latency.py — latência por DECISÃO do rollout (h = 3, K = 64) e de um passo direto de uma rede 64–64 (DESVIOS.md 2026-10-18).

  python -X utf8 scripts/31_latency.py

Mede 1 episódio de teste do rollout de cabeçalho de cada escala (32 decisões) e 20 000 passos diretos de uma rede com pesos aleatórios do mesmo
tamanho do DQN (2n+2 entradas, 64–64, 5n saídas, ativação ReLU, numpy, 1 thread). Grava runs/latency.json. As medidas dependem da CPU e da
carga da máquina na hora (anotada em `others_busy_note`: o script não a detecta — quem roda a preenche com --note); servem para a ORDEM DE GRANDEZA.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

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


def measure(name: str, n_forward: int = 20_000) -> dict:                      # (cópia de desenvolvimento; ver scripts/31_latency.py no repo)
    inst, sc = RP.get_bundle(name)
    env = E.BSRPEnv(inst)
    with open(ROOT / "runs" / f"rollout_{name}_h3K64.jsonl", encoding="utf-8") as f:
        row = json.loads(f.readline())
    lo, hi, d = map(float, re.findall(r"[\d.]+", row["base_policy"])[:3])
    ro = P.Rollout(P.GreedyFamily(lo, hi, int(d)), h=row["h"], K=row["K"], seed=0)
    t0 = time.perf_counter()
    r = E.run_episode(env, ro, sc["test"][0])
    rollout_s = (time.perf_counter() - t0) / r["steps"]
    rng = np.random.default_rng(0)
    n_in, n_out = 2 * inst.n + 2, inst.n_actions
    W1, W2, W3 = rng.normal(size=(n_in, 64)), rng.normal(size=(64, 64)), rng.normal(size=(64, n_out))
    x = rng.normal(size=(n_in,))
    t1 = time.perf_counter()
    for _ in range(n_forward):
        q = np.maximum(np.maximum(x @ W1, 0) @ W2, 0) @ W3
        int(np.argmax(q))
    net_s = (time.perf_counter() - t1) / n_forward
    return {"h": row["h"], "K": row["K"], "decisions": r["steps"], "rollout_ms": rollout_s * 1e3, "net_us": net_s * 1e6, "ratio": rollout_s / net_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="condição da máquina na hora da medida (ex.: 'CPU compartilhada com outros projetos')")
    a = ap.parse_args()
    out = {"others_busy_note": a.note}
    for name in ("cb4wd", "cb8wd", "cb12wd"):
        out[name] = measure(name)
        print(f"{name}: rollout {out[name]['rollout_ms']:.1f} ms/decisão | rede {out[name]['net_us']:.1f} µs/decisão | razão {out[name]['ratio']:,.0f}×", flush=True)
    (ROOT / "runs" / "latency.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
