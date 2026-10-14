"""
17_pipeline_final.py — experimento final de UMA instância, desacoplado e retomável.

Espera runs/tuned_<inst>.json conter os 5 métodos (16_tune.py) e então roda, para cada (método, semente 0–9), um
subprocesso de `13_run_protocol.py learners --tuned` com orçamento de 500.000 passos, gravando em runs/final_<inst>.jsonl.
Pula pares (método, semente) já gravados, então pode ser relançado após interrupção (ex.: notebook desligado).

Modo --ext (DESVIOS.md 2026-10-14, P1): usa runs/tuned_ext_<inst>.json (pool estendido) e roda SÓ os métodos cuja
configuração escolhida em validação difere da original (runs/tuned_<inst>.json); grava em runs/final_ext_<inst>.jsonl.
Se a original continuar vencendo, o método não é rerodado (regra pré-registrada).

  python scripts/17_pipeline_final.py cb4wd
  python scripts/17_pipeline_final.py --ext cb4wd
  python scripts/17_pipeline_final.py --ext cb4wd --seeds 0 1 2 3 4 --methods dqn      # fatia paralela
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # BLAS de 1 thread: 2,5x mais rapido sob disputa de CPU
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["dqn", "linear", "tab_station", "tab_cluster", "pg"]
SEEDS = list(range(10))
BUDGET = 500_000
EVAL_EVERY = 50_000


def done_pairs(path):
    s = set()
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                s.add((r["method"], r["seed"]))
    return s


def tuned_ready(path, methods):
    """O ajuste dos métodos pedidos terminou? (permite iniciar o DQN sem esperar os demais métodos)"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return all(m in d for m in methods)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--ext", action="store_true", help="pool estendido (P1): roda só os métodos cuja config escolhida difere da original")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="subconjunto de sementes (para rodar em fatias paralelas)")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="subconjunto de métodos (fatias paralelas)")
    a = ap.parse_args()
    ext, inst = a.ext, a.instance
    tuned_path = ROOT / "runs" / (f"tuned_ext_{inst}.json" if ext else f"tuned_{inst}.json")
    out = ROOT / "runs" / (f"final_ext_{inst}.jsonl" if ext else f"final_{inst}.jsonl")
    while not tuned_ready(tuned_path, a.methods):
        time.sleep(60)
    methods = list(a.methods)
    if ext:
        with open(ROOT / "runs" / f"tuned_{inst}.json", encoding="utf-8") as f:
            orig = json.load(f)
        with open(tuned_path, encoding="utf-8") as f:
            new = json.load(f)
        methods = [m for m in a.methods if new[m]["cfg_id"] != orig[m]["cfg_id"]]
        print(f"[{inst}] ext: rerodar {methods}; mantêm a original: {[m for m in a.methods if m not in methods]}", flush=True)
    print(f"[{inst}] ajuste concluído; iniciando runs finais (métodos {methods}, sementes {a.seeds})", flush=True)
    for seed in a.seeds:                     # semente no laço externo: se for interrompido, todos os métodos têm mais sementes iguais
        for m in methods:
            if (m, seed) in done_pairs(out):
                continue
            cmd = [sys.executable, "-I", str(ROOT / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst,
                   "--methods", m, "--seeds", str(seed), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY),
                   "--tuned", "--tuned-file", str(tuned_path), "--out", str(out)]
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=ROOT)
            print(f"[{inst}] {m} seed {seed}: rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{inst}] FIM", flush=True)


if __name__ == "__main__":
    main()
