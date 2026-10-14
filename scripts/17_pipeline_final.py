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
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # BLAS de 1 thread: 2,5x mais rapido sob disputa de CPU
    os.environ.setdefault(_v, "1")

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


def tuned_ready(path):
    try:
        return len(json.load(open(path, encoding="utf-8"))) >= len(METHODS)
    except Exception:
        return False


def main():
    args = [x for x in sys.argv[1:] if x != "--ext"]
    ext = "--ext" in sys.argv[1:]
    inst = args[0]
    tuned_path = ROOT / "runs" / (f"tuned_ext_{inst}.json" if ext else f"tuned_{inst}.json")
    out = ROOT / "runs" / (f"final_ext_{inst}.jsonl" if ext else f"final_{inst}.jsonl")
    while not tuned_ready(tuned_path):
        time.sleep(60)
    methods = METHODS
    if ext:
        orig = json.load(open(ROOT / "runs" / f"tuned_{inst}.json", encoding="utf-8"))
        new = json.load(open(tuned_path, encoding="utf-8"))
        methods = [m for m in METHODS if new[m]["cfg_id"] != orig[m]["cfg_id"]]
        print(f"[{inst}] ext: rerodar {methods}; mantêm a original: {[m for m in METHODS if m not in methods]}", flush=True)
    print(f"[{inst}] ajuste concluído; iniciando runs finais", flush=True)
    for seed in SEEDS:                       # semente no laço externo: se for interrompido, todos os métodos têm mais sementes iguais
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
