"""
22_queue.py — fila de execuções com N trabalhadores: reruns finais do P1 (`runs/final_ext_<inst>.jsonl`) e ablação do DQN
(P2, `runs/ablation_<inst>.jsonl`). Retomável: as pendências saem dos próprios arquivos de resultado.

* Cada execução é um subprocesso (`13_run_protocol.py learners` / `20_ablation.py`) que grava num arquivo TEMPORÁRIO em
  runs/_q/; só este processo anexa ao arquivo de resultado (um escritor por arquivo: linhas de ~10 KB não se intercalam).
* Ordem: finais antes da ablação; dentro de cada tipo, a mais cara primeiro (balanceia o fim da fila).
* Nº de trabalhadores: `--workers`, sobrescrito a cada ciclo pelo inteiro em runs/_q/WORKERS (se existir). `runs/_q/STOP`
  = não inicia novas execuções e sai quando as em curso terminarem. Falhou 2 vezes → desiste daquela execução e segue.
* `sens` (P3, DESVIOS.md 2026-10-14): 4 aprendizes × 5 sementes × variantes de cb4wd/cb12wd e braço legado de 4 est., com a config
  escolhida no P1 na escala-base; só entram instâncias cujos cenários val/test já existem em scenarios/.
* `sensrt` (P3b, DESVIOS.md 2026-10-18): DQN com a config escolhida em validação NA PRÓPRIA variante (`runs/tuned_ext_<variante>.json`),
  5 sementes (`runs/sens_retuned_<variante>.jsonl`); só entra a variante cujo arquivo de ajuste já existe.
* Os reruns finais só entram para métodos cuja config escolhida em validação difere da original (regra do DESVIOS.md).

  python scripts/22_queue.py --kinds finals ablation --workers 4
  python scripts/22_queue.py --kinds sens --instances cb4wd cb12wd --workers 4      # P3: sensibilidade (runs/sens_<instância>.jsonl)
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dqn", "linear", "tab_station", "tab_cluster", "pg")
SEEDS = tuple(range(10))
BUDGET, EVAL_EVERY = 500_000, 50_000
INSTANCES = ("cb4wd", "cb8wd", "cb12wd")
ABL_INSTANCES = ("cb12wd", "cb4wd")
ABL_VARIANTS = ("noreplay", "notarget", "nodouble", "vanilla")
# P3 (sensibilidade; DESVIOS.md 2026-10-14): escala-base -> [(especificação da instância, nome da instância)]
SENS = {"cb4wd": [("cb4wdx0.5", "cb4wdx0.5"), ("cb4wdx0.75", "cb4wdx0.75"), ("cb4wdx1.5", "cb4wdx1.5"), ("cb4we", "cb4we"),
                  ("legacy4x1.5", "legacy4x1.5"), ("legacy4x2.5", "legacy4x2.5"), ("4", "legacy4"), ("legacy4x5", "legacy4x5")],
        "cb12wd": [("cb12wdx0.5", "cb12wdx0.5"), ("cb12wdx0.75", "cb12wdx0.75"), ("cb12wdx1.5", "cb12wdx1.5"), ("cb12we", "cb12we")]}
SENS_METHODS = ("dqn", "linear", "tab_station", "tab_cluster")
SENS_SEEDS = tuple(range(5))
SENSRT = {"cb12wd": ("cb12wdx0.5", "cb12we")}   # P3b (DESVIOS.md 2026-10-18): DQN reajustado nas variantes em que a vantagem sumiu; 5 sementes
SEO_VARIANTS = ("published", "gamma1")        # P4b (DESVIOS.md 2026-10-14): A2C adaptado de Seo et al. (2022); 10 sementes
KIND_RANK = {"finals": 0, "sens": 1, "sensrt": 2, "seo": 3, "ablation": 4}


@dataclass
class Job:
    key: tuple
    kind: str            # "finals" | "sens" | "seo" | "ablation"
    cost: float          # custo relativo (só para ordenar)
    cmd: list
    tmp: Path
    target: Path


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_rows(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def est_cost(method, cfg):
    """Custo relativo (minutos de 500k passos medidos em 14/out com a CPU compartilhada). Só ordena a fila."""
    be = cfg.get("batch", 64) / cfg.get("train_every", 1)
    return 3.6 + 0.057 * be if method == "dqn" else (2.0 + 0.01 * be if method == "linear" else 1.5)


def build_jobs(root, insts=INSTANCES, kinds=("finals", "ablation")):
    root = Path(root)
    qdir, py = root / "runs" / "_q", sys.executable
    jobs = []
    for inst in insts:
        p_o, p_e = root / "runs" / f"tuned_{inst}.json", root / "runs" / f"tuned_ext_{inst}.json"
        if not (p_o.exists() and p_e.exists()):
            continue
        orig, ext = load_json(p_o), load_json(p_e)
        if "seo" in kinds:
            target = root / "runs" / f"seo_a2c_{inst}.jsonl"
            done = {(r["variant"], r["seed"]) for r in read_rows(target)}
            variants = SEO_VARIANTS + (("tuned",) if (root / "runs" / f"tuned_a2c_{inst}.json").exists() else ())
            for v in variants:
                for s in SEEDS:
                    if (v, s) not in done:
                        tmp = qdir / f"{inst}_seo_{v}_{s}.jsonl"
                        cmd = [py, "-I", str(root / "scripts" / "24_seo_a2c.py"), "--instances", inst, "--variants", v,
                               "--seeds", str(s), "--out", str(tmp)]
                        jobs.append(Job((inst, "seo", v, s), "seo", 3.0, cmd, tmp, target))
        if "sens" in kinds and inst in SENS:
            for spec, name in SENS[inst]:
                if not all((root / "scenarios" / f"{k}_{name}.npz").exists() for k in ("val", "test")):
                    continue                     # cenários ainda não gerados (evita corrida na geração preguiçosa)
                target = root / "runs" / f"sens_{name}.jsonl"
                done = {(r["method"], r["seed"]) for r in read_rows(target)}
                for m in SENS_METHODS:
                    if m not in ext:
                        continue
                    for s in SENS_SEEDS:
                        if (m, s) not in done:
                            tmp = qdir / f"{name}_sens_{m}_{s}.jsonl"
                            cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", spec, "--methods", m,
                                   "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                                   "--tuned-file", str(p_e), "--out", str(tmp)]
                            jobs.append(Job((name, "sens", m, s), "sens", est_cost(m, ext[m]["cfg"]), cmd, tmp, target))
        if "sensrt" in kinds and inst in SENSRT:
            for name in SENSRT[inst]:
                p_v = root / "runs" / f"tuned_ext_{name}.json"          # config do DQN escolhida em validação NA PRÓPRIA variante
                if not p_v.exists() or "dqn" not in load_json(p_v):
                    continue
                target = root / "runs" / f"sens_retuned_{name}.jsonl"
                done = {r["seed"] for r in read_rows(target)}
                for s in SENS_SEEDS:
                    if s not in done:
                        tmp = qdir / f"{name}_sensrt_dqn_{s}.jsonl"
                        cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", name, "--methods", "dqn",
                               "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                               "--tuned-file", str(p_v), "--out", str(tmp)]
                        jobs.append(Job((name, "sensrt", "dqn", s), "sensrt", est_cost("dqn", load_json(p_v)["dqn"]["cfg"]), cmd, tmp, target))
        if "finals" in kinds:
            target = root / "runs" / f"final_ext_{inst}.jsonl"
            done = {(r["method"], r["seed"]) for r in read_rows(target)}
            for m in METHODS:
                if m in ext and m in orig and ext[m]["cfg_id"] != orig[m]["cfg_id"]:
                    for s in SEEDS:
                        if (m, s) not in done:
                            tmp = qdir / f"{inst}_finals_{m}_{s}.jsonl"
                            cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", m,
                                   "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                                   "--tuned-file", str(p_e), "--out", str(tmp)]
                            jobs.append(Job((inst, "finals", m, s), "finals", est_cost(m, ext[m]["cfg"]), cmd, tmp, target))
        if "ablation" in kinds and inst in ABL_INSTANCES and "dqn" in ext:
            target = root / "runs" / f"ablation_{inst}.jsonl"
            done = {(r["variant"], r["seed"]) for r in read_rows(target)}
            base = est_cost("dqn", ext["dqn"]["cfg"])
            for v in ABL_VARIANTS:
                for s in SEEDS:
                    if (v, s) not in done:
                        tmp = qdir / f"{inst}_ablation_{v}_{s}.jsonl"
                        cmd = [py, "-I", str(root / "scripts" / "20_ablation.py"), "--instance", inst, "--variants", v,
                               "--seeds", str(s), "--out", str(tmp)]
                        jobs.append(Job((inst, "ablation", v, s), "ablation", base * (0.6 if v in ("noreplay", "vanilla") else 1.0), cmd, tmp, target))
    return jobs


def order(jobs):
    return sorted(jobs, key=lambda j: (KIND_RANK[j.kind], -j.cost, j.key))


def merge(job):
    """Anexa as linhas do arquivo temporário ao arquivo de resultado (único escritor)."""
    with open(job.tmp, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    job.target.parent.mkdir(parents=True, exist_ok=True)
    with open(job.target, "a", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in lines))
    return len(lines)


def run_queue(get_jobs, workers_fn, qdir, complete_fn=lambda: True, poll=20.0, retries=2, log=print):
    qdir = Path(qdir)
    qdir.mkdir(parents=True, exist_ok=True)
    running, fails, dead = {}, defaultdict(int), set()
    while True:
        for key, (p, t0, job, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[key]
            lf.close()
            ok = rc == 0 and job.tmp.exists() and job.tmp.stat().st_size > 0
            if ok:
                n = merge(job)
                job.tmp.unlink()
                Path(lf.name).unlink(missing_ok=True)
                log(f"[fila] fim {key} ({time.time() - t0:.0f}s, {n} linha(s))")
            else:
                fails[key] += 1
                job.tmp.unlink(missing_ok=True)
                log(f"[fila] FALHA {key} rc={rc} (tentativa {fails[key]}/{retries}); log em {lf.name}")
                if fails[key] >= retries:
                    dead.add(key)
        stop = (qdir / "STOP").exists()
        pending = order([j for j in get_jobs() if j.key not in running and j.key not in dead])
        slots = max(0, workers_fn() - len(running))
        if not stop:
            for job in pending[:slots]:
                job.tmp.parent.mkdir(parents=True, exist_ok=True)
                job.tmp.unlink(missing_ok=True)
                lf = open(qdir / ("_".join(map(str, job.key)) + ".log"), "a", encoding="utf-8")
                running[job.key] = (subprocess.Popen(job.cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT), time.time(), job, lf)
                log(f"[fila] início {job.key} (custo ~{job.cost:.0f}); em curso {len(running)}, pendentes {len(pending) - min(slots, len(pending))}")
        if not running and (stop or (not pending and complete_fn())):
            break
        time.sleep(poll)
    if dead:
        log(f"[fila] desistiu de {sorted(dead)}")
    log("[fila] FIM" + (" (STOP)" if stop else ""))
    return dead


def workers_from_file(qdir, default):
    def fn():
        try:
            return int((Path(qdir) / "WORKERS").read_text().strip())
        except (OSError, ValueError):
            return default
    return fn


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out and "python" in out.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["finals", "ablation"], choices=["finals", "sens", "sensrt", "seo", "ablation"])
    ap.add_argument("--instances", nargs="+", default=list(INSTANCES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    qdir = ROOT / "runs" / "_q"
    qdir.mkdir(parents=True, exist_ok=True)
    lock = qdir / "LOCK"
    if lock.exists() and pid_alive(int(lock.read_text().strip() or 0)):
        raise SystemExit(f"já há uma fila em execução (PID {lock.read_text().strip()})")
    lock.write_text(str(os.getpid()))
    required = {"finals": METHODS, "sens": SENS_METHODS, "sensrt": (), "seo": (), "ablation": ("dqn",)}

    def complete():
        for inst in a.instances:
            p = ROOT / "runs" / f"tuned_ext_{inst}.json"
            if not p.exists():
                return False
            d = load_json(p)
            if not all(m in d for k in a.kinds for m in required[k]
                       if not ((k == "ablation" and inst not in ABL_INSTANCES) or (k == "sens" and inst not in SENS))):
                return False
        return True

    def stamp(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    try:
        run_queue(lambda: build_jobs(ROOT, a.instances, a.kinds), workers_from_file(qdir, a.workers), qdir, complete, a.poll, log=stamp)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
