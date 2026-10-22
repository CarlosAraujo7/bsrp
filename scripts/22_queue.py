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
* `budget4x` (P8b, DESVIOS.md 2026-10-18): DQN com a config do P1 e 2 milhões de passos, 5 sementes (`runs/budget4x_<inst>.jsonl`).
* `sensx` (P8c): DQN com sementes 5–9 nas variantes de `SENSRT`, com a config do P1 (acrescenta a `sens_<variante>.jsonl`) e com a
  reajustada (acrescenta a `sens_retuned_<variante>.jsonl`; não entra se for a mesma config do P1).
* `reseedfinals` (P8c): 10 sementes × 500 mil passos do DQN com a config de `runs/tuned_reseed_<inst>.json` (só existe se a reavaliação com 4
  sementes mudou a escolha do P1) → `runs/final_reseed_<inst>.jsonl`.
* `scale` (P9): DQN com a config de 12 est. do P1 (sem reajuste) em `cb20wd`/`cb30wd`, 5 sementes, 500 mil (`runs/scale_<inst>.jsonl`) e 2 milhões de passos
  (`runs/scale2m_<inst>.jsonl`); entra a instância cujos cenários já existem em scenarios/.
* `arch` (P10): DQN com rede 128-128 e 256-256 (config nº 9 do P1 + `hidden`; `runs/tuned_arch_<h128|h256>.json`), 500 mil passos, 5 sementes, em `cb30wd` e `cb12wd`
  (`runs/arch_<tag>_<inst>.jsonl`).
* `dprl` (P11): adaptação do DPRL (`bsrp/dprl.py`) com a config do DQN plano da escala (`runs/tuned_dprl_<inst>.json`), 500 mil passos, 5 sementes
  (`runs/dprl_<inst>.jsonl`).
* `es` (P12a): neuroevolução (OpenAI-ES, rede 64–64) com a config escolhida em validação por escala (`runs/tuned_es_<inst>.json`): `es` = 500 mil passos × 10
  sementes e `es2m` = 2 milhões × 5 sementes (`runs/<tag>_<inst>.jsonl`).
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
SENSX_SEEDS = tuple(range(5, 10))             # P8c (DESVIOS.md 2026-10-18): DQN com 10 sementes nas duas variantes de 12 est. em que a conclusão mudou
BUDGET4X, B4X_SEEDS = 2_000_000, tuple(range(5))   # P8b (DESVIOS.md 2026-10-18): DQN com 4x o orçamento (2 milhões de passos), 5 sementes
SEO_VARIANTS = ("published", "gamma1")        # P4b (DESVIOS.md 2026-10-14): A2C adaptado de Seo et al. (2022); 10 sementes
SCALE_INSTANCES, SCALE_SEEDS = ("cb20wd", "cb30wd"), tuple(range(5))   # P9 (DESVIOS.md 2026-10-19): DQN com a config de 12 est. em 20 e 30 estações
SCALE_BUDGETS = (("scale", 500_000), ("scale2m", 2_000_000))
ARCH_INSTANCES, ARCH_SEEDS = ("cb30wd", "cb12wd"), tuple(range(5))   # P10 (DESVIOS.md 2026-10-20): DQN com rede mais larga, 500 mil passos
ARCH_WIDTHS = {"h128": 1.8, "h256": 3.4}                                # custo relativo aproximado vs 64-64 (medido em 30 est.)
DPRL_INSTANCES, DPRL_SEEDS = ("cb30wd", "cb12wd", "cb20wd", "cb8wd", "cb4wd"), tuple(range(5))   # P11 (DESVIOS.md 2026-10-20): adaptação do DPRL
ES_INSTANCES = ("cb30wd", "cb20wd", "cb12wd", "cb8wd", "cb4wd")        # P12a (DESVIOS.md 2026-10-22): neuroevolução (OpenAI-ES); config escolhida em val por escala
ES_RUNS = (("es", 500_000, tuple(range(10))), ("es2m", 2_000_000, tuple(range(5))))      # (tag, passos, sementes)
KIND_RANK = {"finals": 0, "sens": 1, "sensrt": 2, "reseedfinals": 3, "budget4x": 4, "scale": 5, "dprl": 6, "es": 7, "arch": 8, "sensx": 9, "seo": 10, "ablation": 11}


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
                if load_json(p_v)["dqn"]["cfg_id"] == ext["dqn"]["cfg_id"]:
                    continue        # o pool é o mesmo (config_list só depende do método): mesma config e sementes = o resultado de sens_<variante>.jsonl
                target = root / "runs" / f"sens_retuned_{name}.jsonl"
                done = {r["seed"] for r in read_rows(target)}
                for s in SENS_SEEDS:
                    if s not in done:
                        tmp = qdir / f"{name}_sensrt_dqn_{s}.jsonl"
                        cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", name, "--methods", "dqn",
                               "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                               "--tuned-file", str(p_v), "--out", str(tmp)]
                        jobs.append(Job((name, "sensrt", "dqn", s), "sensrt", est_cost("dqn", load_json(p_v)["dqn"]["cfg"]), cmd, tmp, target))
        if "reseedfinals" in kinds:                      # P8c (1): só existe se a reavaliação com 4 sementes mudou a config escolhida
            p_r = root / "runs" / f"tuned_reseed_{inst}.json"
            if p_r.exists() and "dqn" in load_json(p_r) and load_json(p_r)["dqn"]["cfg_id"] != ext["dqn"]["cfg_id"]:
                target = root / "runs" / f"final_reseed_{inst}.jsonl"
                done = {r["seed"] for r in read_rows(target)}
                for s in SEEDS:
                    if s not in done:
                        tmp = qdir / f"{inst}_reseedfinals_dqn_{s}.jsonl"
                        cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "dqn",
                               "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                               "--tuned-file", str(p_r), "--out", str(tmp)]
                        jobs.append(Job((inst, "reseedfinals", "dqn", s), "reseedfinals", est_cost("dqn", load_json(p_r)["dqn"]["cfg"]), cmd, tmp, target))
        if "budget4x" in kinds and "dqn" in ext:
            target = root / "runs" / f"budget4x_{inst}.jsonl"
            done = {r["seed"] for r in read_rows(target)}
            for s in B4X_SEEDS:
                if s not in done:
                    tmp = qdir / f"{inst}_budget4x_dqn_{s}.jsonl"
                    cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "dqn",
                           "--seeds", str(s), "--budget", str(BUDGET4X), "--eval-every", str(BUDGET4X // 10), "--tuned",
                           "--tuned-file", str(p_e), "--out", str(tmp)]
                    jobs.append(Job((inst, "budget4x", "dqn", s), "budget4x", 4 * est_cost("dqn", ext["dqn"]["cfg"]), cmd, tmp, target))
        if "sensx" in kinds and inst in SENSRT:
            for name in SENSRT[inst]:
                p_v = root / "runs" / f"tuned_ext_{name}.json"
                variants = [("orig", p_e, root / "runs" / f"sens_{name}.jsonl")]
                if p_v.exists() and "dqn" in load_json(p_v) and load_json(p_v)["dqn"]["cfg_id"] != ext["dqn"]["cfg_id"]:
                    variants.append(("retuned", p_v, root / "runs" / f"sens_retuned_{name}.jsonl"))
                for vname, p_cfg, target in variants:
                    done = {r["seed"] for r in read_rows(target) if r.get("method", "dqn") == "dqn"}
                    for s in SENSX_SEEDS:
                        if s not in done:
                            tmp = qdir / f"{name}_sensx_{vname}_{s}.jsonl"
                            cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", name, "--methods", "dqn",
                                   "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned",
                                   "--tuned-file", str(p_cfg), "--out", str(tmp)]
                            jobs.append(Job((name, "sensx", f"dqn_{vname}", s), "sensx", est_cost("dqn", load_json(p_cfg)["dqn"]["cfg"]), cmd, tmp, target))
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
    if "scale" in kinds:
        jobs += scale_jobs(root)
    if "arch" in kinds:
        jobs += arch_jobs(root)
    if "dprl" in kinds:
        jobs += dprl_jobs(root)
    if "es" in kinds:
        jobs += es_jobs(root)
    return jobs


def scale_jobs(root):
    """P9: DQN com a config escolhida no P1 para 12 estações (runs/tuned_ext_cb12wd.json; SEM reajuste) em cb20wd/cb30wd, 5 sementes, 500 mil e 2 milhões
    de passos. Só entra a instância cujos cenários val/test já existem em scenarios/ (evita corrida na geração preguiçosa)."""
    root = Path(root)
    p_e = root / "runs" / "tuned_ext_cb12wd.json"
    if not p_e.exists() or "dqn" not in load_json(p_e):
        return []
    cfg = load_json(p_e)["dqn"]["cfg"]
    qdir, py, jobs = root / "runs" / "_q", sys.executable, []
    for inst in SCALE_INSTANCES:
        if not all((root / "scenarios" / f"{k}_{inst}.npz").exists() for k in ("val", "test")):
            continue
        n = int("".join(ch for ch in inst if ch.isdigit()))
        for tag, budget in SCALE_BUDGETS:
            target = root / "runs" / f"{tag}_{inst}.jsonl"
            done = {r["seed"] for r in read_rows(target)}
            for s in SCALE_SEEDS:
                if s not in done:
                    tmp = qdir / f"{inst}_{tag}_dqn_{s}.jsonl"
                    cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "dqn",
                           "--seeds", str(s), "--budget", str(budget), "--eval-every", str(budget // 10), "--tuned",
                           "--tuned-file", str(p_e), "--out", str(tmp)]
                    jobs.append(Job((inst, "scale", f"dqn_{tag}", s), "scale", est_cost("dqn", cfg) * (budget / BUDGET) * (n / 12), cmd, tmp, target))
    return jobs


def arch_jobs(root):
    """P10: DQN com `hidden` = 128-128 / 256-256 (config nº 9 do P1 + hidden: runs/tuned_arch_<tag>.json), 500 mil passos, 5 sementes, em cb30wd e cb12wd.
    Só entra a instância cujos cenários val/test já existem em scenarios/."""
    root = Path(root)
    qdir, py, jobs = root / "runs" / "_q", sys.executable, []
    for inst in ARCH_INSTANCES:
        if not all((root / "scenarios" / f"{k}_{inst}.npz").exists() for k in ("val", "test")):
            continue
        n = int("".join(ch for ch in inst if ch.isdigit()))
        for tag, mult in ARCH_WIDTHS.items():
            p_cfg = root / "runs" / f"tuned_arch_{tag}.json"
            if not p_cfg.exists():
                continue
            cfg = load_json(p_cfg)["dqn"]["cfg"]
            target = root / "runs" / f"arch_{tag}_{inst}.jsonl"
            done = {r["seed"] for r in read_rows(target)}
            for s in ARCH_SEEDS:
                if s not in done:
                    tmp = qdir / f"{inst}_arch_{tag}_{s}.jsonl"
                    cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "dqn",
                           "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned", "--tuned-file", str(p_cfg), "--out", str(tmp)]
                    cost = est_cost("dqn", {k: v for k, v in cfg.items() if k != "hidden"}) * mult * (n / 12 if n > 12 else 1.0)
                    jobs.append(Job((inst, "arch", f"dqn_{tag}", s), "arch", cost, cmd, tmp, target))
    return jobs


def dprl_jobs(root):
    """P11: adaptação do DPRL (método `dprl`) com a config do DQN plano da escala (runs/tuned_dprl_<inst>.json, sem ajuste próprio), 500 mil passos, 5 sementes.
    Só entra a instância cujos cenários val/test existem em scenarios/ e cujo arquivo de config existe. Custo ~1,8x o do DQN plano."""
    root = Path(root)
    qdir, py, jobs = root / "runs" / "_q", sys.executable, []
    for inst in DPRL_INSTANCES:
        p_cfg = root / "runs" / f"tuned_dprl_{inst}.json"
        if not p_cfg.exists() or not all((root / "scenarios" / f"{k}_{inst}.npz").exists() for k in ("val", "test")):
            continue
        cfg = load_json(p_cfg)["dprl"]["cfg"]
        n = int("".join(ch for ch in inst if ch.isdigit()))
        target = root / "runs" / f"dprl_{inst}.jsonl"
        done = {r["seed"] for r in read_rows(target)}
        for s in DPRL_SEEDS:
            if s not in done:
                tmp = qdir / f"{inst}_dprl_{s}.jsonl"
                cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "dprl",
                       "--seeds", str(s), "--budget", str(BUDGET), "--eval-every", str(EVAL_EVERY), "--tuned", "--tuned-file", str(p_cfg), "--out", str(tmp)]
                jobs.append(Job((inst, "dprl", "dprl", s), "dprl", 1.8 * est_cost("dqn", cfg) * (n / 12 if n > 12 else 1.0), cmd, tmp, target))
    return jobs


def es_jobs(root, tags=("es", "es2m")):
    """P12a: ES (neuroevolução) com a config escolhida em validação por escala (runs/tuned_es_<inst>.json): `es` = 500 mil passos × 10 sementes,
    `es2m` = 2 milhões × 5 sementes (runs/<tag>_<inst>.jsonl). Só entra a instância cujo arquivo de config e cenários existem."""
    root = Path(root)
    qdir, py, jobs = root / "runs" / "_q", sys.executable, []
    for inst in ES_INSTANCES:
        p_cfg = root / "runs" / f"tuned_es_{inst}.json"
        if not p_cfg.exists() or not all((root / "scenarios" / f"{k}_{inst}.npz").exists() for k in ("val", "test")):
            continue
        for tag, budget, seeds in ES_RUNS:
            if tag not in tags:
                continue
            target = root / "runs" / f"{tag}_{inst}.jsonl"
            done = {r["seed"] for r in read_rows(target)}
            for s in seeds:
                if s not in done:
                    tmp = qdir / f"{inst}_{tag}_{s}.jsonl"
                    cmd = [py, "-I", str(root / "scripts" / "13_run_protocol.py"), "learners", "--instances", inst, "--methods", "es",
                           "--seeds", str(s), "--budget", str(budget), "--eval-every", str(budget // 10), "--tuned", "--tuned-file", str(p_cfg), "--out", str(tmp)]
                    jobs.append(Job((inst, "es", tag, s), "es", 0.4 * (budget / BUDGET), cmd, tmp, target))
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
    # errors="replace": com `python -X utf8` a mensagem localizada do tasklist (cp850) não decodifica e `out` viria None
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, errors="replace").stdout or ""
    return str(pid) in out and "python" in out.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["finals", "ablation"], choices=["finals", "sens", "sensrt", "sensx", "budget4x", "reseedfinals", "scale", "arch", "dprl", "es", "seo", "ablation"])
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
    required = {"finals": METHODS, "sens": SENS_METHODS, "sensrt": (), "sensx": (), "budget4x": ("dqn",), "reseedfinals": (), "scale": (), "arch": (), "dprl": (), "es": (), "seo": (), "ablation": ("dqn",)}

    def complete():
        for inst in a.instances:
            p = ROOT / "runs" / f"tuned_ext_{inst}.json"
            if not p.exists():
                return False
            d = load_json(p)
            if not all(m in d for k in a.kinds for m in required[k]
                       if not ((k == "ablation" and inst not in ABL_INSTANCES) or (k == "sens" and inst not in SENS)
                               or (k in ("sensrt", "sensx") and inst not in SENSRT))):
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
