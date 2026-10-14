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


N_ORIG = 8          # tamanho do pool original (DESVIOS.md 2026-10-13); o pool estendido o contém como prefixo


def _bal(rng, values, n):
    """n valores balanceados (cada valor aparece floor/ceil(n/k) vezes), em ordem aleatória: nenhum valor novo fica sem teste."""
    idx = np.resize(np.arange(len(values)), n)
    rng.shuffle(idx)
    return [values[i] for i in idx]


def _bal_log(rng, lo, hi, n):
    """n valores em escala log10 de [lo, hi], um por estrato igual (hipercubo latino), em ordem aleatória."""
    u = (rng.permutation(n) + rng.random(n)) / n
    return [float(10 ** (lo + (hi - lo) * x)) for x in u]


def ext_block(method, rng, n):
    """Espaço ESTENDIDO (DESVIOS.md 2026-10-14): n configs com desenho BALANCEADO (hipercubo latino discreto), em torno/além
    da borda onde o vencedor caiu. Batch <= 256 (custo)."""
    if method == "dqn":      # vencedor nas 3 escalas: sync/batch/buffer no máximo, train_every=2 -> zoom nessa região
        cols = {"lr": _bal_log(rng, np.log10(1.5e-4), -3, n), "target_sync": _bal(rng, [4000, 8000, 16000], n),
                "train_every": _bal(rng, [1, 2], n), "batch": _bal(rng, [128, 256], n),
                "buffer": _bal(rng, [50_000, 100_000, 200_000], n), "eps_decay_frac": _bal(rng, [0.3, 0.5, 0.7, 0.9], n)}
    elif method == "linear":  # vencedores em bordas opostas conforme a escala -> união ampliada nos dois lados
        cols = {"lr": _bal_log(rng, -4, -2.5, n), "target_sync": _bal(rng, [250, 500, 1000, 2000, 4000, 8000, 16000], n),
                "train_every": _bal(rng, [1, 2, 4, 8], n), "batch": _bal(rng, [16, 32, 64, 128, 256], n),
                "buffer": _bal(rng, [5_000, 10_000, 50_000, 100_000, 200_000], n),
                "eps_decay_frac": _bal(rng, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], n)}
    elif method.startswith("tab_"):
        cols = {"alpha": _bal(rng, [0.02, 0.05, 0.1, 0.2, 0.5, 0.8], n),
                "eps_decay_frac": _bal(rng, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], n), "eps_end": _bal(rng, [0.01, 0.05, 0.1, 0.2], n)}
    elif method == "pg":     # vencedor nas 3 escalas: entropia=0,1 (máx.) e episodes_per_update=4 (mín.)
        cols = {"lr_actor": _bal_log(rng, -4, -2.5, n), "lr_critic": _bal_log(rng, -3.5, -2, n),
                "entropy": _bal(rng, [0.1, 0.3, 1.0], n), "episodes_per_update": _bal(rng, [1, 2, 4], n)}
    else:
        raise ValueError(method)
    return [{k: col[i] for k, col in cols.items()} for i in range(n)]


def config_list(method, n_cfg, space="orig"):
    """space='orig': padrão + n_cfg-1 sorteios do espaço original. space='ext': as N_ORIG originais (idênticas, mesmo
    gerador) + (n_cfg - N_ORIG) configs balanceadas do espaço estendido, continuando o mesmo gerador."""
    rng = np.random.default_rng(np.random.SeedSequence([4242, sum(map(ord, method))]))   # fixa por método
    if space == "orig":
        return [DEFAULTS[method]] + [sample_cfg(method, rng) for _ in range(n_cfg - 1)]
    assert space == "ext" and n_cfg > N_ORIG
    cfgs = [DEFAULTS[method]] + [sample_cfg(method, rng) for _ in range(N_ORIG - 1)]
    return cfgs + ext_block(method, rng, n_cfg - N_ORIG)


def run_one(inst, sc, method, cfg, seed, budget, eval_every):
    agent_kw = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    train_kw = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    ag = T.make_agent(method, inst, seed, **agent_kw)
    info = T.train(inst, ag, seed, budget, sc["val"], eval_every, **train_kw)
    return info


def est_cost(method, cfg):
    """Custo relativo aproximado de uma execução (medido em 14/out: DQN 500k passos ≈ 7 min com batch/every = 64, ≈ 17 min com 256).
    Serve só para equilibrar as fatias (--shard); não entra em nenhuma decisão."""
    be = cfg.get("batch", 64) / cfg.get("train_every", 1)
    return 3.6 + 0.057 * be if method == "dqn" else (2.0 + 0.01 * be if method == "linear" else 1.0)


def shard_jobs(jobs, n_shards):
    """Atribuição estática (LPT: maior custo primeiro, para a fatia menos carregada) de (custo, método, cfg_id, semente) a fatias."""
    loads, owner = [0.0] * n_shards, {}
    for c, m, ci, s in sorted(jobs, key=lambda j: (-j[0], j[1], j[2], j[3])):
        k = int(np.argmin(loads))
        loads[k] += c
        owner[(m, ci, s)] = k
    return owner


def load_done(out):
    done = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    done[(r["method"], r["cfg_id"], r["seed"])] = r
    return done


def check_reuse(r, cfg, a, sc, inst, method, ci, seed):
    """Reaproveita uma linha existente só se for exatamente a mesma execução."""
    if r["cfg"] != cfg or r["budget"] != a.budget or r["val_digest"] != sc["val"].digest():
        raise SystemExit(f"[{inst.name}] {method} cfg {ci} seed {seed}: linha existente não corresponde à config/orçamento/cenários atuais")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULTS))
    ap.add_argument("--configs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--space", choices=["orig", "ext"], default="orig",
                    help="ext: pool estendido (8 originais reaproveitadas de tune_<inst>.jsonl + novas); grava tuned_ext_<inst>.json")
    ap.add_argument("--shard", default="0/1", help="i/N: executa só a fatia i de N das execuções pendentes (use com --fill-only)")
    ap.add_argument("--fill-only", action="store_true",
                    help="só executa/grava as execuções da fatia; NÃO escolhe a melhor config (rode depois sem --fill-only para escolher)")
    a = ap.parse_args()
    shard_i, shard_n = map(int, a.shard.split("/"))
    if shard_n > 1 and not a.fill_only:
        raise SystemExit("--shard i/N com N > 1 exige --fill-only (a escolha da melhor config é feita depois, sobre o pool completo)")
    inst, sc = RP.get_bundle(a.instance)
    out = ROOT / "runs" / f"tune_{inst.name}.jsonl"           # log único: as 8 originais ficam intactas, as novas são anexadas
    best_path = ROOT / "runs" / (f"tuned_ext_{inst.name}.json" if a.space == "ext" else f"tuned_{inst.name}.json")
    best = json.load(open(best_path)) if best_path.exists() else {}
    all_cfgs = {m: config_list(m, a.configs, a.space) for m in a.methods}

    def execute(method, ci, cfg, seed):
        t0 = time.time()
        info = run_one(inst, sc, method, cfg, seed, a.budget, a.eval_every)
        vf = float(np.mean([c["val"] for c in info["curve"][-2:]]))           # média dos 2 últimos pontos (reduz ruído)
        r = {"instance": inst.name, "method": method, "cfg_id": ci, "cfg": cfg, "seed": seed, "budget": a.budget,
             "space": "ext" if ci >= N_ORIG else "orig", "val_final": vf, "curve": info["curve"],
             "seconds": round(time.time() - t0, 1), "val_digest": sc["val"].digest()}
        RP.log(out, r)
        print(f"[{inst.name}] {method:<12} cfg {ci} seed {seed}: val {vf:8.3f} ({time.time()-t0:.0f}s)", flush=True)
        return r

    if a.fill_only:                                            # modo fatiado: só preenche o log (várias fatias em paralelo)
        jobs = [(est_cost(m, cfg), m, ci, s) for m in a.methods for ci, cfg in enumerate(all_cfgs[m]) for s in a.seeds]
        owner = shard_jobs(jobs, shard_n)
        mine = [j for j in sorted(jobs, key=lambda j: (-j[0], j[1], j[2], j[3])) if owner[j[1:]] == shard_i]
        print(f"[{inst.name}] fatia {shard_i}/{shard_n}: {len(mine)} de {len(jobs)} execuções (custo relativo {sum(j[0] for j in mine):.0f})", flush=True)
        for _, m, ci, s in mine:
            r = load_done(out).get((m, ci, s))                 # relê o log: outra fatia/cadeia pode ter feito esta execução
            if r is not None:
                check_reuse(r, all_cfgs[m][ci], a, sc, inst, m, ci, s)
                continue
            execute(m, ci, all_cfgs[m][ci], s)
        print(f"[{inst.name}] FILL-FIM fatia {shard_i}/{shard_n}", flush=True)
        return

    done = load_done(out)
    for method in a.methods:
        cfgs = all_cfgs[method]
        scores = []
        for ci, cfg in enumerate(cfgs):
            vals = []
            for seed in a.seeds:
                if (method, ci, seed) in done:                   # reaproveita, mas só se for exatamente a mesma execução
                    r = done[(method, ci, seed)]
                    check_reuse(r, cfg, a, sc, inst, method, ci, seed)
                    vals.append(r["val_final"])
                    continue
                vals.append(execute(method, ci, cfg, seed)["val_final"])
            scores.append(float(np.mean(vals)))
        b = int(np.argmax(scores))
        best[method] = {"cfg_id": b, "cfg": cfgs[b], "val_mean": scores[b], "all_val_means": scores, "n_configs": len(cfgs),
                        "seeds": a.seeds, "budget": a.budget, "space": a.space}
        json.dump(best, open(best_path, "w"), indent=1)
        print(f"[{inst.name}] {method}: melhor cfg {b} (val {scores[b]:.3f}; padrão {scores[0]:.3f}) -> {cfgs[b]}", flush=True)


if __name__ == "__main__":
    main()