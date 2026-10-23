"""
33_cmaes_heuristics.py — calibração EVOLUTIVA (CMA-ES) das heurísticas, no lugar da grade (DESVIOS.md 2026-10-22, P12b).

  python scripts/33_cmaes_heuristics.py --instance cb12wd --families greedy threshold --seeds 0 1 2 3 4 5 6 7 8 9 --evals 200 --shard 0/8

Para cada (instância, família, semente): CMA-ES (λ padrão, σ0 = 1 no espaço logit, x0 = centro da caixa) minimiza a demanda perdida média nos cenários de VALIDAÇÃO
(fixos: números aleatórios comuns ⇒ fitness determinístico), com `--evals` avaliações; a melhor solução por validação é avaliada UMA vez no teste e nos dias reais.
Uma linha por execução em runs/cmaes_<inst>.jsonl (retomável; fatias em paralelo). A unidade de análise é a execução (semente) do CMA-ES.

`--search random` (P14b, DESVIOS.md 2026-10-23): controle com busca ALEATÓRIA na mesma caixa e com o mesmo número de avaliações; grava runs/rsearch_<inst>.jsonl.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
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
from bsrp import env as E                                                    # noqa: E402
from bsrp import policies as P                                               # noqa: E402
from bsrp.evo import CMAES                                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("runproto", ROOT / "scripts" / "13_run_protocol.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

DIM = {"greedy": 3, "threshold": 5}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _lo_hi(z_lo, z_hi):
    lo, hi = 0.05 + 0.9 * z_lo, 0.05 + 0.9 * z_hi
    return lo, max(hi, min(lo + 0.05, 0.99))


def decode(family: str, x):
    """x ∈ R^k (logit) -> política. greedy: (lo, hi, d); threshold: (t, lo, hi, d, w). d ∈ {5, 10} por limiar em 0,5."""
    z = _sig(np.asarray(x, float))
    if family == "greedy":
        lo, hi = _lo_hi(z[0], z[1])
        return P.GreedyFamily(round(float(lo), 4), round(float(hi), 4), 10 if z[2] > 0.5 else 5)
    if family == "threshold":
        lo, hi = _lo_hi(z[1], z[2])
        return P.ThresholdFamily(round(float(0.3 + 0.4 * z[0]), 4), round(float(lo), 4), round(float(hi), 4), 10 if z[3] > 0.5 else 5, round(float(0.5 * z[4]), 4))
    raise ValueError(family)


def run_cma(env, val, family, seed, evals):
    """Um CMA-ES; devolve (x_melhor, política_melhor, val_melhor (reward), n_avaliações, curva [(avaliações, melhor val)])."""
    cma = CMAES(np.zeros(DIM[family]), 1.0, None, np.random.default_rng(np.random.SeedSequence([20261022, seed, DIM[family]])))
    best_x, best_f, n, curve = None, np.inf, 0, []
    while n < evals:
        X = cma.ask()
        f = np.array([-float(E.evaluate(env, decode(family, x), val).mean()) for x in X])      # minimizar demanda perdida
        cma.tell(X, f)
        n += len(X)
        if f.min() < best_f:
            best_f, best_x = float(f.min()), X[int(f.argmin())].copy()
        curve.append((n, -best_f))
    return best_x, decode(family, best_x), -best_f, n, curve


def run_random(env, val, family, seed, evals):
    """Controle do CMA-ES (P14b): busca ALEATÓRIA com o mesmo número de avaliações na mesma caixa (u ~ U(0,1)^k, x = logit(u), logo a decodificação é a mesma);
    devolve (x_melhor, política_melhor, val_melhor (reward), n_avaliações, curva [(avaliações, melhor val)] a cada 10 avaliações)."""
    rng = np.random.default_rng(np.random.SeedSequence([20261023, seed, DIM[family]]))
    best_x, best_f, curve = None, np.inf, []
    for n in range(1, evals + 1):
        u = rng.uniform(1e-6, 1 - 1e-6, DIM[family])
        x = np.log(u / (1.0 - u))
        f = -float(E.evaluate(env, decode(family, x), val).mean())
        if f < best_f:
            best_f, best_x = f, x
        if n % 10 == 0 or n == evals:
            curve.append((n, -best_f))
    return best_x, decode(family, best_x), -best_f, evals, curve


def done_keys(path):
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["family"], r["seed"]) for r in map(json.loads, (l for l in f if l.strip()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--families", nargs="+", default=["greedy", "threshold"], choices=list(DIM))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--search", choices=["cma", "random"], default="cma", help="random: controle do P14b (runs/rsearch_<inst>.jsonl)")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    inst, sc = RP.get_bundle(a.instance)
    env = E.BSRPEnv(inst)
    out = ROOT / "runs" / (f"cmaes_{inst.name}.jsonl" if a.search == "cma" else f"rsearch_{inst.name}.jsonl")
    done = done_keys(out)
    jobs = [(f, s) for f in a.families for s in a.seeds]
    mine = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
    print(f"[{inst.name}] {'CMA-ES' if a.search == 'cma' else 'busca aleatória'} {a.evals} avaliações; fatia {si}/{sn}: {len(mine)} de {len(jobs)} execuções", flush=True)
    git, dirty = RP.git_state()
    for fam, seed in mine:
        t0 = time.time()
        x, pol, val, n, curve = (run_cma if a.search == "cma" else run_random)(env, sc["val"], fam, seed, a.evals)
        r = E.evaluate(env, pol, sc["test"])
        rec = {"instance": inst.name, "family": fam, "seed": seed, "evals": n, "policy": pol.name, "x": [float(v) for v in x], "val_best": float(val),
               "test_mean": float(r.mean()), "test_rewards": r.round(1).tolist(), "curve": [(int(k), float(v)) for k, v in curve],
               "val_digest": sc["val"].digest(), "test_digest": sc["test"].digest(), "seconds": round(time.time() - t0, 1), "commit": git, "dirty": dirty}
        if "real_test" in sc:
            rr = E.evaluate(env, pol, sc["real_test"])
            rec.update({"real_test_mean": float(rr.mean()), "real_test_rewards": rr.round(1).tolist()})
        RP.log(out, rec)
        print(f"[{inst.name}] {fam} seed {seed}: val {val:8.3f} -> test {rec['test_mean']:8.3f} ({rec['seconds']:.0f}s) {pol.name}", flush=True)
    print(f"[{inst.name}] FIM fatia {si}/{sn}", flush=True)


if __name__ == "__main__":
    main()
