"""
13_run_protocol.py — executor do protocolo (PROTOCOLO_RIGOR.md). Uma linha JSON por execução em runs/*.jsonl.

Instâncias (--instances):  4 | 8 | 12 | legacy4x1.5   -> perfis legados × escala (padrão 3,5)
                           cb4wd | cb8we | cb12wd ...    -> calibradas em dados reais (Citi Bike 2024, semanas de treino)

  python scripts/13_run_protocol.py scenarios  --instances cb4wd cb8wd cb12wd
  python scripts/13_run_protocol.py heuristics --instances cb4wd --out runs/heuristics_cb.jsonl
  python scripts/13_run_protocol.py learners   --instances cb4wd --methods dqn --seeds 0 1 2 --budget 300000 --out runs/x.jsonl

Cenários sintéticos: val = 200, test = 1000 por instância (SeedSequence própria, disjunta dos treinos), salvos em
scenarios/ com hash. Instâncias `cb*`: avaliação adicional em DIAS REAIS de teste (replay de fluxos observados nas
semanas de teste). Heurísticas: calibradas em val (sintético), avaliadas uma vez em test.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsrp import data as D                                                   # noqa: E402
from bsrp import env as E                                                    # noqa: E402
from bsrp import policies as P                                               # noqa: E402
from bsrp import train as T                                                  # noqa: E402

N_VAL, N_TEST = 200, 1000
SCEN_DIR = ROOT / "scenarios"
SCEN_SALT = 777001


def git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "bsrp", "scripts", "tests"], cwd=ROOT, text=True).strip())
        return h, dirty
    except Exception:
        return "unknown", True


_FLOWS = None


def get_bundle(spec: str, scale: float = 3.5):
    """spec -> (Instance, {'val','test'[, 'real_test']}). Cenários sintéticos salvos/recarregados com hash."""
    global _FLOWS
    spec = str(spec)
    real = None
    m = re.fullmatch(r"cb(\d+)(wd|we)(?:x([\d.]+))?", spec)          # calibrada em dados reais; xK = fator de sensibilidade
    if m:
        if _FLOWS is None:
            _FLOWS = D.Flows()
        n, dt, k = int(m[1]), (0 if m[2] == "wd" else 1), (float(m[3]) if m[3] else 1.0)
        inst = D.calibrated_instance(n, _FLOWS, dt, name=spec, demand_scale=k)
        real = D.real_scenarios(n, _FLOWS, "test", dt)
    else:
        mm = re.fullmatch(r"(?:legacy)?(\d+)(?:x([\d.]+))?", spec)    # legado: "4", "legacy4", "legacy4x1.5"
        if not mm:
            raise ValueError(f"instância desconhecida: {spec}")
        n, k = int(mm[1]), (float(mm[2]) if mm[2] else scale)
        inst = E.legacy_instance(n, demand_scale=k)
        if k != 3.5:
            inst = E.Instance(f"legacy{n}x{k:g}", inst.cap, inst.cluster, inst.lat, inst.lon, inst.lam_dep, inst.lam_arr, inst.station_names)
    SCEN_DIR.mkdir(exist_ok=True)
    sc = {}
    for kind, n_ep, k in (("val", N_VAL, 1), ("test", N_TEST, 2)):
        p = SCEN_DIR / f"{kind}_{inst.name}.npz"
        if p.exists():
            sc[kind] = E.ScenarioSet.load(p)
        else:
            ss = np.random.SeedSequence([SCEN_SALT, k, sum(map(ord, inst.name))])
            sc[kind] = E.sample_scenarios(inst, n_ep, np.random.default_rng(ss))
            sc[kind].save(p)
    if real is not None:
        sc["real_test"] = real
    return inst, sc


def log(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def base_rec(inst, sc, kind):
    h, dirty = git_state()
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": h, "dirty": dirty, "instance": inst.name,
           "kind": kind, "val_digest": sc["val"].digest(), "test_digest": sc["test"].digest()}
    if "real_test" in sc:
        rec["real_test_digest"] = sc["real_test"].digest()
        rec["real_test_days"] = sc["real_test"].meta.get("days")
    return rec


def real_eval(env, pol, sc):
    if "real_test" not in sc:
        return {}
    r = E.evaluate(env, pol, sc["real_test"])
    return {"real_test_mean": float(r.mean()), "real_test_std_scen": float(r.std(ddof=1)), "real_test_rewards": r.round(1).tolist()}


def cmd_scenarios(a):
    for spec in a.instances:
        inst, sc = get_bundle(spec, a.scale)
        print(inst.name, "val", sc["val"].digest(), "test", sc["test"].digest(),
              ("real_test %d dias" % len(sc["real_test"])) if "real_test" in sc else "")


def cmd_heuristics(a):
    for spec in a.instances:
        inst, sc = get_bundle(spec, a.scale)
        env = E.BSRPEnv(inst)
        refs = [P.Stay(), P.RandomPolicy(np.random.default_rng(1)), P.LegacyGreedy(), P.LegacyThreshold(0.5)]
        cal = {}
        if a.only_greedy2:      # reexecução só da grade estendida da gulosa (DESVIOS.md)
            refs = []
            fams = (("greedy_cal2", P.greedy_grid2()),)
        elif a.skip_threshold:
            fams = (("greedy_cal", P.greedy_grid()), ("greedy_cal2", P.greedy_grid2()))
        else:
            fams = (("greedy_cal", P.greedy_grid()), ("greedy_cal2", P.greedy_grid2()), ("threshold_cal", P.threshold_grid()))
        for fam, grid in fams:
            t0 = time.time()
            best, table = P.calibrate(env, grid, sc["val"])
            cal[fam] = (best, table)
            print(f"[{inst.name}] {fam}: melhor em val = {best.name}  ({time.time()-t0:.0f}s, {len(grid)} configs)", flush=True)
        labeled = [(p.name, p) for p in refs] + [(f, cal[f][0]) for f in cal]
        for fam, pol in labeled:
            t0 = time.time()
            r = E.evaluate(env, pol, sc["test"])
            rec = base_rec(inst, sc, "heuristic")
            rec.update({"method": fam, "policy": pol.name, "demand_scale": a.scale, "test_mean": float(r.mean()),
                        "test_std_scen": float(r.std(ddof=1)), "test_rewards": r.round(1).tolist(),
                        "val_table": dict(cal[fam][1]) if fam in cal else None, **real_eval(env, pol, sc),
                        "seconds": round(time.time() - t0, 1)})
            log(Path(a.out), rec)
            print(f"[{inst.name}] test {fam:<18} {r.mean():8.3f} ± {r.std(ddof=1):.2f}"
                  + (f"   real-test {rec['real_test_mean']:8.3f}" if "real_test_mean" in rec else ""), flush=True)


def cmd_learners(a):
    for spec in a.instances:
        inst, sc = get_bundle(spec, a.scale)
        env = E.BSRPEnv(inst)
        for method in a.methods:
            for seed in a.seeds:
                agent = T.make_agent(method, inst, seed)
                info = T.train(inst, agent, seed, a.budget, sc["val"], a.eval_every)
                fe = T.final_eval(inst, agent, sc["test"])
                rec = base_rec(inst, sc, "learner")
                rec.update({"method": method, "seed": seed, "budget_steps": a.budget, "demand_scale": a.scale,
                            "eval_every": a.eval_every, **info,
                            **{k: v for k, v in fe.items() if k != "test_rewards"},
                            "test_rewards": fe["test_rewards"].round(1).tolist(), **real_eval(env, agent, sc)})
                log(Path(a.out), rec)
                print(f"[{inst.name}] {method:<12} seed {seed}: test {fe['test_mean']:7.3f}  "
                      f"val(last) {info['curve'][-1]['val']:7.3f}"
                      + (f"  real-test {rec['real_test_mean']:7.3f}" if "real_test_mean" in rec else "")
                      + f"  {info['train_seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scenarios", cmd_scenarios), ("heuristics", cmd_heuristics), ("learners", cmd_learners)):
        p = sub.add_parser(name)
        p.add_argument("--instances", nargs="+", default=["4"])
        p.add_argument("--scale", type=float, default=3.5)
        p.add_argument("--out", default=str(ROOT / "runs" / f"{name}.jsonl"))
        if name == "heuristics":
            p.add_argument("--only-greedy2", action="store_true", help="só a grade estendida da gulosa")
            p.add_argument("--skip-threshold", action="store_true")
        if name == "learners":
            p.add_argument("--methods", nargs="+", default=["tab_cluster", "tab_station", "linear", "dqn"])
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
            p.add_argument("--budget", type=int, default=150_000)
            p.add_argument("--eval-every", type=int, default=15_000)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
