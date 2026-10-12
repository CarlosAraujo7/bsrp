"""
13_run_protocol.py — executor do protocolo (PROTOCOLO_RIGOR.md). Uma linha JSON por execução em runs/*.jsonl.

  python scripts/13_run_protocol.py scenarios --instances 4 8 12
  python scripts/13_run_protocol.py heuristics --instances 4 8 12 --out runs/heuristics.jsonl
  python scripts/13_run_protocol.py learners --instances 4 --methods tab_cluster tab_station linear dqn \
        --seeds 0 1 2 3 4 --budget 150000 --out runs/pilot_g1.jsonl

Cenários: val = 200, test = 1000 por instância, de SeedSequence própria (disjunta dos treinos); salvos em
scenarios/ com hash. Heurísticas: calibradas em val (grade), avaliadas uma vez em test.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def instance(n, scale=3.5):
    return E.legacy_instance(n, demand_scale=scale)


def scen_path(inst_name, kind):
    return SCEN_DIR / f"{kind}_{inst_name}.npz"


def make_scenarios(inst):
    """Gera (uma vez) e salva val/test. As sementes dependem só do nome da instância e do tipo: reproduzível."""
    SCEN_DIR.mkdir(exist_ok=True)
    out = {}
    for kind, n_ep, k in (("val", N_VAL, 1), ("test", N_TEST, 2)):
        p = scen_path(inst.name, kind)
        if p.exists():
            out[kind] = E.ScenarioSet.load(p)
            continue
        ss = np.random.SeedSequence([SCEN_SALT, k, sum(map(ord, inst.name))])
        s = E.sample_scenarios(inst, n_ep, np.random.default_rng(ss))
        s.save(p)
        out[kind] = s
    return out


def log(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def base_rec(inst, sc, kind):
    h, dirty = git_state()
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": h, "dirty": dirty, "instance": inst.name,
            "kind": kind, "val_digest": sc["val"].digest(), "test_digest": sc["test"].digest()}


def cmd_scenarios(a):
    for n in a.instances:
        inst = instance(n)
        sc = make_scenarios(inst)
        print(inst.name, "val", sc["val"].digest(), "test", sc["test"].digest())


def cmd_heuristics(a):
    for n in a.instances:
        inst = instance(n, a.scale)
        sc = make_scenarios(inst)
        env = E.BSRPEnv(inst)
        refs = [P.Stay(), P.RandomPolicy(np.random.default_rng(1)), P.LegacyGreedy(), P.LegacyThreshold(0.5)]
        cal = {}
        if a.only_greedy2:      # reexecução só da grade estendida da gulosa (DESVIOS.md)
            refs = []
            fams = (("greedy_cal2", P.greedy_grid2()),)
        else:
            fams = (("greedy_cal", P.greedy_grid()), ("threshold_cal", P.threshold_grid()))
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
                        "val_table": dict(cal[fam][1]) if fam in cal else None, "seconds": round(time.time() - t0, 1)})
            log(Path(a.out), rec)
            print(f"[{inst.name}] test {fam:<14} {r.mean():8.3f} ± {r.std(ddof=1):.2f}", flush=True)


def cmd_learners(a):
    for n in a.instances:
        inst = instance(n, a.scale)
        sc = make_scenarios(inst)
        for method in a.methods:
            for seed in a.seeds:
                agent = T.make_agent(method, inst, seed)
                info = T.train(inst, agent, seed, a.budget, sc["val"], a.eval_every)
                fe = T.final_eval(inst, agent, sc["test"])
                rec = base_rec(inst, sc, "learner")
                rec.update({"method": method, "seed": seed, "budget_steps": a.budget, "demand_scale": a.scale,
                            "eval_every": a.eval_every, **info,
                            **{k: v for k, v in fe.items() if k != "test_rewards"},
                            "test_rewards": fe["test_rewards"].round(1).tolist()})
                log(Path(a.out), rec)
                print(f"[{inst.name}] {method:<12} seed {seed}: test {fe['test_mean']:7.3f}  "
                      f"val(last) {info['curve'][-1]['val']:7.3f}  {info['train_seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scenarios", cmd_scenarios), ("heuristics", cmd_heuristics), ("learners", cmd_learners)):
        p = sub.add_parser(name)
        p.add_argument("--instances", type=int, nargs="+", default=[4])
        p.add_argument("--scale", type=float, default=3.5)
        p.add_argument("--out", default=str(ROOT / "runs" / f"{name}.jsonl"))
        if name == "heuristics":
            p.add_argument("--only-greedy2", action="store_true", help="só a grade estendida da gulosa")
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
