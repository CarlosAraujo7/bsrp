"""
21_analyze_p1_p2.py — análise do P1 (pool estendido de 16 configs) e da ablação do DQN (P2: C7–C9).

  python -X utf8 scripts/21_analyze_p1_p2.py --which p1 --out tables/p1_extended_search.md
  python -X utf8 scripts/21_analyze_p1_p2.py --which p2 --out tables/p2_ablation.md

P1: por (escala, método), config original × escolhida no pool de 16 (só VALIDAÇÃO), ganho em val, e quais hiperparâmetros da
escolhida estão no mínimo/máximo explorado no pool (critério de aceite do HANDOFF: ótimo no interior ou ganho marginal
documentado). Onde houve rerun (`final_ext_<inst>.jsonl`), compara o teste original × estendido (sementes independentes, Welch).
P2: diferença PAREADA por semente (DQN completo − variante); IC95% t, p, Holm sobre C7–C9 (noreplay, notarget, nodouble),
TOST (δ = 0,5). `vanilla` é exploratória e fica fora da família de Holm. Unidade de análise = semente de treino.
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze14", ROOT / "scripts" / "14_analyze_runs.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
fmt, ci_t, holm, read, DELTA = A.fmt, A.ci_t, A.holm, A.read, A.DELTA_EQ

INSTANCES = ("cb4wd", "cb8wd", "cb12wd")
METHODS = ("dqn", "linear", "tab_station", "tab_cluster", "pg")
CONTINUOUS = {"lr", "lr_actor", "lr_critic"}
FAMILY = (("noreplay", "C7: replay"), ("notarget", "C8: rede-alvo"), ("nodouble", "C9: Double DQN"))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rows_of(p):
    return read(p) if Path(p).exists() else []


def welch(a, b):
    """diferença de médias (a − b), IC95% de Welch e p (t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = se ** 4 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return d, (d - h, d + h), stats.ttest_ind(a, b, equal_var=False).pvalue


def tost_paired(d, delta=DELTA):
    """TOST de equivalência para a diferença pareada (|média| < delta). Retorna p_TOST."""
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    p1 = 1 - stats.t.cdf((d.mean() + delta) / se, len(d) - 1)
    p2 = stats.t.cdf((d.mean() - delta) / se, len(d) - 1)
    return max(p1, p2)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# P1
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def pool_stats(rows, method):
    """cfg_id -> (cfg, val médio sobre as sementes de ajuste, n sementes)."""
    by, cfgs = defaultdict(list), {}
    for r in rows:
        if r["method"] == method:
            by[r["cfg_id"]].append(r["val_final"])
            cfgs[r["cfg_id"]] = r["cfg"]
    return {i: (cfgs[i], float(np.mean(v)), len(v)) for i, v in by.items()}


def border_flags(cfg, pool):
    """hiperparâmetros (não contínuos) da config que valem o mínimo/máximo explorado no pool."""
    out = []
    for k, v in cfg.items():
        if k in CONTINUOUS:
            continue
        vals = [c[k] for c, _, _ in pool.values()]
        lo, hi = min(vals), max(vals)
        if lo != hi and v == hi:
            out.append(f"{k}={v} (máx.)")
        elif lo != hi and v == lo:
            out.append(f"{k}={v} (mín.)")
    return out


def p1_report(runs):
    runs = Path(runs)
    L = ["# P1 — busca de hiperparâmetros estendida (gerado por `scripts/21_analyze_p1_p2.py --which p1`)\n",
         "> Pool de 16 configs por método (8 originais + 8 novas; DESVIOS.md 2026-10-14). Escolha por **validação** (2 sementes de ajuste × 200 mil passos); "
         "o teste só entra nos reruns. 'Limite' = o hiperparâmetro da escolhida vale o mínimo/máximo explorado no pool de 16 "
         "(lr/lr_actor/lr_critic, contínuos, não entram).\n"]
    sel, tests, marg = [], [], []
    for inst in INSTANCES:
        p_o, p_e, p_t = runs / f"tuned_{inst}.json", runs / f"tuned_ext_{inst}.json", runs / f"tune_{inst}.jsonl"
        if not (p_o.exists() and p_e.exists() and p_t.exists()):
            continue
        orig, ext, trows = load_json(p_o), load_json(p_e), read(p_t)
        f_o = [r for r in rows_of(runs / f"final_{inst}.jsonl")]
        f_e = [r for r in rows_of(runs / f"final_ext_{inst}.jsonl")]
        for m in METHODS:
            if m not in ext or m not in orig:
                continue
            pool = pool_stats(trows, m)
            complete = len(pool) == ext[m]["n_configs"] and all(n == len(ext[m]["seeds"]) for _, _, n in pool.values())
            o, e = orig[m], ext[m]
            changed = e["cfg_id"] != o["cfg_id"]
            fl = border_flags(e["cfg"], pool)
            sel.append(f"| {inst} | {m} | {o['cfg_id']} ({fmt(o['val_mean'])}) | {e['cfg_id']} ({fmt(e['val_mean'])}) | "
                       f"{fmt(e['val_mean'] - o['val_mean'])} | {'sim' if changed else 'não (original mantida)'} | "
                       f"{', '.join(fl) if fl else 'nenhum (interior)'}{'' if complete else ' — pool incompleto'} |")
            xo = np.array([r["test_mean"] for r in f_o if r["method"] == m])
            xe = np.array([r["test_mean"] for r in f_e if r["method"] == m])
            if changed and len(xo) > 1 and len(xe) > 1:
                d, (lo, hi), p = welch(xe, xo)
                ro = np.array([r["real_test_mean"] for r in f_o if r["method"] == m and "real_test_mean" in r])
                re_ = np.array([r["real_test_mean"] for r in f_e if r["method"] == m and "real_test_mean" in r])
                dr = welch(re_, ro) if len(ro) > 1 and len(re_) > 1 else None
                tests.append(f"| {inst} | {m} | {len(xo)} / {len(xe)} | {fmt(xo.mean())} | {fmt(xe.mean())} | {fmt(d)} [{fmt(lo)}; {fmt(hi)}] | "
                             f"{fmt(p, 4)} | " + (f"{fmt(dr[0])} [{fmt(dr[1][0])}; {fmt(dr[1][1])}]" if dr else "—") + " |")
            if m == "dqn":
                for k in e["cfg"]:
                    if k in CONTINUOUS:
                        continue
                    lv = defaultdict(list)
                    for c, v, _ in pool.values():
                        lv[c[k]].append(v)
                    marg.append(f"| {inst} | {k} | " + "; ".join(f"{x}: {fmt(float(np.mean(v)), 1)} (n={len(v)})" for x, v in sorted(lv.items())) + " |")
    if not sel:
        L.append("(ainda não há `tuned_ext_<inst>.json` — o ajuste estendido não terminou)\n")
    else:
        L += ["### Configuração escolhida (validação)\n",
              "| escala | método | original: cfg (val) | pool de 16: cfg (val) | ganho em val | escolha mudou? | hiperparâmetros da escolhida no limite do pool |",
              "|---|---|---|---|---|---|---|"] + sel
    if tests:
        L += ["\n### Reruns: teste original × estendido (reward de teste por semente; sementes independentes, Welch)\n",
              "| escala | método | n sem. (orig / ext) | média original | média estendida | ext − orig [IC95%] | p (t) | dias reais: ext − orig [IC95%] |",
              "|---|---|---|---|---|---|---|---|"] + tests
    if marg:
        L += ["\n### Exploratório: val médio do DQN por nível de cada hiperparâmetro (16 configs; efeitos confundidos, só descritivo)\n",
              "| escala | hiperparâmetro | nível: val médio (n configs) |", "|---|---|---|"] + marg
    return L


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# P2
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def reference_dqn(runs, inst):
    """DQN completo de referência: rerun estendido se existir, senão o final original. Retorna (fonte, linhas por semente)."""
    ext = [r for r in rows_of(runs / f"final_ext_{inst}.jsonl") if r["method"] == "dqn"]
    if ext:
        return f"final_ext_{inst}.jsonl", {r["seed"]: r for r in ext}
    return f"final_{inst}.jsonl", {r["seed"]: r for r in rows_of(runs / f"final_{inst}.jsonl") if r["method"] == "dqn"}


def reading(d_mean, p_holm, p_tost):
    if p_holm < 0.05 and d_mean > 0:
        return "necessária (completo melhor)"
    if p_holm < 0.05 and d_mean < 0:
        return "prejudica (variante melhor)"
    if p_tost < 0.05:
        return f"equivalente (TOST, δ = {fmt(DELTA, 1)})"
    return "inconclusivo (não detectamos diferença)"


def p2_report(runs, instances=("cb12wd", "cb4wd")):
    runs = Path(runs)
    L = ["# P2 — ablação do DQN (gerado por `scripts/21_analyze_p1_p2.py --which p2`)\n",
         "> Diferença **pareada por semente** (mesma semente → mesmos pesos iniciais e mesmo ambiente de treino): Δ = DQN completo − variante, em reward de TESTE "
         "(maior = melhor). IC95% t e p (t) sobre a diferença por semente; Holm sobre C7–C9 por escala; TOST (δ = 0,5) sobre a diferença pareada. "
         "Leitura: 'necessária' só com p de Holm < 0,05 e Δ > 0. `vanilla` (as três removidas) é exploratória e fica fora da família. "
         "Lembrete de desenho (DESVIOS.md): `−target` ≡ `−target −double`; `−replay` faz 1 atualização por transição com lote 1.\n"]
    any_data = False
    for inst in instances:
        abl = rows_of(runs / f"ablation_{inst}.jsonl")
        if not abl:
            continue
        any_data = True
        src, ref = reference_dqn(runs, inst)
        by_v = defaultdict(dict)
        for r in abl:
            by_v[r["variant"]][r["seed"]] = r
        cfg_abl = {json.dumps(r["cfg"], sort_keys=True) for r in abl}
        cfg_ref = {json.dumps(r["cfg"], sort_keys=True) for r in ref.values()}
        L.append(f"\n## {inst}\n")
        L.append(f"Referência (DQN completo): `{src}`, {len(ref)} sementes; config-base da ablação: {len(cfg_abl)} config(s) distinta(s)"
                 + ("" if cfg_abl == cfg_ref else " — **ATENÇÃO: a config da ablação difere da do DQN de referência**") + ".\n")

        def contrast(v, key):
            seeds = sorted(set(by_v[v]) & set(ref))
            full = np.array([ref[s][key] for s in seeds if key in ref[s] and key in by_v[v][s]])
            var = np.array([by_v[v][s][key] for s in seeds if key in ref[s] and key in by_v[v][s]])
            return full, var

        rows, pv = [], []
        for v, lab in list(FAMILY) + [("vanilla", "exploratória: vanilla")]:
            if v not in by_v:
                continue
            full, var = contrast(v, "test_mean")
            if len(full) < 2:
                continue
            d = full - var
            lo, hi = ci_t(d)
            p = stats.ttest_1samp(d, 0.0).pvalue
            rows.append([v, lab, len(d), full.mean(), var.mean(), d.mean(), (lo, hi), p, tost_paired(d), float(np.mean(d > 0)), v != "vanilla"])
            if v != "vanilla":
                pv.append(p)
        adj = iter(holm(pv)) if pv else iter(())
        L += ["### Reward de teste (1000 cenários sintéticos)\n",
              "| variante | contraste | n sem. | completo | variante | Δ = completo − variante | IC95% t (pareado) | p (t) | p Holm | TOST p | P(completo > variante) | leitura |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, lab, n, mf, mv, dm, (lo, hi), p, pt, ps, in_fam in rows:
            ph = next(adj) if in_fam else None
            L.append(f"| {v} | {lab} | {n} | {fmt(mf)} | {fmt(mv)} | {fmt(dm)} | [{fmt(lo)}; {fmt(hi)}] | {fmt(p, 4)} | "
                     f"{fmt(ph, 4) if ph is not None else '—'} | {fmt(pt, 4)} | {fmt(ps)} | "
                     f"{reading(dm, ph, pt) if ph is not None else 'exploratória'} |")
        rrows = []
        for v, lab in list(FAMILY) + [("vanilla", "vanilla")]:
            if v not in by_v:
                continue
            full, var = contrast(v, "real_test_mean")
            if len(full) >= 2:
                d = full - var
                lo, hi = ci_t(d)
                rrows.append(f"| {v} | {len(d)} | {fmt(d.mean())} | [{fmt(lo)}; {fmt(hi)}] | {fmt(stats.ttest_1samp(d, 0.0).pvalue, 4)} |")
        if rrows:
            L += ["\n### Dias reais de teste (24 dias úteis; ocupação inicial desconhecida)\n",
                  "| variante | n sem. | Δ = completo − variante | IC95% t (pareado) | p (t) |", "|---|---|---|---|---|"] + rrows
    if not any_data:
        L.append("(ainda não há `runs/ablation_<inst>.jsonl`)\n")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["p1", "p2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    lines = p1_report(a.runs_dir) if a.which == "p1" else p2_report(a.runs_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print(f"(tabela gravada em {a.out})")


if __name__ == "__main__":
    main()
