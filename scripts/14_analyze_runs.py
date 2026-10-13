"""
14_analyze_runs.py — análise estatística dos runs/*.jsonl conforme PROTOCOLO_RIGOR.md §6.

  python scripts/14_analyze_runs.py --learners runs/pilot_g1.jsonl --heuristics runs/heuristics_legacy_scale3.5.jsonl \
         --out tables/pilot_g1.md

Unidade de análise = semente de treino. Escolha do "melhor aprendido" e da "melhor heurística" por VALIDAÇÃO
(nunca por teste). Famílias de comparações com correção de Holm. Margem de equivalência (TOST): δ = 0,5.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DELTA_EQ = 0.5
RNG = np.random.default_rng(12345)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ci_t(x, level=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def ci_boot(x, B=10000):
    x = np.asarray(x, float)
    m = RNG.choice(x, size=(B, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def tost_welch(a, b, delta=DELTA_EQ):
    """TOST de equivalência (|média(a)-média(b)| < delta) com Welch. Retorna p_TOST (= max dos dois unilaterais)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    p1 = 1 - stats.t.cdf((d + delta) / se, df)      # H0: d <= -delta
    p2 = stats.t.cdf((d - delta) / se, df)          # H0: d >= +delta
    return max(p1, p2)


def n_needed(sd, delta, paired):
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    return int(np.ceil((1 if paired else 2) * z ** 2 * sd ** 2 / delta ** 2))


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace(".", ",")


def analyze(learners, heur, out):
    L = defaultdict(lambda: defaultdict(list))          # inst -> method -> [records]
    for r in learners:
        L[r["instance"]][r["method"]].append(r)
    H = defaultdict(dict)
    for r in heur:
        H[r["instance"]][r["method"]] = r
    lines = ["# Análise de runs (gerado por `scripts/14_analyze_runs.py`)\n",
             f"> Unidade = semente de treino. Escolha do melhor por **validação**. TOST δ = {DELTA_EQ}. Holm por família.\n"]
    for inst in sorted(L):
        lines.append(f"\n## {inst}\n")
        rows, seed_means, val_means, aucs = [], {}, {}, {}
        for m, recs in sorted(L[inst].items()):
            x = np.array([r["test_mean"] for r in recs])
            v = np.array([r["curve"][-1]["val"] for r in recs])
            auc = np.array([np.mean([c["val"] for c in r["curve"]]) for r in recs])
            seed_means[m], val_means[m], aucs[m] = x, v, auc
            lo, hi = ci_t(x)
            blo, bhi = ci_boot(x) if len(x) > 1 else (np.nan, np.nan)
            w_var = np.mean([r["test_std_scen"] ** 2 for r in recs]) / len(recs[0]["test_rewards"])
            b_var = x.var(ddof=1) if len(x) > 1 else np.nan
            icc = b_var / (b_var + w_var) if len(x) > 1 else np.nan
            rows.append([m, len(x), recs[0]["budget_steps"], fmt(x.mean()), fmt(x.std(ddof=1) if len(x) > 1 else np.nan),
                         f"[{fmt(lo)}; {fmt(hi)}]", f"[{fmt(blo)}; {fmt(bhi)}]", fmt(v.mean()), fmt(auc.mean()), fmt(icc, 3)])
        lines.append("### Aprendizes (reward de TESTE por semente; maior = melhor)\n")
        lines.append("| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]

        hrows = []
        for m, r in sorted(H[inst].items()):
            hrows.append([m, fmt(r["test_mean"]), fmt(r["test_std_scen"]), r.get("policy", "")])
        lines.append("\n### Referências sem aprendizado (teste, 1000 cenários compartilhados)\n")
        lines.append("| política | reward médio | DP por cenário | configuração |")
        lines.append("|---|---|---|---|")
        lines += ["| " + " | ".join(r) + " |" for r in hrows]

        # melhor aprendido e melhor heurística, ambos escolhidos em VALIDAÇÃO
        best_l = max(val_means, key=lambda m: val_means[m].mean())
        cal = {k: v for k, v in H[inst].items() if k.endswith(("_cal", "_cal2", "_cal3"))}
        best_h = None
        if cal:
            best_h = max(cal, key=lambda k: max((cal[k]["val_table"] or {0: -1e9}).values()))
        lines.append("\n### Comparações pré-definidas\n")
        pv, comp = [], []
        if best_h:
            hv = H[inst][best_h]["test_mean"]
            d = seed_means[best_l] - hv
            t, p = stats.ttest_1samp(d, 0.0)
            try:
                wp = stats.wilcoxon(d).pvalue
            except ValueError:
                wp = np.nan
            comp.append((f"C1: melhor aprendido ({best_l}, por val) − melhor heurística ({best_h}, por val)", d, p, wp,
                         float(np.mean(d > 0))))
            pv.append(p)
        pairs = [("tab_station", "tab_cluster", "C4: R1 − R0"), ("linear", "tab_station", "C5: R2 − R1"), ("dqn", "linear", "C6: R3 − R2")]
        eq_rows = []
        for a, b, lab in pairs:
            if a in seed_means and b in seed_means and len(seed_means[a]) > 1 and len(seed_means[b]) > 1:
                d = seed_means[a].mean() - seed_means[b].mean()
                t, p = stats.ttest_ind(seed_means[a], seed_means[b], equal_var=False)
                mw = stats.mannwhitneyu(seed_means[a], seed_means[b]).pvalue
                comp.append((lab + " (sementes independentes)", np.array([d]), p, mw, float(np.mean(seed_means[a][:, None] > seed_means[b][None, :]))))
                pv.append(p)
                eq_rows.append((lab, d, tost_welch(seed_means[a], seed_means[b])))
        adj = holm(pv) if pv else []
        lines.append("| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |")
        lines.append("|---|---|---|---|---|---|")
        for (lab, d, p, wp, ps), pa in zip(comp, adj):
            lines.append(f"| {lab} | {fmt(float(np.mean(d)))} | {fmt(p, 4)} | {fmt(pa, 4)} | {fmt(wp, 4)} | {fmt(ps)} |")
        if eq_rows:
            lines.append(f"\nEquivalência (TOST, δ = {DELTA_EQ}): " + "; ".join(f"{lab}: diff {fmt(d)}, p_TOST {fmt(p, 4)}" for lab, d, p in eq_rows))

        # --- dias REAIS de teste (hold-out), quando existem -----------------------------------------------------
        if any("real_test_mean" in r for recs in L[inst].values() for r in recs):
            lines.append("\n### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente\n")
            lines.append("| método | n sem. | média | DP entre sem. | IC95% t |")
            lines.append("|---|---|---|---|---|")
            real_means = {}
            for m, recs in sorted(L[inst].items()):
                xr = np.array([r["real_test_mean"] for r in recs if "real_test_mean" in r])
                real_means[m] = xr
                lo, hi = ci_t(xr)
                lines.append(f"| {m} | {len(xr)} | {fmt(xr.mean())} | {fmt(xr.std(ddof=1) if len(xr) > 1 else np.nan)} | [{fmt(lo)}; {fmt(hi)}] |")
            lines.append("")
            lines.append("| referência (sem aprendizado) | dias reais |")
            lines.append("|---|---|")
            for m, r in sorted(H[inst].items()):
                if "real_test_mean" in r:
                    lines.append(f"| {m} | {fmt(r['real_test_mean'])} |")
            if best_h and best_l in real_means and "real_test_mean" in H[inst][best_h]:
                d = real_means[best_l] - H[inst][best_h]["real_test_mean"]
                t, p = stats.ttest_1samp(d, 0.0)
                lines.append(f"\nC1r (dias reais): {best_l} − {best_h} (ambos escolhidos por val): diferença média {fmt(float(d.mean()))}, "
                             f"p(t) = {fmt(p, 4)}, fração de sementes melhores que a heurística = {fmt(float(np.mean(d > 0)))}. "
                             f"Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).")

        sds = {m: x.std(ddof=1) for m, x in seed_means.items() if len(x) > 1}
        if sds:
            lines.append("\n### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)\n")
            lines.append("| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |")
            lines.append("|---|---|---|---|---|---|")
            for m, sd in sorted(sds.items()):
                lines.append(f"| {m} | {fmt(sd)} | {n_needed(sd,1,True)} | {n_needed(sd,0.5,True)} | {n_needed(sd,1,False)} | {n_needed(sd,0.5,False)} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:       # console cp1252 no Windows; a tabela já foi gravada em UTF-8
        print(f"(tabela gravada em {out})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="+", required=True)
    ap.add_argument("--heuristics", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    analyze([r for p in a.learners for r in read(p)], [r for p in a.heuristics for r in read(p)], a.out)
