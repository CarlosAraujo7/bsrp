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


def primary_rows(runs, inst):
    """Execuções finais PRIMÁRIAS de uma escala (DESVIOS.md 2026-10-14): para cada método, o rerun estendido
    (`final_ext_<inst>.jsonl`) se a config escolhida no pool de 16 difere da original; senão, o final original."""
    runs = Path(runs)
    orig, ext = load_json(runs / f"tuned_{inst}.json"), load_json(runs / f"tuned_ext_{inst}.json")
    r_ext, r_orig = rows_of(runs / f"final_ext_{inst}.jsonl"), rows_of(runs / f"final_{inst}.jsonl")
    out = []
    for m in METHODS:
        changed = m in ext and m in orig and ext[m]["cfg_id"] != orig[m]["cfg_id"]
        src = r_ext if changed else r_orig
        for r in src:
            if r["method"] == m:
                out.append({**r, "p1_source": "ext" if changed else "orig"})
    return out

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
    last, cfgs = {}, {}
    for r in rows:                                   # execução repetida (mesma config e semente): vale a última
        if r["method"] == method:
            last[(r["cfg_id"], r["seed"])] = r["val_final"]
            cfgs[r["cfg_id"]] = r["cfg"]
    by = defaultdict(list)
    for (i, _), v in last.items():
        by[i].append(v)
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


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# P3 — sensibilidade do simulador (DESVIOS.md 2026-10-14)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
P3_VARIANTS = {
    "cb4wd": [("K = 0,5", "cb4wdx0.5"), ("K = 0,75", "cb4wdx0.75"), ("K = 1 (primário)", "cb4wd"), ("K = 1,5", "cb4wdx1.5"),
              ("fim de semana", "cb4we"), ("legado × 1,5", "legacy4x1.5"), ("legado × 2,5", "legacy4x2.5"),
              ("legado × 3,5", "legacy4"), ("legado × 5", "legacy4x5")],
    "cb12wd": [("K = 0,5", "cb12wdx0.5"), ("K = 0,75", "cb12wdx0.75"), ("K = 1 (primário)", "cb12wd"), ("K = 1,5", "cb12wdx1.5"),
               ("fim de semana", "cb12we")],
}
P3_SEEDS = 5
P3_LEARNERS = ("dqn", "linear", "tab_station", "tab_cluster")
P3_HEUR_FILES = ("heuristics_cb_wd.jsonl", "heuristics_cb_wd_g3.jsonl", "heuristics_sens_a.jsonl", "heuristics_sens_a_g3.jsonl",
                 "heuristics_sens_b.jsonl", "heuristics_sens_b_g3.jsonl")


def best_heuristic(H):
    """Melhor heurística calibrada, escolhida por VALIDAÇÃO (mesma regra de 14_analyze_runs.py)."""
    cal = {k: v for k, v in H.items() if k.endswith(("_cal", "_cal2", "_cal3"))}
    if not cal:
        return None
    return max(cal, key=lambda k: max((cal[k].get("val_table") or {0: -1e9}).values()))


def p3_data(runs, name, base):
    runs = Path(runs)
    if name == base:         # K = 1: resultados primários do P1, restritos às mesmas 5 sementes
        rows = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == name and r["seed"] < P3_SEEDS]
    else:
        rows = [r for r in rows_of(runs / f"sens_{name}.jsonl") if r["seed"] < P3_SEEDS]
    by = {m: {r["seed"]: r for r in rows if r["method"] == m} for m in P3_LEARNERS}
    H = {}
    for f in P3_HEUR_FILES:
        for r in rows_of(runs / f):
            if r["instance"] == name:
                H[r["method"]] = r
    return by, H


def p3_report(runs, bases=("cb4wd", "cb12wd")):
    L = ["# P3 — sensibilidade do simulador (gerado por `scripts/21_analyze_p1_p2.py --which p3`)\n",
         "> Desenho e critério fixados em `DESVIOS.md` (2026-10-14), **antes** de rodar. 5 sementes (0–4); aprendizes com a config escolhida no P1 na "
         "escala-base, **sem reajuste por variante**; a heurística é **recalibrada em validação em cada variante** (viés contra os aprendizes). "
         "Reward de TESTE (1000 cenários sintéticos novos por variante); maior = melhor. C1 = DQN − melhor heurística, diferença por semente "
         "(a heurística é determinística), IC95% t; 'significativo' = IC exclui 0. Análise exploratória: sem correção entre variantes. "
         "K = 1 usa as mesmas 5 primeiras sementes dos resultados primários.\n"]
    flips, total = [], 0
    for base in bases:
        L += [f"\n## Escala-base {base}\n",
              "| variante | n sem. | parada | melhor heurística (por val) | DQN [IC95%] | C1: DQN − heurística [IC95%] | redução vs heurística | linear | tab. estação | tab. cluster |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        real_rows = []
        for label, name in P3_VARIANTS[base]:
            by, H = p3_data(runs, name, base)
            bh = best_heuristic(H)
            if bh is None or len(by["dqn"]) < 2:
                L.append(f"| {label} (`{name}`) | {len(by['dqn'])} | (pendente) | | | | | | | |")
                continue
            total += 1
            h = H[bh]["test_mean"]
            stay = H["stay"]["test_mean"] if "stay" in H else float("nan")
            x = np.array([r["test_mean"] for _, r in sorted(by["dqn"].items())])
            d = x - h
            lo, hi = ci_t(d)
            red = (abs(h) - abs(x.mean())) / abs(h) if h else float("nan")
            means = {m: (np.mean([r["test_mean"] for r in by[m].values()]) if by[m] else float("nan")) for m in P3_LEARNERS}
            beaten = [m for m in ("linear", "tab_station", "tab_cluster") if by[m] and means[m] > h]
            c1_ok = bool(lo > 0)
            L.append(f"| {label} (`{name}`) | {len(x)} | {fmt(stay)} | {fmt(h)} ({bh}) | {fmt(x.mean())} [{fmt(ci_t(x)[0])}; {fmt(ci_t(x)[1])}] | "
                     f"{fmt(d.mean())} [{fmt(lo)}; {fmt(hi)}]{'' if c1_ok else ' ⚠'} | {(fmt(100 * red, 0) + '%') if abs(h) >= 0.5 else 'n/d (heurística ≈ 0)'} | "
                     + " | ".join(fmt(means[m]) + (" ⚠" if m in beaten else "") for m in ("linear", "tab_station", "tab_cluster")) + " |")
            if not c1_ok:
                if hi < 0:
                    flips.append(f"{name}: o DQN é SIGNIFICATIVAMENTE PIOR que a melhor heurística (DQN − heurística = {fmt(d.mean())} [{fmt(lo)}; {fmt(hi)}])")
                else:
                    flips.append(f"{name}: C1 NÃO é significativo (DQN − heurística = {fmt(d.mean())} [{fmt(lo)}; {fmt(hi)}])")
            if beaten:
                flips.append(f"{name}: {', '.join(beaten)} supera(m) a melhor heurística")
            if name.endswith("we") and "real_test_mean" in H.get(bh, {}) and all("real_test_mean" in r for r in by["dqn"].values()):
                xr = np.array([r["real_test_mean"] for _, r in sorted(by["dqn"].items())])
                real_rows.append(f"| {label} | {len(xr)} | {fmt(H['stay']['real_test_mean'])} | {fmt(H[bh]['real_test_mean'])} | {fmt(xr.mean())} | "
                                 f"{fmt((xr - H[bh]['real_test_mean']).mean())} |")
        if real_rows:
            L += [f"\n**Dias reais de fim de semana ({base})** (única variante em que o replay de dias reais é interpretável):\n",
                  "| variante | n sem. | parada | melhor heurística | DQN | DQN − heurística |", "|---|---|---|---|---|---|"] + real_rows
    L += ["\n### Onde a conclusão muda (⚠ nas tabelas)\n"]
    L += [f"- {s}" for s in flips] if flips else ["- Em nenhuma variante: C1 significativo e nenhum tabular/linear supera a heurística."]
    L.append(f"\nVariantes analisadas: {total}; com alguma mudança de conclusão: {len({s.split(':')[0] for s in flips})}.")
    return L

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# P4 — rollout, threshold e A2C adaptado de Seo et al. (DESVIOS.md 2026-10-14)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
P4_HEUR_FILES = ("heuristics_cb_wd.jsonl", "heuristics_cb_wd_g3.jsonl", "heuristics_cb_wd_threshold.jsonl",
                 "heuristics_cb_wd_threshold2_cb4wd.jsonl", "heuristics_cb_wd_threshold2_cb8wd.jsonl", "heuristics_cb_wd_threshold2_cb12wd.jsonl")
P4_FAMILIES = ("greedy_cal", "greedy_cal2", "greedy_cal3", "threshold_cal", "threshold_cal2")


def p4_report(runs, bases=("cb4wd", "cb8wd", "cb12wd")):
    runs = Path(runs)
    L = ["# P4 — rollout, threshold e A2C adaptado de Seo et al. 2022 (gerado por `scripts/21_analyze_p1_p2.py --which p4`)\n",
         "> Reward de TESTE (1000 cenários sintéticos) e dias reais; maior = melhor. Referência de heurística = a de maior reward em **validação** entre as "
         "gulosas (grades 1–3) e a `threshold` (grades 1–2). Rollout: futuros do modelo (nunca da demanda real); config de cabeçalho = a de maior "
         "reward em val. Contrastes com aprendizes: média por semente − constante (referência determinística) ou Welch (aprendiz × aprendiz). "
         "O A2C é uma **adaptação** (hiperparâmetros como publicados; entrada e ação do nosso simulador), não uma reprodução.\n"]
    for base in bases:
        H = {}
        for f in P4_HEUR_FILES:
            for r in rows_of(runs / f):
                if r["instance"] == base:
                    H[r["method"]] = r
        fams = {k: v for k, v in H.items() if k in P4_FAMILIES and v.get("val_table")}
        if not fams:
            continue
        ref = max(fams, key=lambda k: max(fams[k]["val_table"].values()))
        h, stay = H[ref]["test_mean"], H["stay"]["test_mean"]
        h_real = H[ref].get("real_test_mean")
        L += [f"\n## {base}\n",
              f"Referência por val: **{ref}** = `{H[ref]['policy']}` (teste {fmt(h)}; parada {fmt(stay)}). "
              "Famílias, melhor val → teste: " + "; ".join(f"{k}: {fmt(max(v['val_table'].values()))} → {fmt(v['test_mean'])}" for k, v in sorted(fams.items())) + "\n"]
        ro = sorted([r for f in sorted(runs.glob(f"rollout_{base}*.jsonl")) for r in rows_of(f)], key=lambda r: (r["h"], r["K"]))      # inclui rollout_<inst>_h3K64.jsonl
        dq = np.array([r["test_mean"] for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == base and r["method"] == "dqn"])
        L += ["| método | val | teste | redução vs referência | dias reais |", "|---|---|---|---|---|"]
        best_ro = max(ro, key=lambda r: r["val_mean"]) if ro else None
        for r in ro:
            star = " **(cabeçalho: melhor em val)**" if r is best_ro else ""
            L.append(f"| rollout h={r['h']}, K={r['K']}{star} | {fmt(r['val_mean'])} | {fmt(r['test_mean'])} | "
                     f"{fmt(100 * (abs(h) - abs(r['test_mean'])) / abs(h), 0)}% | {fmt(r.get('real_test_mean'))} |")
        for v, lab in (("published", "A2C Seo (publicado, γ = 0,9)"), ("gamma1", "A2C Seo (γ = 1,0)"), ("tuned", "A2C Seo (ajustado)")):
            a = [r for r in rows_of(runs / f"seo_a2c_{base}.jsonl") if r["variant"] == v]
            if len(a) >= 2:
                x = np.array([r["test_mean"] for r in a])
                lo, hi = ci_t(x)
                rr = [r["real_test_mean"] for r in a if "real_test_mean" in r]
                L.append(f"| {lab}, n = {len(x)} | {fmt(np.mean([r['curve'][-1]['val'] for r in a]))} | {fmt(x.mean())} [{fmt(lo)}; {fmt(hi)}] | "
                         f"{fmt(100 * (abs(h) - abs(x.mean())) / abs(h), 0)}% | {fmt(np.mean(rr)) if rr else '—'} |")
        if len(dq) > 1:
            lo, hi = ci_t(dq)
            L.append(f"| DQN (primário), n = {len(dq)} | — | {fmt(dq.mean())} [{fmt(lo)}; {fmt(hi)}] | "
                     f"{fmt(100 * (abs(h) - abs(dq.mean())) / abs(h), 0)}% | — |")
            if best_ro:
                d = dq - best_ro["test_mean"]
                lo, hi = ci_t(d)
                L.append(f"\nDQN − rollout de cabeçalho (h={best_ro['h']}, K={best_ro['K']}): {fmt(d.mean())} [{fmt(lo)}; {fmt(hi)}] "
                         f"(10/10 sementes melhores: {'sim' if np.all(d > 0) else 'não'}).")
            for v, lab in (("published", "A2C Seo (publicado)"), ("gamma1", "A2C Seo (γ = 1,0)"), ("tuned", "A2C Seo (ajustado)")):
                a = np.array([r["test_mean"] for r in rows_of(runs / f"seo_a2c_{base}.jsonl") if r["variant"] == v])
                if len(a) >= 2:
                    dd, (lo, hi), p = welch(dq, a)
                    L.append(f"\nDQN − {lab}: {fmt(dd)} [{fmt(lo)}; {fmt(hi)}], p(Welch) = {fmt(p, 4)}; "
                             f"{lab} − referência: {fmt((a - h).mean())} [{fmt(ci_t(a - h)[0])}; {fmt(ci_t(a - h)[1])}].")
    return L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["p1", "p2", "p3", "p4", "primary"], required=True,
                    help="primary: grava (em --out, jsonl) as execuções finais primárias das escalas de --instances")
    ap.add_argument("--instances", nargs="+", default=list(INSTANCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    if a.which == "primary":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(a.out, "w", encoding="utf-8") as f:
            for inst in a.instances:
                for r in primary_rows(a.runs_dir, inst):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        print(f"{n} execuções primárias gravadas em {a.out}")
        return
    lines = {"p1": p1_report, "p2": p2_report, "p3": p3_report, "p4": p4_report}[a.which](a.runs_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print(f"(tabela gravada em {a.out})")


if __name__ == "__main__":
    main()
