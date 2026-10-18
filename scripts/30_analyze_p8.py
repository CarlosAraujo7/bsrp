"""
30_analyze_p8.py — análise do P8 (DESVIOS.md 2026-10-18): ancoragem e robustez da conclusão central.

  python -X utf8 scripts/30_analyze_p8.py --out tables/p8_anchoring.md

(a) limite clarividente: valor do planejador com conhecimento do futuro (bsrp/oracle.py) × parada, heurística, rollout e DQN nos mesmos cenários;
(b) DQN com 4× o orçamento (2 milhões de passos) × 500 mil passos × rollout;
(c) reavaliação das 4 melhores configs com 4 sementes de ajuste e sensibilidade do DQN com 10 sementes (P3c).
Convenções: reward ≤ 0 (maior é melhor); unidade de análise = semente de treino; IC95% t; Welch entre execuções independentes.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("an21", ROOT / "scripts" / "21_analyze_p1_p2.py")
AN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(AN)
rows_of, ci_t, welch, fmt = AN.rows_of, AN.ci_t, AN.welch, AN.fmt

BASES = ("cb4wd", "cb8wd", "cb12wd")
WIDTHS = (300, 1000)
NEAR_ZERO = 0.05           # clarividente ≥ −5% do valor de "parar" ⇒ limite ≈ "nenhuma demanda perdida" (DESVIOS 2026-10-18)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def reference_rows(runs, inst):
    """stay, heurística de referência (maior val entre as famílias, como em p4_report) e rollout de cabeçalho (maior val)."""
    runs = Path(runs)
    H = {}
    for f in AN.P4_HEUR_FILES:
        for r in rows_of(runs / f):
            if r["instance"] == inst:
                H[r["method"]] = r
    fams = {k: v for k, v in H.items() if k in AN.P4_FAMILIES and v.get("val_table")}
    ref = max(fams, key=lambda k: max(fams[k]["val_table"].values())) if fams else None
    ro = [r for f in sorted(runs.glob(f"rollout_{inst}*.jsonl")) for r in rows_of(f)]
    best_ro = max(ro, key=lambda r: r["val_mean"]) if ro else None
    return H.get("stay"), (H[ref] if ref else None), best_ro


# ──────────────────────────────────────────────────────────────────────────────
# (a) limite clarividente
# ──────────────────────────────────────────────────────────────────────────────
def oracle_values(runs, inst):
    """{(conjunto, idx): melhor valor entre as larguras} e {largura: {(conjunto, idx): valor}}; só índices presentes."""
    per_w = {}
    for W in WIDTHS:
        p = Path(runs) / f"oracle_{inst}_w{W}.jsonl"
        d = {}
        for r in rows_of(p):
            d[(r["set"], r["idx"])] = r["value"]       # execução repetida: vale a última
        per_w[W] = d
    best = {}
    for d in per_w.values():
        for k, v in d.items():
            best[k] = max(best.get(k, -np.inf), v)
    return best, per_w


def oracle_section(runs, bases=BASES):
    runs = Path(runs)
    L = ["## (a) Limite clarividente (P8a)\n",
         "> Planejador que **conhece a demanda inteira do cenário** (busca em feixe sobre as ações do simulador; `bsrp/oracle.py`). Seu valor é um "
         "limite **inferior** do ótimo clarividente; este, por sua vez, **não é atingível** por política que não vê o futuro. 'Ganho capturado' = "
         "(método − parada) / (clarividente − parada): contra um limite inferior do clarividente, é um limite **superior** da fração verdadeira. "
         "Reward de teste nos primeiros cenários sintéticos e nos 24 dias reais; maior = melhor.\n"]
    for inst in bases:
        best, per_w = oracle_values(runs, inst)
        stay, heur, ro = reference_rows(runs, inst)
        if not best or stay is None or heur is None or ro is None:
            L.append(f"\n### {inst}\n\n(pendente)\n")
            continue
        prim = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
        L += [f"\n### {inst}\n",
              "| conjunto | n cen. | parada | heurística | rollout | DQN [IC95%] | clarividente (W = 300) | (W = 1000) | clarividente (melhor) | ganho capturado: heurística | rollout | DQN | clarividente ≥ −5% de parada? |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, key in (("teste sintético", "test"), ("dias reais", "real")):
            idx = sorted(k for s, k in best if s == key)
            if not idx:
                continue
            need = [(key, k) for k in idx]
            if key == "test":
                n = max(i + 1 for i in idx)
                if len(idx) != n:                                   # índices 0..n−1 completos
                    continue
                sl = slice(0, n)
                rk = "test_rewards"
            else:
                sl, rk = slice(0, len(idx)), "real_test_rewards"
            if len(idx) < (24 if key == "real" else 1):
                continue
            m_stay = float(np.mean(stay[rk][sl]))
            m_heur = float(np.mean(heur[rk][sl]))
            m_ro = float(np.mean(ro[rk][sl]))
            dq = np.array([float(np.mean(r[rk][sl])) for r in prim])
            V = float(np.mean([best[k] for k in need]))
            cw = {W: (float(np.mean([per_w[W][k] for k in need])) if all(k in per_w[W] for k in need) else None) for W in WIDTHS}
            den = V - m_stay

            def frac(m):
                return f"{100 * (m - m_stay) / den:.0f}%" if den > 1e-9 else "n/d"

            lo, hi = ci_t(dq)
            frac_dq = f"{100 * (dq.mean() - m_stay) / den:.0f}% [{100 * (lo - m_stay) / den:.0f}; {100 * (hi - m_stay) / den:.0f}]" if den > 1e-9 else "n/d"
            near = V >= -NEAR_ZERO * abs(m_stay)
            L.append(f"| {name} | {len(need)} | {fmt(m_stay)} | {fmt(m_heur)} | {fmt(m_ro)} | {fmt(dq.mean())} [{fmt(lo)}; {fmt(hi)}] | "
                     f"{fmt(cw[300]) if cw[300] is not None else '—'} | {fmt(cw[1000]) if cw[1000] is not None else '—'} | {fmt(V)} | "
                     f"{frac(m_heur)} | {frac(m_ro)} | {frac_dq} | {'sim: limite ≈ nenhuma demanda perdida' if near else 'não'} |")
    L += ["\n**Leitura (fixada antes de ver os dados):** onde o clarividente chega a ≥ −5% do valor de 'parar', o limite é praticamente 'nenhuma demanda "
          "perdida' e **não** se afirma distância ao ótimo não antecipativo; a coluna 'ganho capturado' vira apenas a redução relativa à parada. "
          "A largura 1000 melhorando a 300 indica que o feixe ainda não convergiu (o valor é limite inferior do clarividente).\n"]
    return L


# ──────────────────────────────────────────────────────────────────────────────
# (b) orçamento 4×
# ──────────────────────────────────────────────────────────────────────────────
def budget_rows(runs, inst):
    return {r["seed"]: r for r in rows_of(Path(runs) / f"budget4x_{inst}.jsonl")}


def budget_section(runs, bases=BASES):
    runs = Path(runs)
    L = ["## (b) DQN com 4× o orçamento de treino (P8b)\n",
         "> Config do P1 de cada escala, **2 milhões de passos** (o ε-decay estica junto; sem reajuste), sementes 0–4. 'Δ orçamento' = DQN(2 M) − DQN(500 mil), "
         "Welch (execuções independentes; as 10 sementes de 500 mil passos). 'DQN(2 M) − rollout' = por semente contra o rollout de cabeçalho (constante), "
         "IC95% t. Reward de teste; maior = melhor.\n",
         "| escala | n (2 M) | DQN 500 mil [IC95%] | DQN 2 M [IC95%] | Δ orçamento [IC95% Welch] | rollout | DQN(2 M) − rollout [IC95%] | leitura (regra pré-fixada) |",
         "|---|---|---|---|---|---|---|---|"]
    notes = []
    for inst in bases:
        b4 = budget_rows(runs, inst)
        _, _, ro = reference_rows(runs, inst)
        prim = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
        if len(b4) < 2 or ro is None or not prim:
            L.append(f"| {inst} | {len(b4)} | (pendente) | | | | | |")
            continue
        x4 = np.array([r["test_mean"] for _, r in sorted(b4.items())])
        x1 = np.array([r["test_mean"] for r in prim])
        dw, (wl, wh), _ = welch(x4, x1)
        d = x4 - ro["test_mean"]
        lo, hi = ci_t(d)
        lo1, hi1 = ci_t(x1)
        lo4, hi4 = ci_t(x4)
        lim = wl > 0
        vs = "DQN(2 M) supera o rollout" if lo > 0 else ("o rollout continua melhor" if hi < 0 else "sem diferença detectável com o rollout")
        reading = ("o orçamento de 500 mil passos limita o DQN; " if lim else "sem ganho detectável com 4× o orçamento; ") + vs
        L.append(f"| {inst} | {len(x4)} | {fmt(x1.mean())} [{fmt(lo1)}; {fmt(hi1)}] | {fmt(x4.mean())} [{fmt(lo4)}; {fmt(hi4)}] | "
                 f"{fmt(dw)} [{fmt(wl)}; {fmt(wh)}] | {fmt(ro['test_mean'])} | {fmt(d.mean())} [{fmt(lo)}; {fmt(hi)}] | {reading} |")
        notes.append((inst, lim, lo, hi))
    return L + [""]


# ──────────────────────────────────────────────────────────────────────────────
# (c) reavaliação das top-4 e sensibilidade com 10 sementes
# ──────────────────────────────────────────────────────────────────────────────
def reseed_section(runs, bases=BASES):
    runs = Path(runs)
    L = ["## (c1) Reavaliação das 4 melhores configs do DQN com 4 sementes de ajuste (P8c)\n",
         "> As 4 configs de maior val médio (sementes 1000, 1001) recebem as sementes 1002 e 1003; escolha pela média das 4. Se a escolhida difere da do P1, "
         "rodam-se 10 sementes × 500 mil passos com ela e o resultado é reportado ao lado do primário (Welch).\n",
         "| escala | cfg do P1 | médias (4 sementes) das 4 configs | escolhida | resultado |", "|---|---|---|---|---|"]
    for inst in bases:
        p = runs / f"reseed_{inst}_decision.json"
        if not p.exists():
            L.append(f"| {inst} | | | | (pendente) |")
            continue
        d = load_json(p)
        means = ", ".join(f"nº {k}: {fmt(v)}" for k, v in sorted(d["means_all_seeds"].items(), key=lambda kv: -kv[1]))
        if not d["changed"]:
            res = "seleção confirmada"
        else:
            fr = rows_of(runs / f"final_reseed_{inst}.jsonl")
            prim = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
            if len(fr) < 2:
                res = "mudou; finais pendentes"
            else:
                x2 = np.array([r["test_mean"] for r in fr])
                x1 = np.array([r["test_mean"] for r in prim])
                dw, (wl, wh), _ = welch(x2, x1)
                res = f"mudou; finais (n = {len(x2)}): {fmt(x2.mean())} × primário {fmt(x1.mean())}; diferença {fmt(dw)} [{fmt(wl)}; {fmt(wh)}]"
        L.append(f"| {inst} | nº {d['p1_cfg_id']} | {means} | nº {d['selected_cfg_id']} | {res} |")
    L += ["\n## (c2) Sensibilidade do DQN com 10 sementes (P3c)\n",
          "> Variantes de 12 estações em que a conclusão mudou no P3. Sementes 0–9 do DQN com a config do P1 ('sem reajuste') e com a reajustada no P3b; "
          "referência = heurística recalibrada na variante. C1 = DQN − heurística, por semente, IC95% t. As tabelas de 5 sementes (P3/P3b) continuam sendo as pré-registradas.\n",
          "| variante | n (sem reajuste) | C1 sem reajuste [IC95%] | n (reajustado) | C1 reajustado [IC95%] | reajustado − sem reajuste [IC95% Welch] |", "|---|---|---|---|---|---|"]
    for base, lst in AN.P3B_VARIANTS.items():
        for label, name in lst:
            by, H = AN.p3_data(runs, name, base)
            bh = AN.best_heuristic(H)
            d = AN.p3b_data(runs, name, base)
            if bh is None or d is None:
                continue
            h = H[bh]["test_mean"]
            t, base_id, same, _ = d
            orig = {r["seed"]: r["test_mean"] for r in rows_of(runs / f"sens_{name}.jsonl") if r["method"] == "dqn" and r["seed"] < 10}
            rt_src = runs / (f"sens_{name}.jsonl" if same else f"sens_retuned_{name}.jsonl")
            rt = {r["seed"]: r["test_mean"] for r in rows_of(rt_src) if r["method"] == "dqn" and r["seed"] < 10}
            if len(orig) < 2 or len(rt) < 2:
                L.append(f"| {label} (`{name}`) | {len(orig)} | (pendente) | {len(rt)} | | |")
                continue
            xo = np.array([v for _, v in sorted(orig.items())])
            xr = np.array([v for _, v in sorted(rt.items())])
            lo_o, hi_o = ci_t(xo - h)
            lo_r, hi_r = ci_t(xr - h)
            dw, (wl, wh), _ = welch(xr, xo)
            L.append(f"| {label} (`{name}`) | {len(xo)} | {fmt((xo - h).mean())} [{fmt(lo_o)}; {fmt(hi_o)}] | {len(xr)}{' (= sem reajuste)' if same else ''} | "
                     f"{fmt((xr - h).mean())} [{fmt(lo_r)}; {fmt(hi_r)}] | " + ("—" if same else f"{fmt(dw)} [{fmt(wl)}; {fmt(wh)}]") + " |")
    return L + [""]


def p8_report(runs):
    L = ["# P8 — ancoragem e robustez da conclusão central (gerado por `scripts/30_analyze_p8.py`)\n",
         "> Desenho e leituras fixados em `DESVIOS.md` (2026-10-18), **antes** de rodar. Reward ≤ 0; maior = melhor; unidade de análise = semente de treino.\n"]
    return L + oracle_section(runs) + budget_section(runs) + reseed_section(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    lines = p8_report(a.runs_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print(f"(tabela gravada em {a.out})")


if __name__ == "__main__":
    main()
