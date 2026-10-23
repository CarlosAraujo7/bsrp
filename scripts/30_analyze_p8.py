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
    for f in list(AN.P4_HEUR_FILES) + sorted(q.name for q in runs.glob(f"heuristics_scale_{inst}.jsonl")):
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
    for it in p3c_data(runs):
        if "pending" in it:
            L.append(f"| {it['label']} (`{it['name']}`) | {it['n_orig']} | (pendente) | {it['n_rt']} | | |")
            continue
        L.append(f"| {it['label']} (`{it['name']}`) | {it['n_orig']} | {fmt(it['c1_orig'])} [{fmt(it['c1_orig_lo'])}; {fmt(it['c1_orig_hi'])}] | "
                 f"{it['n_rt']}{' (= sem reajuste)' if it['same'] else ''} | {fmt(it['c1_rt'])} [{fmt(it['c1_rt_lo'])}; {fmt(it['c1_rt_hi'])}] | "
                 + ("—" if it["same"] else f"{fmt(it['delta'])} [{fmt(it['delta_lo'])}; {fmt(it['delta_hi'])}]") + " |")
    return L + [""]


def p3c_data(runs):
    """P3c: DQN com até 10 sementes nas variantes de P3B_VARIANTS, com a config do P1 e com a reajustada (mesma heurística recalibrada)."""
    runs = Path(runs)
    out = []
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
                out.append({"label": label, "name": name, "n_orig": len(orig), "n_rt": len(rt), "pending": True})
                continue
            xo = np.array([v for _, v in sorted(orig.items())])
            xr = np.array([v for _, v in sorted(rt.items())])
            lo_o, hi_o = ci_t(xo - h)
            lo_r, hi_r = ci_t(xr - h)
            dw, (wl, wh), _ = welch(xr, xo)
            out.append({"label": label, "name": name, "same": bool(same), "n_orig": len(xo), "n_rt": len(xr), "heur": h,
                        "c1_orig": float((xo - h).mean()), "c1_orig_lo": float(lo_o), "c1_orig_hi": float(hi_o),
                        "c1_rt": float((xr - h).mean()), "c1_rt_lo": float(lo_r), "c1_rt_hi": float(hi_r),
                        "delta": float(dw), "delta_lo": float(wl), "delta_hi": float(wh)})
    return out


def reseed_data(runs, bases=BASES):
    """Resultado da reavaliação com 4 sementes por escala: mudou? e, se mudou e há finais, a diferença (reseed − primário, Welch)."""
    runs = Path(runs)
    out = {}
    for inst in bases:
        p = runs / f"reseed_{inst}_decision.json"
        if not p.exists():
            continue
        d = load_json(p)
        rec = {"changed": bool(d["changed"]), "p1": d["p1_cfg_id"], "selected": d["selected_cfg_id"]}
        if d["changed"]:
            fr = rows_of(runs / f"final_reseed_{inst}.jsonl")
            prim = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
            if len(fr) >= 2 and prim:
                x2, x1 = np.array([r["test_mean"] for r in fr]), np.array([r["test_mean"] for r in prim])
                dw, (wl, wh), _ = welch(x2, x1)
                rec.update({"n": len(x2), "diff": float(dw), "lo": float(wl), "hi": float(wh)})
        out[inst] = rec
    return out


def p8_data(runs, bases=BASES, n_test=200):
    """Números do P8 em forma de dicionário (para 25_make_report.py): demanda perdida POSITIVA (= −reward), por escala.
    'oracle': parada, heurística, rollout, DQN [IC], clarividente (melhor largura e por largura) nos `n_test` primeiros cenários de teste
    e a fração do ganho até o clarividente; 'budget': DQN 500 mil × 2 M × rollout; só entram escalas com dados suficientes."""
    runs = Path(runs)
    out = {"oracle": {}, "budget": {}}
    for inst in bases:
        stay, heur, ro = reference_rows(runs, inst)
        prim = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
        best, per_w = oracle_values(runs, inst)
        if stay is not None and heur is not None and ro is not None and prim and all(("test", k) in best for k in range(n_test)):
            sl = slice(0, n_test)
            m_stay, m_heur, m_ro = (float(np.mean(r["test_rewards"][sl])) for r in (stay, heur, ro))
            dq = np.array([float(np.mean(r["test_rewards"][sl])) for r in prim])
            V = float(np.mean([best[("test", k)] for k in range(n_test)]))
            cw = {W: float(np.mean([per_w[W][("test", k)] for k in range(n_test)])) for W in WIDTHS if all(("test", k) in per_w[W] for k in range(n_test))}
            den = V - m_stay
            lo, hi = ci_t(dq)
            out["oracle"][inst] = {"n": n_test, "stay": -m_stay, "heur": -m_heur, "rollout": -m_ro, "dqn": -float(dq.mean()), "clair": -V,
                                   "clair_w": {W: -v for W, v in cw.items()}, "near_zero": bool(V >= -NEAR_ZERO * abs(m_stay)),
                                   "frac_heur": (m_heur - m_stay) / den, "frac_rollout": (m_ro - m_stay) / den, "frac_dqn": (float(dq.mean()) - m_stay) / den,
                                   "frac_dqn_lo": (lo - m_stay) / den, "frac_dqn_hi": (hi - m_stay) / den}
        b4 = budget_rows(runs, inst)
        if len(b4) >= 2 and ro is not None and prim:
            x4 = np.array([r["test_mean"] for _, r in sorted(b4.items())])
            x1 = np.array([r["test_mean"] for r in prim])
            dw, (wl, wh), _ = welch(x4, x1)
            d = x4 - ro["test_mean"]
            dlo, dhi = ci_t(d)
            lo4, hi4 = ci_t(x4)
            out["budget"][inst] = {"n": len(x4), "dqn_500k": -float(x1.mean()), "dqn_2m": -float(x4.mean()), "dqn_2m_lo": -float(hi4), "dqn_2m_hi": -float(lo4),
                                   "delta": float(dw), "delta_lo": float(wl), "delta_hi": float(wh), "rollout": -float(ro["test_mean"]),
                                   "vs_roll": float(d.mean()), "vs_roll_lo": float(dlo), "vs_roll_hi": float(dhi),
                                   "gap_500k": float(x1.mean() - ro["test_mean"])}
    return out


SCALES = ("cb4wd", "cb8wd", "cb12wd", "cb20wd", "cb30wd")
N_ST = {"cb4wd": 4, "cb8wd": 8, "cb12wd": 12, "cb20wd": 20, "cb30wd": 30}


def scale_rollout(runs, inst, heur_row, rows500k, rows2m):
    """P9c (DESVIOS.md 2026-10-19): rollout (h, K) fixos em cb20wd/cb30wd, por episódio (runs/rollout_scale_<inst>.jsonl), nos n primeiros cenários de teste;
    heurística e DQN comparados NOS MESMOS n cenários. None se faltar algum episódio."""
    p = Path(runs) / f"rollout_scale_{inst}.jsonl"
    if not p.exists():
        return None
    last = {}
    for r in rows_of(p):
        last[(r["set"], r["idx"])] = r["reward"]
    n_test = max((k for s_, k in last if s_ == "test"), default=-1) + 1
    n_real = max((k for s_, k in last if s_ == "real_test"), default=-1) + 1
    if n_test < 2 or any(("test", k) not in last for k in range(n_test)) or n_real != 24 or any(("real_test", k) not in last for k in range(n_real)):
        return None
    roll = np.array([last[("test", k)] for k in range(n_test)])
    hs = np.array(heur_row["test_rewards"][:n_test])
    m_roll, m_heur = float(roll.mean()), float(hs.mean())
    out = {"n": n_test, "mean": -m_roll, "red": 100 * (m_heur - m_roll) / m_heur, "heur_sub": -m_heur,
           "real": -float(np.mean([last[("real_test", k)] for k in range(n_real)])), "subset": True}
    for tag, rows in (("500k", rows500k), ("2m", rows2m)):
        if len(rows) >= 2:
            x = np.array([float(np.mean(r["test_rewards"][:n_test])) for r in rows])
            d = x - m_roll                                   # reward: positivo favorece o DQN
            lo, hi = ci_t(d)
            out[tag] = {"dqn_red": float(100 * (m_heur - x.mean()) / m_heur), "c": float(d.mean()), "c_lo": float(lo), "c_hi": float(hi)}
    return out


def scale_data(runs, scales=SCALES):
    """P9 (DESVIOS.md 2026-10-19): DQN × heurística calibrada ao longo da escala (4 → 30 estações), com 500 mil e 2 milhões de passos.
    Demanda perdida POSITIVA. 4/8/12: 500 mil = 10 sementes do experimento principal; 2 M = budget4x (5 sementes). 20/30: scale_/scale2m_ (5 sementes).
    Rollout só existe em 4/8/12. Só entram escalas com a referência e ≥ 2 sementes."""
    runs = Path(runs)
    out = {}
    for inst in scales:
        H = {}
        for f in list(AN.P4_HEUR_FILES) + sorted(p.name for p in runs.glob(f"heuristics_scale_{inst}.jsonl")):
            for r in rows_of(runs / f):
                if r["instance"] == inst:
                    H[r["method"]] = r
        fams = {k: v for k, v in H.items() if k in AN.P4_FAMILIES and v.get("val_table")}
        if not fams or "stay" not in H:
            continue
        ref = max(fams, key=lambda k: max(fams[k]["val_table"].values()))
        h, st = H[ref], H["stay"]
        if inst in ("cb4wd", "cb8wd", "cb12wd"):
            s5 = [r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"]
            s2 = list(budget_rows(runs, inst).values())
        else:
            s5 = [r for r in rows_of(runs / f"scale_{inst}.jsonl") if r["method"] == "dqn"]
            s2 = [r for r in rows_of(runs / f"scale2m_{inst}.jsonl") if r["method"] == "dqn"]
        rec = {"n_st": N_ST[inst], "stay": -st["test_mean"], "heur": -h["test_mean"], "heur_family": ref,
               "stay_real": -st.get("real_test_mean", np.nan), "heur_real": -h.get("real_test_mean", np.nan)}
        for tag, rows in (("500k", s5), ("2m", s2)):
            if len(rows) < 2:
                continue
            x = np.array([r["test_mean"] for r in rows])
            xr = np.array([r.get("real_test_mean", np.nan) for r in rows])
            lo, hi = ci_t(x)
            c1 = x - h["test_mean"]
            clo, chi = ci_t(c1)
            rec[tag] = {"n": len(x), "mean": -float(x.mean()), "lo": -float(hi), "hi": -float(lo), "c1": float(c1.mean()), "c1_lo": float(clo), "c1_hi": float(chi),
                        "red": float(100 * (h["test_mean"] - x.mean()) / h["test_mean"]), "red_stay": float(100 * (st["test_mean"] - x.mean()) / st["test_mean"]),
                        "red_lo": float(ci_t(100 * (h["test_mean"] - x) / h["test_mean"])[0]), "red_hi": float(ci_t(100 * (h["test_mean"] - x) / h["test_mean"])[1]),
                        "real": -float(np.nanmean(xr)) if np.isfinite(xr).any() else float("nan")}
        ro = [r for f in sorted(runs.glob(f"rollout_{inst}*.jsonl")) for r in rows_of(f)]
        if ro:
            b = max(ro, key=lambda r: r["val_mean"])
            rec["rollout"] = {"mean": -b["test_mean"], "red": float(100 * (h["test_mean"] - b["test_mean"]) / h["test_mean"]), "n": len(b["test_rewards"])}
        rs = scale_rollout(runs, inst, h, s5, s2)
        if rs:
            rec["rollout"] = rs
        out[inst] = rec
    return out


def scale_retuned_data(runs, inst="cb30wd"):
    """P9b (DESVIOS.md 2026-10-19): DQN reajustado em `inst` (pool de 16, escolha por val) × DQN com a config de 12 estações, 500 mil passos, 5 sementes.
    Se a escolhida for a mesma config do P1 de 12 estações, não há finais (resultado = scale_<inst>.jsonl). None se ainda não há dados."""
    runs = Path(runs)
    p_t, p12 = runs / f"tuned_ext_{inst}.json", runs / "tuned_ext_cb12wd.json"
    sd = scale_data(runs, scales=(inst,)).get(inst)
    if not (p_t.exists() and p12.exists() and sd and "500k" in sd):
        return None
    t, base_id = load_json(p_t)["dqn"], load_json(p12)["dqn"]["cfg_id"]
    same = t["cfg_id"] == base_id
    base = np.array([r["test_mean"] for r in rows_of(runs / f"scale_{inst}.jsonl") if r["method"] == "dqn"])
    rt_rows = rows_of(runs / ("scale_" + inst + ".jsonl" if same else f"scale_retuned_{inst}.jsonl"))
    x = np.array([r["test_mean"] for r in rt_rows if r["method"] == "dqn"])
    pool = AN.pool_stats(rows_of(runs / f"tune_{inst}.jsonl"), "dqn") if (runs / f"tune_{inst}.jsonl").exists() else {}
    rec = {"cfg_id": int(t["cfg_id"]), "same": bool(same), "val_mean": float(t["val_mean"]), "val_base": float(pool[base_id][1]) if base_id in pool else float("nan"),
           "n_configs": int(t["n_configs"]), "limits": AN.border_flags(t["cfg"], pool) if pool else [], "cfg": t["cfg"],
           "val_second": float(sorted([v[1] for v in pool.values()], reverse=True)[1]) if len(pool) > 1 else float("nan")}
    if len(x) >= 2:
        h = -sd["heur"]
        dw, (wl, wh), _ = welch(x, base)
        lo, hi = ci_t(x)
        c1 = x - h
        clo, chi = ci_t(c1)
        rec.update({"n": len(x), "mean": -float(x.mean()), "lo": -float(hi), "hi": -float(lo), "delta": float(dw), "delta_lo": float(wl), "delta_hi": float(wh),
                    "c1": float(c1.mean()), "c1_lo": float(clo), "c1_hi": float(chi), "red": float(100 * (h - x.mean()) / h)})
    return rec


def scale_section(runs):
    d = scale_data(runs)
    L = ["## (d) Escala maior: 20 e 30 estações (P9)\n",
         "> Instâncias `cb20wd` e `cb30wd` (12 legadas + as de maior volume; DESVIOS.md 2026-10-19). DQN com a config do P1 de 12 estações, **sem reajuste**, 5 sementes "
         "(4/8/12: as 10 sementes do experimento principal para 500 mil passos e as 5 de `budget4x` para 2 milhões). Referência = heurística calibrada em validação. "
         "Demanda perdida por dia no teste sintético (1000 cenários); C1 = DQN − heurística em reward (positivo favorece o DQN); redução = relativa à heurística. "
         "O rollout em 20/30 estações usa (h, K) = (3, 64) sem ajuste e só os n primeiros cenários de teste (heurística e DQN comparados nos mesmos n; coluna 'rollout'). As estações acrescentadas têm volume maior que as 12 legadas: o efeito do tamanho não está isolado.\n",
         "| escala | parada | heurística | DQN 500 mil [IC95%] | C1 500 mil [IC95%] | redução | DQN 2 M [IC95%] | C1 2 M [IC95%] | redução | rollout (redução) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for inst, r in d.items():
        cells = []
        for tag in ("500k", "2m"):
            if tag in r:
                t = r[tag]
                cells += [f"{fmt(t['mean'])} [{fmt(t['lo'])}; {fmt(t['hi'])}] (n = {t['n']})", f"{fmt(t['c1'])} [{fmt(t['c1_lo'])}; {fmt(t['c1_hi'])}]", f"{t['red']:.0f}%"]
            else:
                cells += ["(pendente)", "", ""]
        ro = (f"{fmt(r['rollout']['mean'])} ({r['rollout']['red']:.0f}%" + (f"; {r['rollout']['n']} cen." if r["rollout"].get("subset") else "") + ")") if "rollout" in r else "não rodado"
        L.append(f"| {r['n_st']} est. | {fmt(r['stay'])} | {fmt(r['heur'])} | " + " | ".join(cells) + f" | {ro} |")
    rt = scale_retuned_data(runs)
    if rt is not None:
        L += ["### (d2) DQN reajustado em 30 estações (P9b)\n"]
        if rt["same"]:
            L += [f"A reavaliação do pool de {rt['n_configs']} configs escolheu a mesma config do P1 de 12 estações (nº {rt['cfg_id']}; val {fmt(rt['val_mean'])} contra {fmt(rt['val_second'])} da segunda melhor): "
                  "seleção confirmada; o resultado é o de 500 mil passos acima. Leitura pré-fixada: a configuração herdada de 12 estações não explica a queda da vantagem (dentro deste espaço de busca).\n"]
        elif "n" in rt:
            c = rt["cfg"]
            L += [f"Config escolhida (nº {rt['cfg_id']}): lr {c['lr']:.2e}, sync {c['target_sync']}, every {c['train_every']}, batch {c['batch']}, buffer {c['buffer']}, ε-decay {c['eps_decay_frac']}; "
                  f"val {fmt(rt['val_mean'])} (a da config de 12 estações: {fmt(rt['val_base'])}); limites do pool: {', '.join(rt['limits']) or 'nenhum'}.\n",
                  "| DQN reajustado (500 mil passos) [IC95%] | n | reajustado − sem reajuste [IC95% Welch] | C1 reajustado [IC95%] | redução vs heurística |", "|---|---|---|---|---|",
                  f"| {fmt(rt['mean'])} [{fmt(rt['lo'])}; {fmt(rt['hi'])}] | {rt['n']} | {fmt(rt['delta'])} [{fmt(rt['delta_lo'])}; {fmt(rt['delta_hi'])}] | "
                  f"{fmt(rt['c1'])} [{fmt(rt['c1_lo'])}; {fmt(rt['c1_hi'])}] | {rt['red']:.0f}% |", ""]
        else:
            L += ["Reajuste escolhido; finais pendentes.\n"]
    return L + [""]


ARCH_TAGS = (("h128", "128–128", "Wide"), ("h256", "256–256", "Wider"))


def arch_data(runs, insts=("cb30wd", "cb12wd")):
    """P10 (DESVIOS.md 2026-10-20): DQN com rede mais larga (config nº 9 do P1 + hidden), 500 mil passos, × DQN 64–64 do mesmo orçamento, × rollout.
    Demanda perdida POSITIVA. 64–64: 5 sementes (`scale_cb30wd.jsonl`) em 30 est.; 10 sementes (principal) em 12 est. Rollout: subconjunto de 300 cenários em
    30 est. (heurística e DQN nos mesmos cenários), 1000 em 12 est."""
    runs = Path(runs)
    out = {}
    for inst in insts:
        sd = scale_data(runs, scales=(inst,)).get(inst)
        if not sd or "500k" not in sd:
            continue
        base_rows = ([r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"] if inst == "cb12wd"
                      else [r for r in rows_of(runs / f"scale_{inst}.jsonl") if r["method"] == "dqn"])
        base = np.array([r["test_mean"] for r in base_rows])
        h = -sd["heur"]
        ro = sd.get("rollout")
        n_sub = ro["n"] if ro and ro.get("subset") else None
        rec = {"n_st": sd["n_st"], "base_n": len(base), "base": -float(base.mean()), "heur": sd["heur"], "stay": sd["stay"],
               "rollout": ro["mean"] if ro else None, "rollout_subset": n_sub}
        for tag, label, nm in ARCH_TAGS:
            rows = [r for r in rows_of(runs / f"arch_{tag}_{inst}.jsonl") if r["method"] == "dqn"]
            if len(rows) < 2:
                continue
            x = np.array([r["test_mean"] for r in rows])
            dw, (wl, wh), _ = welch(x, base)
            lo, hi = ci_t(x)
            t = {"label": label, "n": len(x), "mean": -float(x.mean()), "lo": -float(hi), "hi": -float(lo), "delta": float(dw), "delta_lo": float(wl), "delta_hi": float(wh),
                 "red": float(100 * (h - x.mean()) / h), "train_seconds": float(np.mean([r.get("train_seconds", np.nan) for r in rows]))}
            if ro:
                if n_sub:
                    xs = np.array([float(np.mean(r["test_rewards"][:n_sub])) for r in rows])
                    rr = -ro["mean"]
                else:
                    xs, rr = x, -ro["mean"]
                d = xs - rr
                dlo, dhi = ci_t(d)
                t.update({"vs_roll": float(d.mean()), "vs_roll_lo": float(dlo), "vs_roll_hi": float(dhi)})
            rec[nm] = t
        out[inst] = rec
    return out


def arch_section(runs):
    d = arch_data(runs)
    L = ["## (e) Capacidade da rede: DQN com 128–128 e 256–256 (P10)\n",
         "> Config nº 9 do P1 (sem reajuste dos demais hiperparâmetros) com `hidden` mais largo, 500 mil passos, 5 sementes; 64–64 = mesmo orçamento (5 sementes em 30 est., 10 em 12 est.). "
         "'Δ vs 64–64' = Welch (larga − 64–64, em reward; positivo favorece a rede larga). 'DQN − rollout' em reward (negativo = o rollout continua melhor; em 30 est. no subconjunto de 300 cenários). "
         "Leitura pré-fixada em `DESVIOS.md` (2026-10-20).\n",
         "| escala | rede | n | demanda perdida [IC95%] | Δ vs 64–64 [IC95% Welch] | redução vs heurística | DQN − rollout [IC95%] | leitura |", "|---|---|---|---|---|---|---|---|"]
    for inst, r in d.items():
        L.append(f"| {r['n_st']} est. | 64–64 (n = {r['base_n']}) | {r['base_n']} | {fmt(r['base'])} | — | {100 * (r['heur'] - r['base']) / r['heur']:.0f}% | — | referência |")
        for tag, label, nm in ARCH_TAGS:
            t = r.get(nm)
            if not t:
                L.append(f"| {r['n_st']} est. | {label} | 0 | (pendente) | | | | |")
                continue
            reading = ("a capacidade da rede limita o DQN" if t["delta_lo"] > 0 else ("a rede maior piora (demais hiperparâmetros não reajustados)" if t["delta_hi"] < 0
                                                                                           else "não limita, dentro do testado"))
            vs = f"{fmt(t['vs_roll'])} [{fmt(t['vs_roll_lo'])}; {fmt(t['vs_roll_hi'])}]" if "vs_roll" in t else "—"
            L.append(f"| {r['n_st']} est. | {label} | {t['n']} | {fmt(t['mean'])} [{fmt(t['lo'])}; {fmt(t['hi'])}] | {fmt(t['delta'])} [{fmt(t['delta_lo'])}; {fmt(t['delta_hi'])}] | "
                     f"{t['red']:.0f}% | {vs} | {reading} |")
    return L + [""]


def dprl_data(runs, scales=SCALES):
    """P11 (DESVIOS.md 2026-10-20): adaptação do DPRL (inventário + roteamento) × DQN plano do mesmo orçamento (500 mil passos), × heurística, × rollout.
    Demanda perdida POSITIVA. Plano: 10 sementes (principal) em 4/8/12; 5 (`scale_<inst>.jsonl`) em 20/30. Rollout em 20/30: subconjunto de cenários."""
    runs = Path(runs)
    out = {}
    for inst in scales:
        rows = [r for r in rows_of(runs / f"dprl_{inst}.jsonl") if r["method"] == "dprl"]
        sd = scale_data(runs, scales=(inst,)).get(inst)
        if len(rows) < 2 or not sd or "500k" not in sd:
            continue
        flat_rows = ([r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"] if inst in ("cb4wd", "cb8wd", "cb12wd")
                     else [r for r in rows_of(runs / f"scale_{inst}.jsonl") if r["method"] == "dqn"])
        flat = np.array([r["test_mean"] for r in flat_rows])
        x = np.array([r["test_mean"] for r in rows])
        h = -sd["heur"]
        dw, (wl, wh), _ = welch(x, flat)
        lo, hi = ci_t(x)
        c1 = x - h
        clo, chi = ci_t(c1)
        rec = {"n_st": sd["n_st"], "n": len(x), "flat_n": len(flat), "flat": -float(flat.mean()), "mean": -float(x.mean()), "lo": -float(hi), "hi": -float(lo),
               "delta": float(dw), "delta_lo": float(wl), "delta_hi": float(wh), "c1": float(c1.mean()), "c1_lo": float(clo), "c1_hi": float(chi),
               "red": float(100 * (h - x.mean()) / h), "flat_red": float(100 * (h - flat.mean()) / h), "heur": sd["heur"],
               "train_seconds": float(np.mean([r.get("train_seconds", np.nan) for r in rows]))}
        ro = sd.get("rollout")
        if ro:
            n_sub = ro["n"] if ro.get("subset") else None
            xs = np.array([float(np.mean(r["test_rewards"][:n_sub])) for r in rows]) if n_sub else x
            d = xs - (-ro["mean"])
            vlo, vhi = ci_t(d)
            rec.update({"vs_roll": float(d.mean()), "vs_roll_lo": float(vlo), "vs_roll_hi": float(vhi), "rollout": ro["mean"], "rollout_red": ro["red"]})
        out[inst] = rec
    return out


def dprl_section(runs):
    d = dprl_data(runs)
    L = ["## (f) Adaptação do DPRL: duas políticas (inventário + roteamento) (P11)\n",
         "> Adaptação (não reprodução) de Liang et al. (2024): `bsrp/dprl.py`; config do DQN plano da escala, **sem ajuste próprio**, 500 mil passos, 5 sementes. 'Δ vs plano' = Welch "
         "(dual − DQN plano do mesmo orçamento, em reward; positivo favorece o dual). 'dual − rollout' em reward (negativo = o rollout continua melhor; em 20/30 est. nos primeiros cenários de teste). "
         "Leitura pré-fixada em `DESVIOS.md` (2026-10-20).\n",
         "| escala | DQN plano [n] | dual [IC95%] (n) | Δ vs plano [IC95% Welch] | redução vs heurística (plano → dual) | dual − rollout [IC95%] | leitura |", "|---|---|---|---|---|---|---|"]
    for inst, r in d.items():
        reading = ("a fatoração em duas políticas melhora o DQN" if r["delta_lo"] > 0 else ("a fatoração em duas políticas piora o DQN" if r["delta_hi"] < 0 else "empate"))
        vs = f"{fmt(r['vs_roll'])} [{fmt(r['vs_roll_lo'])}; {fmt(r['vs_roll_hi'])}]" if "vs_roll" in r else "—"
        L.append(f"| {r['n_st']} est. | {fmt(r['flat'])} [{r['flat_n']}] | {fmt(r['mean'])} [{fmt(r['lo'])}; {fmt(r['hi'])}] ({r['n']}) | {fmt(r['delta'])} [{fmt(r['delta_lo'])}; {fmt(r['delta_hi'])}] | "
                 f"{r['flat_red']:.0f}% → {r['red']:.0f}% | {vs} | {reading} |")
    if not d:
        L.append("| (pendente) | | | | | | |")
    return L + [""]


N_SUB = 300          # primeiros cenários de teste usados em TODAS as comparações do artigo do CEC (RHEA/rollout em 20/30 est. só têm 300)
CEC_METHODS = ("stay", "heur", "cma", "rollout", "rhea_s", "rhea_l", "rhea_s_v", "rhea_l_v", "dqn", "dqn2m", "dprl", "es", "es2m", "sepcma", "sepcma2m")
CEC_LABEL = {"stay": "Stay", "heur": "Reference heuristic", "grid": "Grid-calibrated heuristic", "cma": "CMA-ES heuristic", "rollout": "MC rollout", "rhea_s": "RHEA-S", "rhea_l": "RHEA-L", "rhea_s_v": "RHEA-S + DQN value", "rhea_l_v": "RHEA-L + DQN value",
             "dqn": "DQN", "dqn2m": "DQN (4x budget)", "dprl": "Dual-policy (DPRL-style)", "es": "ES (neuroevolution)", "es2m": "ES (4x budget)",
             "sepcma": "sep-CMA-ES (neuroevolution)", "sepcma2m": "sep-CMA-ES (4x budget)"}


def _sub_means(rows, n):
    """média por linha (semente) dos n primeiros cenários de teste e dos dias reais; ignora linhas sem test_rewards."""
    t = np.array([float(np.mean(r["test_rewards"][:n])) for r in rows if "test_rewards" in r])
    real = np.array([r["real_test_mean"] for r in rows if "real_test_mean" in r], float)
    return t, real


def _episode_rows(path, cfg=None):
    """{(conjunto, índice): recompensa} de um arquivo por episódio (rollout_scale / rhea), opcionalmente filtrando a config."""
    out = {}
    for r in rows_of(path):
        if cfg is None or r.get("config") == cfg:
            out[(r["set"], r["idx"])] = r["reward"]
    return out


def cec_reference(runs, inst):
    """P12b (DESVIOS.md 2026-10-22): referência de heurística = a melhor por VALIDAÇÃO entre a de grade e as execuções do CMA-ES (o teste não entra na escolha).
    Retorna (stay, grade, referência, origem 'grid'|'cma'); a linha é a do arquivo (test_rewards, real_test_mean...)."""
    runs = Path(runs)
    stay, grid, _ = reference_rows(runs, inst)
    if grid is None:
        return stay, None, None, None
    gval = max(grid["val_table"].values())
    cm = rows_of(runs / f"cmaes_{inst}.jsonl")
    best = max(cm, key=lambda r: r["val_best"]) if cm else None
    if best is not None and best["val_best"] > gval:
        return stay, grid, best, "cma"
    return stay, grid, grid, "grid"


HYB_CFG = (("S", "rhea_s_v"), ("L", "rhea_l_v"))
HYB_CFG_BASE = (("S", "rhea_s_v", "rhea_s"), ("L", "rhea_l_v", "rhea_l"))


def cec_hybrid_rows(runs, inst, cfg, n=N_SUB):
    """P13: linhas por SEMENTE DO DQN do RHEA com valor terminal (runs/rheav_<inst>.jsonl, um episódio por linha): {seed, test_rewards (n primeiros cenários), real_test_mean}
    só para as sementes com os n episódios de teste e os 24 dias reais completos. Execução repetida do mesmo episódio: vale a última."""
    ep = {}
    for r in rows_of(Path(runs) / f"rheav_{inst}.jsonl"):
        if r["config"] == cfg:
            ep[(r["dqn_seed"], r["set"], r["idx"])] = r["reward"]
    out = []
    for s in sorted({k[0] for k in ep}):
        if all((s, "test", k) in ep for k in range(n)) and all((s, "real_test", k) in ep for k in range(24)):
            out.append({"seed": s, "test_rewards": [ep[(s, "test", k)] for k in range(n)], "real_test_mean": float(np.mean([ep[(s, "real_test", k)] for k in range(24)]))})
    return out


def cec_agent_rows(runs, inst):
    """seed -> linha do DQN retreinado (runs/agents/dqn_<inst>.jsonl; o MESMO agente cujos pesos alimentam o RHEA-V); a última linha de cada semente."""
    return {r["seed"]: r for r in rows_of(Path(runs) / "agents" / f"dqn_{inst}.jsonl")}


def cec_data(runs, scales=SCALES, n=N_SUB):
    """Tabela única do artigo do CEC: todos os métodos nos n primeiros cenários de teste e nos 24 dias reais, por escala. Demanda perdida POSITIVA.
    Cada método: {kind: 'seeds'|'det', n, mean, lo, hi, red, red_lo, red_hi, real}. `red` = redução vs a heurística de grade de referência (%).
    Aprendizes: unidade = semente; determinísticos (heurística, rollout, RHEA): um valor por escala; 'cma' = por execução do CMA-ES (melhor família por validação)."""
    runs = Path(runs)
    out = {}
    for inst in scales:
        stay, grid, heur, src = cec_reference(runs, inst)
        if stay is None or heur is None:
            continue
        ro = reference_rows(runs, inst)[2]
        hs = float(np.mean(heur["test_rewards"][:n]))                        # reward (negativo) da heurística de referência no subconjunto
        res = {"n_st": N_ST[inst]}

        def det(name, test_vals, real_mean=None):
            m = float(np.mean(test_vals))
            res[name] = {"kind": "det", "n": 1, "mean": -m, "red": 100 * (hs - m) / hs, "real": -real_mean if real_mean is not None else float("nan"), "sub": len(test_vals)}

        def seeds(name, rows):
            t, real = _sub_means(rows, n)
            if len(t) < 2:
                return
            lo, hi = ci_t(t)
            red = 100 * (hs - t) / hs
            rlo, rhi = ci_t(red)
            res[name] = {"kind": "seeds", "n": len(t), "mean": -float(t.mean()), "lo": -float(hi), "hi": -float(lo), "red": float(red.mean()), "red_lo": float(rlo), "red_hi": float(rhi),
                         "real": -float(real.mean()) if len(real) else float("nan")}

        det("stay", stay["test_rewards"][:n], stay.get("real_test_mean"))
        det("heur", heur["test_rewards"][:n], heur.get("real_test_mean"))
        res["heur"]["src"] = src
        det("grid", grid["test_rewards"][:n], grid.get("real_test_mean"))
        # CMA-ES: por execução (semente), a família com melhor validação
        cm = rows_of(runs / f"cmaes_{inst}.jsonl")
        by = {}
        for r in cm:
            by.setdefault(r["seed"], {})[r["family"]] = r
        sel = [max(d.values(), key=lambda r: r["val_best"]) for d in by.values() if len(d) == 2]
        if len(sel) >= 2:
            seeds("cma", sel)
        # rollout
        if inst in ("cb4wd", "cb8wd", "cb12wd") and ro is not None:
            det("rollout", ro["test_rewards"][:n], ro.get("real_test_mean"))
        else:
            ep = _episode_rows(runs / f"rollout_scale_{inst}.jsonl")
            if all(("test", k) in ep for k in range(n)) and all(("real_test", k) in ep for k in range(24)):
                det("rollout", [ep[("test", k)] for k in range(n)], float(np.mean([ep[("real_test", k)] for k in range(24)])))
        # RHEA
        for cfg, name in (("S", "rhea_s"), ("L", "rhea_l")):
            ep = _episode_rows(runs / f"rhea_{inst}.jsonl", cfg)
            if all(("test", k) in ep for k in range(n)) and all(("real_test", k) in ep for k in range(24)):
                det(name, [ep[("test", k)] for k in range(n)], float(np.mean([ep[("real_test", k)] for k in range(24)])))
        # híbrido: RHEA com valor terminal do DQN (por semente do DQN)
        for cfg, name in HYB_CFG:
            seeds(name, cec_hybrid_rows(runs, inst, cfg, n))
        # aprendizes
        flat = ([r for r in rows_of(runs / "final_primary.jsonl") if r["instance"] == inst and r["method"] == "dqn"] if inst in ("cb4wd", "cb8wd", "cb12wd")
                else [r for r in rows_of(runs / f"scale_{inst}.jsonl") if r["method"] == "dqn"])
        seeds("dqn", flat)
        seeds("dqn2m", list(budget_rows(runs, inst).values()) if inst in ("cb4wd", "cb8wd", "cb12wd") else [r for r in rows_of(runs / f"scale2m_{inst}.jsonl") if r["method"] == "dqn"])
        seeds("dprl", [r for r in rows_of(runs / f"dprl_{inst}.jsonl") if r["method"] == "dprl"])
        seeds("es", [r for r in rows_of(runs / f"es_{inst}.jsonl") if r["method"] == "es"])
        seeds("es2m", [r for r in rows_of(runs / f"es2m_{inst}.jsonl") if r["method"] == "es"])
        seeds("sepcma", [r for r in rows_of(runs / f"sepcma_{inst}.jsonl") if r["method"] == "sepcma"])
        seeds("sepcma2m", [r for r in rows_of(runs / f"sepcma2m_{inst}.jsonl") if r["method"] == "sepcma"])
        out[inst] = res
    return out


def cec_contrasts(runs, scales=SCALES, n=N_SUB):
    """Contrastes do artigo (por escala): aprendiz − aprendiz por Welch (reward; positivo favorece o primeiro) e aprendiz × determinístico por semente."""
    d = cec_data(runs, scales, n)
    out = {}
    for inst, r in d.items():
        rows = {}
        pairs = (("es", "dqn"), ("dprl", "dqn"), ("es", "dprl"), ("es2m", "dqn2m"), ("sepcma", "es"), ("sepcma", "dqn"), ("sepcma2m", "es2m"), ("sepcma2m", "dqn2m"))
        raw = {}
        for name, path, key in (("dqn", "final_primary.jsonl" if inst in ("cb4wd", "cb8wd", "cb12wd") else f"scale_{inst}.jsonl", "dqn"),
                                ("dprl", f"dprl_{inst}.jsonl", "dprl"), ("es", f"es_{inst}.jsonl", "es"), ("es2m", f"es2m_{inst}.jsonl", "es"),
                                ("sepcma", f"sepcma_{inst}.jsonl", "sepcma"), ("sepcma2m", f"sepcma2m_{inst}.jsonl", "sepcma")):
            rr = [x for x in rows_of(Path(runs) / path) if x["method"] == key and (key != "dqn" or x.get("instance", inst) == inst)]
            raw[name] = np.array([float(np.mean(x["test_rewards"][:n])) for x in rr if "test_rewards" in x])
        if inst in ("cb4wd", "cb8wd", "cb12wd"):
            raw["dqn2m"] = np.array([float(np.mean(x["test_rewards"][:n])) for x in budget_rows(runs, inst).values()])
        else:
            raw["dqn2m"] = np.array([float(np.mean(x["test_rewards"][:n])) for x in rows_of(Path(runs) / f"scale2m_{inst}.jsonl") if x["method"] == "dqn"])
        for a, b in pairs:
            if len(raw.get(a, [])) >= 2 and len(raw.get(b, [])) >= 2:
                dw, (lo, hi), _ = welch(raw[a], raw[b])
                rows[f"{a}_vs_{b}"] = {"delta": float(dw), "lo": float(lo), "hi": float(hi), "na": len(raw[a]), "nb": len(raw[b])}
        # P13: híbrido pareado com o PRÓPRIO DQN (mesmos pesos, mesmos n cenários), por semente do DQN; positivo favorece o híbrido
        ag = cec_agent_rows(runs, inst)
        for cfg, name in HYB_CFG:
            hr = cec_hybrid_rows(runs, inst, cfg, n)
            raw[name] = np.array([float(np.mean(x["test_rewards"])) for x in hr])
            same = [s for s in (x["seed"] for x in hr) if s in ag]
            if len(same) >= 2:
                hv = np.array([float(np.mean(x["test_rewards"])) for x in hr if x["seed"] in same])
                dq = np.array([float(np.mean(ag[s]["test_rewards"][:n])) for s in same])
                dd = hv - dq
                lo, hi = ci_t(dd)
                rows[f"{name}_vs_dqnsame"] = {"delta": float(dd.mean()), "lo": float(lo), "hi": float(hi), "na": len(dd), "dqn": -float(dq.mean())}
        for a in ("dqn", "dprl", "es", "dqn2m", "es2m", "rhea_s_v", "rhea_l_v", "sepcma", "sepcma2m"):
            for ref in ("rollout", "rhea_s", "rhea_l", "heur"):
                if ref in r and len(raw.get(a, [])) >= 2:
                    rr = -r[ref]["mean"]
                    dd = raw[a] - rr
                    lo, hi = ci_t(dd)
                    rows[f"{a}_vs_{ref}"] = {"delta": float(dd.mean()), "lo": float(lo), "hi": float(hi), "na": len(dd)}
        out[inst] = rows
    return out


def cec_agent_check(runs, scales=SCALES):
    """P13: o DQN retreinado (runs/agents) reproduz a linha original da tabela principal? {inst: {n, n_match}} (n_match = nº de sementes com teste idêntico)."""
    out = {}
    for inst in scales:
        ag = cec_agent_rows(runs, inst)
        if ag:
            out[inst] = {"n": len(ag), "n_match": sum(1 for r in ag.values() if r.get("matches_original")), "n_orig": sum(1 for r in ag.values() if r.get("matches_original") is not None)}
    return out


def cec_calibration(runs, scales=SCALES, delta=0.5):
    """P12b: heurística evoluída (CMA-ES, melhor família por validação em cada execução) × grade, no teste inteiro (1000 cenários). Unidade = execução do CMA-ES;
    a grade é determinística. Leitura pré-fixada: IC95% t > 0 → melhor; < 0 → pior; senão equivalente só se TOST (δ = 0,5) rejeitar, senão inconclusivo."""
    out = {}
    for inst in scales:
        stay, grid, ref, src = cec_reference(runs, inst)
        if grid is None:
            continue
        cm = rows_of(Path(runs) / f"cmaes_{inst}.jsonl")
        by = {}
        for r in cm:
            by.setdefault(r["seed"], {})[r["family"]] = r
        sel = [max(d.values(), key=lambda r: r["val_best"]) for d in by.values() if len(d) == 2]
        if len(sel) < 2:
            continue
        t = np.array([r["test_mean"] for r in sel])
        v = np.array([r["val_best"] for r in sel])
        d = t - grid["test_mean"]                                         # reward: positivo favorece o CMA-ES
        lo, hi = ci_t(d)
        tlo, thi = ci_t(t)
        ptost = float(AN.tost_paired(d, delta)) if d.std(ddof=1) > 0 else 1.0
        verdict = "better" if lo > 0 else ("worse" if hi < 0 else ("equivalent" if ptost < 0.05 else "inconclusive"))
        out[inst] = {"n_st": N_ST[inst], "n": len(sel), "grid_val": -max(grid["val_table"].values()), "grid_test": -float(grid["test_mean"]),
                     "cma_val": -float(v.mean()), "cma_test": -float(t.mean()), "cma_lo": -float(thi), "cma_hi": -float(tlo),
                     "delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_tost": ptost, "verdict": verdict, "src": src,
                     "ref_test": -float(ref["test_mean"]), "n_greedy": sum(r["family"] == "greedy" for r in sel)}
    return out


ES_BUDGETS = (("500k", "es_{}.jsonl"), ("2M", "es2m_{}.jsonl"), ("8M", "es8m_{}.jsonl"), ("32M", "es32m_{}.jsonl"))
ES_STEPS = {"500k": 5e5, "2M": 2e6, "8M": 8e6, "32M": 3.2e7}


SEP_BUDGETS = (("500k", "sepcma_{}.jsonl"), ("2M", "sepcma2m_{}.jsonl"), ("8M", "sepcma8m_{}.jsonl"), ("32M", "sepcma32m_{}.jsonl"))


def cec_sep_budget(runs, scales=SCALES):
    """P15: o mesmo que `cec_es_budget` para o sep-CMA-ES (arquivos sepcma*_<inst>.jsonl)."""
    return cec_es_budget(runs, scales, budgets=SEP_BUDGETS, method="sepcma")


def cec_es_budget(runs, scales=SCALES, budgets=ES_BUDGETS, method="es"):
    """P12d: neuroevolução (ES) × orçamento de passos, no teste inteiro (1000 cenários), unidade = semente. Contraste ES(B) − DQN(500 mil) por Welch (reward; positivo
    favorece o ES). 'reached' = menor orçamento cujo IC95% do contraste inclui ou passa de 0 (None se nenhum). `budgets`/`method`: outro algoritmo de neuroevolução (P15)."""
    out = {}
    for inst in scales:
        stay, grid, ref, src = cec_reference(runs, inst)
        if ref is None:
            continue
        dq = [r for r in rows_of(Path(runs) / ("final_primary.jsonl" if inst in BASES else f"scale_{inst}.jsonl")) if r["method"] == "dqn" and r.get("instance", inst) == inst]
        dqx = np.array([r["test_mean"] for r in dq], float)
        rec = {"n_st": N_ST[inst], "heur": -float(ref["test_mean"]), "dqn": -float(dqx.mean()) if len(dqx) else float("nan"), "budgets": {}, "reached": None,
               "dqn_secs": float(np.mean([r.get("train_seconds", float("nan")) for r in dq])) if dq else float("nan")}
        for tag, pat in budgets:
            er = [r for r in rows_of(Path(runs) / pat.format(inst)) if r["method"] == method]
            xs = np.array([r["test_mean"] for r in er], float)
            if len(xs) < 2:
                continue
            lo, hi = ci_t(xs)
            b = {"n": len(xs), "mean": -float(xs.mean()), "lo": -float(hi), "hi": -float(lo), "steps": ES_STEPS[tag], "secs": float(np.mean([r.get("train_seconds", float("nan")) for r in er]))}
            hd = xs - float(ref["test_mean"])                                # reward: positivo favorece o ES contra a heurística de referência (teste inteiro)
            hlo, hhi = ci_t(hd)
            b.update({"vs_heur": float(hd.mean()), "vs_heur_lo": float(hlo), "vs_heur_hi": float(hhi)})
            if len(dqx) >= 2:
                dw, (wlo, whi), _ = welch(xs, dqx)
                b.update({"vs_dqn": float(dw), "vs_dqn_lo": float(wlo), "vs_dqn_hi": float(whi)})
                if rec["reached"] is None and whi >= 0:
                    rec["reached"] = tag
            rec["budgets"][tag] = b
        out[inst] = rec
    return out


def _curve_stats(rows):
    """Curva de validação (demanda perdida, positiva) média por ponto de avaliação, com IC95% t entre sementes; os pontos de avaliação têm de coincidir entre as sementes."""
    rows = [r for r in rows if r.get("curve")]
    if not rows:
        return None
    n = min(len(r["curve"]) for r in rows)
    steps = np.array([np.mean([r["curve"][k]["steps"] for r in rows]) for k in range(n)])
    v = np.array([[-r["curve"][k]["val"] for k in range(n)] for r in rows], float)
    lo, hi = zip(*[ci_t(v[:, k]) if len(rows) >= 2 else (v[0, k], v[0, k]) for k in range(n)])
    return {"n": len(rows), "steps": steps, "mean": v.mean(axis=0), "lo": np.array(lo), "hi": np.array(hi)}


def cec_curves(runs, scales=("cb4wd", "cb12wd", "cb30wd")):
    """Curvas de validação durante o treino: DQN (500 mil passos) e ES (cada orçamento), demanda perdida na validação × passos de simulador; unidade = semente."""
    out = {}
    for inst in scales:
        dq = [r for r in rows_of(Path(runs) / ("final_primary.jsonl" if inst in BASES else f"scale_{inst}.jsonl")) if r["method"] == "dqn" and r.get("instance", inst) == inst]
        rec = {"n_st": N_ST[inst], "dqn": _curve_stats(dq), "es": {}}
        for tag, pat in ES_BUDGETS:
            st = _curve_stats([r for r in rows_of(Path(runs) / pat.format(inst)) if r["method"] == "es"])
            if st is not None:
                rec["es"][tag] = st
        out[inst] = rec
    return out


def cec_es_config(runs, scales=SCALES):
    """Configurações do ES escolhidas em validação (pool de 16) por escala."""
    out = {}
    for inst in scales:
        p = Path(runs) / f"tuned_es_{inst}.json"
        if p.exists():
            d = load_json(p)["es"]
            out[inst] = {"cfg_id": d["cfg_id"], **d["cfg"], "val": -float(d["val_mean"])}
    return out


def cec_controls(runs, scales=SCALES, n=N_SUB, boot=4000, delta=0.5):
    """P14 (DESVIOS.md 2026-10-23): controles SEM evolução. (a) RHEA × tiro aleatório com o mesmo número de avaliações (config S × RS, L × RL), pareado por cenário nos n primeiros cenários de
    teste, IC95% bootstrap sobre cenários; positivo (reward) = RHEA melhor. (b) CMA-ES × busca aleatória (mesma caixa, mesmas avaliações), por execução (melhor família por validação em cada
    execução), teste inteiro; Welch; positivo = CMA-ES melhor; 'equivalent' só se o IC95% couber em ±delta."""
    out = {}
    rng = np.random.default_rng(20261023)
    for inst in scales:
        rec = {"n_st": N_ST[inst]}
        for cfg, ctl, key in (("S", "RS", "rhea_vs_rs_s"), ("L", "RL", "rhea_vs_rs_l")):
            a, b = _episode_rows(Path(runs) / f"rhea_{inst}.jsonl", cfg), _episode_rows(Path(runs) / f"rhea_{inst}.jsonl", ctl)
            if not (all(("test", k) in a and ("test", k) in b for k in range(n))):
                continue
            x = np.array([a[("test", k)] for k in range(n)], float)
            y = np.array([b[("test", k)] for k in range(n)], float)
            d = x - y
            bm = d[rng.integers(0, n, size=(boot, n))].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rec[key] = {"rhea": -float(x.mean()), "control": -float(y.mean()), "delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}
        sel = {}
        for tag in ("cmaes", "rsearch"):
            by = {}
            for r in rows_of(Path(runs) / f"{tag}_{inst}.jsonl"):
                by.setdefault(r["seed"], {})[r["family"]] = r
            sel[tag] = [max(d_.values(), key=lambda r: r["val_best"]) for d_ in by.values() if len(d_) == 2]
        if len(sel["cmaes"]) >= 2 and len(sel["rsearch"]) >= 2:
            tc = np.array([r["test_mean"] for r in sel["cmaes"]], float)
            tr = np.array([r["test_mean"] for r in sel["rsearch"]], float)
            dw, (lo, hi), _ = welch(tc, tr)
            verdict = "better" if lo > 0 else ("worse" if hi < 0 else ("equivalent" if (lo > -delta and hi < delta) else "inconclusive"))
            rec["cma_vs_random"] = {"cma": -float(tc.mean()), "random": -float(tr.mean()), "n_cma": len(tc), "n_random": len(tr), "delta": float(dw), "lo": float(lo), "hi": float(hi), "verdict": verdict,
                                    "val_cma": -float(np.mean([r["val_best"] for r in sel["cmaes"]])), "val_random": -float(np.mean([r["val_best"] for r in sel["rsearch"]]))}
        if len(rec) > 1:
            out[inst] = rec
    return out


def cec_rhea_vs_rollout(runs, scales=SCALES, n=N_SUB, boot=4000):
    """P12c: RHEA − rollout (3, 64), pareado por cenário de teste (os n primeiros), IC95% bootstrap sobre cenários; positivo = RHEA melhor (reward)."""
    out = {}
    rng = np.random.default_rng(20261022)
    for inst in scales:
        ro = reference_rows(runs, inst)[2]
        if inst in BASES and ro is not None:
            roll = np.array(ro["test_rewards"][:n], float)
        else:
            ep = _episode_rows(Path(runs) / f"rollout_scale_{inst}.jsonl")
            if not all(("test", k) in ep for k in range(n)):
                continue
            roll = np.array([ep[("test", k)] for k in range(n)], float)
        rec = {"n_st": N_ST[inst], "rollout": -float(roll.mean())}
        for cfg in ("S", "L"):
            ep = _episode_rows(Path(runs) / f"rhea_{inst}.jsonl", cfg)
            if not all(("test", k) in ep for k in range(n)):
                continue
            x = np.array([ep[("test", k)] for k in range(n)], float)
            d = x - roll
            idx = rng.integers(0, n, size=(boot, n))
            bm = d[idx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rec[cfg] = {"mean": -float(x.mean()), "delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}
        out[inst] = rec
    return out


def p8_report(runs):
    L = ["# P8 — ancoragem e robustez da conclusão central (gerado por `scripts/30_analyze_p8.py`)\n",
         "> Desenho e leituras fixados em `DESVIOS.md` (2026-10-18), **antes** de rodar. Reward ≤ 0; maior = melhor; unidade de análise = semente de treino.\n"]
    return L + oracle_section(runs) + budget_section(runs) + reseed_section(runs) + scale_section(runs) + arch_section(runs) + dprl_section(runs)


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
