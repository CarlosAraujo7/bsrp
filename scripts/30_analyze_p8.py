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


def p8_report(runs):
    L = ["# P8 — ancoragem e robustez da conclusão central (gerado por `scripts/30_analyze_p8.py`)\n",
         "> Desenho e leituras fixados em `DESVIOS.md` (2026-10-18), **antes** de rodar. Reward ≤ 0; maior = melhor; unidade de análise = semente de treino.\n"]
    return L + oracle_section(runs) + budget_section(runs) + reseed_section(runs) + scale_section(runs)


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
