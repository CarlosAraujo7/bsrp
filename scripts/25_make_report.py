"""
25_make_report.py — figuras, tabelas LaTeX, macros numéricas e 06_resultados.md, TUDO gerado de runs/*.jsonl
(nenhum número digitado à mão). Idioma: português; decimais com vírgula.

  python -X utf8 scripts/25_make_report.py

Saídas: figuras/*.pdf|png, figuras/dados/*.csv (visão em tabela de cada figura), artigo/tabelas/*.tex,
        artigo/numeros.tex (macros \\dqnA, \\redB, ...; sufixos A/B/C = 4/8/12 estações) e 06_resultados.md.
Convenções: "demanda perdida" = −reward (menor é melhor); IC95% t sobre sementes de treino (n = 10 salvo indicação);
heurística de referência e rollout de cabeçalho escolhidos em VALIDAÇÃO (DESVIOS.md).
"""
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
from matplotlib.ticker import FuncFormatter                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("an21", ROOT / "scripts" / "21_analyze_p1_p2.py")
AN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(AN)
rows_of, ci_t, welch = AN.rows_of, AN.ci_t, AN.welch
_s30 = importlib.util.spec_from_file_location("an30", ROOT / "scripts" / "30_analyze_p8.py")
AN30 = importlib.util.module_from_spec(_s30)
_s30.loader.exec_module(AN30)

BASES = ("cb4wd", "cb8wd", "cb12wd")
LETTER = {"cb4wd": "A", "cb8wd": "B", "cb12wd": "C"}
NST = {"cb4wd": 4, "cb8wd": 8, "cb12wd": 12}
# cores por ENTIDADE (ordem fixa da paleta validada em light) + neutros
COL = {"dqn": "#2a78d6", "rollout": "#eb6834", "linear": "#1baf7a", "tab_station": "#eda100", "pg": "#e87ba4",
       "tab_cluster": "#008300", "a2c": "#4a3aa7", "a2c_t": "#9085e9", "heur": "#52514e", "stay": "#898781"}
LABEL = {"stay": "Parada", "heur": "Heurística calibrada", "rollout": "Rollout (modelo)", "dqn": "DQN", "pg": "Policy gradient",
         "a2c": "A2C (Seo, publicado)", "a2c_t": "A2C (Seo, ajustado)", "linear": "Linear", "tab_station": "Tabular (estação)", "tab_cluster": "Tabular (cluster)"}
ORDER = ("stay", "heur", "rollout", "dqn", "pg", "a2c_t", "a2c", "linear", "tab_station", "tab_cluster")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"


def pt(x, d=1):
    """número com vírgula decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def tex_num(s: str) -> str:
    """sinal de menos tipográfico (modo matemático) em números negativos dentro de macros LaTeX."""
    return re.sub(r"-(?=\d)", "$-$", s)


def setup_style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                         "figure.dpi": 100, "savefig.dpi": 220, "pdf.fonttype": 42, "axes.titleweight": "regular"})


def comma_axis(ax, axis="x"):
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


# ──────────────────────────────────────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────────────────────────────────────
def unmet(rows, key="test_mean"):
    return np.array([-r[key] for r in rows if key in r], float)


def collect(runs: Path) -> dict:
    prim = rows_of(runs / "final_primary.jsonl")
    out = {}
    for b in BASES:
        H = {}
        for f in AN.P4_HEUR_FILES:
            for r in rows_of(runs / f):
                if r["instance"] == b:
                    H[r["method"]] = r
        fams = {k: v for k, v in H.items() if k in AN.P4_FAMILIES and v.get("val_table")}
        ref = max(fams, key=lambda k: max(fams[k]["val_table"].values()))
        d = {"stay": {"test": -H["stay"]["test_mean"], "real": -H["stay"].get("real_test_mean", np.nan)},
             "heur": {"family": ref, "policy": H[ref]["policy"], "test": -H[ref]["test_mean"], "real": -H[ref].get("real_test_mean", np.nan),
                      "val": -max(fams[ref]["val_table"].values())}}
        ro = [r for f in sorted(runs.glob(f"rollout_{b}*.jsonl")) for r in rows_of(f)]
        best = max(ro, key=lambda r: r["val_mean"])
        d["rollout"] = {"h": best["h"], "K": best["K"], "test": -best["test_mean"], "real": -best.get("real_test_mean", np.nan),
                        "val": -best["val_mean"], "all": [(r["h"], r["K"], -r["val_mean"], -r["test_mean"]) for r in sorted(ro, key=lambda r: (r["h"], r["K"]))]}
        for m in ("dqn", "linear", "tab_station", "tab_cluster", "pg"):
            rr = sorted([r for r in prim if r["instance"] == b and r["method"] == m], key=lambda r: r["seed"])
            d[m] = {"test": unmet(rr), "real": unmet(rr, "real_test_mean"), "seeds": [r["seed"] for r in rr],
                    "steps": [c["steps"] for c in rr[0]["curve"]], "curve": np.array([[-c["val"] for c in r["curve"]] for r in rr])}
        for key, var in (("a2c", "published"), ("a2c_g1", "gamma1"), ("a2c_t", "tuned")):
            rr = sorted([r for r in rows_of(runs / f"seo_a2c_{b}.jsonl") if r["variant"] == var], key=lambda r: r["seed"])
            d[key] = {"test": unmet(rr), "real": unmet(rr, "real_test_mean"), "steps": [c["steps"] for c in rr[0]["curve"]],
                      "curve": np.array([[-c["val"] for c in r["curve"]] for r in rr])}
        orig = {}
        for m in ("dqn", "linear", "tab_station", "tab_cluster", "pg"):
            rr = [r for r in rows_of(runs / f"final_{b}.jsonl") if r["method"] == m]
            orig[m] = unmet(rr)
        d["orig"] = orig
        out[b] = d
    return out


def ablation(runs: Path) -> dict:
    res = {}
    for b in ("cb12wd", "cb4wd"):
        abl = rows_of(runs / f"ablation_{b}.jsonl")
        _, ref = AN.reference_dqn(runs, b)
        res[b] = {}
        for v in ("noreplay", "notarget", "nodouble", "vanilla"):
            by = {r["seed"]: r for r in abl if r["variant"] == v}
            seeds = sorted(set(by) & set(ref))
            full = np.array([ref[s]["test_mean"] for s in seeds])
            var = np.array([by[s]["test_mean"] for s in seeds])
            dlt = full - var                               # > 0: remover a componente PIORA o resultado
            lo, hi = ci_t(dlt)
            res[b][v] = {"mean": float(dlt.mean()), "lo": float(lo), "hi": float(hi), "n": len(seeds), "var_mean": float(-var.mean()),
                         "full_mean": float(-full.mean())}
    return res


def sensitivity(runs: Path) -> dict:
    res = {}
    for base in ("cb4wd", "cb12wd"):
        res[base] = []
        for label, name in AN.P3_VARIANTS[base]:
            by, H = AN.p3_data(runs, name, base)
            bh = AN.best_heuristic(H)
            if bh is None or len(by["dqn"]) < 2:
                continue
            h = H[bh]["test_mean"]
            x = np.array([r["test_mean"] for _, r in sorted(by["dqn"].items())])
            d = x - h
            lo, hi = ci_t(d)
            res[base].append({"label": label, "name": name, "mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(x),
                              "heur": -h, "stay": -H["stay"]["test_mean"] if "stay" in H else np.nan, "dqn": -float(x.mean())})
    return res


def retuned(runs: Path) -> dict:
    """P3b (DESVIOS.md 2026-10-18): DQN reajustado (pool de 16, escolha por val) nas variantes do P3 em que a conclusão mudou."""
    res = {}
    for base, lst in AN.P3B_VARIANTS.items():
        for label, name in lst:
            d = AN.p3b_data(runs, name, base)
            if d is None:
                continue
            t, base_id, same, rows = d
            by, H = AN.p3_data(runs, name, base)
            bh = AN.best_heuristic(H)
            x = np.array([r["test_mean"] for _, r in sorted(rows.items())])
            x0 = np.array([r["test_mean"] for _, r in sorted(by["dqn"].items())])
            if bh is None or len(x) < 2:
                continue
            h = H[bh]["test_mean"]
            lo, hi = ci_t(x - h)
            dw, (wl, wh), _ = welch(x, x0)
            res[name] = {"label": label, "name": name, "mean": float(np.mean(x - h)), "lo": float(lo), "hi": float(hi), "n": len(x), "same": bool(same),
                         "cfg_id": int(t["cfg_id"]), "delta": float(dw), "dlo": float(wl), "dhi": float(wh), "dqn": -float(x.mean())}
    return res


# ──────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ──────────────────────────────────────────────────────────────────────────────
def fig_scale(SC, outdir, RT=None):
    """P9: redução da demanda perdida em relação à heurística calibrada, de 4 a 30 estações (DQN 500 mil e 2 milhões de passos; rollout só até 12)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    csv_rows = []
    for tag, label, ls, filled in (("500k", "DQN, 500 mil passos", "-", True), ("2m", "DQN, 2 milhões de passos", "--", False)):
        pts = [(r["n_st"], r[tag]) for r in SC.values() if tag in r]
        if not pts:
            continue
        xs = [n for n, _ in pts]
        ys = [t["red"] for _, t in pts]
        ax.fill_between(xs, [t["red_lo"] for _, t in pts], [t["red_hi"] for _, t in pts], color=COL["dqn"], alpha=0.12, lw=0)
        ax.plot(xs, ys, ls, color=COL["dqn"], lw=1.6, marker="o", ms=5, mfc=COL["dqn"] if filled else "white", mec=COL["dqn"], label=label, zorder=3)
        csv_rows += [[tag, n, f"{t['red']:.1f}", f"{t['red_lo']:.1f}..{t['red_hi']:.1f}", t["n"]] for n, t in pts]
    if RT and "n" in RT and not RT["same"]:
        ax.scatter([30], [RT["red"]], marker="D", s=34, color=COL["dqn"], edgecolor=INK, lw=0.8, zorder=4, label="DQN reajustado em 30 est. (500 mil)")
        csv_rows.append(["reajustado_30", 30, f"{RT['red']:.1f}", "", RT["n"]])
    ro = [(r["n_st"], r["rollout"]["red"]) for r in SC.values() if "rollout" in r]    # 20/30: subconjunto de cenários (ver legenda)
    if ro:
        ax.plot([n for n, _ in ro], [v for _, v in ro], "-", color=COL["rollout"], lw=1.6, marker="s", ms=4.5, label="Rollout (modelo)", zorder=3)
        csv_rows += [["rollout", n, f"{v:.1f}", "", ""] for n, v in ro]
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_xticks([r["n_st"] for r in SC.values()])
    ax.set_xlabel("estações")
    ax.set_ylabel("redução da demanda perdida vs heurística (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_escala", outdir)
    table_csv(outdir / "dados" / "fig_escala.csv", ["serie", "estacoes", "reducao_pct", "ic95", "n_sementes"], csv_rows)


def save(fig, name, outdir):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def fig_main(R, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.1))
    csv_rows = []
    for r, (key, title) in enumerate((("test", "teste sintético (1000 cenários)"), ("real", "24 dias reais de teste"))):
        for c, b in enumerate(BASES):
            ax = axes[r, c]
            vals, errs = [], []
            for m in ORDER:
                d = R[b][m] if m != "heur" and m != "stay" and m != "rollout" else R[b][m]
                if m in ("stay", "heur", "rollout"):
                    vals.append(d[key]); errs.append(None)
                else:
                    x = d[key]
                    lo, hi = ci_t(x)
                    vals.append(float(x.mean())); errs.append((float(lo), float(hi)))
            y = np.arange(len(ORDER))
            for yi, m, v, e in zip(y, ORDER, vals, errs):
                ax.barh(yi, v, height=0.62, color=COL[m], zorder=2)
                if e is not None:
                    ax.plot([e[0], e[1]], [yi, yi], color=INK, lw=1.0, zorder=3)
                    ax.plot([e[0], e[0]], [yi - 0.15, yi + 0.15], color=INK, lw=1.0, zorder=3)
                    ax.plot([e[1], e[1]], [yi - 0.15, yi + 0.15], color=INK, lw=1.0, zorder=3)
                x_txt = (e[1] if e is not None else v) + 0.035 * max(vals)
                ax.text(x_txt, yi, pt(v, 1), va="center", ha="left", fontsize=7, color=INK)
                csv_rows.append([b, key, m, pt(v, 2), "" if e is None else f"{pt(e[0], 2)}..{pt(e[1], 2)}"])
            ax.set_yticks(y)
            ax.set_yticklabels([LABEL[m] for m in ORDER] if c == 0 else [], fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlim(0, max(vals) * 1.22)
            ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            comma_axis(ax)
            if r == 0:
                ax.set_title(f"{NST[b]} estações", loc="left", color=INK)
            if r == 1:
                ax.set_xlabel("demanda perdida por dia")
            if c == 0:
                ax.annotate(title, xy=(0, 1.0), xycoords="axes fraction", xytext=(-92, 22), textcoords="offset points", fontsize=8, color=INK2)
    fig.tight_layout(h_pad=1.4, w_pad=0.6)
    save(fig, "fig_principal", outdir)
    table_csv(outdir / "dados" / "fig_principal.csv", ["escala", "conjunto", "metodo", "demanda_perdida_media", "ic95_sementes"], csv_rows)


def fig_curves(R, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    csv_rows = []
    learners = ("dqn", "pg", "a2c_t", "linear", "tab_station")
    for ax, b in zip(axes, BASES):
        for m in learners:
            d = R[b][m]
            x = np.array(d["steps"]) / 1000.0
            mu = d["curve"].mean(0)
            half = np.array([(ci_t(d["curve"][:, k])[1] - ci_t(d["curve"][:, k])[0]) / 2 for k in range(d["curve"].shape[1])])
            ax.fill_between(x, mu - half, mu + half, color=COL[m], alpha=0.16, lw=0, zorder=1)
            ax.plot(x, mu, color=COL[m], lw=1.6, zorder=2, label=LABEL[m])
            for k in range(len(x)):
                csv_rows.append([b, m, int(x[k] * 1000), pt(mu[k], 2), pt(half[k], 2)])
        ax.axhline(R[b]["heur"]["val"], color=INK2, lw=1.0, ls="--", zorder=2, label=LABEL["heur"])
        ax.axhline(R[b]["rollout"]["val"], color=COL["rollout"], lw=1.0, ls="--", zorder=2, label=LABEL["rollout"])
        lows = [R[b]["heur"]["val"], R[b]["rollout"]["val"]] + [float(R[b][m]["curve"].mean(0).min()) for m in learners]
        ax.set_ylim(bottom=max(0.0, min(lows) * 0.85))
        ax.set_title(f"{NST[b]} estações", loc="left")
        ax.set_xlabel("passos de treino (mil)")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        comma_axis(ax, "y")
    axes[0].set_ylabel("demanda perdida (validação)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_curvas", outdir)
    table_csv(outdir / "dados" / "fig_curvas.csv", ["escala", "metodo", "passos", "demanda_perdida_val_media", "meia_largura_ic95"], csv_rows)


def fig_pool(R, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    methods = ("dqn", "pg", "linear", "tab_station", "tab_cluster")
    csv_rows = []
    for ax, b in zip(axes, BASES):
        for yi, m in enumerate(methods):
            o, n = R[b]["orig"][m].mean(), R[b][m]["test"].mean()
            same = abs(o - n) < 1e-9
            ax.plot([o, n], [yi, yi], color=AXIS, lw=1.2, zorder=1)
            ax.scatter([o], [yi], s=34, facecolor="white", edgecolor=COL[m], lw=1.4, zorder=3)
            ax.scatter([n], [yi], s=34, color=COL[m], zorder=4)
            csv_rows.append([b, m, pt(o, 2), pt(n, 2), "mantida" if same else "trocada"])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([LABEL[m] for m in methods] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(f"{NST[b]} estações", loc="left")
        ax.set_xlabel("demanda perdida por dia (teste)")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        comma_axis(ax)
    fig.text(0.5, -0.04, "círculo vazio: melhor das 8 configurações originais · círculo cheio: melhor das 16 (escolha por validação)",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig_busca", outdir)
    table_csv(outdir / "dados" / "fig_busca.csv", ["escala", "metodo", "pool_8", "pool_16", "escolha"], csv_rows)


def fig_ablation(A, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    csv_rows = []
    names = (("noreplay", "sem replay"), ("notarget", "sem rede-alvo"), ("nodouble", "sem Double DQN"), ("vanilla", "as três (exploratória)"))
    for ax, b in zip(axes, ("cb4wd", "cb12wd")):
        for yi, (v, lab) in enumerate(names):
            d = A[b][v]
            sig = d["lo"] > 0
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=INK, lw=1.1, zorder=2)
            ax.scatter([d["mean"]], [yi], s=36, color=POS if sig else "white", edgecolor=POS if sig else MUTED, lw=1.3, zorder=3)
            csv_rows.append([b, v, pt(d["mean"], 2), f"{pt(d['lo'], 2)}..{pt(d['hi'], 2)}", d["n"]])
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n for _, n in names])
        ax.invert_yaxis()
        ax.set_title(f"{NST[b]} estações", loc="left")
        ax.set_xlabel("piora ao remover (demanda perdida, IC 95%)")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        comma_axis(ax)
    fig.tight_layout()
    save(fig, "fig_ablacao", outdir)
    table_csv(outdir / "dados" / "fig_ablacao.csv", ["escala", "variante", "delta_medio", "ic95", "n_sementes"], csv_rows)


def fig_sensitivity(S, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [9, 5]})
    csv_rows = []
    for ax, b in zip(axes, ("cb4wd", "cb12wd")):
        items = S[b]
        for yi, it in enumerate(items):
            sig_pos, sig_neg = it["lo"] > 0, it["hi"] < 0
            col = POS if sig_pos else (NEG if sig_neg else MUTED)
            ax.plot([it["lo"], it["hi"]], [yi, yi], color=INK, lw=1.1, zorder=2)
            ax.scatter([it["mean"]], [yi], s=36, color=col if (sig_pos or sig_neg) else "white", edgecolor=col, lw=1.3, zorder=3)
            csv_rows.append([b, it["name"], pt(it["mean"], 2), f"{pt(it['lo'], 2)}..{pt(it['hi'], 2)}", it["n"]])
            rt = S.get("rt", {}).get(it["name"])
            if rt is not None:                       # P3b: losango = DQN reajustado na variante (linha abaixo do círculo)
                col_rt = POS if rt["lo"] > 0 else (NEG if rt["hi"] < 0 else MUTED)
                ax.plot([rt["lo"], rt["hi"]], [yi + 0.28, yi + 0.28], color=INK, lw=0.9, alpha=0.55, zorder=2)
                ax.scatter([rt["mean"]], [yi + 0.28], s=30, marker="D", color=col_rt if (rt["lo"] > 0 or rt["hi"] < 0) else "white",
                           edgecolor=col_rt, lw=1.2, zorder=3)
                csv_rows.append([b, it["name"] + "_reajustado", pt(rt["mean"], 2), f"{pt(rt['lo'], 2)}..{pt(rt['hi'], 2)}", rt["n"]])
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([it["label"] for it in items])
        ax.invert_yaxis()
        ax.set_title(f"{NST[b]} estações", loc="left")
        ax.set_xlabel("DQN − heurística (ganho em demanda perdida)")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        comma_axis(ax)
    fig.tight_layout()
    save(fig, "fig_sensibilidade", outdir)
    table_csv(outdir / "dados" / "fig_sensibilidade.csv", ["escala", "variante", "dqn_menos_heuristica", "ic95", "n_sementes"], csv_rows)


# ──────────────────────────────────────────────────────────────────────────────
# TABELAS LaTeX, MACROS E RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────
def mean_ci(x):
    lo, hi = ci_t(x)
    return float(np.mean(x)), float(lo), float(hi)


def write_tables_and_macros(R, A, S, root, runs):
    tdir = root / "artigo" / "tabelas"
    tdir.mkdir(parents=True, exist_ok=True)
    mac = {}
    for b in BASES:
        L = LETTER[b]
        d = R[b]
        h, st, ro = d["heur"]["test"], d["stay"]["test"], d["rollout"]
        dq = d["dqn"]["test"]
        mu, lo, hi = mean_ci(dq)
        mac[f"stay{L}"], mac[f"heur{L}"], mac[f"dqn{L}"] = pt(st), pt(h), pt(mu)
        mac[f"dqnLo{L}"], mac[f"dqnHi{L}"] = pt(lo), pt(hi)
        mac[f"red{L}"] = f"{100 * (h - mu) / h:.0f}"
        mac[f"redStay{L}"] = f"{100 * (st - mu) / st:.0f}"
        mac[f"cum{L}"] = pt(h - mu)
        mac[f"roll{L}"], mac[f"rollH{L}"], mac[f"rollK{L}"] = pt(ro["test"]), str(ro["h"]), str(ro["K"])
        mac[f"redRoll{L}"] = f"{100 * (h - ro['test']) / h:.0f}"
        dr = dq - ro["test"]
        dm, dlo, dhi = mean_ci(-dr)                           # > 0: DQN melhor (menos demanda perdida)
        mac[f"dqnMenosRoll{L}"], mac[f"dqnMenosRollLo{L}"], mac[f"dqnMenosRollHi{L}"] = pt(dm), pt(dlo), pt(dhi)
        mac[f"dqnR{L}"] = pt(float(np.mean(d["dqn"]["real"])))
        mac[f"heurR{L}"], mac[f"rollR{L}"] = pt(d["heur"]["real"]), pt(ro["real"])
        mac[f"redR{L}"] = f"{100 * (d['heur']['real'] - np.mean(d['dqn']['real'])) / d['heur']['real']:.0f}"
        for m, nm in (("pg", "pg"), ("linear", "lin"), ("tab_station", "tabs"), ("tab_cluster", "tabc"), ("a2c", "aca"), ("a2c_t", "acaT")):
            mac[f"{nm}{L}"] = pt(float(np.mean(d[m]["test"])))
        o, n = d["orig"]["dqn"].mean(), d["dqn"]["test"].mean()
        mac[f"dqnOrig{L}"], mac[f"ganhoBusca{L}"] = pt(o), pt(o - n)
        mac[f"nMelhor{L}"] = str(int((dq < h).sum()))
        mac[f"nSem{L}"] = str(len(dq))
        mac[f"pgSd{L}"] = pt(float(np.std(d["pg"]["test"], ddof=1)), 0)
        to, te = json.load(open(runs / f"tuned_{b}.json", encoding="utf-8")), json.load(open(runs / f"tuned_ext_{b}.json", encoding="utf-8"))
        mac[f"valGanho{L}"] = pt(float(te["dqn"]["val_mean"] - to["dqn"]["val_mean"]))
    for b in ("cb4wd", "cb12wd"):
        L = LETTER[b]
        for v, nm in (("noreplay", "Replay"), ("notarget", "Alvo"), ("nodouble", "Double")):
            mac[f"abl{nm}{L}"] = pt(A[b][v]["mean"])
            mac[f"abl{nm}Lo{L}"], mac[f"abl{nm}Hi{L}"] = pt(A[b][v]["lo"]), pt(A[b][v]["hi"])
    k05 = next(it for it in S["cb12wd"] if it["name"] == "cb12wdx0.5")
    mac["sensKmeioC"], mac["sensKmeioCLo"], mac["sensKmeioCHi"] = pt(k05["mean"], 2), pt(k05["lo"], 2), pt(k05["hi"], 2)
    we = next(it for it in S["cb12wd"] if it["name"] == "cb12we")
    mac["sensFdsC"], mac["sensFdsCLo"], mac["sensFdsCHi"] = pt(we["mean"], 2), pt(we["lo"], 2), pt(we["hi"], 2)
    for name, nm in (("cb12wdx0.5", "Kmeio"), ("cb12we", "Fds")):                    # P3b: DQN reajustado nessas variantes
        rt = S["rt"][name]
        mac[f"sens{nm}RtC"], mac[f"sens{nm}RtCLo"], mac[f"sens{nm}RtCHi"] = pt(rt["mean"], 2), pt(rt["lo"], 2), pt(rt["hi"], 2)
        mac[f"sens{nm}Delta"], mac[f"sens{nm}DeltaLo"], mac[f"sens{nm}DeltaHi"] = pt(rt["delta"], 2), pt(rt["dlo"], 2), pt(rt["dhi"], 2)
    lat_p = runs / "latency.json"                                                     # P8: latência por decisão (scripts/31_latency.py)
    if lat_p.exists():
        lat = json.load(open(lat_p, encoding="utf-8"))
        for b in BASES:
            L = LETTER[b]
            mac[f"latRoll{L}"], mac[f"latNet{L}"] = pt(lat[b]["rollout_ms"], 0), pt(lat[b]["net_us"], 0)
            mac[f"latOrder{L}"] = f"{int(np.floor(np.log10(lat[b]['ratio'])))}"
    P8 = S["p8"]
    for b in BASES:                                                                   # P8a: limite clarividente; P8b: orçamento 4x
        L = LETTER[b]
        if b in P8["oracle"]:
            o = P8["oracle"][b]
            mac[f"clvStay{L}"], mac[f"clvHeur{L}"], mac[f"clvRoll{L}"] = pt(o["stay"]), pt(o["heur"]), pt(o["rollout"])
            mac[f"clvDqn{L}"], mac[f"clv{L}"] = pt(o["dqn"]), pt(o["clair"])
            mac[f"clvPct{L}"] = f"{100 * o['clair'] / o['stay']:.0f}"
            mac[f"fracHeur{L}"], mac[f"fracRoll{L}"], mac[f"fracDqn{L}"] = (f"{100 * o[k]:.0f}" for k in ("frac_heur", "frac_rollout", "frac_dqn"))
            mac[f"leftDqn{L}"] = f"{100 * (1 - o['frac_dqn']):.0f}"
            mac[f"leftRoll{L}"] = f"{100 * (1 - o['frac_rollout']):.0f}"
        if b in P8["budget"]:
            bd = P8["budget"][b]
            mac[f"bdgN{L}"], mac[f"bdgDqn{L}"] = str(bd["n"]), pt(bd["dqn_2m"])
            mac[f"bdgDqnLo{L}"], mac[f"bdgDqnHi{L}"] = pt(bd["dqn_2m_lo"]), pt(bd["dqn_2m_hi"])
            mac[f"bdgDelta{L}"], mac[f"bdgDeltaLo{L}"], mac[f"bdgDeltaHi{L}"] = pt(bd["delta"], 2), pt(bd["delta_lo"], 2), pt(bd["delta_hi"], 2)
            mac[f"bdgRoll{L}"] = pt(bd["rollout"])
            mac[f"bdgVsRoll{L}"], mac[f"bdgVsRollLo{L}"], mac[f"bdgVsRollHi{L}"] = pt(bd["vs_roll"], 2), pt(bd["vs_roll_lo"], 2), pt(bd["vs_roll_hi"], 2)
            mac[f"bdgGap{L}"] = pt(bd["gap_500k"], 2)
    SC_LETTER = {"cb4wd": "A", "cb8wd": "B", "cb12wd": "C", "cb20wd": "D", "cb30wd": "E"}
    for inst, r in P8["scale"].items():                                                 # P9: escala maior (4 -> 30 estações)
        L = SC_LETTER[inst]
        mac[f"scStay{L}"], mac[f"scHeur{L}"] = pt(r["stay"]), pt(r["heur"])
        if "rollout" in r:
            ro = r["rollout"]
            mac[f"scRollRed{L}"] = f"{ro['red']:.0f}"
            if ro.get("subset"):
                mac["scRollN"] = str(ro["n"])
                for tag, nm in (("500k", "Five"), ("2m", "TwoM")):
                    if tag in ro:
                        mac[f"scRollDqnRed{nm}{L}"] = f"{ro[tag]['dqn_red']:.0f}"
                        mac[f"scRollC{nm}{L}"], mac[f"scRollC{nm}Lo{L}"], mac[f"scRollC{nm}Hi{L}"] = pt(ro[tag]["c"]), pt(ro[tag]["c_lo"]), pt(ro[tag]["c_hi"])
        for tag, nm in (("500k", "Five"), ("2m", "TwoM")):
            if tag in r:
                t = r[tag]
                mac[f"scN{nm}{L}"] = str(t["n"])
                mac[f"scDqn{nm}{L}"], mac[f"scDqn{nm}Lo{L}"], mac[f"scDqn{nm}Hi{L}"] = pt(t["mean"]), pt(t["lo"]), pt(t["hi"])
                mac[f"scC{nm}{L}"], mac[f"scC{nm}Lo{L}"], mac[f"scC{nm}Hi{L}"] = pt(t["c1"]), pt(t["c1_lo"]), pt(t["c1_hi"])
                mac[f"scRed{nm}{L}"] = f"{t['red']:.0f}"
    rt30 = P8.get("scale_rt")
    if rt30 and "n" in rt30:                                                              # P9b: DQN reajustado em 30 estações
        mac["scRtCfg"], mac["scRtSame"] = str(rt30["cfg_id"]), "sim" if rt30["same"] else "n\\~ao"
        mac["scRtVal"], mac["scRtValSecond"], mac["scRtNConf"] = pt(-rt30["val_mean"]), pt(-rt30["val_second"]), str(rt30["n_configs"])
        mac["scRtN"], mac["scRtDqn"], mac["scRtDqnLo"], mac["scRtDqnHi"] = str(rt30["n"]), pt(rt30["mean"]), pt(rt30["lo"]), pt(rt30["hi"])
        mac["scRtDelta"], mac["scRtDeltaLo"], mac["scRtDeltaHi"] = pt(rt30["delta"]), pt(rt30["delta_lo"]), pt(rt30["delta_hi"])
        mac["scRtC"], mac["scRtCLo"], mac["scRtCHi"], mac["scRtRed"] = pt(rt30["c1"]), pt(rt30["c1_lo"]), pt(rt30["c1_hi"]), f"{rt30['red']:.0f}"
    for inst, r in P8["scale"].items():
        mac[f"scHeurPct{SC_LETTER[inst]}"] = f"{100 * (r['stay'] - r['heur']) / r['stay']:.0f}"
    for it in P8["p3c"]:                                                                  # P3c: sensibilidade do DQN com 10 sementes
        if "pending" in it:
            continue
        nm = {"cb12wdx0.5": "Kmeio", "cb12we": "Fds"}[it["name"]]
        mac[f"sens{nm}TenN"] = str(it["n_orig"])
        mac[f"sens{nm}TenC"], mac[f"sens{nm}TenCLo"], mac[f"sens{nm}TenCHi"] = pt(it["c1_orig"], 2), pt(it["c1_orig_lo"], 2), pt(it["c1_orig_hi"], 2)
        mac[f"sens{nm}TenRtC"], mac[f"sens{nm}TenRtCLo"], mac[f"sens{nm}TenRtCHi"] = pt(it["c1_rt"], 2), pt(it["c1_rt_lo"], 2), pt(it["c1_rt_hi"], 2)
        mac[f"sens{nm}TenDelta"], mac[f"sens{nm}TenDeltaLo"], mac[f"sens{nm}TenDeltaHi"] = pt(it["delta"], 2), pt(it["delta_lo"], 2), pt(it["delta_hi"], 2)
    for b, rs in P8["reseed"].items():                                                    # P8c: reavaliação das 4 melhores configs
        L = LETTER[b]
        mac[f"reseedChg{L}"] = "sim" if rs["changed"] else "n\\~ao"
        if rs.get("n"):
            mac[f"reseedDiff{L}"], mac[f"reseedDiffLo{L}"], mac[f"reseedDiffHi{L}"] = pt(rs["diff"], 2), pt(rs["lo"], 2), pt(rs["hi"], 2)
    cfgs = {b: (R[b]["rollout"]["h"], R[b]["rollout"]["K"]) for b in BASES}
    same = len(set(cfgs.values())) == 1
    h0, k0 = cfgs["cb4wd"]
    mac["rollCfg"] = (f"$h={h0}$ e $K={k0}$ nas tr\\^es escalas" if same else "; ".join(f"$h={cfgs[b][0]}$, $K={cfgs[b][1]}$ com {NST[b]} esta\\c{{c}}\\~oes" for b in BASES))
    with open(root / "artigo" / "numeros.tex", "w", encoding="utf-8") as f:
        f.write("% GERADO por scripts/25_make_report.py — não editar. Sufixos A/B/C = 4/8/12 estações. Requer \\usepackage{xspace}.\n")
        for k, v in mac.items():
            f.write(f"\\newcommand{{\\{k}}}{{{tex_num(v)}\\xspace}}\n")

    # tabela principal
    with open(tdir / "resultados.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n& \\multicolumn{3}{c}{Teste sintético} & \\multicolumn{3}{c}{24 dias reais} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMétodo & 4 est. & 8 est. & 12 est. & 4 est. & 8 est. & 12 est. \\\\\n\\midrule\n")
        for m in ORDER:
            cells = []
            for key in ("test", "real"):
                for b in BASES:
                    d = R[b][m]
                    if m in ("stay", "heur", "rollout"):
                        cells.append(pt(d[key]))
                    else:
                        mu, lo, hi = mean_ci(d[key])
                        cells.append(f"{pt(mu)} [{pt(lo)}; {pt(hi)}]")
            f.write(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if P8["oracle"]:
        with open(tdir / "clarividente.tex", "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
                    "Escala & Parada & Heur\\'istica & \\emph{Rollout} & DQN & Clarividente & \\multicolumn{2}{c}{Ganho capturado (\\%)} \\\\\n"
                    "\\cmidrule(lr){7-8}\n & & & & & & \\emph{Rollout} & DQN \\\\\n\\midrule\n")
            for b in BASES:
                if b in P8["oracle"]:
                    o = P8["oracle"][b]
                    f.write(f"{NST[b]} est. & {pt(o['stay'])} & {pt(o['heur'])} & {pt(o['rollout'])} & {pt(o['dqn'])} & {pt(o['clair'])} & "
                            f"{100 * o['frac_rollout']:.0f} & {100 * o['frac_dqn']:.0f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    return mac


def write_results_md(R, A, S, root):
    L = ["# 06 — Resultados (gerado por `scripts/25_make_report.py`; nenhum número digitado à mão)\n",
         "> Demanda perdida por dia (menor é melhor), média de 10 sementes de treino com IC 95% t, salvo indicação. Referência de heurística e rollout "
         "de cabeçalho escolhidos em validação. Fonte: `runs/*.jsonl`; configurações e desvios: `DESVIOS.md`.\n",
         "## 1. Resultado principal\n",
         "![principal](figuras/fig_principal.png)\n",
         "| escala | parada | heurística calibrada | rollout (h, K) | DQN [IC95%] | redução vs heurística | DQN − rollout [IC95%] | dias reais: heurística | rollout | DQN |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for b in BASES:
        d = R[b]
        mu, lo, hi = mean_ci(d["dqn"]["test"])
        ro = d["rollout"]
        dm, dlo, dhi = mean_ci(ro["test"] - d["dqn"]["test"])
        L.append(f"| {NST[b]} est. | {pt(d['stay']['test'])} | {pt(d['heur']['test'])} | {pt(ro['test'])} ({ro['h']}, {ro['K']}) | "
                 f"{pt(mu)} [{pt(lo)}; {pt(hi)}] | {100 * (d['heur']['test'] - mu) / d['heur']['test']:.0f}% | {pt(dm)} [{pt(dlo)}; {pt(dhi)}] | "
                 f"{pt(d['heur']['real'])} | {pt(ro['real'])} | {pt(float(np.mean(d['dqn']['real'])))} |")
    L += ["\n(DQN − rollout > 0: o DQN perde menos demanda que o rollout.)\n", "## 2. Todos os métodos\n", "![curvas](figuras/fig_curvas.png)\n",
          "| método | " + " | ".join(f"{NST[b]} est. (teste)" for b in BASES) + " | " + " | ".join(f"{NST[b]} est. (dias reais)" for b in BASES) + " |",
          "|---|" + "---|" * 6]
    for m in ORDER:
        cells = []
        for key in ("test", "real"):
            for b in BASES:
                d = R[b][m]
                if m in ("stay", "heur", "rollout"):
                    cells.append(pt(d[key]))
                else:
                    mu, lo, hi = mean_ci(d[key])
                    cells.append(f"{pt(mu)} [{pt(lo)}; {pt(hi)}]")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["\n## 3. Efeito de estender a busca de hiperparâmetros (P1)\n", "![busca](figuras/fig_busca.png)\n",
          "| escala | DQN: 8 configs | DQN: 16 configs | ganho |", "|---|---|---|---|"]
    for b in BASES:
        o, n = R[b]["orig"]["dqn"].mean(), R[b]["dqn"]["test"].mean()
        L.append(f"| {NST[b]} est. | {pt(o)} | {pt(n)} | {pt(o - n)} |")
    L += ["\n## 4. Ablação do DQN (P2)\n", "![ablação](figuras/fig_ablacao.png)\n",
          "| escala | variante | piora ao remover [IC95%] | n |", "|---|---|---|---|"]
    for b in ("cb4wd", "cb12wd"):
        for v, nm in (("noreplay", "sem replay"), ("notarget", "sem rede-alvo"), ("nodouble", "sem Double DQN"), ("vanilla", "as três (exploratória)")):
            d = A[b][v]
            L.append(f"| {NST[b]} est. | {nm} | {pt(d['mean'], 2)} [{pt(d['lo'], 2)}; {pt(d['hi'], 2)}] | {d['n']} |")
    L += ["\n## 5. Sensibilidade do simulador (P3; n = 5, exploratória)\n", "![sensibilidade](figuras/fig_sensibilidade.png)\n",
          "| escala | variante | parada | heurística | DQN | DQN − heurística [IC95%] |", "|---|---|---|---|---|---|"]
    for b in ("cb4wd", "cb12wd"):
        for it in S[b]:
            L.append(f"| {NST[b]} est. | {it['label']} | {pt(it['stay'])} | {pt(it['heur'])} | {pt(it['dqn'])} | "
                     f"{pt(it['mean'], 2)} [{pt(it['lo'], 2)}; {pt(it['hi'], 2)}] |")
    if S.get("rt"):
        L += ["\n### 5.1 DQN reajustado nas variantes em que a conclusão mudou (P3b; losangos da figura)\n",
              "| variante | config (nº no pool de 16) | DQN reajustado − heurística [IC95%] | reajustado − sem reajuste [IC95% Welch] | n |", "|---|---|---|---|---|"]
        for name, rt in S["rt"].items():
            L.append(f"| {rt['label']} | {rt['cfg_id']}{' (= a do P1)' if rt['same'] else ''} | {pt(rt['mean'], 2)} [{pt(rt['lo'], 2)}; {pt(rt['hi'], 2)}] | "
                     + ("—" if rt["same"] else f"{pt(rt['delta'], 2)} [{pt(rt['dlo'], 2)}; {pt(rt['dhi'], 2)}]") + f" | {rt['n']} |")
    P8 = S.get("p8") or {}
    if P8.get("oracle") or P8.get("budget") or P8.get("scale"):
        L += ["\n## 6. Âncoras e robustez da conclusão central (P8; DESVIOS.md 2026-10-18)\n"]
        if P8.get("oracle"):
            L += ["### 6.1 Limite clarividente (planejador que conhece a demanda do cenário; 200 primeiros cenários de teste; demanda perdida, menor é melhor)\n",
                  "| escala | parada | heurística | rollout | DQN | clarividente (limite inferior do ótimo clarividente) | ganho capturado: rollout | DQN | limite ≈ 0? |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for b in BASES:
                if b in P8["oracle"]:
                    o = P8["oracle"][b]
                    L.append(f"| {NST[b]} est. | {pt(o['stay'])} | {pt(o['heur'])} | {pt(o['rollout'])} | {pt(o['dqn'])} | {pt(o['clair'])} | "
                             f"{100 * o['frac_rollout']:.0f}% | {100 * o['frac_dqn']:.0f}% | {'sim (≤ 5% da parada)' if o['near_zero'] else 'não'} |")
            L += ["\n(O clarividente não é atingível por política que não vê o futuro; com largura finita é limite inferior do ótimo clarividente, logo 'ganho capturado' é "
                  "limite superior da fração verdadeira. Não se afirma distância ao ótimo não antecipativo.)\n"]
        if P8.get("budget"):
            L += ["### 6.2 DQN com 4× o orçamento de treino (2 milhões de passos; config do P1; sem reajuste)\n",
                  "| escala | n (2 M) | DQN 500 mil | DQN 2 M [IC95%] | redução da demanda perdida com 4× [IC95% Welch] | rollout | DQN(2 M) − rollout [IC95%] |", "|---|---|---|---|---|---|---|"]
            for b in BASES:
                if b in P8["budget"]:
                    bd = P8["budget"][b]
                    L.append(f"| {NST[b]} est. | {bd['n']} | {pt(bd['dqn_500k'])} | {pt(bd['dqn_2m'])} [{pt(bd['dqn_2m_lo'])}; {pt(bd['dqn_2m_hi'])}] | "
                             f"{pt(bd['delta'], 2)} [{pt(bd['delta_lo'], 2)}; {pt(bd['delta_hi'], 2)}] | {pt(bd['rollout'])} | "
                             f"{pt(bd['vs_roll'], 2)} [{pt(bd['vs_roll_lo'], 2)}; {pt(bd['vs_roll_hi'], 2)}] |")
            L += ["\n(DQN(2 M) − rollout em reward: negativo = o rollout continua melhor.)\n"]
        if P8.get("scale"):
            L += ["### 6.0 Escala maior: 20 e 30 estações (P9; DQN com a config de 12 estações, sem reajuste)\n",
                  "| escala | parada | heurística | DQN 500 mil | redução | DQN 2 M | redução | C1 2 M [IC95%] |", "|---|---|---|---|---|---|---|---|"]
            for inst, r in P8["scale"].items():
                c = []
                for tag in ("500k", "2m"):
                    t = r.get(tag)
                    c += [f"{pt(t['mean'])} [{pt(t['lo'])}; {pt(t['hi'])}] (n = {t['n']})", f"{t['red']:.0f}%"] if t else ["(pendente)", ""]
                t2 = r.get("2m")
                L.append(f"| {r['n_st']} est. | {pt(r['stay'])} | {pt(r['heur'])} | " + " | ".join(c) + " | "
                         + (f"{pt(t2['c1'], 1)} [{pt(t2['c1_lo'], 1)}; {pt(t2['c1_hi'], 1)}]" if t2 else "") + " |")
            L += ["\n(Rollout em 20/30 estações: (h, K) = (3, 64) sem ajuste, nos primeiros cenários de teste; o clarividente não foi rodado nessas escalas; as estações acrescentadas têm volume maior que as 12 legadas.)\n"]
        if P8.get("p3c"):
            L += ["### 6.3 Sensibilidade do DQN com 10 sementes (variantes de 12 est. em que a conclusão mudou)\n",
                  "| variante | n | C1 sem reajuste [IC95%] | C1 reajustado [IC95%] | reajustado − sem reajuste [IC95% Welch] |", "|---|---|---|---|---|"]
            for it in P8["p3c"]:
                if "pending" not in it:
                    L.append(f"| {it['label']} | {it['n_orig']} / {it['n_rt']} | {pt(it['c1_orig'], 2)} [{pt(it['c1_orig_lo'], 2)}; {pt(it['c1_orig_hi'], 2)}] | "
                             f"{pt(it['c1_rt'], 2)} [{pt(it['c1_rt_lo'], 2)}; {pt(it['c1_rt_hi'], 2)}] | {pt(it['delta'], 2)} [{pt(it['delta_lo'], 2)}; {pt(it['delta_hi'], 2)}] |")
        if P8.get("reseed"):
            L += ["\n### 6.4 Reavaliação das 4 melhores configs com 4 sementes de ajuste\n"]
            for b, rs in P8["reseed"].items():
                txt = "seleção confirmada" if not rs["changed"] else (f"mudou (nº {rs['p1']} → nº {rs['selected']}); finais de 10 sementes − primário = "
                                                                      f"{pt(rs['diff'], 2)} [{pt(rs['lo'], 2)}; {pt(rs['hi'], 2)}]" if rs.get("n") else "mudou; finais pendentes")
                L.append(f"- {NST[b]} est.: {txt}")
    L += ["\n## 7. Limitações a declarar\n",
          "- Simulador sem roteamento: todo deslocamento custa 1 intervalo; ocupação inicial dos dias reais desconhecida (50% ± 2); "
          "o modelo de Poisson superestima a dificuldade vs dias reais.",
          "- O rollout usa o modelo verdadeiro da demanda; o DQN aprende só de amostras. O melhor rollout estava no limite da grade (h, K).",
          "- O A2C é uma adaptação (entrada e ação do nosso simulador); a variante \"publicado\" não foi ajustada e a \"ajustado\" usa o mesmo orçamento (16 configs, 2 sementes de ajuste) dos demais aprendizes; nenhuma prova que o método original falha no problema de Seo et al.",
          "- Sensibilidade: aprendizes sem reajuste por variante (exceto o DQN nas duas variantes de 12 estações em que a conclusão mudou, reajustado no P3b), "
          "5 sementes; a heurística é recalibrada em todas as variantes. Dados Tembici: ainda sem hold-out (janela < 4 semanas completas)."]
    (root / "06_resultados.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def check_claims(R, A, S):
    """Afirmações QUALITATIVAS do texto (artigo/main.tex) confrontadas com os dados; ALERTA = texto desatualizado."""
    out = []
    for b in BASES:
        d = R[b]
        dq = d["dqn"]["test"]
        out.append((bool((dq < d["heur"]["test"]).all()), f"{b}: DQN melhor que a heurística em todas as sementes"))
        out.append((all(float(d[m]["test"].mean()) > d["heur"]["test"] for m in ("linear", "tab_station", "tab_cluster")), f"{b}: linear e tabulares não superam a heurística"))
        diff = d["rollout"]["test"] - dq                       # > 0: DQN melhor
        lo, hi = ci_t(diff)
        want = "dqn" if b == "cb4wd" else "rollout"
        got = "dqn" if lo > 0 else ("rollout" if hi < 0 else "empate")
        out.append((got == want, f"{b}: vantagem no teste sintético = {got} (texto: {want})"))
        out.append((float(np.mean(d["dqn"]["real"])) <= d["rollout"]["real"] + 1e-9 or abs(float(np.mean(d["dqn"]["real"])) - d["rollout"]["real"]) < 2.0,
                    f"{b}: nos dias reais o DQN fica à frente ou empata com o rollout"))
    out.append((float(R["cb4wd"]["pg"]["test"].mean()) < R["cb4wd"]["heur"]["test"] and
                all(float(R[b]["pg"]["test"].mean()) > R[b]["heur"]["test"] for b in ("cb8wd", "cb12wd")), "PG supera a heurística só em 4 estações"))
    out.append((all(float(R[b]["a2c_t"]["test"].mean()) < float(R[b]["a2c"]["test"].mean()) for b in BASES), "A2C ajustado melhora o publicado nas 3 escalas"))
    out.append((float(R["cb4wd"]["a2c_t"]["test"].mean()) < R["cb4wd"]["heur"]["test"]
                and all(float(R[b]["a2c_t"]["test"].mean()) > R[b]["heur"]["test"] for b in ("cb8wd", "cb12wd")), "A2C ajustado supera a heurística só em 4 estações"))
    out.append((all(float(R[b]["a2c_t"]["test"].mean()) > float(R[b]["dqn"]["test"].mean()) for b in BASES), "A2C ajustado fica atrás do DQN nas 3 escalas"))
    out.append((all(float(R[b]["a2c_t"]["test"].mean()) < R[b]["stay"]["test"] for b in BASES), "A2C ajustado fica melhor que a parada nas 3 escalas"))
    out.append((all(abs(float(R[b]["a2c"]["test"].mean()) - R[b]["stay"]["test"]) / R[b]["stay"]["test"] < 0.10 for b in ("cb8wd", "cb12wd")),
                "A2C publicado termina a menos de 10% da parada em 8 e 12 estações"))
    out.append((all(A[b]["noreplay"]["lo"] > 0 for b in A), "ablação: sem replay piora (IC exclui 0) em 4 e 12 estações"))
    out.append((all(A[b][v]["lo"] <= 0 <= A[b][v]["hi"] for b in A for v in ("notarget", "nodouble")), "ablação: rede-alvo e Double DQN sem efeito detectável"))
    out.append((abs(A["cb12wd"]["noreplay"]["var_mean"] - R["cb12wd"]["stay"]["test"]) / R["cb12wd"]["stay"]["test"] < 0.05,
                "ablação: sem replay em 12 estações termina a menos de 5% da parada (o agente não aprende)"))
    cal4 = [it for it in S["cb4wd"] if not it["name"].startswith("legacy")]
    out.append((all(it["lo"] > 0 for it in cal4), "sensibilidade 4 est.: DQN supera a heurística em todas as variantes calibradas"))
    k05 = next(it for it in S["cb12wd"] if it["name"] == "cb12wdx0.5")
    we = next(it for it in S["cb12wd"] if it["name"] == "cb12we")
    out.append((k05["hi"] < 0, "sensibilidade 12 est.: DQN significativamente pior com demanda × 0,5"))
    out.append((we["lo"] <= 0 <= we["hi"], "sensibilidade 12 est.: fim de semana sem diferença detectável"))
    P8 = S["p8"]
    for b in BASES:
        if b in P8["oracle"]:
            o = P8["oracle"][b]
            out.append((o["clair"] < 0.10 * o["stay"], f"P8a {b}: o clarividente perde menos de 10% da demanda perdida da parada (limite frouxo)"))
    if {"cb4wd", "cb8wd", "cb12wd"} <= set(P8["budget"]):
        b4, b8, b12 = (P8["budget"][k] for k in ("cb4wd", "cb8wd", "cb12wd"))
        out.append((b4["delta_lo"] > 0 and b8["delta_lo"] > 0 and b12["delta_lo"] > 0, "P8b: 4× o orçamento melhora o DQN nas três escalas (IC de Welch exclui 0)"))
        out.append((b4["vs_roll_lo"] > 0, "P8b 4 est.: com 4× o orçamento o DQN continua melhor que o rollout (IC exclui 0)"))
        out.append((b8["vs_roll_lo"] <= 0 <= b8["vs_roll_hi"], "P8b 8 est.: com 4× o orçamento o DQN empata com o rollout (IC inclui 0)"))
        out.append((b12["vs_roll_hi"] < 0, "P8b 12 est.: com 4× o orçamento o rollout continua melhor que o DQN (IC exclui 0)"))
    for b, rs in P8["reseed"].items():
        if rs.get("n"):
            out.append((rs["lo"] <= 0 <= rs["hi"], f"P8c {b}: os finais com a config reavaliada não diferem detectavelmente dos primários"))
    sc = P8["scale"]
    if P8.get("scale_rt"):
        out.append((P8["scale_rt"]["same"], "P9b: em 30 estações o pool de 16 configs escolhe a mesma config de 12 estações (seleção confirmada)"))
    if {"cb4wd", "cb8wd", "cb12wd", "cb20wd", "cb30wd"} <= set(sc) and all("500k" in sc[k] and "2m" in sc[k] for k in sc):
        order = ("cb4wd", "cb8wd", "cb12wd", "cb20wd", "cb30wd")
        for tag, nm in (("500k", "500 mil"), ("2m", "2 milhões")):
            reds = [sc[k][tag]["red"] for k in order]
            out.append((all(a > b for a, b in zip(reds, reds[1:])), f"P9: a redução relativa do DQN ({nm} de passos) vs heurística diminui monotonicamente de 4 a 30 estações"))
        rr = [sc[k]["rollout"]["red"] for k in order if "rollout" in sc[k]]
        out.append((len(rr) == 5 and all(x > y for x, y in zip(rr, rr[1:])), "P9c: a redução relativa do rollout também diminui monotonicamente de 4 a 30 estações"))
        out.append((all(sc[k]["rollout"][tag]["c_hi"] < 0 for k in ("cb20wd", "cb30wd") for tag in ("500k", "2m") if tag in sc[k].get("rollout", {})),
                    "P9c: com 20 e 30 estações o rollout (sem ajuste) é melhor que o DQN, com 500 mil e 2 milhões de passos (IC exclui 0)"))
        out.append((all(sc[k][tag]["c1_lo"] > 0 for k in ("cb20wd", "cb30wd") for tag in ("500k", "2m")),
                    "P9: o DQN supera a heurística calibrada em 20 e 30 estações (IC de C1 exclui 0), com 500 mil e 2 milhões de passos"))
    p3c = {it["name"]: it for it in P8["p3c"] if "pending" not in it}
    if {"cb12wdx0.5", "cb12we"} <= set(p3c):
        k, w = p3c["cb12wdx0.5"], p3c["cb12we"]
        out.append((k["c1_orig_hi"] < 0 and k["c1_rt_hi"] < 0, "P3c (10 sementes): com demanda × 0,5 o DQN segue significativamente pior, com e sem reajuste"))
        out.append((w["c1_orig_lo"] > 0 and w["c1_rt_lo"] > 0, "P3c (10 sementes): no fim de semana o DQN passa a ser significativamente melhor, com e sem reajuste"))
        out.append((k["delta_lo"] <= 0 <= k["delta_hi"] and w["delta_lo"] <= 0 <= w["delta_hi"], "P3c: o reajuste não altera detectavelmente o DQN (10 sementes) nas duas variantes"))
    rt = S["rt"]
    out.append((all(rt[n]["lo"] <= 0 <= rt[n]["hi"] for n in ("cb12wdx0.5", "cb12we")),
                "P3b: com o DQN reajustado, o IC95% de C1 inclui 0 nas duas variantes"))
    out.append((all(rt[n]["dlo"] <= 0 <= rt[n]["dhi"] for n in ("cb12wdx0.5", "cb12we")),
                "P3b: o reajuste não altera detectavelmente o DQN (IC de Welch inclui 0) nas duas variantes"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    setup_style()
    outdir = ROOT / "figuras"
    (outdir / "dados").mkdir(parents=True, exist_ok=True)
    R, A, S = collect(runs), ablation(runs), sensitivity(runs)
    S["rt"] = retuned(runs)
    S["p8"] = {**AN30.p8_data(runs), "p3c": AN30.p3c_data(runs), "reseed": AN30.reseed_data(runs), "scale": AN30.scale_data(runs), "scale_rt": AN30.scale_retuned_data(runs)}
    fig_main(R, outdir)
    fig_curves(R, outdir)
    fig_pool(R, outdir)
    fig_ablation(A, outdir)
    fig_sensitivity(S, outdir)
    if len(S["p8"]["scale"]) >= 4 and any("500k" in r for k, r in S["p8"]["scale"].items() if k in ("cb20wd", "cb30wd")):
        fig_scale(S["p8"]["scale"], outdir, S["p8"].get("scale_rt"))
    mac = write_tables_and_macros(R, A, S, ROOT, runs)
    for ok, msg in check_claims(R, A, S):
        print(("ok     " if ok else "ALERTA ") + msg)
    write_results_md(R, A, S, ROOT)
    print(f"figuras em {outdir}; {len(mac)} macros em artigo/numeros.tex; 06_resultados.md atualizado")


if __name__ == "__main__":
    main()
