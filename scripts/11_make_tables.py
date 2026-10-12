"""
11_make_tables.py — gera tables/legacy_results.md a partir de results/**/*.json
(somente biblioteca padrão; nenhum número é digitado à mão).

Escopo: resultados LEGADOS (pré-protocolo; ver PROTOCOLO_RIGOR.md). Todos usam UMA semente de
treino (42); as 3 "sementes" abaixo (123/456/789) são de AVALIAÇÃO do simulador. Servem para
formular hipóteses, não para afirmações.

Uso: python scripts/11_make_tables.py
"""
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "tables" / "legacy_results.md"


def load(rel):
    p = R / rel
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def ms_stats(ms):
    """multiseed {seed: {reward, std}} -> (média das 3, amplitude entre sementes de avaliação, std por episódio)."""
    r = [v["reward"] for v in ms.values()]
    return mean(r), (max(r) - min(r)), mean(v["std"] for v in ms.values())


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def scale_table():
    rows = []
    spec = {
        4: [("Q-learning (estado por cluster)", "scaling/ql_4st.json"), ("DQN", "scaling/dqn_4st.json"),
            ("Threshold", "baselines/baselines_4st.json:threshold"), ("Rollout 1 passo", "baselines/baselines_4st.json:rollout")],
        8: [("Q-learning (estado por cluster)", "scaling/ql_8st.json"), ("DQN", "scaling/dqn_8st.json"),
            ("Threshold", "baselines/baselines_8st.json:threshold")],
        12: [("Q-learning (estado por cluster)", "scaling/ql_12st.json"), ("DQN", "scaling/dqn_12st.json"),
             ("Threshold", "baselines/baselines_12st.json:threshold")],
    }
    for n, items in spec.items():
        for label, ref in items:
            path, _, key = ref.partition(":")
            d = load(path)
            if d is None:
                continue
            d = d[key] if key else d
            m, rng, sd = ms_stats(d["multiseed"])
            ep = d.get("n_episodes", "—")
            rows.append([n, label, ep if key == "" else "—", f(m), f(rng), f(sd), Path(path).name + (f":{key}" if key else "")])
        rows.append([n, "Aleatória / Gulosa (multi-semente)", "—", "não salvo", "—", "—", "só stdout; ausente dos JSONs"])
    return md(["Estações", "Política", "Ep. treino", "Reward médio (3×100 ep)", "Amplitude entre sem. de AVALIAÇÃO",
               "Std por episódio", "Fonte"], rows)


def single_seed_table():
    rows = []
    d = load("qtable_improved_A_fix_aliasing_summary.json")
    for k, v in d["baselines"].items():
        rows.append([k, "12", "123", "100", f(v["reward"]), f(v["std"]), "qtable_improved_*_summary.json"])
    return md(["Política", "Estações", "Semente aval.", "Episódios", "Reward", "Std", "Fonte"], rows)


def dqn8_table():
    rows = []
    paths = ["scaling/dqn_8st.json"] + [f"scaling/dqn_8st_ep{e}.json" for e in (11500, 12000, 13500, 15500, 17500)]
    for p in paths:
        d = load(p)
        if d is None:
            continue
        m, rng, sd = ms_stats(d["multiseed"])
        rows.append([d["n_episodes"], f(m), f(rng), Path(p).name])
    return md(["Ep. treino (checkpoint)", "Reward médio (3×100 ep)", "Amplitude entre sem. aval.", "Fonte"], rows)


def ablation_table():
    full = load("scaling/dqn_12st.json")
    a9 = load("scaling/ablation_9500ep.json")
    rows = [["Full DQN", full["n_episodes"], f(ms_stats(full["multiseed"])[0]), "dqn_12st.json"]]
    for k in ("no_replay", "no_target", "no_double"):
        rows.append([k, a9[k]["n_episodes"], f(a9[k]["mean"]), "ablation_9500ep.json"])
    for k in ("no_replay", "no_target", "no_double"):
        d = load(f"scaling/abl_{k}.json")
        rows.append([k + " (abl_*.json)", "não registrado", f(d["reward"]), f"abl_{k}.json"])
    return md(["Variante", "Ep. treino", "Reward médio (3×100 ep)", "Fonte"], rows)


def control_table():
    c = load("scaling/ql_control.json")
    q4 = load("scaling/ql_4st.json")
    rows = []
    for key, label in (("cluster", "cluster (ql_control)"), ("perstation", "por estação binário (ql_control)"),
                       ("perstation_18k", "por estação binário, 18k (ql_control)")):
        d = c[key]
        last = d["history"][-1]
        rows.append([label, last["ep"], d["n_states"], f(d["mean"]), f(d["std"]), f(last["eval_r"], 2)])
    m4 = ms_stats(q4["multiseed"])
    rows.append(["cluster (ql_4st.json)", q4["n_episodes"], "—", f(m4[0]), f(m4[2]), "—"])
    return md(["Representação (fonte)", "Ep. treino", "Estados visitados", "Reward médio", "Std", "Último eval de 20 ep"], rows)


def space_table():
    rows = []
    for n in (4, 8, 12):
        with open(ROOT / f"station_selected_{n}.csv", encoding="utf-8") as fh:
            cl = [int(r["cluster"]) for r in csv.DictReader(fh)]
        k = max(cl) + 1
        rows.append([n, k, f"{4**k * n * 4 * 32:,}".replace(",", "."), f"{2**n * n * 4 * 32:,}".replace(",", ".")])
    return md(["Estações", "Clusters no simulador", "Espaço cluster (4 níveis)", "Espaço por estação (binário)"], rows)


def variants_table():
    rows = []
    for tag in ("A_fix_aliasing", "B_qlambda", "C_shaping", "D_combined"):
        d = load(f"qtable_improved_{tag}_summary.json")
        rows.append([tag, d["episodes"], d["occ_levels"], f(d["qlearning"]["reward"]), f(d["qlearning"]["std"]),
                     f(d["coverage_pct"], 1) + "%"])
    return md(["Variante", "Ep.", "L", "Reward (sem. 123, 100 ep)", "Std", "Cobertura"], rows)


def main():
    parts = [
        "# Resultados legados — gerado por `scripts/11_make_tables.py`\n",
        "> **Não editar à mão.** Todos os resultados abaixo vêm de **uma única semente de treino (42)**; "
        "as \"sementes\" 123/456/789 são de avaliação no simulador (medem ruído do simulador, não do algoritmo). "
        "Servem para formular hipóteses, não para afirmações.\n",
        "## T1. Escalabilidade (como salvo)\n", scale_table(),
        "\n## T2. Política gulosa/aleatória — única fonte salva (semente 123, 100 ep, 12 estações)\n", single_seed_table(),
        "\n## T3. DQN 8 estações — checkpoints avaliados nas MESMAS sementes de avaliação\n", dqn8_table(),
        "\n## T4. Ablação do DQN (12 estações) — registros coexistentes e não equivalentes\n", ablation_table(),
        "\n## T5. Controle de representação (4 estações)\n", control_table(),
        "\n## T6. Tamanho do espaço de estados tabular por instância\n", space_table(),
        "\n## T7. Variantes tabulares A–D (12 estações)\n", variants_table(),
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("escrito", OUT)


if __name__ == "__main__":
    main()
