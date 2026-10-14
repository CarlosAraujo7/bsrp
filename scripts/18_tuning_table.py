"""
18_tuning_table.py — resume runs/tuned_<inst>.json em tables/tuned_configs.md (configuração escolhida por método/escala,
val do ajuste da escolhida × da padrão). Somente stdlib.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = ["# Configurações ajustadas (gerado por `scripts/18_tuning_table.py`)\n",
         "> Ajuste: 8 configurações por método (a nº 0 é a padrão) × 2 sementes de ajuste (1000, 1001) × 200.000 passos; "
         "escolha pelo val médio dos 2 últimos pontos da curva. Maior (menos negativo) = melhor.\n",
         "| escala | método | cfg escolhida | val ajuste (escolhida) | val ajuste (padrão) | ganho | configuração |",
         "|---|---|---|---|---|---|---|"]
for inst in ("cb4wd", "cb8wd", "cb12wd"):
    d = json.load(open(ROOT / "runs" / f"tuned_{inst}.json", encoding="utf-8"))
    for m in ("dqn", "linear", "tab_station", "tab_cluster", "pg"):
        r = d[m]
        base = r["all_val_means"][0]
        cfg = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["cfg"].items())
        lines.append(f"| {inst} | {m} | {r['cfg_id']} | {r['val_mean']:.2f} | {base:.2f} | {r['val_mean']-base:+.2f} | {cfg} |")
(ROOT / "tables" / "tuned_configs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("ok")
