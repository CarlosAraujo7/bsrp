"""
15_real_data_checks.py — checagens dos fluxos reais do Citi Bike contra as suposições do simulador legado.

Responde: (a) chegadas ≈ 5× partidas nas 12 estações era defeito do pré-processamento legado?
          (b) que relação há entre os λ legados (com DEMAND_SCALE = 3,5) e os λ reais (escala 1)?
          (c) qual a fração de fluxo INTERNO entre as estações selecionadas (importa a conservação de bicicletas)?
          (d) quão desequilibrado é o fluxo líquido real (o "problema" de rebalanceamento)?
Saída: tables/real_data_checks.md (somente numpy).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import data as D                                      # noqa: E402
from bsrp import env as E                                       # noqa: E402


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def main():
    fl = D.Flows()
    out = ["# Checagens dos fluxos reais (Citi Bike, Jan–Jun 2024) — `scripts/15_real_data_checks.py`\n"]
    tr_all = fl.days("train")
    out.append(f"Dias: treino {len(tr_all)}, val {len(fl.days('val'))}, teste {len(fl.days('test'))}; "
               f"dias úteis de treino {len(fl.days('train', 0))}, fim de semana/feriado {len(fl.days('train', 1))}.\n")

    out.append("## (a)+(d) Partidas × chegadas e desequilíbrio líquido por instância (semanas de TREINO)\n")
    out.append("| instância | tipo de dia | partidas/estação/dia | chegadas/estação/dia | razão chegadas/partidas | |fluxo líquido| médio por estação-dia | pico |líquido| acumulado 06–22h (bikes) |")
    out.append("|---|---|---|---|---|---|---|")
    for n in (4, 8, 12):
        base = E.legacy_instance(n)
        rows = fl.rows_for(list(base.station_names))
        for dt, lab in ((0, "dia útil"), (1, "fim de semana")):
            days = fl.days("train", dt)
            sl = slice(D.OFFSET, D.OFFSET + E.N_INTERVALS)
            dep = fl.dep[rows][:, days, sl].astype(float)          # (n, days, 32)
            arr = fl.arr[rows][:, days, sl].astype(float)
            dep_d, arr_d = dep.sum(2).mean(1), arr.sum(2).mean(1)  # por estação
            net_cum = np.cumsum(arr - dep, axis=2)                  # acumulado ao longo do dia
            out.append(f"| {n} est. | {lab} | {f(dep_d.mean(),1)} | {f(arr_d.mean(),1)} | {f(arr_d.sum()/dep_d.sum(),3)} | "
                       f"{f(np.abs(arr.sum(2) - dep.sum(2)).mean(),1)} | {f(np.abs(net_cum).max(2).mean(),1)} |")

    out.append("\n## (b) λ legado (DEMAND_SCALE = 3,5) × λ real (escala 1), dia útil, 06–22h\n")
    out.append("| instância | partidas/estação/dia legado ×3,5 | partidas/estação/dia real | fator real/legado (×3,5 incluso) | chegadas legado ×3,5 | chegadas real |")
    out.append("|---|---|---|---|---|---|")
    for n in (4, 8, 12):
        leg = E.legacy_instance(n, 3.5)
        cal = D.calibrated_instance(n, fl, 0)
        ld, cd = leg.lam_dep.sum(0).mean(), cal.lam_dep.sum(0).mean()
        la, ca = leg.lam_arr.sum(0).mean(), cal.lam_arr.sum(0).mean()
        out.append(f"| {n} est. | {f(ld,1)} | {f(cd,1)} | {f(cd/ld,2)} | {f(la,1)} | {f(ca,1)} |")

    out.append("\n## (c) Fluxo interno entre as estações da instância (O–D, semanas de TREINO, viagens Manhattan→Manhattan)\n")
    out.append("| instância | partidas totais (treino) | viagens internas | % das partidas que terminam DENTRO da instância | % das chegadas que vêm de DENTRO |")
    out.append("|---|---|---|---|---|")
    for n in (4, 8, 12):
        base = E.legacy_instance(n)
        rows = np.array(fl.rows_for(list(base.station_names)))
        days = fl.days("train")
        dep_tot = fl.dep[rows][:, days].sum()
        arr_tot = fl.arr[rows][:, days].sum()
        internal = fl.od_cnt[:, :, rows][:, :, :, rows].sum()
        out.append(f"| {n} est. | {int(dep_tot):,} | {int(internal):,} | {f(100*internal/dep_tot,2)}% | {f(100*internal/arr_tot,2)}% |".replace(",", "."))
    out.append("\n> Se a fração interna é pequena, quase toda chegada vem de fora da instância e quase toda partida sai dela:"
               " o subsistema é **aberto** e o modelo de Poisson independente (sem conservação) é uma aproximação razoável"
               " para ele; a conservação só importaria para instâncias que cobrissem boa parte da rede.")
    p = ROOT / "tables" / "real_data_checks.md"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"escrito {p}")


if __name__ == "__main__":
    main()
