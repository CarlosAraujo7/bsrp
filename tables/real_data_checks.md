# Checagens dos fluxos reais (Citi Bike, Jan–Jun 2024) — `scripts/15_real_data_checks.py`

Dias: treino 112, val 35, teste 35; dias úteis de treino 76, fim de semana/feriado 36.

## (a)+(d) Partidas × chegadas e desequilíbrio líquido por instância (semanas de TREINO)

| instância | tipo de dia | partidas/estação/dia | chegadas/estação/dia | razão chegadas/partidas | |fluxo líquido| médio por estação-dia | pico |líquido| acumulado 06–22h (bikes) |
|---|---|---|---|---|---|---|
| 4 est. | dia útil | 199,8 | 199,6 | 0,999 | 14,4 | 29,1 |
| 4 est. | fim de semana | 155,7 | 155,6 | 1,000 | 13,1 | 19,7 |
| 8 est. | dia útil | 196,1 | 196,5 | 1,002 | 10,3 | 23,6 |
| 8 est. | fim de semana | 137,0 | 138,6 | 1,012 | 9,6 | 15,8 |
| 12 est. | dia útil | 205,0 | 204,3 | 0,996 | 11,3 | 28,0 |
| 12 est. | fim de semana | 141,8 | 141,2 | 0,995 | 10,6 | 17,9 |

## (b) λ legado (DEMAND_SCALE = 3,5) × λ real (escala 1), dia útil, 06–22h

| instância | partidas/estação/dia legado ×3,5 | partidas/estação/dia real | fator real/legado (×3,5 incluso) | chegadas legado ×3,5 | chegadas real |
|---|---|---|---|---|---|
| 4 est. | 66,2 | 199,8 | 3,02 | 66,2 | 199,6 |
| 8 est. | 76,3 | 196,1 | 2,57 | 76,3 | 196,5 |
| 12 est. | 80,3 | 205,0 | 2,55 | 80,3 | 204,3 |

## (c) Fluxo interno entre as estações da instância (O–D, semanas de TREINO, viagens Manhattan→Manhattan)

| instância | partidas totais (treino) | viagens internas | % das partidas que terminam DENTRO da instância | % das chegadas que vêm de DENTRO |
|---|---|---|---|---|
| 4 est. | 91.694 | 1.841 | 2.01% | 2.02% |
| 8 est. | 172.900 | 5.367 | 3.10% | 3.11% |
| 12 est. | 270.489 | 12.217 | 4.52% | 4.54% |

> Se a fração interna é pequena, quase toda chegada vem de fora da instância e quase toda partida sai dela: o subsistema é **aberto** e o modelo de Poisson independente (sem conservação) é uma aproximação razoável para ele; a conservação só importaria para instâncias que cobrissem boa parte da rede.
