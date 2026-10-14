# P3 — sensibilidade do simulador (gerado por `scripts/21_analyze_p1_p2.py --which p3`)

> Desenho e critério fixados em `DESVIOS.md` (2026-10-14), **antes** de rodar. 5 sementes (0–4); aprendizes com a config escolhida no P1 na escala-base, **sem reajuste por variante**; a heurística é **recalibrada em validação em cada variante** (viés contra os aprendizes). Reward de TESTE (1000 cenários sintéticos novos por variante); maior = melhor. C1 = DQN − melhor heurística, diferença por semente (a heurística é determinística), IC95% t; 'significativo' = IC exclui 0. Análise exploratória: sem correção entre variantes. K = 1 usa as mesmas 5 primeiras sementes dos resultados primários.


## Escala-base cb4wd

| variante | n sem. | parada | melhor heurística (por val) | DQN [IC95%] | C1: DQN − heurística [IC95%] | redução vs heurística | linear | tab. estação | tab. cluster |
|---|---|---|---|---|---|---|---|---|---|
| K = 0,5 (`cb4wdx0.5`) | 5 | -7,34 | -2,62 (greedy_cal2) | -0,63 [-0,68; -0,58] | 1,99 [1,94; 2,04] | 76% | -7,59 | -8,20 | -10,95 |
| K = 0,75 (`cb4wdx0.75`) | 5 | -23,48 | -10,58 (greedy_cal2) | -3,02 [-3,26; -2,79] | 7,56 [7,32; 7,79] | 71% | -22,23 | -21,37 | -23,47 |
| K = 1 (primário) (`cb4wd`) | 5 | -46,94 | -24,46 (greedy_cal2) | -9,85 [-10,59; -9,12] | 14,61 [13,87; 15,34] | 60% | -36,58 | -40,44 | -42,81 |
| K = 1,5 (`cb4wdx1.5`) | 5 | -105,12 | -71,44 (greedy_cal2) | -35,55 [-36,52; -34,57] | 35,89 [34,92; 36,87] | 50% | -75,50 | -91,99 | -90,07 |
| fim de semana (`cb4we`) | 5 | -17,50 | -8,51 (greedy_cal2) | -2,59 [-2,99; -2,18] | 5,93 [5,52; 6,33] | 70% | -20,68 | -17,31 | -20,76 |
| legado × 1,5 (`legacy4x1.5`) | 5 | -0,38 | -0,09 (greedy_cal2) | -0,18 [-0,32; -0,04] | -0,10 [-0,24; 0,04] ⚠ | n/d (heurística ≈ 0) | -2,54 | -0,73 | -1,57 |
| legado × 2,5 (`legacy4x2.5`) | 5 | -1,68 | -0,37 (greedy_cal2) | -0,33 [-0,59; -0,08] | 0,03 [-0,22; 0,29] ⚠ | n/d (heurística ≈ 0) | -3,45 | -2,33 | -3,39 |
| legado × 3,5 (`legacy4`) | 5 | -3,41 | -1,13 (greedy_cal2) | -0,38 [-0,66; -0,10] | 0,75 [0,46; 1,03] | 66% | -4,07 | -4,07 | -5,60 |
| legado × 5 (`legacy4x5`) | 5 | -7,63 | -3,86 (greedy_cal) | -0,80 [-0,89; -0,72] | 3,06 [2,97; 3,14] | 79% | -8,26 | -9,26 | -10,97 |

**Dias reais de fim de semana (cb4wd)** (única variante em que o replay de dias reais é interpretável):

| variante | n sem. | parada | melhor heurística | DQN | DQN − heurística |
|---|---|---|---|---|---|
| fim de semana | 5 | -20,36 | -21,55 | -12,71 | 8,84 |

## Escala-base cb12wd

| variante | n sem. | parada | melhor heurística (por val) | DQN [IC95%] | C1: DQN − heurística [IC95%] | redução vs heurística | linear | tab. estação | tab. cluster |
|---|---|---|---|---|---|---|---|---|---|
| K = 0,5 (`cb12wdx0.5`) | 5 | -20,92 | -6,84 (greedy_cal2) | -7,31 [-7,60; -7,01] | -0,47 [-0,76; -0,17] ⚠ | -7% | -30,75 | -25,08 | -27,13 |
| K = 0,75 (`cb12wdx0.75`) | 5 | -61,30 | -30,58 (greedy_cal2) | -20,96 [-22,08; -19,83] | 9,62 [8,50; 10,75] | 31% | -65,73 | -67,56 | -67,66 |
| K = 1 (primário) (`cb12wd`) | 5 | -131,34 | -80,45 (greedy_cal2) | -56,00 [-58,37; -53,63] | 24,46 [22,09; 26,82] | 30% | -129,38 | -137,96 | -137,30 |
| K = 1,5 (`cb12wdx1.5`) | 5 | -327,40 | -259,16 (greedy_cal2) | -210,86 [-216,24; -205,49] | 48,30 [42,92; 53,68] | 19% | -327,68 | -331,91 | -331,17 |
| fim de semana (`cb12we`) | 5 | -39,84 | -18,30 (greedy_cal) | -17,67 [-18,41; -16,94] | 0,63 [-0,10; 1,36] ⚠ | 3% | -43,46 | -44,36 | -46,00 |

**Dias reais de fim de semana (cb12wd)** (única variante em que o replay de dias reais é interpretável):

| variante | n sem. | parada | melhor heurística | DQN | DQN − heurística |
|---|---|---|---|---|---|
| fim de semana | 5 | -39,09 | -30,00 | -28,15 | 1,85 |

### Onde a conclusão muda (⚠ nas tabelas)

- legacy4x1.5: C1 NÃO é significativo (DQN − heurística = -0,10 [-0,24; 0,04])
- legacy4x2.5: C1 NÃO é significativo (DQN − heurística = 0,03 [-0,22; 0,29])
- cb12wdx0.5: o DQN é SIGNIFICATIVAMENTE PIOR que a melhor heurística (DQN − heurística = -0,47 [-0,76; -0,17])
- cb12we: C1 NÃO é significativo (DQN − heurística = 0,63 [-0,10; 1,36])

Variantes analisadas: 14; com alguma mudança de conclusão: 4.
