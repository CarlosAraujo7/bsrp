# Análise de runs (gerado por `scripts/14_analyze_runs.py`)

> Unidade = semente de treino. Escolha do melhor por **validação**. TOST δ = 0.5. Holm por família.


## cb4wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 3 | 300000 | -13,70 | 1,89 | [-18,39; -9,00] | [-15,87; -12,44] | -11,12 | -18,13 | 0,947 |
| linear | 3 | 300000 | -37,46 | 2,29 | [-43,16; -31,77] | [-39,56; -35,02] | -35,81 | -38,74 | 0,938 |
| pg | 3 | 300000 | -45,46 | 15,56 | [-84,11; -6,81] | [-56,24; -27,62] | -43,59 | -47,38 | 0,998 |
| tab_cluster | 3 | 300000 | -46,16 | 0,96 | [-48,53; -43,78] | [-46,80; -45,05] | -42,78 | -42,38 | 0,664 |
| tab_station | 3 | 300000 | -41,77 | 1,08 | [-44,45; -39,09] | [-42,90; -40,75] | -38,95 | -42,04 | 0,742 |

### Referências sem aprendizado (teste, 1000 cenários compartilhados)

| política | reward médio | DP por cenário | configuração |
|---|---|---|---|
| greedy_cal | -25,55 | 17,08 | greedy(lo=0.35,hi=0.6,d=10) |
| greedy_cal2 | -24,46 | 15,78 | greedy(lo=0.25,hi=0.55,d=10) |
| legacy_greedy | -29,01 | 18,25 | legacy_greedy |
| legacy_threshold | -29,68 | 18,19 | legacy_threshold |
| random | -49,90 | 22,81 | random |
| stay | -46,94 | 20,23 | stay |

### Comparações pré-definidas

| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |
|---|---|---|---|---|---|
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 10,76 | 0,0101 | 0,0203 | 0,2500 | 1,00 |
| C4: R1 − R0 (sementes independentes) | 4,39 | 0,0065 | 0,0194 | 0,1000 | 1,00 |
| C5: R2 − R1 (sementes independentes) | 4,30 | 0,0646 | 0,0646 | 0,1000 | 1,00 |
| C6: R3 − R2 (sementes independentes) | 23,77 | 0,0002 | 0,0008 | 0,1000 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff 4,39, p_TOST 0,9951; C5: R2 − R1: diff 4,30, p_TOST 0,9575; C6: R3 − R2: diff 23,77, p_TOST 0,9999

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 3 | -13,00 | 1,73 | [-17,30; -8,70] |
| linear | 3 | -27,54 | 0,30 | [-28,29; -26,80] |
| pg | 3 | -38,65 | 17,28 | [-81,57; 4,26] |
| tab_cluster | 3 | -34,42 | 2,95 | [-41,74; -27,09] |
| tab_station | 3 | -33,92 | 2,32 | [-39,69; -28,15] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -25,46 |
| greedy_cal2 | -25,42 |
| legacy_greedy | -28,29 |
| legacy_threshold | -29,04 |
| random | -43,21 |
| stay | -32,67 |

C1r (dias reais): dqn − greedy_cal2 (ambos escolhidos por val): diferença média 12,42, p(t) = 0,0064, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 1,89 | 29 | 113 | 57 | 225 |
| linear | 2,29 | 42 | 166 | 83 | 331 |
| pg | 15,56 | 1901 | 7601 | 3801 | 15202 |
| tab_cluster | 0,96 | 8 | 29 | 15 | 58 |
| tab_station | 1,08 | 10 | 37 | 19 | 74 |
