# Análise de runs (gerado por `scripts/14_analyze_runs.py`)

> Unidade = semente de treino. Escolha do melhor por **validação**. TOST δ = 0.5. Holm por família.


## cb12wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -64,30 | 3,05 | [-66,48; -62,11] | [-66,27; -62,61] | -63,47 | -85,68 | 0,930 |
| linear | 10 | 500000 | -129,99 | 4,78 | [-133,40; -126,57] | [-132,82; -127,20] | -130,21 | -131,48 | 0,946 |
| pg | 10 | 500000 | -124,94 | 10,43 | [-132,40; -117,48] | [-131,43; -119,25] | -124,55 | -126,25 | 0,988 |
| tab_cluster | 10 | 500000 | -136,06 | 2,45 | [-137,81; -134,31] | [-137,47; -134,63] | -135,72 | -136,92 | 0,820 |
| tab_station | 10 | 500000 | -137,91 | 0,21 | [-138,06; -137,77] | [-138,03; -137,79] | -139,15 | -136,56 | 0,031 |

### Referências sem aprendizado (teste, 1000 cenários compartilhados)

| política | reward médio | DP por cenário | configuração |
|---|---|---|---|
| greedy_cal | -80,82 | 28,12 | greedy(lo=0.35,hi=0.6,d=10) |
| greedy_cal2 | -80,45 | 28,00 | greedy(lo=0.35,hi=0.5,d=10) |
| greedy_cal3 | -80,45 | 28,00 | greedy(lo=0.35,hi=0.5,d=10) |
| legacy_greedy | -83,85 | 28,76 | legacy_greedy |
| legacy_threshold | -88,95 | 29,22 | legacy_threshold |
| random | -136,81 | 37,45 | random |
| stay | -131,34 | 35,75 | stay |

### Comparações pré-definidas

| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |
|---|---|---|---|---|---|
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 16,16 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | -1,86 | 0,0401 | 0,0401 | 0,0073 | 0,14 |
| C5: R2 − R1 (sementes independentes) | 7,93 | 0,0005 | 0,0010 | 0,0002 | 1,00 |
| C6: R3 − R2 (sementes independentes) | 65,69 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff -1,86, p_TOST 0,9429; C5: R2 − R1: diff 7,93, p_TOST 0,9996; C6: R3 − R2: diff 65,69, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -45,07 | 3,14 | [-47,32; -42,82] |
| linear | 10 | -85,00 | 4,54 | [-88,24; -81,75] |
| pg | 10 | -87,52 | 9,77 | [-94,51; -80,53] |
| tab_cluster | 10 | -98,12 | 3,69 | [-100,76; -95,48] |
| tab_station | 10 | -95,60 | 2,80 | [-97,60; -93,60] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -71,75 |
| greedy_cal2 | -67,83 |
| greedy_cal3 | -67,83 |
| legacy_greedy | -74,25 |
| legacy_threshold | -68,83 |
| random | -98,04 |
| stay | -84,54 |

C1r (dias reais): dqn − greedy_cal2 (ambos escolhidos por val): diferença média 22,76, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 3,05 | 74 | 293 | 147 | 586 |
| linear | 4,78 | 180 | 717 | 359 | 1433 |
| pg | 10,43 | 854 | 3415 | 1708 | 6830 |
| tab_cluster | 2,45 | 47 | 188 | 94 | 376 |
| tab_station | 0,21 | 1 | 2 | 1 | 3 |

## cb4wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -10,43 | 0,49 | [-10,78; -10,08] | [-10,72; -10,15] | -8,00 | -9,61 | 0,606 |
| linear | 10 | 500000 | -37,59 | 4,34 | [-40,70; -34,48] | [-40,43; -35,34] | -35,07 | -43,01 | 0,976 |
| pg | 10 | 500000 | -37,71 | 12,72 | [-46,81; -28,61] | [-45,93; -30,91] | -35,67 | -42,78 | 0,997 |
| tab_cluster | 10 | 500000 | -42,97 | 2,22 | [-44,56; -41,38] | [-44,37; -41,76] | -40,85 | -40,84 | 0,920 |
| tab_station | 10 | 500000 | -40,38 | 1,44 | [-41,41; -39,35] | [-41,22; -39,56] | -38,23 | -40,18 | 0,835 |

### Referências sem aprendizado (teste, 1000 cenários compartilhados)

| política | reward médio | DP por cenário | configuração |
|---|---|---|---|
| greedy_cal | -25,55 | 17,08 | greedy(lo=0.35,hi=0.6,d=10) |
| greedy_cal2 | -24,46 | 15,78 | greedy(lo=0.25,hi=0.55,d=10) |
| greedy_cal3 | -24,99 | 16,63 | greedy(lo=0.45,hi=0.5,d=5) |
| legacy_greedy | -29,01 | 18,25 | legacy_greedy |
| legacy_threshold | -29,68 | 18,19 | legacy_threshold |
| random | -49,90 | 22,81 | random |
| stay | -46,94 | 20,23 | stay |

### Comparações pré-definidas

| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |
|---|---|---|---|---|---|
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 14,03 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | 2,59 | 0,0072 | 0,0144 | 0,0073 | 0,86 |
| C5: R2 − R1 (sementes independentes) | 2,79 | 0,0801 | 0,0801 | 0,0211 | 0,81 |
| C6: R3 − R2 (sementes independentes) | 27,16 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff 2,59, p_TOST 0,9878; C5: R2 − R1: diff 2,79, p_TOST 0,9291; C6: R3 − R2: diff 27,16, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -9,03 | 0,87 | [-9,65; -8,41] |
| linear | 10 | -33,34 | 5,97 | [-37,61; -29,07] |
| pg | 10 | -33,82 | 16,06 | [-45,31; -22,34] |
| tab_cluster | 10 | -35,85 | 3,95 | [-38,67; -33,02] |
| tab_station | 10 | -34,24 | 2,64 | [-36,12; -32,35] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -25,46 |
| greedy_cal2 | -25,42 |
| greedy_cal3 | -26,00 |
| legacy_greedy | -28,29 |
| legacy_threshold | -29,04 |
| random | -43,21 |
| stay | -32,67 |

C1r (dias reais): dqn − greedy_cal2 (ambos escolhidos por val): diferença média 16,39, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 0,49 | 2 | 8 | 4 | 16 |
| linear | 4,34 | 148 | 592 | 296 | 1184 |
| pg | 12,72 | 1271 | 5082 | 2541 | 10163 |
| tab_cluster | 2,22 | 39 | 156 | 78 | 311 |
| tab_station | 1,44 | 17 | 65 | 33 | 130 |

## cb8wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -31,96 | 1,12 | [-32,76; -31,16] | [-32,59; -31,28] | -30,77 | -45,36 | 0,802 |
| linear | 10 | 500000 | -87,75 | 3,46 | [-90,22; -85,27] | [-89,70; -85,66] | -89,01 | -88,76 | 0,933 |
| pg | 10 | 500000 | -101,77 | 20,23 | [-116,24; -87,30] | [-114,70; -91,64] | -101,46 | -92,54 | 0,997 |
| tab_cluster | 10 | 500000 | -92,75 | 1,34 | [-93,71; -91,78] | [-93,57; -92,02] | -93,68 | -94,80 | 0,670 |
| tab_station | 10 | 500000 | -94,50 | 0,49 | [-94,85; -94,15] | [-94,76; -94,18] | -95,59 | -96,45 | 0,214 |

### Referências sem aprendizado (teste, 1000 cenários compartilhados)

| política | reward médio | DP por cenário | configuração |
|---|---|---|---|
| greedy_cal | -52,79 | 21,54 | greedy(lo=0.45,hi=0.6,d=10) |
| greedy_cal2 | -52,79 | 21,54 | greedy(lo=0.45,hi=0.6,d=10) |
| greedy_cal3 | -52,79 | 21,54 | greedy(lo=0.45,hi=0.6,d=10) |
| legacy_greedy | -52,77 | 22,28 | legacy_greedy |
| legacy_threshold | -55,72 | 22,60 | legacy_threshold |
| random | -92,98 | 29,63 | random |
| stay | -89,77 | 29,18 | stay |

### Comparações pré-definidas

| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |
|---|---|---|---|---|---|
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal, por val) | 20,83 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | -1,75 | 0,0024 | 0,0024 | 0,0046 | 0,12 |
| C5: R2 − R1 (sementes independentes) | 6,75 | 0,0001 | 0,0003 | 0,0002 | 1,00 |
| C6: R3 − R2 (sementes independentes) | 55,79 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff -1,75, p_TOST 0,9911; C5: R2 − R1: diff 6,75, p_TOST 0,9999; C6: R3 − R2: diff 55,79, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -19,59 | 1,83 | [-20,90; -18,28] |
| linear | 10 | -55,77 | 4,79 | [-59,19; -52,34] |
| pg | 10 | -71,45 | 21,39 | [-86,75; -56,14] |
| tab_cluster | 10 | -63,15 | 3,58 | [-65,71; -60,59] |
| tab_station | 10 | -66,43 | 3,72 | [-69,10; -63,77] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -47,54 |
| greedy_cal2 | -47,54 |
| greedy_cal3 | -47,54 |
| legacy_greedy | -45,79 |
| legacy_threshold | -45,79 |
| random | -60,21 |
| stay | -54,04 |

C1r (dias reais): dqn − greedy_cal (ambos escolhidos por val): diferença média 27,95, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 1,12 | 10 | 40 | 20 | 79 |
| linear | 3,46 | 94 | 376 | 188 | 751 |
| pg | 20,23 | 3212 | 12847 | 6424 | 25694 |
| tab_cluster | 1,34 | 15 | 57 | 29 | 114 |
| tab_station | 0,49 | 2 | 8 | 4 | 16 |
