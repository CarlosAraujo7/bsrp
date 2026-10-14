# Análise de runs (gerado por `scripts/14_analyze_runs.py`)

> Unidade = semente de treino. Escolha do melhor por **validação**. TOST δ = 0.5. Holm por família.


## cb12wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -55,94 | 1,68 | [-57,14; -54,74] | [-56,91; -54,97] | -55,61 | -73,13 | 0,822 |
| linear | 10 | 500000 | -129,99 | 4,78 | [-133,40; -126,57] | [-132,82; -127,20] | -130,21 | -131,48 | 0,946 |
| pg | 10 | 500000 | -157,95 | 18,19 | [-170,97; -144,94] | [-168,10; -146,97] | -157,30 | -138,19 | 0,996 |
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
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 24,51 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | -1,86 | 0,0401 | 0,0401 | 0,0073 | 0,14 |
| C5: R2 − R1 (sementes independentes) | 7,93 | 0,0005 | 0,0010 | 0,0002 | 1,00 |
| C6: R3 − R2 (sementes independentes) | 74,04 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff -1,86, p_TOST 0,9429; C5: R2 − R1: diff 7,93, p_TOST 0,9996; C6: R3 − R2: diff 74,04, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -41,25 | 2,14 | [-42,78; -39,71] |
| linear | 10 | -85,00 | 4,54 | [-88,24; -81,75] |
| pg | 10 | -121,70 | 15,99 | [-133,14; -110,26] |
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

C1r (dias reais): dqn − greedy_cal2 (ambos escolhidos por val): diferença média 26,59, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 1,68 | 23 | 89 | 45 | 178 |
| linear | 4,78 | 180 | 717 | 359 | 1433 |
| pg | 18,19 | 2598 | 10391 | 5196 | 20781 |
| tab_cluster | 2,45 | 47 | 188 | 94 | 376 |
| tab_station | 0,21 | 1 | 2 | 1 | 3 |

## cb4wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -9,77 | 0,57 | [-10,18; -9,37] | [-10,14; -9,47] | -7,51 | -8,50 | 0,693 |
| linear | 10 | 500000 | -37,59 | 4,34 | [-40,70; -34,48] | [-40,43; -35,34] | -35,07 | -43,01 | 0,976 |
| pg | 10 | 500000 | -19,88 | 1,31 | [-20,81; -18,94] | [-20,70; -19,17] | -17,67 | -31,37 | 0,879 |
| tab_cluster | 10 | 500000 | -41,39 | 2,17 | [-42,94; -39,84] | [-42,71; -40,20] | -38,80 | -40,24 | 0,915 |
| tab_station | 10 | 500000 | -39,74 | 1,11 | [-40,53; -38,94] | [-40,39; -39,07] | -37,47 | -40,35 | 0,755 |

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
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 14,69 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | 1,65 | 0,0510 | 0,1020 | 0,0452 | 0,77 |
| C5: R2 − R1 (sementes independentes) | 2,15 | 0,1600 | 0,1600 | 0,0257 | 0,80 |
| C6: R3 − R2 (sementes independentes) | 27,82 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff 1,65, p_TOST 0,9208; C5: R2 − R1: diff 2,15, p_TOST 0,8643; C6: R3 − R2: diff 27,82, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -9,06 | 1,54 | [-10,17; -7,96] |
| linear | 10 | -33,34 | 5,97 | [-37,61; -29,07] |
| pg | 10 | -15,91 | 2,99 | [-18,05; -13,77] |
| tab_cluster | 10 | -34,35 | 4,41 | [-37,50; -31,19] |
| tab_station | 10 | -32,75 | 1,35 | [-33,71; -31,78] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -25,46 |
| greedy_cal2 | -25,42 |
| greedy_cal3 | -26,00 |
| legacy_greedy | -28,29 |
| legacy_threshold | -29,04 |
| random | -43,21 |
| stay | -32,67 |

C1r (dias reais): dqn − greedy_cal2 (ambos escolhidos por val): diferença média 16,35, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 0,57 | 3 | 11 | 6 | 21 |
| linear | 4,34 | 148 | 592 | 296 | 1184 |
| pg | 1,31 | 14 | 54 | 27 | 108 |
| tab_cluster | 2,17 | 37 | 148 | 74 | 296 |
| tab_station | 1,11 | 10 | 39 | 20 | 78 |

## cb8wd

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 10 | 500000 | -29,69 | 0,91 | [-30,34; -29,03] | [-30,20; -29,14] | -29,32 | -38,88 | 0,737 |
| linear | 10 | 500000 | -87,75 | 3,46 | [-90,22; -85,27] | [-89,70; -85,66] | -89,01 | -88,76 | 0,933 |
| pg | 10 | 500000 | -105,48 | 21,46 | [-120,83; -90,13] | [-118,28; -92,93] | -106,84 | -94,86 | 0,998 |
| tab_cluster | 10 | 500000 | -92,10 | 1,30 | [-93,03; -91,17] | [-92,91; -91,40] | -93,65 | -94,84 | 0,655 |
| tab_station | 10 | 500000 | -94,37 | 0,73 | [-94,89; -93,85] | [-94,77; -93,93] | -95,08 | -96,08 | 0,368 |

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
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal, por val) | 23,10 | 0,0000 | 0,0000 | 0,0020 | 1,00 |
| C4: R1 − R0 (sementes independentes) | -2,27 | 0,0003 | 0,0003 | 0,0022 | 0,09 |
| C5: R2 − R1 (sementes independentes) | 6,63 | 0,0002 | 0,0003 | 0,0002 | 1,00 |
| C6: R3 − R2 (sementes independentes) | 58,06 | 0,0000 | 0,0000 | 0,0002 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff -2,27, p_TOST 0,9990; C5: R2 − R1: diff 6,63, p_TOST 0,9999; C6: R3 − R2: diff 58,06, p_TOST 1,0000

### Dias REAIS de teste (replay dos fluxos observados nas semanas de teste), por semente

| método | n sem. | média | DP entre sem. | IC95% t |
|---|---|---|---|---|
| dqn | 10 | -19,69 | 2,12 | [-21,21; -18,18] |
| linear | 10 | -55,77 | 4,79 | [-59,19; -52,34] |
| pg | 10 | -73,00 | 22,56 | [-89,14; -56,86] |
| tab_cluster | 10 | -63,83 | 2,75 | [-65,80; -61,87] |
| tab_station | 10 | -63,90 | 2,24 | [-65,50; -62,30] |

| referência (sem aprendizado) | dias reais |
|---|---|
| greedy_cal | -47,54 |
| greedy_cal2 | -47,54 |
| greedy_cal3 | -47,54 |
| legacy_greedy | -45,79 |
| legacy_threshold | -45,79 |
| random | -60,21 |
| stay | -54,04 |

C1r (dias reais): dqn − greedy_cal (ambos escolhidos por val): diferença média 27,85, p(t) = 0,0000, fração de sementes melhores que a heurística = 1,00. Obs.: 24 dias úteis de teste; ocupação inicial desconhecida (50% ± 2).

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 0,91 | 7 | 27 | 14 | 53 |
| linear | 3,46 | 94 | 376 | 188 | 751 |
| pg | 21,46 | 3614 | 14453 | 7227 | 28906 |
| tab_cluster | 1,30 | 14 | 53 | 27 | 106 |
| tab_station | 0,73 | 5 | 17 | 9 | 34 |
