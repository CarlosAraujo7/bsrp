# Análise de runs (gerado por `scripts/14_analyze_runs.py`)

> Unidade = semente de treino. Escolha do melhor por **validação**. TOST δ = 0.5. Holm por família.


## legacy4

### Aprendizes (reward de TESTE por semente; maior = melhor)

| método | n sem. | passos | média | DP entre sem. | IC95% t | IC95% bootstrap | val final | AUC val | var entre sem./(entre+cenário) |
|---|---|---|---|---|---|---|---|---|---|
| dqn | 5 | 150000 | -0,34 | 0,12 | [-0,48; -0,19] | [-0,44; -0,27] | -0,34 | -1,01 | 0,878 |
| linear | 5 | 150000 | -5,49 | 2,82 | [-8,99; -1,99] | [-7,99; -3,66] | -5,08 | -5,82 | 0,995 |
| pg | 5 | 150000 | -3,58 | 0,24 | [-3,88; -3,29] | [-3,78; -3,41] | -3,21 | -4,05 | 0,679 |
| tab_cluster | 5 | 150000 | -5,46 | 0,73 | [-6,37; -4,55] | [-6,07; -4,92] | -5,06 | -5,45 | 0,932 |
| tab_station | 5 | 150000 | -4,68 | 0,90 | [-5,80; -3,56] | [-5,43; -4,06] | -4,52 | -4,94 | 0,962 |

### Referências sem aprendizado (teste, 1000 cenários compartilhados)

| política | reward médio | DP por cenário | configuração |
|---|---|---|---|
| greedy_cal | -1,44 | 3,57 | greedy(lo=0.25,hi=0.6,d=10) |
| greedy_cal2 | -1,13 | 2,81 | greedy(lo=0.25,hi=0.55,d=10) |
| legacy_greedy | -2,25 | 4,64 | legacy_greedy |
| legacy_threshold | -2,60 | 5,14 | legacy_threshold |
| random | -6,92 | 7,05 | random |
| stay | -3,41 | 5,18 | stay |
| threshold_cal | -2,05 | 4,33 | threshold(t=0.6,lo=0.35,hi=0.9,d=10,w=0.0) |

### Comparações pré-definidas

| comparação | diferença média (unid. de reward) | p (t) | p Holm | p (não-paramétrico) | P(superioridade) |
|---|---|---|---|---|---|
| C1: melhor aprendido (dqn, por val) − melhor heurística (greedy_cal2, por val) | 0,79 | 0,0001 | 0,0005 | 0,0625 | 1,00 |
| C4: R1 − R0 (sementes independentes) | 0,78 | 0,1716 | 0,3431 | 0,1508 | 0,80 |
| C5: R2 − R1 (sementes independentes) | -0,81 | 0,5664 | 0,5664 | 1,0000 | 0,52 |
| C6: R3 − R2 (sementes independentes) | 5,15 | 0,0149 | 0,0448 | 0,0079 | 1,00 |

Equivalência (TOST, δ = 0.5): C4: R1 − R0: diff 0,78, p_TOST 0,7000; C5: R2 − R1: diff -0,81, p_TOST 0,5889; C6: R3 − R2: diff 5,15, p_TOST 0,9896

### Tamanho de amostra (80% de poder, α = 0,05, Δ = 1,0 e 0,5 unidade de reward)

| método | DP entre sementes | n (vs heurística, Δ=1) | n (vs heurística, Δ=0,5) | n (2 amostras, Δ=1) | n (2 amostras, Δ=0,5) |
|---|---|---|---|---|---|
| dqn | 0,12 | 1 | 1 | 1 | 1 |
| linear | 2,82 | 63 | 250 | 125 | 499 |
| pg | 0,24 | 1 | 2 | 1 | 4 |
| tab_cluster | 0,73 | 5 | 17 | 9 | 34 |
| tab_station | 0,90 | 7 | 26 | 13 | 52 |
