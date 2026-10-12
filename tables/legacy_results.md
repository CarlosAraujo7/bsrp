# Resultados legados — gerado por `scripts/11_make_tables.py`

> **Não editar à mão.** Todos os resultados abaixo vêm de **uma única semente de treino (42)**; as "sementes" 123/456/789 são de avaliação no simulador (medem ruído do simulador, não do algoritmo). Servem para formular hipóteses, não para afirmações.

## T1. Escalabilidade (como salvo)

| Estações | Política | Ep. treino | Reward médio (3×100 ep) | Amplitude entre sem. de AVALIAÇÃO | Std por episódio | Fonte |
|---|---|---|---|---|---|---|
| 4 | Q-learning (estado por cluster) | 10000 | -7,13 | 1,39 | 6,71 | ql_4st.json |
| 4 | DQN | 5000 | -3,49 | 0,74 | 5,19 | dqn_4st.json |
| 4 | Threshold | — | -2,08 | 0,20 | 4,16 | baselines_4st.json:threshold |
| 4 | Rollout 1 passo | — | -3,70 | 0,52 | 4,61 | baselines_4st.json:rollout |
| 4 | Aleatória / Gulosa (multi-semente) | — | não salvo | — | — | só stdout; ausente dos JSONs |
| 8 | Q-learning (estado por cluster) | 3500 | -13,92 | 1,87 | 9,96 | ql_8st.json |
| 8 | DQN | 9500 | -6,76 | 1,38 | 6,67 | dqn_8st.json |
| 8 | Threshold | — | -5,26 | 1,10 | 5,67 | baselines_8st.json:threshold |
| 8 | Aleatória / Gulosa (multi-semente) | — | não salvo | — | — | só stdout; ausente dos JSONs |
| 12 | Q-learning (estado por cluster) | 20000 | -12,23 | 2,14 | 10,05 | ql_12st.json |
| 12 | DQN | 9500 | -3,20 | 0,70 | 4,70 | dqn_12st.json |
| 12 | Threshold | — | -6,97 | 0,15 | 6,81 | baselines_12st.json:threshold |
| 12 | Aleatória / Gulosa (multi-semente) | — | não salvo | — | — | só stdout; ausente dos JSONs |

## T2. Política gulosa/aleatória — única fonte salva (semente 123, 100 ep, 12 estações)

| Política | Estações | Semente aval. | Episódios | Reward | Std | Fonte |
|---|---|---|---|---|---|---|
| Aleatoria | 12 | 123 | 100 | -14,54 | 10,89 | qtable_improved_*_summary.json |
| Gulosa | 12 | 123 | 100 | -4,60 | 5,50 | qtable_improved_*_summary.json |

## T3. DQN 8 estações — checkpoints avaliados nas MESMAS sementes de avaliação

| Ep. treino (checkpoint) | Reward médio (3×100 ep) | Amplitude entre sem. aval. | Fonte |
|---|---|---|---|
| 9500 | -6,76 | 1,38 | dqn_8st.json |
| 11500 | -7,09 | 0,94 | dqn_8st_ep11500.json |
| 12000 | -8,37 | 1,19 | dqn_8st_ep12000.json |
| 13500 | -6,85 | 0,89 | dqn_8st_ep13500.json |
| 15500 | -6,14 | 1,42 | dqn_8st_ep15500.json |
| 17500 | -6,57 | 1,19 | dqn_8st_ep17500.json |

## T4. Ablação do DQN (12 estações) — registros coexistentes e não equivalentes

| Variante | Ep. treino | Reward médio (3×100 ep) | Fonte |
|---|---|---|---|
| Full DQN | 9500 | -3,20 | dqn_12st.json |
| no_replay | 9500 | -12,09 | ablation_9500ep.json |
| no_target | 9500 | -10,74 | ablation_9500ep.json |
| no_double | 9500 | -10,53 | ablation_9500ep.json |
| no_replay (abl_*.json) | não registrado | -10,77 | abl_no_replay.json |
| no_target (abl_*.json) | não registrado | -10,74 | abl_no_target.json |
| no_double (abl_*.json) | não registrado | -10,06 | abl_no_double.json |

## T5. Controle de representação (4 estações)

| Representação (fonte) | Ep. treino | Estados visitados | Reward médio | Std | Último eval de 20 ep |
|---|---|---|---|---|---|
| cluster (ql_control) | 10000 | 4740 | -6,15 | 6,99 | -7,40 |
| por estação binário (ql_control) | 10000 | 7403 | -5,35 | 6,13 | -2,35 |
| por estação binário, 18k (ql_control) | 18000 | 7649 | -3,73 | 5,18 | -4,07 |
| cluster (ql_4st.json) | 10000 | — | -7,13 | 6,71 | — |

## T6. Tamanho do espaço de estados tabular por instância

| Estações | Clusters no simulador | Espaço cluster (4 níveis) | Espaço por estação (binário) |
|---|---|---|---|
| 4 | 2 | 8.192 | 8.192 |
| 8 | 4 | 262.144 | 262.144 |
| 12 | 4 | 393.216 | 6.291.456 |

## T7. Variantes tabulares A–D (12 estações)

| Variante | Ep. | L | Reward (sem. 123, 100 ep) | Std | Cobertura |
|---|---|---|---|---|---|
| A_fix_aliasing | 5000 | 2 | -16,22 | 10,58 | 68,1% |
| B_qlambda | 5000 | 2 | -15,95 | 9,53 | 73,2% |
| C_shaping | 5000 | 2 | -14,81 | 8,00 | 77,9% |
| D_combined | 5000 | 2 | -15,99 | 10,23 | 83,6% |
