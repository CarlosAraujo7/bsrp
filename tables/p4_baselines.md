# P4 — rollout, threshold e A2C adaptado de Seo et al. 2022 (gerado por `scripts/21_analyze_p1_p2.py --which p4`)

> Reward de TESTE (1000 cenários sintéticos) e dias reais; maior = melhor. Referência de heurística = a de maior reward em **validação** entre as gulosas (grades 1–3) e a `threshold` (grades 1–2). Rollout: futuros do modelo (nunca da demanda real); config de cabeçalho = a de maior reward em val. Contrastes com aprendizes: média por semente − constante (referência determinística) ou Welch (aprendiz × aprendiz). O A2C é uma **adaptação** (hiperparâmetros como publicados; entrada e ação do nosso simulador), não uma reprodução.


## cb4wd

Referência por val: **greedy_cal2** = `greedy(lo=0.25,hi=0.55,d=10)` (teste -24,46; parada -46,94). Famílias, melhor val → teste: greedy_cal: -22,57 → -25,55; greedy_cal2: -21,71 → -24,46; greedy_cal3: -22,30 → -24,99; threshold_cal: -22,91 → -26,27; threshold_cal2: -22,38 → -25,09

| método | val | teste | redução vs referência | dias reais |
|---|---|---|---|---|
| rollout h=1, K=4 | -18,14 | -20,30 | 17% | -21,96 |
| rollout h=1, K=16 | -14,18 | -16,66 | 32% | -18,17 |
| rollout h=2, K=4 | -16,86 | -18,91 | 23% | -19,54 |
| rollout h=2, K=16 | -12,92 | -14,88 | 39% | -15,71 |
| rollout h=2, K=64 | -11,41 | -13,65 | 44% | -13,50 |
| rollout h=3, K=16 | -11,22 | -13,66 | 44% | -12,88 |
| rollout h=3, K=64 **(cabeçalho: melhor em val)** | -8,97 | -11,83 | 52% | -10,88 |
| A2C Seo (publicado, γ = 0,9), n = 10 | -25,62 | -28,01 [-36,19; -19,83] | -14% | -20,12 |
| A2C Seo (γ = 1,0), n = 10 | -37,57 | -39,99 [-47,61; -32,38] | -64% | -27,81 |
| A2C Seo (ajustado), n = 10 | -13,81 | -16,10 [-17,21; -14,99] | 34% | -13,80 |
| DQN (primário), n = 10 | — | -9,77 [-10,18; -9,37] | 60% | — |

DQN − rollout de cabeçalho (h=3, K=64): 2,06 [1,65; 2,47] (10/10 sementes melhores: sim).

DQN − A2C Seo (publicado): 18,23 [10,05; 26,42], p(Welch) = 0,0007; A2C Seo (publicado) − referência: -3,55 [-11,73; 4,63].

DQN − A2C Seo (γ = 1,0): 30,22 [22,60; 37,84], p(Welch) = 0,0000; A2C Seo (γ = 1,0) − referência: -15,53 [-23,15; -7,92].

DQN − A2C Seo (ajustado): 6,33 [5,18; 7,48], p(Welch) = 0,0000; A2C Seo (ajustado) − referência: 8,36 [7,25; 9,48].

## cb8wd

Referência por val: **greedy_cal** = `greedy(lo=0.45,hi=0.6,d=10)` (teste -52,79; parada -89,77). Famílias, melhor val → teste: greedy_cal: -52,70 → -52,79; greedy_cal2: -52,70 → -52,79; greedy_cal3: -52,70 → -52,79; threshold_cal: -57,23 → -55,62; threshold_cal2: -55,89 → -54,71

| método | val | teste | redução vs referência | dias reais |
|---|---|---|---|---|
| rollout h=1, K=4 | -43,43 | -41,84 | 21% | -31,71 |
| rollout h=1, K=16 | -34,51 | -33,44 | 37% | -29,67 |
| rollout h=2, K=4 | -41,84 | -40,15 | 24% | -31,83 |
| rollout h=2, K=16 | -34,07 | -32,06 | 39% | -26,50 |
| rollout h=2, K=64 | -30,09 | -29,65 | 44% | -23,04 |
| rollout h=3, K=16 | -31,68 | -30,34 | 43% | -23,21 |
| rollout h=3, K=64 **(cabeçalho: melhor em val)** | -27,41 | -27,08 | 49% | -20,17 |
| A2C Seo (publicado, γ = 0,9), n = 10 | -86,79 | -85,99 [-91,66; -80,31] | -63% | -50,74 |
| A2C Seo (γ = 1,0), n = 10 | -85,68 | -84,98 [-90,64; -79,32] | -61% | -50,01 |
| A2C Seo (ajustado), n = 10 | -76,18 | -75,70 [-83,41; -67,99] | -43% | -46,25 |
| DQN (primário), n = 10 | — | -29,69 [-30,34; -29,03] | 44% | — |

DQN − rollout de cabeçalho (h=3, K=64): -2,61 [-3,26; -1,96] (10/10 sementes melhores: não).

DQN − A2C Seo (publicado): 56,30 [50,61; 61,99], p(Welch) = 0,0000; A2C Seo (publicado) − referência: -33,20 [-38,87; -27,52].

DQN − A2C Seo (γ = 1,0): 55,30 [49,62; 60,97], p(Welch) = 0,0000; A2C Seo (γ = 1,0) − referência: -32,19 [-37,85; -26,53].

DQN − A2C Seo (ajustado): 46,01 [38,29; 53,73], p(Welch) = 0,0000; A2C Seo (ajustado) − referência: -22,91 [-30,62; -15,20].

## cb12wd

Referência por val: **greedy_cal2** = `greedy(lo=0.35,hi=0.5,d=10)` (teste -80,45; parada -131,34). Famílias, melhor val → teste: greedy_cal: -82,02 → -80,82; greedy_cal2: -81,27 → -80,45; greedy_cal3: -81,27 → -80,45; threshold_cal: -85,64 → -84,81; threshold_cal2: -82,44 → -82,45

| método | val | teste | redução vs referência | dias reais |
|---|---|---|---|---|
| rollout h=1, K=4 | -70,17 | -68,37 | 15% | -55,92 |
| rollout h=1, K=16 | -59,47 | -57,84 | 28% | -50,75 |
| rollout h=2, K=4 | -67,31 | -66,90 | 17% | -55,62 |
| rollout h=2, K=16 | -56,50 | -56,02 | 30% | -50,04 |
| rollout h=2, K=64 | -52,27 | -51,41 | 36% | -42,83 |
| rollout h=3, K=16 | -53,47 | -53,37 | 34% | -42,58 |
| rollout h=3, K=64 **(cabeçalho: melhor em val)** | -48,91 | -48,10 | 40% | -42,08 |
| A2C Seo (publicado, γ = 0,9), n = 10 | -131,61 | -131,36 [-131,40; -131,31] | -63% | -84,61 |
| A2C Seo (γ = 1,0), n = 10 | -131,37 | -131,26 [-138,83; -123,69] | -63% | -85,40 |
| A2C Seo (ajustado), n = 10 | -111,62 | -111,31 [-119,54; -103,08] | -38% | -70,26 |
| DQN (primário), n = 10 | — | -55,94 [-57,14; -54,74] | 30% | — |

DQN − rollout de cabeçalho (h=3, K=64): -7,85 [-9,05; -6,64] (10/10 sementes melhores: não).

DQN − A2C Seo (publicado): 75,42 [74,21; 76,62], p(Welch) = 0,0000; A2C Seo (publicado) − referência: -50,90 [-50,95; -50,86].

DQN − A2C Seo (γ = 1,0): 75,32 [67,71; 82,93], p(Welch) = 0,0000; A2C Seo (γ = 1,0) − referência: -50,81 [-58,38; -43,24].

DQN − A2C Seo (ajustado): 55,37 [47,11; 63,63], p(Welch) = 0,0000; A2C Seo (ajustado) − referência: -30,86 [-39,08; -22,63].
