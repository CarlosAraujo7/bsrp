# P8 — ancoragem e robustez da conclusão central (gerado por `scripts/30_analyze_p8.py`)

> Desenho e leituras fixados em `DESVIOS.md` (2026-10-18), **antes** de rodar. Reward ≤ 0; maior = melhor; unidade de análise = semente de treino.

## (a) Limite clarividente (P8a)

> Planejador que **conhece a demanda inteira do cenário** (busca em feixe sobre as ações do simulador; `bsrp/oracle.py`). Seu valor é um limite **inferior** do ótimo clarividente; este, por sua vez, **não é atingível** por política que não vê o futuro. 'Ganho capturado' = (método − parada) / (clarividente − parada): contra um limite inferior do clarividente, é um limite **superior** da fração verdadeira. Reward de teste nos primeiros cenários sintéticos e nos 24 dias reais; maior = melhor.


### cb4wd

| conjunto | n cen. | parada | heurística | rollout | DQN [IC95%] | clarividente (W = 300) | (W = 1000) | clarividente (melhor) | ganho capturado: heurística | rollout | DQN | clarividente ≥ −5% de parada? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teste sintético | 200 | -45,70 | -23,73 | -12,09 | -9,44 [-9,83; -9,04] | -2,35 | -2,08 | -2,08 | 50% | 77% | 83% [82; 84] | sim: limite ≈ nenhuma demanda perdida |
| dias reais | 24 | -32,67 | -25,42 | -10,88 | -9,06 [-10,17; -7,96] | -1,46 | -1,42 | -1,42 | 23% | 70% | 76% [72; 79] | sim: limite ≈ nenhuma demanda perdida |

### cb8wd

| conjunto | n cen. | parada | heurística | rollout | DQN [IC95%] | clarividente (W = 300) | (W = 1000) | clarividente (melhor) | ganho capturado: heurística | rollout | DQN | clarividente ≥ −5% de parada? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teste sintético | 200 | -90,50 | -52,07 | -26,80 | -29,04 [-29,68; -28,39] | -3,29 | -2,73 | -2,69 | 44% | 73% | 70% [69; 71] | sim: limite ≈ nenhuma demanda perdida |
| dias reais | 24 | -54,04 | -47,54 | -20,17 | -19,69 [-21,21; -18,18] | -2,21 | -1,79 | -1,79 | 12% | 65% | 66% [63; 69] | sim: limite ≈ nenhuma demanda perdida |

### cb12wd

| conjunto | n cen. | parada | heurística | rollout | DQN [IC95%] | clarividente (W = 300) | (W = 1000) | clarividente (melhor) | ganho capturado: heurística | rollout | DQN | clarividente ≥ −5% de parada? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teste sintético | 200 | -130,07 | -80,19 | -47,60 | -55,35 [-56,81; -53,90] | -10,65 | -9,21 | -8,91 | 41% | 68% | 62% [60; 63] | não |
| dias reais | 24 | -84,54 | -67,83 | -42,08 | -41,25 [-42,78; -39,71] | -14,00 | -12,83 | -12,83 | 23% | 59% | 60% [58; 63] | não |

**Leitura (fixada antes de ver os dados):** onde o clarividente chega a ≥ −5% do valor de 'parar', o limite é praticamente 'nenhuma demanda perdida' e **não** se afirma distância ao ótimo não antecipativo; a coluna 'ganho capturado' vira apenas a redução relativa à parada. A largura 1000 melhorando a 300 indica que o feixe ainda não convergiu (o valor é limite inferior do clarividente).

## (b) DQN com 4× o orçamento de treino (P8b)

> Config do P1 de cada escala, **2 milhões de passos** (o ε-decay estica junto; sem reajuste), sementes 0–4. 'Δ orçamento' = DQN(2 M) − DQN(500 mil), Welch (execuções independentes; as 10 sementes de 500 mil passos). 'DQN(2 M) − rollout' = por semente contra o rollout de cabeçalho (constante), IC95% t. Reward de teste; maior = melhor.

| escala | n (2 M) | DQN 500 mil [IC95%] | DQN 2 M [IC95%] | Δ orçamento [IC95% Welch] | rollout | DQN(2 M) − rollout [IC95%] | leitura (regra pré-fixada) |
|---|---|---|---|---|---|---|---|
| cb4wd | 5 | -9,77 [-10,18; -9,37] | -9,28 [-9,49; -9,07] | 0,49 [0,07; 0,92] | -11,83 | 2,55 [2,35; 2,76] | o orçamento de 500 mil passos limita o DQN; DQN(2 M) supera o rollout |
| cb8wd | 5 | -29,69 [-30,34; -29,03] | -27,54 [-28,79; -26,28] | 2,15 [0,89; 3,41] | -27,08 | -0,46 [-1,72; 0,80] | o orçamento de 500 mil passos limita o DQN; sem diferença detectável com o rollout |
| cb12wd | 5 | -55,94 [-57,14; -54,74] | -52,04 [-54,08; -50,01] | 3,90 [1,82; 5,98] | -48,10 | -3,95 [-5,98; -1,91] | o orçamento de 500 mil passos limita o DQN; o rollout continua melhor |

## (c1) Reavaliação das 4 melhores configs do DQN com 4 sementes de ajuste (P8c)

> As 4 configs de maior val médio (sementes 1000, 1001) recebem as sementes 1002 e 1003; escolha pela média das 4. Se a escolhida difere da do P1, rodam-se 10 sementes × 500 mil passos com ela e o resultado é reportado ao lado do primário (Welch).

| escala | cfg do P1 | médias (4 sementes) das 4 configs | escolhida | resultado |
|---|---|---|---|---|
| cb4wd | nº 11 | nº 13: -8,15, nº 11: -8,19, nº 14: -8,46, nº 12: -8,63 | nº 13 | mudou; finais (n = 10): -10,22 × primário -9,77; diferença -0,45 [-0,97; 0,07] |
| cb8wd | nº 9 | nº 9: -33,89, nº 15: -36,92, nº 8: -37,03, nº 14: -37,65 | nº 9 | seleção confirmada |
| cb12wd | nº 9 | nº 9: -69,80, nº 13: -73,13, nº 15: -73,24, nº 14: -80,64 | nº 9 | seleção confirmada |

## (c2) Sensibilidade do DQN com 10 sementes (P3c)

> Variantes de 12 estações em que a conclusão mudou no P3. Sementes 0–9 do DQN com a config do P1 ('sem reajuste') e com a reajustada no P3b; referência = heurística recalibrada na variante. C1 = DQN − heurística, por semente, IC95% t. As tabelas de 5 sementes (P3/P3b) continuam sendo as pré-registradas.

| variante | n (sem reajuste) | C1 sem reajuste [IC95%] | n (reajustado) | C1 reajustado [IC95%] | reajustado − sem reajuste [IC95% Welch] |
|---|---|---|---|---|---|
| K = 0,5 (`cb12wdx0.5`) | 10 | -0,47 [-0,76; -0,19] | 10 | -0,51 [-0,82; -0,21] | -0,04 [-0,43; 0,35] |
| fim de semana (`cb12we`) | 10 | 0,91 [0,49; 1,33] | 10 | 1,04 [0,30; 1,78] | 0,13 [-0,68; 0,94] |

## (d) Escala maior: 20 e 30 estações (P9)

> Instâncias `cb20wd` e `cb30wd` (12 legadas + as de maior volume; DESVIOS.md 2026-10-19). DQN com a config do P1 de 12 estações, **sem reajuste**, 5 sementes (4/8/12: as 10 sementes do experimento principal para 500 mil passos e as 5 de `budget4x` para 2 milhões). Referência = heurística calibrada em validação. Demanda perdida por dia no teste sintético (1000 cenários); C1 = DQN − heurística em reward (positivo favorece o DQN); redução = relativa à heurística. O rollout em 20/30 estações usa (h, K) = (3, 64) sem ajuste e só os n primeiros cenários de teste (heurística e DQN comparados nos mesmos n; coluna 'rollout'). As estações acrescentadas têm volume maior que as 12 legadas: o efeito do tamanho não está isolado.

| escala | parada | heurística | DQN 500 mil [IC95%] | C1 500 mil [IC95%] | redução | DQN 2 M [IC95%] | C1 2 M [IC95%] | redução | rollout (redução) |
|---|---|---|---|---|---|---|---|---|---|
| 4 est. | 46,94 | 24,46 | 9,77 [9,37; 10,18] (n = 10) | 14,69 [14,28; 15,10] | 60% | 9,28 [9,07; 9,49] (n = 5) | 15,18 [14,97; 15,39] | 62% | 11,83 (52%) |
| 8 est. | 89,77 | 52,79 | 29,69 [29,03; 30,34] (n = 10) | 23,10 [22,45; 23,76] | 44% | 27,54 [26,28; 28,79] (n = 5) | 25,25 [24,00; 26,51] | 48% | 27,08 (49%) |
| 12 est. | 131,34 | 80,45 | 55,94 [54,74; 57,14] (n = 10) | 24,51 [23,31; 25,71] | 30% | 52,04 [50,01; 54,08] (n = 5) | 28,41 [26,37; 30,45] | 35% | 48,10 (40%) |
| 20 est. | 359,83 | 277,27 | 219,69 [217,09; 222,29] (n = 10) | 57,58 [54,98; 60,18] | 21% | 206,14 [203,81; 208,47] (n = 5) | 71,13 [68,80; 73,46] | 26% | 202,88 (28%; 300 cen.) |
| 30 est. | 466,81 | 378,52 | 343,63 [337,76; 349,49] (n = 10) | 34,89 [29,03; 40,76] | 9% | 317,32 [311,38; 323,26] (n = 5) | 61,21 [55,27; 67,15] | 16% | 295,74 (23%; 300 cen.) |
### (d2) DQN reajustado em 30 estações (P9b)

A reavaliação do pool de 16 configs escolheu a mesma config do P1 de 12 estações (nº 9; val -372,37 contra -376,41 da segunda melhor): seleção confirmada; o resultado é o de 500 mil passos acima. Leitura pré-fixada: a configuração herdada de 12 estações não explica a queda da vantagem (dentro deste espaço de busca).


## (e) Capacidade da rede: DQN com 128–128 e 256–256 (P10)

> Config nº 9 do P1 (sem reajuste dos demais hiperparâmetros) com `hidden` mais largo, 500 mil passos, 5 sementes; 64–64 = mesmo orçamento (5 sementes em 30 est., 10 em 12 est.). 'Δ vs 64–64' = Welch (larga − 64–64, em reward; positivo favorece a rede larga). 'DQN − rollout' em reward (negativo = o rollout continua melhor; em 30 est. no subconjunto de 300 cenários). Leitura pré-fixada em `DESVIOS.md` (2026-10-20).

| escala | rede | n | demanda perdida [IC95%] | Δ vs 64–64 [IC95% Welch] | redução vs heurística | DQN − rollout [IC95%] | leitura |
|---|---|---|---|---|---|---|---|
| 30 est. | 64–64 (n = 10) | 10 | 343,63 | — | 9% | — | referência |
| 30 est. | 128–128 | 5 | 357,45 [346,68; 368,23] | -13,82 [-24,66; -2,98] | 6% | -68,72 [-79,09; -58,35] | a rede maior piora (demais hiperparâmetros não reajustados) |
| 30 est. | 256–256 | 5 | 391,36 [381,70; 401,01] | -47,73 [-57,62; -37,83] | -3% | -102,28 [-112,74; -91,81] | a rede maior piora (demais hiperparâmetros não reajustados) |
| 12 est. | 64–64 (n = 10) | 10 | 55,94 | — | 30% | — | referência |
| 12 est. | 128–128 | 5 | 62,73 [60,06; 65,39] | -6,79 [-9,42; -4,16] | 22% | -14,63 [-17,30; -11,97] | a rede maior piora (demais hiperparâmetros não reajustados) |
| 12 est. | 256–256 | 5 | 73,15 [71,80; 74,50] | -17,21 [-18,78; -15,64] | 9% | -25,05 [-26,41; -23,70] | a rede maior piora (demais hiperparâmetros não reajustados) |

## (f) Adaptação do DPRL: duas políticas (inventário + roteamento) (P11)

> Adaptação (não reprodução) de Liang et al. (2024): `bsrp/dprl.py`; config do DQN plano da escala, **sem ajuste próprio**, 500 mil passos, 5 sementes (P11) e 10 (P16). 'Δ vs plano' = Welch (dual − DQN plano do mesmo orçamento, em reward; positivo favorece o dual). 'dual − rollout' em reward (negativo = o rollout continua melhor; em 20/30 est. nos primeiros cenários de teste). Leitura pré-fixada em `DESVIOS.md` (2026-10-20).

| escala | DQN plano [n] | dual [IC95%] (n) | Δ vs plano [IC95% Welch] | redução vs heurística (plano → dual) | dual − rollout [IC95%] | leitura |
|---|---|---|---|---|---|---|
| 4 est. | 9,77 [10] | 9,57 [8,89; 10,25] (10) | 0,20 [-0,55; 0,95] | 60% → 61% | 2,26 [1,58; 2,94] | empate |
| 8 est. | 29,69 [10] | 24,79 [24,38; 25,20] (10) | 4,90 [4,17; 5,63] | 44% → 53% | 2,29 [1,88; 2,70] | a fatoração em duas políticas melhora o DQN |
| 12 est. | 55,94 [10] | 47,15 [46,51; 47,80] (10) | 8,79 [7,49; 10,09] | 30% → 41% | 0,94 [0,30; 1,59] | a fatoração em duas políticas melhora o DQN |
| 20 est. | 219,69 [10] | 211,17 [207,00; 215,33] (10) | 8,52 [3,90; 13,15] | 21% → 24% | -12,98 [-17,48; -8,48] | a fatoração em duas políticas melhora o DQN |
| 30 est. | 343,63 [10] | 332,89 [328,57; 337,21] (10) | 10,74 [3,93; 17,55] | 9% → 12% | -43,83 [-48,21; -39,45] | a fatoração em duas políticas melhora o DQN |

