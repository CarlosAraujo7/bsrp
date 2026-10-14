# P1 — busca de hiperparâmetros estendida (gerado por `scripts/21_analyze_p1_p2.py --which p1`)

> Pool de 16 configs por método (8 originais + 8 novas; DESVIOS.md 2026-10-14). Escolha por **validação** (2 sementes de ajuste × 200 mil passos); o teste só entra nos reruns. 'Limite' = o hiperparâmetro da escolhida vale o mínimo/máximo explorado no pool de 16 (lr/lr_actor/lr_critic, contínuos, não entram).

### Configuração escolhida (validação)

| escala | método | original: cfg (val) | pool de 16: cfg (val) | ganho em val | escolha mudou? | hiperparâmetros da escolhida no limite do pool |
|---|---|---|---|---|---|---|
| cb4wd | dqn | 1 (-8,84) | 11 (-8,31) | 0,53 | sim | target_sync=16000 (máx.), batch=256 (máx.), buffer=200000 (máx.) |
| cb4wd | linear | 1 (-29,54) | 1 (-29,54) | 0,00 | não (original mantida) | nenhum (interior) |
| cb4wd | tab_station | 5 (-39,24) | 11 (-38,14) | 1,11 | sim | nenhum (interior) |
| cb4wd | tab_cluster | 0 (-39,53) | 8 (-38,08) | 1,45 | sim | eps_decay_frac=0.1 (mín.), eps_end=0.2 (máx.) |
| cb4wd | pg | 1 (-34,33) | 14 (-25,36) | 8,98 | sim | nenhum (interior) |
| cb8wd | dqn | 1 (-42,11) | 9 (-33,73) | 8,38 | sim | train_every=1 (mín.), batch=256 (máx.) |
| cb8wd | linear | 5 (-82,67) | 5 (-82,67) | 0,00 | não (original mantida) | nenhum (interior) |
| cb8wd | tab_station | 1 (-96,95) | 9 (-95,88) | 1,08 | sim | eps_end=0.01 (mín.) |
| cb8wd | tab_cluster | 1 (-94,81) | 8 (-94,79) | 0,01 | sim | eps_decay_frac=0.1 (mín.), eps_end=0.2 (máx.) |
| cb8wd | pg | 1 (-83,64) | 14 (-77,54) | 6,10 | sim | nenhum (interior) |
| cb12wd | dqn | 1 (-84,94) | 9 (-69,32) | 15,62 | sim | train_every=1 (mín.), batch=256 (máx.) |
| cb12wd | linear | 2 (-128,51) | 2 (-128,51) | 0,00 | não (original mantida) | train_every=1 (mín.) |
| cb12wd | tab_station | 4 (-136,39) | 4 (-136,39) | 0,00 | não (original mantida) | eps_decay_frac=0.9 (máx.) |
| cb12wd | tab_cluster | 3 (-134,20) | 3 (-134,20) | 0,00 | não (original mantida) | eps_end=0.01 (mín.) |
| cb12wd | pg | 1 (-130,49) | 15 (-126,16) | 4,34 | sim | episodes_per_update=1 (mín.) |

### Reruns: teste original × estendido (reward de teste por semente; sementes independentes, Welch)

| escala | método | n sem. (orig / ext) | média original | média estendida | ext − orig [IC95%] | p (t) | dias reais: ext − orig [IC95%] |
|---|---|---|---|---|---|---|---|
| cb4wd | dqn | 10 / 10 | -10,43 | -9,77 | 0,66 [0,16; 1,16] | 0,0129 | -0,03 [-1,23; 1,17] |
| cb4wd | tab_station | 10 / 10 | -40,38 | -39,74 | 0,64 [-0,57; 1,85] | 0,2802 | 1,49 [-0,53; 3,51] |
| cb4wd | tab_cluster | 10 / 10 | -42,97 | -41,39 | 1,58 [-0,48; 3,64] | 0,1251 | 1,50 [-2,44; 5,44] |
| cb4wd | pg | 10 / 10 | -37,71 | -19,88 | 17,83 [8,71; 26,95] | 0,0016 | 17,91 [6,34; 29,48] |
| cb8wd | dqn | 10 / 10 | -31,96 | -29,69 | 2,27 [1,31; 3,23] | 0,0001 | -0,10 [-1,96; 1,76] |
| cb8wd | tab_station | 10 / 10 | -94,50 | -94,37 | 0,13 [-0,46; 0,72] | 0,6509 | 2,53 [-0,40; 5,46] |
| cb8wd | tab_cluster | 10 / 10 | -92,75 | -92,10 | 0,65 [-0,60; 1,89] | 0,2890 | -0,68 [-3,70; 2,33] |
| cb8wd | pg | 10 / 10 | -101,77 | -105,48 | -3,71 [-23,31; 15,89] | 0,6955 | -1,56 [-22,22; 19,10] |
| cb12wd | dqn | 10 / 10 | -64,30 | -55,94 | 8,36 [5,99; 10,72] | 0,0000 | 3,82 [1,27; 6,38] |
| cb12wd | pg | 10 / 10 | -124,94 | -157,95 | -33,01 [-47,20; -18,82] | 0,0002 | -34,19 [-46,83; -21,55] |

### Exploratório: val médio do DQN por nível de cada hiperparâmetro (16 configs; efeitos confundidos, só descritivo)

| escala | hiperparâmetro | nível: val médio (n configs) |
|---|---|---|
| cb4wd | target_sync | 250: -12,7 (n=3); 500: -9,6 (n=2); 1000: -11,6 (n=1); 4000: -8,7 (n=5); 8000: -8,9 (n=3); 16000: -8,4 (n=2) |
| cb4wd | train_every | 1: -10,0 (n=6); 2: -8,7 (n=6); 4: -10,8 (n=4) |
| cb4wd | batch | 32: -10,3 (n=1); 64: -10,3 (n=3); 128: -10,1 (n=8); 256: -8,4 (n=4) |
| cb4wd | buffer | 10000: -12,4 (n=4); 50000: -9,1 (n=7); 100000: -8,4 (n=3); 200000: -8,4 (n=2) |
| cb4wd | eps_decay_frac | 0.3: -8,6 (n=2); 0.5: -8,4 (n=2); 0.7: -9,4 (n=7); 0.9: -11,2 (n=5) |
| cb8wd | target_sync | 250: -64,4 (n=3); 500: -50,7 (n=2); 1000: -60,7 (n=1); 4000: -40,5 (n=5); 8000: -42,9 (n=3); 16000: -38,2 (n=2) |
| cb8wd | train_every | 1: -43,7 (n=6); 2: -41,9 (n=6); 4: -62,4 (n=4) |
| cb8wd | batch | 32: -60,0 (n=1); 64: -56,6 (n=3); 128: -48,1 (n=8); 256: -37,1 (n=4) |
| cb8wd | buffer | 10000: -63,2 (n=4); 50000: -46,2 (n=7); 100000: -36,7 (n=3); 200000: -38,2 (n=2) |
| cb8wd | eps_decay_frac | 0.3: -38,1 (n=2); 0.5: -36,0 (n=2); 0.7: -47,3 (n=7); 0.9: -56,7 (n=5) |
| cb12wd | target_sync | 250: -113,1 (n=3); 500: -102,2 (n=2); 1000: -106,9 (n=1); 4000: -78,6 (n=5); 8000: -91,2 (n=3); 16000: -81,2 (n=2) |
| cb12wd | train_every | 1: -84,7 (n=6); 2: -88,0 (n=6); 4: -110,8 (n=4) |
| cb12wd | batch | 32: -102,7 (n=1); 64: -104,3 (n=3); 128: -96,3 (n=8); 256: -73,4 (n=4) |
| cb12wd | buffer | 10000: -113,6 (n=4); 50000: -92,8 (n=7); 100000: -71,0 (n=3); 200000: -81,2 (n=2) |
| cb12wd | eps_decay_frac | 0.3: -80,6 (n=2); 0.5: -74,9 (n=2); 0.7: -90,5 (n=7); 0.9: -107,0 (n=5) |
