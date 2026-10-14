# P2 — ablação do DQN (gerado por `scripts/21_analyze_p1_p2.py --which p2`)

> Diferença **pareada por semente** (mesma semente → mesmos pesos iniciais e mesmo ambiente de treino): Δ = DQN completo − variante, em reward de TESTE (maior = melhor). IC95% t e p (t) sobre a diferença por semente; Holm sobre C7–C9 por escala; TOST (δ = 0,5) sobre a diferença pareada. Leitura: 'necessária' só com p de Holm < 0,05 e Δ > 0. `vanilla` (as três removidas) é exploratória e fica fora da família. Lembrete de desenho (DESVIOS.md): `−target` ≡ `−target −double`; `−replay` faz 1 atualização por transição com lote 1.


## cb12wd

Referência (DQN completo): `final_ext_cb12wd.jsonl`, 10 sementes; config-base da ablação: 1 config(s) distinta(s).

### Reward de teste (1000 cenários sintéticos)

| variante | contraste | n sem. | completo | variante | Δ = completo − variante | IC95% t (pareado) | p (t) | p Holm | TOST p | P(completo > variante) | leitura |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noreplay | C7: replay | 10 | -55,94 | -132,57 | 76,63 | [68,12; 85,14] | 0,0000 | 0,0000 | 1,0000 | 1,00 | necessária (completo melhor) |
| notarget | C8: rede-alvo | 10 | -55,94 | -63,10 | 7,16 | [-11,70; 26,01] | 0,4128 | 0,8257 | 0,7775 | 0,40 | inconclusivo (não detectamos diferença) |
| nodouble | C9: Double DQN | 10 | -55,94 | -56,18 | 0,24 | [-1,38; 1,86] | 0,7461 | 0,8257 | 0,3622 | 0,60 | inconclusivo (não detectamos diferença) |
| vanilla | exploratória: vanilla | 10 | -55,94 | -128,88 | 72,94 | [62,39; 83,49] | 0,0000 | — | 1,0000 | 1,00 | exploratória |

### Dias reais de teste (24 dias úteis; ocupação inicial desconhecida)

| variante | n sem. | Δ = completo − variante | IC95% t (pareado) | p (t) |
|---|---|---|---|---|
| noreplay | 10 | 48,52 | [39,47; 57,57] | 0,0000 |
| notarget | 10 | 4,71 | [-9,31; 18,73] | 0,4665 |
| nodouble | 10 | -0,30 | [-2,21; 1,60] | 0,7264 |
| vanilla | 10 | 45,18 | [33,68; 56,69] | 0,0000 |

## cb4wd

Referência (DQN completo): `final_ext_cb4wd.jsonl`, 10 sementes; config-base da ablação: 1 config(s) distinta(s).

### Reward de teste (1000 cenários sintéticos)

| variante | contraste | n sem. | completo | variante | Δ = completo − variante | IC95% t (pareado) | p (t) | p Holm | TOST p | P(completo > variante) | leitura |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noreplay | C7: replay | 10 | -9,77 | -25,10 | 15,32 | [11,43; 19,22] | 0,0000 | 0,0000 | 1,0000 | 1,00 | necessária (completo melhor) |
| notarget | C8: rede-alvo | 10 | -9,77 | -9,42 | -0,35 | [-0,78; 0,07] | 0,0906 | 0,1812 | 0,2266 | 0,30 | inconclusivo (não detectamos diferença) |
| nodouble | C9: Double DQN | 10 | -9,77 | -9,60 | -0,18 | [-0,55; 0,20] | 0,3164 | 0,3164 | 0,0401 | 0,40 | equivalente (TOST, δ = 0,5) |
| vanilla | exploratória: vanilla | 10 | -9,77 | -20,63 | 10,85 | [6,56; 15,15] | 0,0003 | — | 0,9998 | 1,00 | exploratória |

### Dias reais de teste (24 dias úteis; ocupação inicial desconhecida)

| variante | n sem. | Δ = completo − variante | IC95% t (pareado) | p (t) |
|---|---|---|---|---|
| noreplay | 10 | 10,99 | [7,66; 14,32] | 0,0000 |
| notarget | 10 | -0,39 | [-1,63; 0,84] | 0,4918 |
| nodouble | 10 | -0,07 | [-1,45; 1,31] | 0,9155 |
| vanilla | 10 | 7,75 | [4,21; 11,28] | 0,0008 |
