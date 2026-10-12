# PROTOCOLO DE RIGOR (B2) — proposta v0.1

**Data:** 2026-10-12 · **Status: PROPOSTA, aguardando aprovação. Nenhum experimento novo foi rodado.**
Depois de aprovado: commit + tag `protocolo-v1` **antes** de qualquer execução; desvios vão em `DESVIOS.md` com data e motivo.

Por que existe: os resultados legados usam 1 semente de treino, validam e testam nas mesmas sementes, e comparam métodos com orçamentos e representações diferentes (ver `04_experimentos.md` §3 e §5). Este protocolo define, **antes de ver resultados**, como se mede, compara e decide.

---

## 0. Perguntas (pré-definidas)

- **Q1.** Em cada escala (4, 8, 12 estações), o melhor método aprendido supera a melhor heurística **calibrada**?
- **Q2.** (hipótese H de `04_experimentos.md` §4) Com algoritmo e orçamento fixos, onde a *representação de estado* faz diferença: tabular-cluster → tabular-estação → linear → rede neural?
- **Q3.** As respostas a Q1–Q2 sobrevivem a variações do simulador (`DEMAND_SCALE`, tipo de dia) e a dias reais não vistos?

Sem pergunta pré-definida, sem teste. Qualquer outra comparação é **exploratória** e será rotulada assim.

## 1. Unidade de replicação

- A unidade é a **semente de treino**. Episódios de avaliação são medidas repetidas *dentro* de uma semente e nunca entram como amostra independente.
- **n ≥ 10 sementes de treino por (método, escala)**; alvo 20. O n final sai do piloto (G1): para teste sobre sementes com poder 80%, α = 0,05, n ≈ 7,85·σ_d²/Δ² (comparação com heurística, σ_d = DP da diferença por semente) ou 15,7·σ²/Δ² (duas amostras independentes de sementes). Com Δ = 1,0 unidade de reward: σ = 1 → n = 8 / 16; σ = 1,5 → n = 18 / 35.
- Custo é baixo: o legado levou ~38 s por 2.000 episódios (8 estações, `elapsed` nos JSONs; outra máquina). 10 mil episódios ≈ minutos; o gargalo é a máquina estar disputada (§9).

## 2. Sementes e cenários (sem vazamento, comparação pareada)

1. **Gerador:** `numpy.random.SeedSequence` com `spawn`; cada semente de treino `s` deriva fluxos **independentes** para `init` (pesos), `explore` (ε-greedy) e `env_train` (demanda). O legado usava a mesma semente 42 para ambiente e exploração.
2. **Conjuntos de cenários fixos e salvos em disco** (`scenarios/{val,test}_<escala>.npz`: ocupação inicial + matriz de partidas e chegadas por intervalo e estação):
   - **val:** 200 episódios · **test:** 1.000 episódios · gerados de uma `SeedSequence` própria, disjunta de qualquer semente de treino, com hash registrado.
   - **Todos os métodos são avaliados nos mesmos cenários** (*common random numbers*). Já vale hoje por construção do gerador (o consumo do `rng` do ambiente não depende da política — leitura de `04_simulator.py`, a confirmar por teste), mas passa a ser explícito e testável.
3. O treino usa cenários novos de `env_train`; **o teste é tocado uma vez por configuração final**.

## 3. Seleção e ajuste

- **Sementes de ajuste** (3, ex.: T0–T2) separadas das **sementes finais** (S0–S19). Hiperparâmetros e regra de checkpoint são escolhidos só em **val**, com as sementes de ajuste.
- **Regra de checkpoint fixada a priori:** reportar o **último** checkpoint (sem seleção). Secundário, rotulado: melhor checkpoint por média móvel em val (nunca em test).
- **Orçamento de ajuste igual por método:** mesmo número de configurações (proposta: 12 por busca aleatória) para cada aprendiz; para heurísticas, grade de tamanho comparável (alvo de ocupação, limiares de delta, regra de destino). Nenhum método fica com "padrão" enquanto outro é ajustado.
- Avaliação periódica durante o treino usa val (≥ 200 episódios; os 20 episódios do legado dão erro-padrão ≈ 1,1).

## 4. Orçamento de treino

- Definido em **passos de ambiente (decisões)**, não em episódios (o tempo de deslocamento muda as decisões por episódio). Referência do legado: ≲ 3·10⁵ passos (9.500 ep × ≤ 32). Valor por escala fixado no piloto e **igual para todos os aprendizes**.
- Reportar **curvas de aprendizado** (reward em val × passos, média e IC 95% sobre sementes, não uma corrida) e duas medidas: desempenho final e área sob a curva (eficiência amostral). Tempo de parede reportado à parte.

## 5. Métodos

**Referências sem aprendizado** (todas em todas as escalas):
`parado` (veículo não age; **hoje não existe** e é o piso real), `aleatória`, `gulosa` (v1), `threshold` **calibrada** por grade em val, `rollout` de 1 e 2 passos sobre a melhor heurística calibrada. (O rollout legado só rodou em 4 estações e foi pior que a threshold.)

**Escada de representação** (mesmo algoritmo base, mesmo orçamento):

| Rótulo | Representação | Viável em |
|---|---|---|
| R0 | tabular, ocupação **por cluster** (legado) | 4, 8, 12 |
| R1 | tabular, ocupação **por estação** (binária; 3 níveis se couber) | 4 (8.192) e 8 (262.144); **12 inviável** (6.291.456) — dito no texto, não escondido |
| R2 | aproximação **linear** nas features do DQN | 4, 8, 12 |
| R3 | rede 64-64 (DQN legado) | 4, 8, 12 |

**Algoritmos na representação R3:** DQN, Double DQN, REINFORCE com baseline, A2C ou PPO (um policy-gradient moderno). **Referência da literatura:** adaptação de Seo et al. (2022, ADP/RL) ou de Liang et al. (2024); a escolha depende de ler os dois e verificar o que é reproduzível (docas × sem docas, frota, dinâmica). Será chamada **"adaptação"**, não "reprodução".

**Ablação do DQN:** só depois de corrigir o bug de ε (`04_experimentos.md` §5.2), com **o mesmo cronograma de treino** em todas as variantes, ≥ 10 sementes, incluindo a variante sem as três componentes. "Cada componente é necessária" só se o IC da diferença pareada por semente excluir 0.

## 6. Estatística

- **Métrica primária:** reward médio nos cenários de test, **por semente**; depois, média e **IC 95% sobre sementes** (t e bootstrap percentil; ambos se n < 20). Secundárias: demanda não atendida (após corrigir o bug do `unmet`), proporção de passos com ação útil.
- **Comparações primárias (família única, correção de Holm):**
  - C1–C3: melhor aprendido × melhor heurística calibrada em cada escala (Q1).
  - C4: R1 × R0 (4 e 8); C5: R2 × R1; C6: R3 × R2 (Q2).
  - C7–C9: ablação (3 contrastes).
- **Contra heurística determinística:** diferença = (média da semente) − (heurística nos mesmos cenários) → uma diferença por semente → t de uma amostra e Wilcoxon sobre sementes. **Entre aprendizes:** sementes independentes (ou pareadas por índice de semente, se a decomposição de variância mostrar ganho; decidir no piloto).
- **Equivalência:** "empata" só com TOST e margem **δ = 0,5 unidade de reward**, fixada agora. Sem TOST, escrever "não detectamos diferença".
- **Tamanho de efeito** sempre junto do p-valor: diferença em unidades de reward e em %, mais probabilidade de superioridade entre sementes.
- **Decomposição de variância** (entre sementes × entre cenários) reportada: mostra de onde vem o ruído.
- **Nada é descartado:** sementes que divergem entram na análise e são sinalizadas. Proibido tratar episódios × sementes como amostra única.

## 7. Portões de decisão

| Portão | Quando | Critério | Se falhar |
|---|---|---|---|
| **G0** | após refatorar e corrigir bugs | testes passam: invariantes do simulador; CRN (dois métodos no mesmo cenário veem a mesma demanda); **regressão**: com configuração legada, números ≈ aos de `tables/legacy_results.md` | não rodar mais nada até corrigir |
| **G1** | piloto: 4 est., 5 sementes, R0–R3 + heurísticas | só decide **n, orçamento e hiperparâmetros**; não decide afirmações | ajustar n |
| **G2** | experimento completo | (A) aprendido > heurística calibrada em ≥ 1 escala, IC excluindo 0 e robusto à sensibilidade de `DEMAND_SCALE` → artigo "quando a representação decide"; (B) aprendido ≈ heurística em todas → artigo de **resultado negativo/benchmark** ("heurísticas calibradas são competitivas"); (C) misto → reportar escala a escala | **(B) também é publicado**; decidido agora |
| **G3** | ≥ 4 semanas de Tembici | o simulador reproduz, em dias de hold-out, a distribuição horária de fluxo líquido e de ocupação dentro de tolerância definida **antes** de ver o hold-out | Tembici vira estudo descritivo |

## 8. Simulador e dados (dependências do B3)

1. **Sensibilidade:** `DEMAND_SCALE` ∈ {1,5; 2,5; 3,5; 5}; conclusões que mudam de sinal são reportadas como dependentes da escala. A calibração de 3,5 hoje é injustificada.
2. **Conservação e O–D:** ou implementar fluxo O–D com conservação (Citi Bike tem O–D), ou justificar por escrito a simplificação e mostrar a sensibilidade. As chegadas legadas foram reescaladas por 0,15–0,26 para igualar as partidas (o original tem ~5× mais chegadas), apagando o desequilíbrio líquido real: refazer a calibração a partir de fluxo líquido real.
3. **Dados Citi Bike:** estender de 2,5 semanas para **≥ 3 meses** (dados abertos), perfis separados dia útil × fim de semana, divisão por **dias** (treino/val/test), não por episódios. *(Download: depende da sua autorização.)*
4. **Tembici (hold-out definido agora, antes de olhar os dados):** teste = **última semana completa (seg–dom)** da janela de coleta; val = semana anterior; treino = o resto; dias marcados (feriados, eleições). Demanda estimada só do treino; o teste é **replay de fluxo líquido real**, com a ressalva de que o fluxo observado já contém a reposição operacional e de que a resposta dos usuários às ações não é capturada.
5. Corrigir antes: `unmet`, ε sem replay, sementes (itens 1–4 de `04_experimentos.md` §5).

## 9. Reprodutibilidade e execução

- Cada execução gera **uma linha** em `runs/*.jsonl`: hash do commit (e flag *dirty*), método, escala, representação, orçamento, semente, hash da configuração e dos cenários, métricas, tempo.
- Tabelas e figuras só por script a partir dos `jsonl`; `04_experimentos.md` não tem número digitado (hoje já vale para `tables/legacy_results.md`).
- Dependências fixadas (`requirements.txt`), `PCG64` explícito.
- **Recursos:** a máquina está sob carga de outros projetos (memória virtual livre ≈ 1–3 GB em 2026-10-12). Usar ≤ 3 processos numpy (~100–150 MB cada) por vez, nunca `pandas` nos runners, e combinar janelas de execução com os outros jobs.

## 10. Esforço e ordem

1. Refatorar simulador (API de cenário), corrigir 4 bugs, `SeedSequence`, runner + testes G0: **3–5 dias de trabalho**.
2. Heurísticas calibradas e referências sem aprendizado (rápido).
3. Piloto G1 (horas). 4. Experimento completo (dezenas de horas de CPU, paralelizável por semente). 5. Figuras e texto.
Prazo: a chamada do ENIAC/BRACIS 2027 ainda não saiu (o de 2026 fechou em 8→22/jun/2026); há folga até o 1º semestre de 2027, mas a coleta Tembici pede ≥ 4 semanas e é melhor 8–12.

## 11. Caminho para periódico (avaliação — inferência, não verificada)

**Não no estado atual.** O que bloquearia, e o que este protocolo faz a respeito:

- **Tembici não tem O–D** (só estoque): serve como estudo de caso com fluxo líquido, não como base de matriz O–D. O–D exige Citi Bike ou similar.
- Um veículo, simulador sem conservação e sem baseline da literatura **não bastam**. Para periódico seriam necessários: multi-veículo ou dinâmica realista, baseline da literatura **reproduzido**, ≥ 3 sistemas reais (incl. Tembici), e um resultado robusto.
- A contribuição defensável é **metodológica/empírica**: escada de representação com estatística correta, heurísticas calibradas fortes, simulador calibrado e **aberto**. Isso só vira artigo de periódico se G2 for "A" (ou um "B" muito bem fundamentado) e se os itens acima forem cumpridos.
- Recomendação: **ENIAC/BRACIS 2027 como alvo firme**; decidir periódico depois de G2, sem prometer. Não verifiquei escopo nem fator de impacto de nenhum periódico específico.

## 12. O que muda em relação ao legado

| Legado | Protocolo |
|---|---|
| 1 semente de treino (42) | ≥ 10 (alvo 20), `SeedSequence` |
| Teste sobre 300 episódios de 1 modelo | teste sobre sementes, 1.000 cenários fixos compartilhados |
| Mesmas sementes para seleção e teste | val e test disjuntos; checkpoint = último |
| Tabular só por cluster | escada R0–R3 com mesmo orçamento |
| Heurísticas sem calibrar; rollout parcial; sem baseline "parado" | grade em val; rollout em todas as escalas; "parado" |
| Orçamentos 3.500 / 5.000 / 9.500 / 10.000 / 20.000 ep | mesmo orçamento em passos |
| "Empata", "toda componente é necessária" sem teste | TOST (δ = 0,5) e IC da diferença pareada |
| Números digitados | tudo gerado de `runs/*.jsonl` |
