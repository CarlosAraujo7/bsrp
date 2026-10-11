# HANDOFF — BSRP (pasta `BSRP ENIAC`)

> Documento autossuficiente para abrir um **chat novo só deste projeto**. Contexto das 6 pesquisas: `../REVISAO_GERAL.md` (§7) e `../BRIEFING_CHATS.md`.
> Data: 2026-10-11. Evidências: **[código]**, **[dados]**, **[web]**, **[inferência]**.

## 0. Objetivo e destino
Reorganizar o BSRP (rebalanceamento de bicicletas compartilhadas com RL) até nível de **evento nacional sólido** e, se os dados brasileiros permitirem, **periódico**.
- **ENIAC 2026 já passou** (prazo 22/jun/2026; evento 19–22/out/2026 em Cuiabá) [web: bracis.sbc.org.br/2026/eniac]. Alvo: **ENIAC/BRACIS 2027** (chamada ainda não publicada — verificar).
- Periódico (ex.: Transportation Research Part C, ITOR, Computers & OR) exige: dados reais com origem-destino, multi-veículo ou rebalanceamento dinâmico realista, e baselines da literatura. **Não é EJOR/CEC no estado atual.**

## 1. O que é o projeto
Rebalanceamento com **1 veículo** (capacidade 20), 32 passos de 30 min (06h–22h), 12 estações de Manhattan (Citi Bike, jan/2024). Q-learning tabular (estado por cluster, 4⁴×12×4×32 = 393.216 estados), 4 melhorias tabulares (A–E), DQN em numpy puro (26→64→64→60, replay, target, Double DQN), escalabilidade 4/8/12 estações, ablação do DQN, controle de aliasing, baselines threshold/rollout, script REINFORCE.

Mapa: `01_escopo.md … 04_experimentos.md` · `scripts/{01_download,02_preprocess,03_collect_tembici_gbfs,04_simulator,05_qlearning,06_improved_qlearning,07_dqn,08_scaling,09_reinforce,10_baselines}` · `results/{qtable*.pkl,dqn_*.npz,scaling/,baselines/}` · `demand_profiles_*`, `station_selected_*`.

## 2. Estado real
Experimentos prontos; **não há `paper/`**; sem git. Dados Tembici **não coletados** (script existe, sem dados). `results/reinforce/` **não existe** (REINFORCE sem resultados). `baselines_8st.json` e `baselines_12st.json` só têm *threshold* (o *rollout* não terminou).

## 3. Achados críticos (verificados)

**B1. Dados fracos** [dados]: 215.897 viagens de **14–31/jan/2024** (meia janela, inverno); o plano dizia jan–jun/2024. Perfis misturam dias úteis e fins de semana.

**B2. Simulador** [código `scripts/04_simulator.py`]: chegadas e partidas Poisson **independentes por estação** (sem fluxo origem-destino, **sem conservação de bicicletas**); `DEMAND_SCALE = 3.5` arbitrário ("ajustado vs. dados reais" sem evidência); todo deslocamento custa ≥ 1 intervalo de 30 min; avaliação **no mesmo simulador** do treino, sem dias reais de teste.

**B3. Uma única semente de treino (42)** [código `08_scaling.py:65`, `07_dqn.py:42`]. Os testes de Welch (300 episódios de avaliação) medem a variância do simulador, **não do algoritmo** (pseudorreplicação). "DQN > gulosa (p = 0,001)" e "toda componente é necessária" precisam de ≥10 sementes de treino com IC.

**B4. Vazamento leve na seleção de checkpoint** [código]: o `eval_r` periódico usa `SEED_EVAL = 123` (`07_dqn.py:43`; `08_scaling.py:83`) e a avaliação final inclui a semente 123. A curva do DQN é instável: em 4 estações `eval_r` varia −1,3 (ep. 500) → −5,4 (ep. 3.500) → −1,5 (ep. 5.000), e o resultado final é −3,63.

**B5. A tese "DQN é necessário" fica fraca pelos próprios dados** [dados `04_experimentos.md §12`, `results/scaling/ql_control.json`]:

| Estações | Gulosa | DQN | Q-learning (cluster) | Q-learning (por estação) |
|---|---|---|---|---|
| 4 | −1,98 | −3,49 | −7,13 | **−3,73** |
| 8 | −3,80 | −6,76 | −13,92 | — |
| 12 | −5,57 | **−3,20** | −12,23 | — |

Gulosa vence o DQN em 4 e 8 estações; só em 12 o DQN vence. Em 4 estações o **tabular com estado por estação empata com o DQN**. O Q-learning "original" usa estado agregado por cluster, que não distingue qual estação do cluster está vazia embora as ações escolham estações — é um adversário construído para perder.

**B6. Narrativas conflitantes no mesmo documento** [leitura]: §4 atribui a falha à complexidade amostral (0,33 atualizações/par; "≈600k episódios"); §12.6 à *aliasing* por cluster; §10.2 interpreta "corrigir aliasing de ações piorou" como benefício do espaço variável (explicação não testada). O controle §12.6 varia ao mesmo tempo resolução, tamanho de espaço e episódios (20k × 18k).

**B7. Baselines incompletos e trabalho não documentado** [dados]: *rollout* só em 4 estações; REINFORCE sem resultados e ausente de `04_experimentos.md`; a *threshold* é pior que a gulosa em 12 estações (−6,97 × −5,57).

**B8. Números inconsistentes no texto**: aleatória = −15,6 / −14,5 / −14,21 conforme a tabela; "+64,1%" vs. "+61,4%"; "4 estações = top-1 por cluster (2 clusters)" quando há 4 clusters.

**B9. Referência errada** [web]: "Pan et al. (2024), arXiv:2402.03589" — autores reais: **Liang, Liu, Jena, Lodi, Vidal** (*Single- vs. Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing*). Conferir as demais (Seo et al. 2022; Liu et al. 2018; Chiariotti et al. 2019).

## 4. O que está sólido (reaproveitar)
Pipeline de dados e seleção de estações; simulador (como base); infraestrutura de treino/avaliação multi-seed (a estender); a observação empírica de que a representação de estado decide entre tabular e profundo é **real e interessante** (B5).

## 5. Plano de lapidação (ordem e critérios de aceite)
1. **B0 – Higiene (1 dia).** `git init`; consolidar `04_experimentos.md` (uma narrativa, números únicos, tabelas geradas por script).
2. **B1 – Coleta Tembici AGORA.** Rodar `scripts/03_collect_tembici_gbfs.py` (verificar endpoints GBFS e estrutura v3.0); coletar ≥4 semanas a cada 5–15 min. É o diferencial brasileiro e leva semanas — não adie.
3. **B2 – Rigor mínimo.** ≥10 sementes de treino; sementes de validação e de teste **separadas**; checkpoint escolhido só na validação; IC 95%; teste sobre sementes (não sobre episódios).
4. **B3 – Simulador.** Fluxo O–D com conservação de bicicletas (ou justificar a simplificação); calibrar `DEMAND_SCALE` contra ocupação real; separar dia útil × fim de semana; **dias reais de teste** (hold-out); mais de 1 mês de dados.
5. **B4 – Baselines.** Q-learning tabular **por estação** (justo), gulosa/threshold **calibradas** (grid), *rollout* em todas as escalas, REINFORCE/PPO/A2C, e uma referência da literatura (ex.: política de Liang et al. 2024 adaptada, ou ADP de Seo et al. 2022).
6. **B5 – Reenquadramento.** "Quando a representação de estado decide entre tabular e profundo no BSRP" é mais defensável que "tabular falha, DQN resolve".
7. **B6 – Escrita.** Artigo no formato do evento-alvo; figuras (curvas com IC, heatmap de escala, ablação) geradas por script.

## 6. Checklist de nível de publicação
- [ ] ≥10 sementes de treino; IC; testes sobre sementes
- [ ] Validação/teste separados; dias reais de teste
- [ ] Simulador justificado (O–D/conservação) e calibrado com dados
- [ ] Tabular justo (por estação) e heurísticas calibradas
- [ ] ≥1 baseline da literatura de BSRP
- [ ] Dados brasileiros (Tembici) ao menos como estudo de caso
- [ ] Documentação sem números conflitantes; referências verificadas

## 7. Armadilhas
- Não escreva "DQN supera a gulosa" sem dizer que só ocorre em 12 estações e com 1 semente.
- Não mantenha a narrativa "falta de amostras" **e** "aliasing" ao mesmo tempo.
- Não assuma os endpoints GBFS do Tembici sem testá-los (podem ter mudado).

## 8. Não verificado
Endpoints GBFS do Tembici; a licença/uso permitido dos dados; se a 2ª metade de jan/2024 é representativa; chamada do ENIAC/BRACIS 2027 (prazos).

## 9. Prompt inicial sugerido para o chat novo
> Diretório de trabalho: `C:\Users\vinic\OneDrive\Documentos\Novas pesquisas\BSRP ENIAC`. Leia `HANDOFF.md` e `../REVISAO_GERAL.md` (§7). Objetivo: lapidar o BSRP para ENIAC/BRACIS 2027 (e avaliar se há caminho para periódico). Comece por B0 e B1 (higiene + coleta Tembici) e proponha o protocolo de rigor (B2) antes de rodar qualquer experimento novo.
