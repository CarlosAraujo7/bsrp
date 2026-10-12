# 04 — Experimentos (consolidado v2)

**Versão:** v2, 2026-10-12 · **Substitui:** v1 de 2026-05-26 (preservada em `archive/04_experimentos_v1_2026-05-26.md` e no commit-base do git).
**Regra deste documento:** nenhum número é digitado à mão. Todos vêm de `results/**/*.json` via `scripts/11_make_tables.py` → [`tables/legacy_results.md`](tables/legacy_results.md). O que não tem arquivo de origem aparece como **"não rastreável"**.

> **Status: resultados legados, pré-protocolo.** Todos usam **uma única semente de treino (42)**. As "sementes" 123/456/789 são de *avaliação* no simulador e medem o ruído do simulador, não do algoritmo. Servem para **formular hipóteses**; nenhuma afirmação de superioridade pode ser feita a partir deles. O desenho que corrige isso está em [`PROTOCOLO_RIGOR.md`](PROTOCOLO_RIGOR.md).

---

## 1. Configuração (conferida no código)

| Item | Valor | Fonte |
|---|---|---|
| Episódio | 1 dia, 32 intervalos de 30 min (06h–22h); 1 veículo, capacidade 20 | `04_simulator.py:30-34` |
| Recompensa | −(partidas não atendidas + chegadas bloqueadas) | `04_simulator.py:306` |
| Demanda | Poisson **independente** por estação (partidas e chegadas), λ = média histórica × `DEMAND_SCALE = 3,5` | `04_simulator.py:283-287` |
| Conservação de bicicletas | **Não há** (chegadas não vêm de partidas); bicicletas são criadas/destruídas | idem |
| `DEMAND_SCALE` | 3,5, "ajustado vs. dados reais" — **sem evidência de ajuste no repositório** | `04_simulator.py:38` |
| Chegadas | **reescaladas** por fatores de 0,15 a 0,26 para igualar as partidas (coluna `arrivals_avg_orig` guarda o original). Nas 12 estações, Σ chegadas originais = 1.478,8 vs Σ partidas = 299,8 por dia (~5×): o desequilíbrio líquido real foi removido da calibração | `demand_profiles_*_normalized.csv` |
| Dados | 215.897 viagens Citi Bike, 14–31/jan/2024 (≈ 2,5 semanas, inverno); dias úteis e fins de semana misturados | `HANDOFF.md` B1 |
| Instâncias | 4 est. = **2 clusters × 2**; 8 est. = 4 clusters × 2; 12 est. = 4 clusters (4,4,2,2) | `station_selected_*.csv` |
| Q-learning | α = 0,1; γ = 1; ε 1,0 → 0,05 linear nos primeiros 80% dos episódios; estado = nível de ocupação **por cluster** (4 níveis) + posição + carga (4 faixas) + t | `08_scaling.py:41-47`, `04_simulator.py:312-337` |
| DQN | (2n+2) → 64 → 64 → 5n (n estações × 5 deltas; 26→64→64→60 com n = 12); Adam lr 3e-4; γ = 1; lote 64; buffer 10.000; treina a cada 4 passos; alvo sincronizado a cada 50 episódios; Double DQN; ε 1,0 → 0,05 em 70% | `08_scaling.py:49-61` |
| Avaliação | 3 sementes (123, 456, 789) × 100 episódios, ε = 0 | `08_scaling.py:64` |
| Avaliação periódica (seleção de checkpoint) | 20 episódios a cada 500, **num único gerador de semente 123 que avança** | `08_scaling.py:60-61, 83` |

## 2. Resultados legados — leitura permitida

Tabelas completas em [`tables/legacy_results.md`](tables/legacy_results.md) (T1–T7). O que se pode ler, **sem inferência estatística**:

1. **4 estações:** threshold (−2,08) e DQN (−3,49) e rollout (−3,70) estão próximos; o Q-learning por cluster (−7,13) fica perto da aleatória do mesmo cenário. Gulosa/aleatória multi-semente não estão salvas.
2. **8 estações:** threshold (−5,26) melhor que DQN (−6,76); Q-learning (−13,92) foi treinado com apenas **3.500** episódios.
3. **12 estações:** DQN (−3,20) melhor que threshold (−6,97). Única referência salva para a gulosa: **−4,60** (semente 123, 100 ep). É a única escala em que o DQN aparece à frente de uma heurística — com uma semente de treino.
4. **Checkpoints do DQN em 8 estações** (T3) variam de −6,14 a −8,37 entre 9.500 e 17.500 episódios, todos avaliados nas mesmas sementes de avaliação: a curva não é monótona. Existe `dqn_8st_best.npz`, mas o critério de escolha não está registrado; se foi olhando estes valores, o resultado final está inflado.
5. **Controle de representação** (T5): com **orçamento igual** (10.000 ep), por-estação binário (−5,35) vs cluster (−6,15) dá Δ = 0,80, com erro-padrão ≈ 0,54 *apenas do simulador* (supondo n = 300 por braço, sem variância de treino). O ganho para −3,73 só aparece com 18.000 episódios.
6. **Tamanho do espaço (T6):** em 4 estações o espaço por cluster e o por-estação binário têm **o mesmo tamanho (8.192)**; em 12 estações o por-estação binário seria 6.291.456 (inviável em tabela).

## 3. Auditoria das afirmações da v1

✔ confirmado · ✖ contradito pelos arquivos · ? não rastreável (sem arquivo de origem) · ⚠ aritmética certa, premissa não testada

| # | Afirmação na v1 | O que os arquivos mostram | |
|---|---|---|---|
| 1 | DQN 12 est. = −3,20 | `dqn_12st.json`: −3,20 | ✔ |
| 2 | DQN 8 est. = −6,76 | `dqn_8st.json` (9.500 ep): −6,76; outros 5 checkpoints avaliados nas mesmas sementes | ✔ |
| 3 | DQN 4 est. = −3,49, "9.500 ep em cada instância" | −3,49 com **5.000** ep | ✔ valor / ✖ episódios |
| 4 | Q-learning 4 est. = −7,13 com 20.000 ep | `ql_4st.json`: −7,13, **10.000** ep | ✖ |
| 5 | Q-learning 8 est. com "10.000–20.000 ep" | `ql_8st.json`: **3.500** ep | ✖ |
| 6 | Q-learning 12 est. = −12,23 | 20.000 ep | ✔ |
| 7 | Gulosa (−1,98 / −3,80 / −5,57) e aleatória (−7,69 / −13,84 / −14,21) multi-semente | nenhum JSON | ? |
| 8 | Aleatória = −15,6 / −14,5 / −14,21; "+64,1%" e "+61,4%" | só −14,54 (sem. 123, 100 ep). Os dois percentuais saem de referências diferentes (−15,6 e −14,5) com gulosa −5,6 | ✖ |
| 9 | Gulosa (12 est.) = −5,6 | −4,60 (sem. 123, 100 ep) | ✖ |
| 10 | DQN 12 est. = −3,4 ±4,9 (§3.1/§11.4) | sem. 123 em `dqn_12st.json`: −3,3 ±4,0 | ✖ |
| 11 | "DQN supera a gulosa em 39%" | depende do −5,6 da linha 9 | ? |
| 12 | "DQN > gulosa, t = 5,33, p = 0,001" | testes sobre 300 episódios de avaliação de **um** modelo treinado: pseudorreplicação; dados por episódio não salvos | ✖ inferência |
| 13 | Espaço Q-learning 4 est. = 131.072 (e "~3,1 atualizações/par") | 4 est. tem 2 clusters → **8.192** | ✖ |
| 14 | "4 est. = top-1 por cluster (2 clusters)" | 2 clusters × 2 estações | ✖ descrição |
| 15 | Por-estação (−3,73, 18k) vs cluster (−7,13, 20k): Δ = +3,40, p < 0,001 | −3,73 é de 18k; −7,13 é de 10k; em 10k: −5,35 vs −6,15 (Δ = 0,80) | ✖ confundido por orçamento |
| 16 | Ablação: −10,74 / −10,53 / −12,09 vs −3,20 | números em `ablation_9500ep.json` ✔; protocolo **não** equivalente (ver §5, itens 2–3) | ✔ valores / ✖ conclusão |
| 17 | Demanda não atendida do DQN = 3,1/ep | campo `unmet` = 0,0 em todos os JSONs do DQN (bug, §5 item 1) | ✖ |
| 18 | Q-learning L=4, 20k ep = −13,8 | sem JSON de avaliação final (`history.json` guarda só a curva) | ? |
| 19 | Variantes A–D (−16,2 / −15,9 / −14,8 / −16,0) | `qtable_improved_*_summary.json` | ✔ |
| 20 | Variante E (−15,5) | sem arquivo de resumo | ? |
| 21 | "0,33 atualizações/par → ≈ 613 mil episódios" | aritmética ✔; premissa (≥ 10 visitas/par, visitas uniformes) não testada | ⚠ |

## 4. Narrativa única (substitui as três da v1)

A v1 explicava a falha do Q-learning de três formas incompatíveis (complexidade amostral §4; *aliasing* por cluster §12.6; "o espaço variável ajuda" §10.2). **Nenhuma está demonstrada**: em 4 estações o espaço por cluster e o por-estação têm o mesmo tamanho (a explicação amostral não distingue as duas representações), e o controle de aliasing mistura orçamentos (10k × 18k).

O que os dados legados sustentam é só uma **hipótese**, a ser testada no protocolo:

> **H.** Em BSRP com um veículo, a *representação de estado* e a classe de função (tabular por cluster → tabular por estação → linear → rede neural) determinam se o aprendizado alcança ou supera heurísticas bem calibradas; o algoritmo (Q-learning × DQN × policy gradient) importa menos.

Consequência: a frase "tabular falha, DQN resolve" **não deve ser escrita**. O que se pode dizer hoje é que, com uma semente de treino, o DQN só ficou à frente das heurísticas na instância de 12 estações.

## 5. Defeitos conhecidos no código (a corrigir antes de qualquer novo experimento)

1. **`unmet` do DQN é sempre 0,0:** `eval_dqn` lê `env._state.unmet_dep`, atributo que `SimState` não tem (`08_scaling.py:350`; `04_simulator.py:55-61`).
2. **ε não decai sem replay:** `if use_replay and eps_fixed is None and ep < decay_end` (`08_scaling.py:454`). No modo `ablation`, `train_dqn` é chamado sem `eps_fixed` → `no_replay` explora com ε = 1,0 o treino inteiro (a menos que o JSON de 9.500 ep tenha sido produzido com outras flags, o que não está registrado).
3. **Ablação não isoepisódica na prática:** o "Full" vem de treino em 3 blocos com LR decrescente; as variantes do modo `ablation` usam LR único e 1 bloco. Dois conjuntos de resultados coexistem (`abl_*.json` e `ablation_9500ep.json`) com valores diferentes e sem registro de episódios nos primeiros.
4. **Sementes:** ambiente de treino e gerador de exploração usam ambos a semente 42 (`08_scaling.py:65, 82, 369`); sem `SeedSequence`.
5. **Seleção de checkpoint:** avaliação periódica de 20 episódios (erro-padrão ≈ 1,1) num gerador de semente 123 que avança; os primeiros ~100 episódios dessa sequência coincidem com os da avaliação final da semente 123 (o consumo do gerador não depende da política — ver `PROTOCOLO_RIGOR.md` §2).
6. **`scripts/ql_control.py`** (citado no §12.7 da v1) **não existe**; só o `ql_control.json`.
7. **Baselines incompletos:** *rollout* só em 4 estações; `09_reinforce.py` sem resultados.
8. **Referências** do `01_escopo.md` corrigidas em 2026-10-12 (3 erros).

## 6. Artefatos

| Arquivo | Conteúdo |
|---|---|
| `scripts/04_simulator.py` | simulador, políticas aleatória e gulosa |
| `scripts/05_qlearning.py`, `06_improved_qlearning.py` | Q-learning tabular e variantes A–E |
| `scripts/07_dqn.py`, `08_scaling.py` | DQN (numpy) e escalabilidade 4/8/12, ablação |
| `scripts/09_reinforce.py`, `10_baselines.py` | REINFORCE (sem resultados), threshold/rollout |
| `scripts/11_make_tables.py` | gera `tables/legacy_results.md` a partir dos JSONs |
| `scripts/03_collect_tembici_gbfs.py` | coletor GBFS v3.0 (ver `05_coleta_tembici.md`) |
| `results/` | JSONs e pesos; `*.pkl` fora do git (hashes em `results/MANIFEST_pkl.sha256`) |
| `tables/legacy_results.md` | tabelas T1–T7 geradas |
| `archive/` | v1 deste documento |

## 7. Próximos passos

Ver [`PROTOCOLO_RIGOR.md`](PROTOCOLO_RIGOR.md) (B2) e [`05_coleta_tembici.md`](05_coleta_tembici.md) (B1). Nenhum experimento novo foi rodado nesta etapa.
