# 01 — Escopo e Fundamentação do Projeto

**Projeto:** Escalonamento de Bicicletas Compartilhadas com Reinforcement Learning Tabular  
**Evento-alvo:** ENIAC 2026  
**Última atualização:** 2026-05-25  

---

## 1. Problema e Motivação

O **Bike Sharing Rebalancing Problem (BSRP)** consiste em redistribuir bicicletas entre estações de um sistema de bicicleta compartilhada usando um ou mais veículos de redistribuição, de modo a minimizar a insatisfação de usuários ao longo do dia. As insatisfações ocorrem em dois sentidos:

- **Demanda não atendida (empty-station failure):** usuário chega a uma estação vazia e não consegue retirar bicicleta.
- **Devolução não atendida (full-station failure):** usuário tenta devolver bicicleta em estação lotada.

### Por que RL e não Otimização Clássica?

| Critério | Otimização Inteira / ILP | Heurísticas (ACO, GA) | RL Tabular |
|---|---|---|---|
| Lida com incerteza de demanda | Não (determinístico) | Parcialmente | **Sim (aprende da distribuição)** |
| Escalabilidade | Ruim (NP-difícil) | Média | Boa para espaços discretizados |
| Interpretabilidade | Alta | Baixa | **Alta (Q-table explícita)** |
| Tempo de treinamento | N/A | Alto | Moderado |
| Requisito de modelo de demanda | Completo | Parcial | **Mínimo (model-free)** |

A formulação como MDP permite aprender uma política de redistribuição diretamente da interação com um simulador baseado em dados históricos, sem necessidade de um modelo analítico explícito da demanda. Isso é especialmente valioso quando a demanda varia por horário, dia da semana e sazonalidade.

**Alinhamento com o ENIAC:** RL tabular é diretamente coberto na área de Reinforcement Learning do evento. A combinação com logística urbana e dados abertos brasileiros fortalece a relevância social do trabalho.

---

## 2. Formulação como MDP

### 2.1 Definição Formal

O processo de decisão é **episódico** e **discreto no tempo**, com um episódio correspondendo a um dia operacional (ex: 06h–22h) dividido em intervalos de Δt = 30 minutos → **T = 32 passos de tempo por episódio**.

#### Estado s_t

```
s_t = (o_1, o_2, ..., o_N, p_t, c_t, t)
```

Onde:
- `o_i ∈ {0, 1, ..., cap_i}` — ocupação (número de bikes) na estação i
- `N` — número de estações no subconjunto considerado
- `p_t` — posição atual do veículo (índice da estação)
- `c_t` — carga atual do veículo (número de bikes no caminhão)
- `t` — índice do intervalo de tempo (0..T-1)

**Decisão de discretização:** O espaço de estados bruto é contínuo/inteiro de dimensão muito alta. Será aplicado k-means sobre perfis históricos de demanda para agrupar os estados em clusters tratáveis (ver Seção 4).

#### Ação a_t

Em cada passo, o agente decide:
```
a_t = (j, Δb)
```

Onde:
- `j` — próxima estação a visitar (índice)
- `Δb ∈ Z` — quantidade de bikes a remover (< 0) ou depositar (> 0) na estação atual antes de partir

**Restrições de ação:**
- `c_t + Δb ≥ 0` (não pode depositar mais do que carrega)
- `o_i - Δb ≥ 0` e `o_i - Δb ≤ cap_i` (não viola capacidade da estação)
- `c_t - Δb ≤ cap_veículo` (não sobrecarrega o veículo)

#### Recompensa r_t

```
r_t = -α · (demanda_não_atendida_t) - β · (devoluções_não_atendidas_t)
```

Onde demanda não atendida e devoluções não atendidas são contadas em todas as estações no intervalo t, independentemente da posição do veículo. Valores padrão: α = β = 1 (pesos iguais). Sensibilidade a α e β será testada nos experimentos.

#### Transição P(s_{t+1} | s_t, a_t)

- O veículo se move de p_t para j com tempo de viagem determinístico ou amostrado da média histórica.
- A demanda (chegadas e partidas em cada estação) é amostrada do simulador baseado nos dados históricos.
- A ocupação é atualizada: `o_i(t+1) = o_i(t) + chegadas_i(t) - partidas_i(t)` (com clipping nos limites).

### 2.2 Horizonte e Desconto

- **Episódico**, sem desconto (γ = 1) — adequado para horizonte finito curto (T = 32).
- Objetivo: maximizar recompensa total acumulada no episódio.

---

## 3. Abordagem: Q-Learning Tabular com ε-greedy

### Por que Q-Learning Tabular?

1. **Escopo do ENIAC:** O evento valoriza clareza metodológica e fundamentos de IA. Q-Learning tabular demonstra domínio da base teórica do RL sem complexidade de deep learning.
2. **Interpretabilidade:** A Q-table resultante pode ser inspecionada e visualizada, facilitando análise dos padrões aprendidos.
3. **Baseline forte:** Serve como referência clara para comparação. Trabalhos futuros podem escalar para DQN/PPO.
4. **Viabilidade em 8 semanas:** Implementação controlável com recursos computacionais modestos.

### Limitações Reconhecidas (a discutir no paper)

- Escala limitada: adequado para N ≤ 15 estações com discretização agressiva.
- Não generaliza para configurações não vistas (lookup na Q-table).
- Convergência pode ser lenta sem ajuste cuidadoso de hiperparâmetros.

### Hiperparâmetros Planejados

| Parâmetro | Valor inicial | Range a testar |
|---|---|---|
| α (taxa de aprendizado) | 0.1 | {0.01, 0.1, 0.3} |
| γ (desconto) | 1.0 | fixo |
| ε inicial | 1.0 | fixo |
| ε final | 0.05 | fixo |
| ε decay | linear over 80% dos episódios | — |
| Episódios de treino | 10.000 | {5k, 10k, 20k} |

---

## 4. Discretização com K-Means

### Justificativa

O espaço de estados completo — incluindo ocupação de cada estação, posição e carga do veículo — é intratável sem discretização. K-means sobre perfis históricos de demanda é uma abordagem padrão na literatura (usado em contextos similares de logística com RL tabular).

### Estratégia

1. Construir o **perfil de demanda** de cada estação: vetor de chegadas e partidas médias por intervalo de 30 minutos (32 bins × 2 = 64 features por estação).
2. Aplicar k-means com k = {4, 8, 16} clusters sobre as estações → grupos de estações com comportamento similar.
3. Representar o estado de cada cluster como a **média de ocupação** dentro do cluster (escalar discretizado em L bins).
4. Estado resultante reduzido: `(cluster_1_occ, ..., cluster_k_occ, posição_veículo, carga_veículo, t)`

### Critério de escolha de k

Silhouette score + análise qualitativa da semântica dos clusters (ex: cluster de estações residenciais vs. comerciais vs. de trânsito).

---

## 5. Políticas Baseline para Comparação

| Política | Descrição |
|---|---|
| **Aleatória** | Escolhe ação uniformemente ao acaso a cada passo |
| **Gulosa (Greedy)** | Move para a estação mais desequilibrada e balanceia ao máximo possível |
| **Q-Learning** | Política aprendida (proposta do trabalho) |

A política gulosa é importante pois simula o raciocínio humano intuitivo e é frequentemente usada como baseline na literatura de BSRP.

---

## 6. Métricas de Avaliação

- **Insatisfações totais por episódio** (demanda não atendida + devoluções não atendidas)
- **Melhoria percentual** do Q-Learning vs. Baseline Aleatório e Guloso
- **Curva de aprendizado** (episódios × recompensa média) — evidência de convergência
- **Análise de sensibilidade** a α, β e k

---

## 7. Escopo de Estações (a decidir após análise dos dados)

- **Mínimo viável para o paper:** 8–12 estações com padrões de demanda distintos.
- **Preferência:** selecionar subconjunto de Manhattan central (Citi Bike NYC) com alta variabilidade de demanda e boa distribuição entre tipos (residencial/comercial).
- **Critério de seleção:** top-N estações por volume de viagens + representação de clusters distintos.

---

## 8. Decisões Tomadas e Justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Algoritmo RL | Q-Learning tabular | Escopo ENIAC, interpretabilidade, viabilidade |
| Discretização | K-means sobre perfis de demanda | Padrão na literatura, reduz espaço de estados |
| Δt (granularidade) | 30 minutos | Balance entre precisão e tratabilidade do estado |
| Horizonte | 1 dia (32 passos) | Episódio natural do problema; evita desconto |
| Dataset primário | NYC Citi Bike | Maior volume, melhor documentado, usado na literatura |
| Dataset secundário | Tembici Brasil (GBFS) | Validação qualitativa, relevância local ENIAC |
| Número de veículos | 1 (single-vehicle) | Simplifica espaço de ação; padrão em trabalhos iniciais |

---

## 9. O que NÃO está no escopo

- Deep RL (DQN, PPO, A3C) — trabalho futuro
- Multi-vehicle rebalancing — complexidade excessiva para 8 semanas
- Otimização de rotas em tempo real (VRPTW) — escopo diferente
- Dockless bike sharing — sistema de estações fixas é mais adequado para MDP tabular

---

## 10. Referências-Chave Identificadas

1. **Seo et al. (2022)** — "Rebalancing Docked Bicycle Sharing System with Approximate Dynamic Programming and Reinforcement Learning". Journal of Advanced Transportation. https://doi.org/10.1155/2022/2780711
2. **Pan et al. (2024)** — "Single- vs. Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing". arXiv:2402.03589. https://arxiv.org/abs/2402.03589
3. **Liu et al. (2018)** — "A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems". arXiv:1802.04592.
4. **Chiariotti et al. (2019)** — "A Bike-sharing Optimization Framework Combining Dynamic Programming and Demand Forecasting".
5. **Efficient Inventory Routing for BSS (2024)** — ScienceDirect, combinatorial RL framework.

---

## 11. Próximos Passos

- [ ] Analisar estrutura do dataset NYC Citi Bike (02_dados.md)
- [ ] Selecionar subconjunto de estações com base em volume e diversidade
- [ ] Confirmar viabilidade do espaço de estados com a discretização proposta
- [ ] Implementar simulador de demanda (Semana 2)
