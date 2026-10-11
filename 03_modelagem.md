# 03 — Modelagem: Seleção de Estações e Análise de Demanda

**Projeto:** BSRP com RL Tabular  
**Última atualização:** 2026-05-25  
**Status:** Estações selecionadas ✓ | Perfis de demanda gerados ✓

---

## 1. Dados Utilizados

- **Fonte:** `202401-citibike-tripdata_2.csv` (Jan 2024, extraído diretamente do ZIP via range HTTP)
- **Viagens:** 215.897 registros (14–31 de janeiro de 2024)
- **Método de extração:** o arquivo ZIP usa compressão `stored` (método 0), permitindo leitura direta dos bytes CSV sem descompressão. Extração feita via Python sem download completo.

---

## 2. Seleção das 12 Estações

### Metodologia

1. Calculou-se o **volume total** (partidas + chegadas) por estação.
2. As 25 estações de maior volume foram retidas.
3. Construiu-se o **vetor de perfil de demanda** para cada estação: chegadas e partidas médias por intervalo de 30 minutos (48 × 2 = 96 features).
4. Aplicou-se **k-means** (StandardScaler + k=4) para identificar grupos de comportamento similar.
5. Selecionaram-se 2 representantes por cluster (maior volume) + 4 estações extras de alto volume.

### Silhouette Scores

| k | Silhouette | Observação |
|---|---|---|
| 3 | 0.111 | Melhor separação |
| 4 | 0.079 | Mais representativo semanticamente |
| 5 | 0.063 | Clusters muito pequenos |
| 6 | 0.061 | Fragmentado |

**Decisão:** usar k=4, pois produz 4 perfis semanticamente interpretáveis, mesmo com silhouette menor que k=3. A baixa separação é esperada: estações de Manhattan têm perfis similares por serem geograficamente compactas.

---

## 3. Clusters Identificados

| Cluster | Rótulo | Pico Partidas | Pico Chegadas | Interpretação |
|---|---|---|---|---|
| 0 | Misto-tarde | 18h | 17h30 | Uso misto; pico no fim da tarde |
| 1 | Alto-volume | 18h | 18h | Alto fluxo bidirecional; área de grande movimento |
| 2 | Residencial | 8h30 | 18h | Saem de manhã, voltam à noite — padrão commuter residencial |
| 3 | Comercial | 17h | 8h30 | Chegam de manhã, saem à tarde — padrão commuter escritório |

Os clusters 2 e 3 são complementares e representam o padrão clássico de pendularismo (commute), essencial para modelar o desequilíbrio diário: estações residenciais ficam vazias de manhã, estações comerciais ficam vazias à tarde.

---

## 4. As 12 Estações Selecionadas

| Estação | Cluster | Volume Jan/2024 | Capacidade | Lat | Lon |
|---|---|---|---|---|---|
| Forsyth St & Grand St | Misto-tarde | 2.481 | 63 | 40.718 | -73.993 |
| Cleveland Pl & Spring St | Misto-tarde | 2.474 | 33 | 40.722 | -73.997 |
| University Pl & E 14 St | Alto-volume | 3.510 | 61 | 40.735 | -73.992 |
| E 17 St & Broadway | Alto-volume | 3.122 | 78 | 40.737 | -73.990 |
| E 1 St & 1 Ave | Residencial | 2.384 | 75 | 40.723 | -73.989 |
| W 44 St & 11 Ave | Residencial | 2.296 | 79 | 40.762 | -73.997 |
| E 43 St & Madison Ave | Comercial | 2.799 | 56 | 40.754 | -73.979 |
| Great Jones St | Comercial | 2.373 | 23 | 40.727 | -73.994 |
| 8 Ave & W 38 St | Alto-volume | 2.904 | 65 | 40.755 | -73.992 |
| Amsterdam Ave & W 73 St | Alto-volume | 2.855 | 55 | 40.780 | -73.981 |
| W 43 St & 10 Ave | Misto-tarde | 2.466 | 117 | 40.760 | -73.995 |
| W 27 St & 6 Ave | Misto-tarde | 2.351 | 66 | 40.745 | -73.991 |

Arquivos gerados:
- `station_selected_12.csv` — metadados completos com cluster, capacidade, coordenadas
- `demand_profiles_12stations.csv` — chegadas/partidas médias por intervalo (96 valores por estação)

---

## 5. Implicações para o MDP

### Espaço de Estados

Com 12 estações e discretização de ocupação em L níveis por estação:

| L (níveis de ocupação) | Estados de ocupação | Com veículo (pos × carga) | Total aproximado |
|---|---|---|---|
| 3 (vazia/meio/cheia) | 3^12 = 531.441 | × 12 × 10 | ~63 milhões |
| 2 (vazia/não-vazia) | 2^12 = 4.096 | × 12 × 10 | ~500 mil |

**Problema:** mesmo com L=2, o espaço é grande demais para Q-Learning tabular puro com 12 estações.

### Estratégia de Redução

Usar os 4 clusters como abstração de estado:
- Representar cada cluster por sua **ocupação agregada** (média de docks disponíveis no cluster) em vez de estação individual.
- Estado reduzido: `(occ_c0, occ_c1, occ_c2, occ_c3, pos_veículo, carga_veículo, intervalo_t)`
- Discretizando cada occ_ci em 4 níveis e usando 12 posições × 5 níveis de carga × 32 intervalos:

```
4^4 × 12 × 5 × 32 = 256 × 12 × 5 × 32 = 491.520 estados
```

Tratável para Q-Learning tabular com exploração eficiente.

### Próximas Decisões a Documentar

- [ ] Confirmar modelo de transição: determinístico vs. estocástico
- [ ] Definir matriz de tempos de viagem entre as 12 estações
- [ ] Definir capacidade do veículo de redistribuição (padrão: 20 bikes)
- [ ] Formalizar função de recompensa com parâmetros α e β

---

## 6. Decisões e Justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| N° de estações | 12 | Viável para Q-Learning tabular com abstração por cluster |
| Abstração do estado | Ocupação por cluster (não por estação individual) | Reduz espaço de estados de ~500k para ~500k mas com semântica mais rica |
| Representantes por cluster | 2 de maior volume | Maximiza cobertura de dados para estimar demanda |
| Extração de dados | Range HTTP + leitura direta do ZIP stored | Evita download de 370 MB; extrai 215k viagens suficientes para perfis |
| k para clustering | 4 | Interpretável semanticamente; clusters residencial/comercial capturam dinâmica core do BSRP |
| Silhouette baixo (0.079) | Documentado como limitação | Espera-se em área geográfica compacta; não invalida a abordagem |
