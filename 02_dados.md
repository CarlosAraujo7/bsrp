# 02 — Dados: Fontes, Aquisição e Decisões

**Projeto:** BSRP com RL Tabular  
**Última atualização:** 2026-05-25  

---

## 1. Estratégia de Dados

A pesquisa usa **dois datasets complementares**:

| Papel | Dataset | Uso |
|---|---|---|
| **Primário** | NYC Citi Bike (histórico) | Treino e avaliação do agente RL; simulador de demanda |
| **Secundário** | Tembici Brasil (GBFS tempo-real) | Validação qualitativa; estudo de caso comparativo |

---

## 2. Dataset Primário — NYC Citi Bike

### 2.1 Fonte e Acesso

- **Fonte oficial:** https://citibikenyc.com/system-data
- **Arquivo S3:** https://s3.amazonaws.com/tripdata/index.html
- **Feed GBFS real-time:** https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json
- **Licença:** NYCBS Data Use Policy (uso para pesquisa e análise é permitido)
- **Formato:** CSV comprimido (.zip), um arquivo por mês

### 2.2 Estrutura dos Dados (formato atual)

| Campo | Descrição |
|---|---|
| `ride_id` | Identificador único da viagem |
| `rideable_type` | Tipo de bicicleta (classic/electric) |
| `started_at` | Timestamp de início |
| `ended_at` | Timestamp de fim |
| `start_station_name` | Nome da estação de origem |
| `start_station_id` | ID da estação de origem |
| `end_station_name` | Nome da estação de destino |
| `end_station_id` | ID da estação de destino |
| `start_lat` / `start_lng` | Coordenadas de origem |
| `end_lat` / `end_lng` | Coordenadas de destino |
| `member_casual` | Tipo de usuário (membro/casual) |

**Nota:** Formato anterior (pré-2021) incluía `tripduration`, `gender`, `birth_year`, `bikeid`. Esses campos foram removidos em reformulação de privacidade.

### 2.3 Volume e Período

- Dados disponíveis desde **julho de 2013**.
- Meses recentes: ~3–5 milhões de viagens/mês.
- Para este projeto: **usar 12 meses de 2023** (Jan–Dez 2023) como período base.
  - Justificativa: pós-pandemia, padrões normalizados; dataset 2023 bem documentado no Kaggle.
  - Kaggle mirror: https://www.kaggle.com/datasets/hassanabsar/nyc-citi-bike-ride-share-system-data-2023

### 2.4 O que Precisamos Derivar

Os arquivos CSV contêm **viagens** (origem → destino). Para o BSRP precisamos de **ocupação de estações ao longo do tempo**. Derivação:

```
Para cada estação i e intervalo t:
  chegadas_i(t) = nº de viagens que terminam em i no intervalo t
  partidas_i(t) = nº de viagens que começam em i no intervalo t
  ocupação_i(t+1) = ocupação_i(t) + chegadas_i(t) - partidas_i(t)
```

**Limitação:** Não temos o estado inicial de ocupação das estações. Estratégia:
- Usar o primeiro intervalo do dia como "aquecimento" e iniciar com ocupação estimada por proporção da capacidade.
- Capacidade de cada estação é obtida via GBFS `station_information.json`.

### 2.5 Seleção de Subconjunto de Estações

Critério de seleção (a ser executado após download):
1. Filtrar estações com volume > 5.000 viagens/mês (alta demanda, dados estatisticamente robustos).
2. Aplicar k-means (k=8–16) sobre perfis de demanda horária.
3. Selecionar **1–2 estações representativas por cluster** → total de **10–15 estações**.
4. Preferir estações geograficamente próximas (subgráfico conexo) para viabilizar roteamento do veículo.

### 2.6 Estrutura Confirmada dos Dados (verificado em 2026-05-25)

Inspeção direta do arquivo `202401-citibike-tripdata.zip` confirmou os campos e formato:

```
ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,
end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual
```

Exemplo real:
```
62EF1AC5BE598131,classic_bike,2024-01-24 09:03:33.533,2024-01-24 09:06:53.535,
E 102 St & 1 Ave,7407.13,E 103 St & Lexington Ave,7463.09,40.7869946,-73.94164802,...,member
```

**Observação importante:** O campo `start_station_id` usa formato numérico com ponto (ex: `7407.13`), diferente dos UUIDs do GBFS atual. O match entre trip data e GBFS será feito **por nome de estação**, não por ID.

### 2.7 Escala do Sistema (GBFS verificado em 2026-05-25)

| Métrica | Valor |
|---|---|
| Total de estações ativas | 2.409 |
| Estações em Manhattan (por coordenada, cap ≥ 15) | 1.171 |
| Capacidade média em Manhattan | 38,1 docks |
| Maior estação | E 40 St & Park Ave — 123 docks |
| Estações com cap ≥ 30 em Manhattan | ~600 |

Arquivo salvo: `manhattan_stations.json` (1.171 estações, com nome, lat/lon, capacidade)

### 2.8 Aquisição — Comandos

```bash
# Dataset 2023 (arquivo anual, ~1.6 GB)
wget https://s3.amazonaws.com/tripdata/2023-citibike-tripdata.zip

# OU meses individuais de 2024 (~370 MB cada) — mesmo formato
wget https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip
wget https://s3.amazonaws.com/tripdata/202402-citibike-tripdata.zip
wget https://s3.amazonaws.com/tripdata/202403-citibike-tripdata.zip
```

**Script automatizado:** `scripts/01_download_citibike.sh`

**Status:** Script criado. Download pendente (executar localmente — arquivo grande demais para sandbox).

### 2.9 Decisão sobre Período de Dados

Após verificação do bucket S3:
- Dados 2023: arquivo **anual único** (`2023-citibike-tripdata.zip`, 1,6 GB)
- Dados 2024: arquivos **mensais** (`202401-citibike-tripdata.zip`, ~370 MB/mês)

**Decisão:** Usar **Jan–Jun 2024** (6 meses, ~2,2 GB total) como dataset primário.
- Justificativa: formato mensal facilita processamento incremental; formato idêntico ao verificado; período suficiente para capturar variação sazonal (inverno + primavera).

### 2.10 Decisões sobre o Dataset Primário

| Decisão | Escolha | Justificativa |
|---|---|---|
| Período | Jan–Jun 2024 (6 meses) | Formato mensal; mesmos padrões de 2023; verificado |
| Granularidade temporal | 30 min | Balanceia precisão de demanda com tratabilidade do MDP |
| Seleção de estações | k-means + volume, top-30 inicial | Garante diversidade de perfis e relevância estatística |
| Capacidade das estações | GBFS station_information (match por nome) | IDs mudaram; nome é o campo estável entre trip data e GBFS |
| Ocupação inicial | 50% da capacidade (warm-start) | Dados históricos não incluem estado inicial direto |
| Match estação trip ↔ GBFS | Por nome de estação | IDs no CSV usam formato diferente dos UUIDs do GBFS atual |

---

## 3. Dataset Secundário — Tembici Brasil (GBFS)

### 3.1 Situação dos Dados Abertos no Brasil

A Tembici (operadora do Bike Itaú) **não disponibiliza dados históricos de viagens** publicamente. Diferente do Citi Bike, não há arquivo CSV de trips para download.

O que está disponível:
- **GBFS real-time feeds** (status atual das estações, atualizado a cada 60s)
- **Dados de localização e capacidade** das estações (station_information)
- **Dados do Portal de Dados Abertos de Recife** (apenas localização de estações, não viagens)

### 3.2 Endpoints GBFS Tembici (confirmados por MobilityData/GBFS systems.csv)

| Cidade | URL Base GBFS |
|---|---|
| Rio de Janeiro | `https://riodejaneiro.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json` |
| São Paulo | `https://saopaulo.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json` |
| Brasília | `https://brasilia.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json` |
| Recife | `https://recife.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json` |
| Salvador | `https://salvador.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json` |

Cada feed expõe:
- `station_information.json` — nome, lat/lon, capacidade de cada estação
- `station_status.json` — bikes disponíveis, vagas disponíveis, status operacional

### 3.3 Estratégia de Uso

Como não há histórico de viagens, o uso dos dados brasileiros no paper será:

1. **Estudo de caso descritivo:** Usar `station_information.json` para comparar o tamanho e distribuição espacial das redes brasileiras com NYC.
2. **Coleta prospectiva (opcional):** Coletar snapshots do `station_status.json` a cada 15 minutos por 2–4 semanas para gerar um histórico mínimo de ocupação.
3. **Simulação com parâmetros calibrados:** Se houver dados suficientes, calibrar o simulador com as taxas de chegada/saída inferidas dos snapshots.
4. **Transferência de política (trabalho futuro):** Avaliar se a política treinada no NYC pode ser transferida para configurações brasileiras com ajuste de escala.

### 3.4 Script de Coleta GBFS (a implementar)

```python
# Coleta periódica de station_status Tembici Rio de Janeiro
import requests, json, time
from datetime import datetime

URL = "https://riodejaneiro.publicbikesystem.net/customer/gbfs/v3.0/station_status.json"

while True:
    resp = requests.get(URL).json()
    timestamp = datetime.now().isoformat()
    with open(f"data/tembici_rio_{timestamp}.json", "w") as f:
        json.dump(resp, f)
    time.sleep(900)  # coleta a cada 15 minutos
```

**Status:** Pendente. Iniciar coleta o quanto antes para acumular histórico.

### 3.5 Decisões sobre o Dataset Secundário

| Decisão | Escolha | Justificativa |
|---|---|---|
| Uso principal | Descritivo + coleta prospectiva | Ausência de histórico público de viagens |
| Cidade alvo | Rio de Janeiro | Maior rede Tembici; relevância para o ENIAC |
| Frequência de coleta | 15 min | Resolução suficiente para inferir padrões diários |
| Período de coleta | Iniciar imediatamente | Quanto mais cedo, mais histórico disponível |

---

## 4. Problemas Identificados e Riscos

| Problema | Risco | Mitigação |
|---|---|---|
| Dados NYC sem estado inicial de ocupação | Simulador impreciso no início do episódio | Usar período de warm-up; normalizar por capacidade |
| Formato NYC mudou em 2021 | Campos diferentes em dados mais antigos | Usar apenas dados 2021+ com formato uniforme |
| Tembici sem histórico público | Não dá para treinar agente com dados brasileiros | Uso descritivo; coleta prospectiva como complemento |
| Volume grande de dados (12 meses NYC) | Processamento pode ser lento | Usar Parquet; amostrar meses representativos |
| Capacidade das estações NYC via GBFS | Feed atual pode não refletir 2023 | Cruzar com dados históricos de station_information |

---

## 5. Estrutura de Diretórios Planejada

```
BSRP ENIAC/
├── 01_escopo.md
├── 02_dados.md
├── 03_modelagem.md          (a criar)
├── 04_experimentos.md       (a criar)
├── data/
│   ├── raw/
│   │   ├── citibike_2023/   (CSVs mensais)
│   │   └── tembici_snapshots/
│   └── processed/
│       ├── demand_profiles.csv
│       ├── station_clusters.csv
│       └── simulation_episodes/
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_preprocess.py
│   ├── 03_cluster_stations.py
│   ├── 04_simulator.py
│   └── 05_qlearning.py
└── paper/
    └── bsrp_eniac_2026.tex
```

---

## 6. Próximos Passos Imediatos

- [x] Verificar estrutura do CSV do Citi Bike (confirmado em 2026-05-25)
- [x] Baixar `station_information.json` via GBFS NYC (2.409 estações, salvo localmente)
- [x] Identificar estações de Manhattan por coordenadas (1.171 estações com cap ≥ 15)
- [ ] **Executar `scripts/01_download_citibike.sh`** para baixar Jan–Jun 2024 (~2.2 GB)
- [ ] **Iniciar `scripts/03_collect_tembici_gbfs.py --city rio`** para começar coleta Tembici
- [ ] Executar `scripts/02_preprocess.py` após download para gerar perfis de demanda
- [ ] Criar notebook EDA: visualizar perfis de demanda das top-30 estações Manhattan
- [ ] Aplicar k-means (k=8,12,16) e escolher número de clusters via silhouette score
