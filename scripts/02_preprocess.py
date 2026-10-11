"""
02_preprocess.py
Pré-processamento do dataset NYC Citi Bike para o BSRP com RL Tabular.

Entrada:  data/raw/citibike/ (arquivos CSV descomprimidos)
Saída:
  - data/processed/demand_profiles.parquet  — chegadas/partidas por estação por intervalo de 30min
  - data/processed/station_meta.csv         — nome, lat, lon, capacidade de cada estação selecionada
  - data/processed/occupancy_series.parquet — série temporal de ocupação estimada por estação

Uso:
    python scripts/02_preprocess.py --input data/raw/citibike --output data/processed
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import json

# ──────────────────────────────────────────────
# PARÂMETROS
# ──────────────────────────────────────────────
INTERVAL_MIN = 30           # granularidade temporal em minutos
MIN_TRIPS_PER_STATION = 500 # volume mínimo para incluir estação (por mês)
TOP_N_STATIONS = 30         # número de estações de alto volume para análise inicial

# ──────────────────────────────────────────────
# 1. CARREGAMENTO
# ──────────────────────────────────────────────
def load_trips(input_dir: str) -> pd.DataFrame:
    """Carrega todos os CSVs do diretório de dados brutos."""
    files = sorted(glob.glob(os.path.join(input_dir, "**/*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {input_dir}")

    print(f"Carregando {len(files)} arquivo(s)...")
    chunks = []
    for f in files:
        print(f"  {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False, usecols=[
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id",
            "end_station_name",   "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng",
            "member_casual"
        ])
        chunks.append(df)

    trips = pd.concat(chunks, ignore_index=True)
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips["ended_at"]   = pd.to_datetime(trips["ended_at"])
    print(f"Total de viagens carregadas: {len(trips):,}")
    return trips


# ──────────────────────────────────────────────
# 2. SELEÇÃO DE ESTAÇÕES
# ──────────────────────────────────────────────
def select_stations(trips: pd.DataFrame, top_n: int = TOP_N_STATIONS) -> list[str]:
    """
    Seleciona as top-N estações por volume total de viagens (origem + destino).
    Filtra estações sem nome ou com ID nulo.
    """
    starts = trips["start_station_id"].dropna().value_counts()
    ends   = trips["end_station_id"].dropna().value_counts()
    volume = (starts.add(ends, fill_value=0)).sort_values(ascending=False)

    # Remover estações de teste (IDs incomuns) e com nome nulo
    valid = trips[trips["start_station_id"].isin(volume.index)]\
                [["start_station_id", "start_station_name"]]\
                .dropna()\
                .drop_duplicates("start_station_id")
    valid_ids = set(valid["start_station_id"])
    top_ids = [sid for sid in volume.index if sid in valid_ids][:top_n]

    print(f"\nEstações selecionadas (top {top_n}):")
    id_to_name = dict(zip(valid["start_station_id"], valid["start_station_name"]))
    for sid in top_ids[:10]:
        print(f"  {sid}: {id_to_name.get(sid, '?')} ({int(volume[sid])} viagens)")
    if len(top_ids) > 10:
        print(f"  ... +{len(top_ids)-10} estações")

    return top_ids


# ──────────────────────────────────────────────
# 3. PERFIS DE DEMANDA
# ──────────────────────────────────────────────
def compute_demand_profiles(trips: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    """
    Calcula chegadas e partidas médias por intervalo de 30min e dia da semana.
    Retorna DataFrame com index = (station_id, day_of_week, time_interval).
    """
    # Filtrar viagens das estações selecionadas
    t = trips[
        trips["start_station_id"].isin(station_ids) |
        trips["end_station_id"].isin(station_ids)
    ].copy()

    # Adicionar colunas de tempo
    t["day_of_week"]    = t["started_at"].dt.dayofweek       # 0=Mon..6=Sun
    t["time_interval"]  = (t["started_at"].dt.hour * 60 + t["started_at"].dt.minute) // INTERVAL_MIN

    # Partidas por estação
    departures = t[t["start_station_id"].isin(station_ids)]\
        .groupby(["start_station_id", "day_of_week", "time_interval"])\
        .size().rename("departures")

    # Chegadas por estação
    arrivals = trips[trips["end_station_id"].isin(station_ids)]\
        .copy()
    arrivals["day_of_week"]   = arrivals["ended_at"].dt.dayofweek
    arrivals["time_interval"] = (arrivals["ended_at"].dt.hour * 60 + arrivals["ended_at"].dt.minute) // INTERVAL_MIN
    arrivals = arrivals\
        .groupby(["end_station_id", "day_of_week", "time_interval"])\
        .size().rename("arrivals")
    arrivals.index.names = ["start_station_id", "day_of_week", "time_interval"]

    # Combinar e calcular média diária
    profile = pd.concat([departures, arrivals], axis=1).fillna(0)

    # Normalizar: dividir pelo número de semanas no dataset para obter média por dia
    n_weeks = (trips["started_at"].max() - trips["started_at"].min()).days / 7
    profile["departures_avg"] = profile["departures"] / max(n_weeks, 1)
    profile["arrivals_avg"]   = profile["arrivals"]   / max(n_weeks, 1)

    return profile.reset_index()


# ──────────────────────────────────────────────
# 4. SÉRIE TEMPORAL DE OCUPAÇÃO ESTIMADA
# ──────────────────────────────────────────────
def estimate_occupancy(trips: pd.DataFrame, station_ids: list[str],
                       station_capacities: dict,
                       sample_date: str = None) -> pd.DataFrame:
    """
    Estima ocupação horária para um dia específico (ou todos os dias).
    Usa ocupação inicial = 50% da capacidade (warm-start).

    Retorna DataFrame com colunas: date, time_interval, station_id, occupancy
    """
    if sample_date:
        t = trips[trips["started_at"].dt.date == pd.to_datetime(sample_date).date()].copy()
    else:
        t = trips.copy()

    t = t[t["start_station_id"].isin(station_ids) | t["end_station_id"].isin(station_ids)]
    t["time_interval"] = (t["started_at"].dt.hour * 60 + t["started_at"].dt.minute) // INTERVAL_MIN
    t["date"] = t["started_at"].dt.date

    records = []
    for date in sorted(t["date"].unique()):
        day_trips = t[t["date"] == date]
        # Inicializar ocupação em 50% da capacidade
        occ = {sid: station_capacities.get(sid, 20) // 2 for sid in station_ids}

        n_intervals = (22 * 60) // INTERVAL_MIN  # 06h-22h = 32 intervalos
        for interval in range(n_intervals):
            for sid in station_ids:
                departures = len(day_trips[
                    (day_trips["start_station_id"] == sid) &
                    (day_trips["time_interval"] == interval)
                ])
                arrivals = len(day_trips[
                    (day_trips["end_station_id"] == sid) &
                    (day_trips["time_interval"] == interval)
                ])
                cap = station_capacities.get(sid, 20)
                occ[sid] = max(0, min(cap, occ[sid] - departures + arrivals))
                records.append({
                    "date": date,
                    "time_interval": interval,
                    "station_id": sid,
                    "occupancy": occ[sid],
                    "capacity": cap,
                    "departures": departures,
                    "arrivals": arrivals
                })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 5. METADADOS DAS ESTAÇÕES
# ──────────────────────────────────────────────
def build_station_meta(trips: pd.DataFrame, station_ids: list[str],
                       gbfs_path: str = None) -> pd.DataFrame:
    """
    Constrói tabela de metadados: nome, coordenadas, capacidade.
    Capacidade vem do GBFS se disponível, senão usa proxy de 20 docks.
    """
    # Extrair lat/lon das viagens
    starts = trips[trips["start_station_id"].isin(station_ids)]\
        [["start_station_id", "start_station_name", "start_lat", "start_lng"]]\
        .dropna()\
        .drop_duplicates("start_station_id")\
        .rename(columns={
            "start_station_id": "station_id",
            "start_station_name": "name",
            "start_lat": "lat",
            "start_lng": "lon"
        })

    # Capacidade do GBFS (station_information.json)
    capacity_map = {}
    if gbfs_path and os.path.exists(gbfs_path):
        with open(gbfs_path) as f:
            gbfs = json.load(f)
        for s in gbfs["data"]["stations"]:
            # Tentar match por nome (GBFS usa UUIDs, CSV usa IDs antigos)
            capacity_map[s.get("name", "")] = s.get("capacity", 20)

        starts["capacity"] = starts["name"].map(capacity_map).fillna(20).astype(int)
    else:
        starts["capacity"] = 20
        print("  AVISO: GBFS não encontrado, capacidade padrão = 20")

    return starts.set_index("station_id")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/citibike", help="Diretório com CSVs")
    parser.add_argument("--output", default="data/processed",    help="Diretório de saída")
    parser.add_argument("--gbfs",   default="station_information_citibike.json",
                        help="Path para station_information.json do GBFS")
    parser.add_argument("--top-n",  type=int, default=TOP_N_STATIONS)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Carregar dados
    trips = load_trips(args.input)

    # 2. Selecionar estações
    station_ids = select_stations(trips, top_n=args.top_n)

    # 3. Metadados
    print("\nConstruindo metadados das estações...")
    meta = build_station_meta(trips, station_ids, args.gbfs)
    meta.to_csv(os.path.join(args.output, "station_meta.csv"))
    print(f"  Salvo: station_meta.csv ({len(meta)} estações)")

    # 4. Perfis de demanda
    print("\nCalculando perfis de demanda...")
    profiles = compute_demand_profiles(trips, station_ids)
    profiles.to_parquet(os.path.join(args.output, "demand_profiles.parquet"), index=False)
    print(f"  Salvo: demand_profiles.parquet ({len(profiles):,} linhas)")

    # 5. Série de ocupação (amostra: primeiro dia disponível)
    sample_date = str(trips["started_at"].dt.date.min())
    print(f"\nEstimando ocupação para {sample_date}...")
    cap_map = dict(zip(meta.index, meta.get("capacity", [20]*len(meta))))
    occ = estimate_occupancy(trips, station_ids, cap_map, sample_date=sample_date)
    occ.to_parquet(os.path.join(args.output, "occupancy_sample.parquet"), index=False)
    print(f"  Salvo: occupancy_sample.parquet ({len(occ):,} linhas)")

    print("\n✓ Pré-processamento concluído.")


if __name__ == "__main__":
    main()
