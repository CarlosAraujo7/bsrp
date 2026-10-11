"""
03_collect_tembici_gbfs.py
Coleta periódica de snapshots GBFS das estações Tembici (Brasil).

Endpoint: station_status.json (GBFS v3.0)
Cidades disponíveis:
  - Rio de Janeiro: https://riodejaneiro.publicbikesystem.net/customer/gbfs/v3.0/station_status.json
  - São Paulo:      https://saopaulo.publicbikesystem.net/customer/gbfs/v3.0/station_status.json
  - Brasília:       https://brasilia.publicbikesystem.net/customer/gbfs/v3.0/station_status.json
  - Recife:         https://recife.publicbikesystem.net/customer/gbfs/v3.0/station_status.json
  - Salvador:       https://salvador.publicbikesystem.net/customer/gbfs/v3.0/station_status.json

Saída: data/raw/tembici/<cidade>/<YYYY-MM-DD>.parquet (um arquivo por dia)

Uso:
    python scripts/03_collect_tembici_gbfs.py --city rio --interval 900
    # Coleta a cada 15 min (900s) para o Rio de Janeiro

    # Para rodar em background no Linux/macOS:
    nohup python scripts/03_collect_tembici_gbfs.py --city rio &

IMPORTANTE: Iniciar a coleta o quanto antes para acumular histórico suficiente.
Recomendado: pelo menos 2-4 semanas de dados para inferir padrões diários.
"""

import os
import time
import argparse
import requests
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIGURAÇÃO DOS ENDPOINTS
# ──────────────────────────────────────────────
CITIES = {
    "rio":       "https://riodejaneiro.publicbikesystem.net/customer/gbfs/v3.0/",
    "saopaulo":  "https://saopaulo.publicbikesystem.net/customer/gbfs/v3.0/",
    "brasilia":  "https://brasilia.publicbikesystem.net/customer/gbfs/v3.0/",
    "recife":    "https://recife.publicbikesystem.net/customer/gbfs/v3.0/",
    "salvador":  "https://salvador.publicbikesystem.net/customer/gbfs/v3.0/",
}


def fetch_station_status(base_url: str) -> list[dict]:
    """Busca station_status.json e retorna lista de estações."""
    url = base_url + "station_status.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # GBFS v3.0: data.stations | GBFS v2.x: data.stations (mesma estrutura)
    stations = data.get("data", {}).get("stations", [])
    # Adicionar timestamp do servidor
    last_updated = data.get("last_updated", int(time.time()))
    for s in stations:
        s["_snapshot_ts"] = last_updated
        s["_collected_at"] = datetime.utcnow().isoformat()
    return stations


def fetch_station_info(base_url: str) -> pd.DataFrame:
    """Busca station_information.json (nome, lat, lon, capacidade)."""
    url = base_url + "station_information.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    stations = data.get("data", {}).get("stations", [])
    return pd.DataFrame(stations)[["station_id", "name", "lat", "lon", "capacity"]]


def collect_loop(city: str, interval_sec: int, output_dir: str):
    """Loop principal de coleta."""
    base_url = CITIES[city]
    city_dir = os.path.join(output_dir, city)
    os.makedirs(city_dir, exist_ok=True)

    # Salvar station_information uma vez
    info_path = os.path.join(city_dir, "station_information.parquet")
    if not os.path.exists(info_path):
        print(f"Buscando informações das estações de {city}...")
        info = fetch_station_info(base_url)
        info.to_parquet(info_path, index=False)
        print(f"  {len(info)} estações salvas em {info_path}")

    print(f"\nIniciando coleta de {city} a cada {interval_sec}s...")
    print(f"  Pressione Ctrl+C para parar.\n")

    snapshots_today = []
    current_date = datetime.utcnow().date()

    try:
        while True:
            now = datetime.utcnow()
            today = now.date()

            # Novo dia: salvar e resetar buffer
            if today != current_date and snapshots_today:
                day_path = os.path.join(city_dir, f"{current_date}.parquet")
                pd.DataFrame(snapshots_today).to_parquet(day_path, index=False)
                print(f"  Dia {current_date}: {len(snapshots_today)} snapshots salvos em {day_path}")
                snapshots_today = []
                current_date = today

            # Coletar snapshot
            try:
                stations = fetch_station_status(base_url)
                snapshots_today.extend(stations)
                n_bikes = sum(s.get("num_bikes_available", 0) for s in stations)
                print(f"  [{now.strftime('%H:%M:%S')}] {len(stations)} estações | {n_bikes} bikes disponíveis")
            except Exception as e:
                print(f"  [{now.strftime('%H:%M:%S')}] ERRO: {e}")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        # Salvar dados do dia atual antes de sair
        if snapshots_today:
            day_path = os.path.join(city_dir, f"{current_date}_partial.parquet")
            pd.DataFrame(snapshots_today).to_parquet(day_path, index=False)
            print(f"\n  Dados parciais de {current_date} salvos ({len(snapshots_today)} snapshots).")
        print("\nColeta encerrada.")


# ──────────────────────────────────────────────
# ANÁLISE DOS DADOS COLETADOS
# ──────────────────────────────────────────────
def analyze_collected(city: str, data_dir: str):
    """Análise exploratória dos snapshots coletados."""
    city_dir = os.path.join(data_dir, city)
    files = sorted([f for f in os.listdir(city_dir) if f.endswith(".parquet") and f != "station_information.parquet"])

    if not files:
        print(f"Nenhum snapshot encontrado em {city_dir}")
        return

    print(f"\n=== Análise dos dados coletados: {city} ===")
    print(f"Arquivos: {files}")

    dfs = [pd.read_parquet(os.path.join(city_dir, f)) for f in files]
    all_data = pd.concat(dfs, ignore_index=True)

    all_data["_collected_at"] = pd.to_datetime(all_data["_collected_at"])
    all_data["hour"] = all_data["_collected_at"].dt.hour

    print(f"Total de registros: {len(all_data):,}")
    print(f"Período: {all_data['_collected_at'].min()} → {all_data['_collected_at'].max()}")

    # Ocupação média por hora
    occ = all_data.groupby(["station_id", "hour"])["num_bikes_available"].mean()
    print(f"\nOcupação média por hora (top 5 estações mais ativas):")
    top_stations = all_data.groupby("station_id")["num_bikes_available"].std().nlargest(5).index
    for sid in top_stations:
        hourly = occ.loc[sid] if sid in occ.index.get_level_values(0) else None
        if hourly is not None:
            print(f"  {sid}: {hourly.to_dict()}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta GBFS Tembici Brasil")
    parser.add_argument("--city",     default="rio",  choices=list(CITIES.keys()))
    parser.add_argument("--interval", type=int, default=900, help="Intervalo em segundos (padrão: 900 = 15min)")
    parser.add_argument("--output",   default="data/raw/tembici")
    parser.add_argument("--analyze",  action="store_true", help="Analisar dados já coletados")
    args = parser.parse_args()

    if args.analyze:
        analyze_collected(args.city, args.output)
    else:
        collect_loop(args.city, args.interval, args.output)
