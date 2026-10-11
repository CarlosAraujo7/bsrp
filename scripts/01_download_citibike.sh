#!/bin/bash
# Script de download dos dados históricos do NYC Citi Bike
# Fonte: https://s3.amazonaws.com/tripdata/
# Formato atual: dados 2023 em um único zip anual; 2024+ em arquivos mensais

set -e
mkdir -p data/raw/citibike

echo "=== Download NYC Citi Bike Trip Data ==="

# OPÇÃO 1: Dados 2023 (arquivo anual, ~1.6GB)
echo "Baixando 2023 (arquivo anual)..."
wget -c "https://s3.amazonaws.com/tripdata/2023-citibike-tripdata.zip" \
     -O data/raw/citibike/2023-citibike-tripdata.zip

# OPÇÃO 2: Dados 2024 mensais (para prototipagem — Jan a Mar 2024, ~1.1GB total)
# Descomente as linhas abaixo se preferir meses individuais:
# for month in 01 02 03; do
#   echo "Baixando 2024-${month}..."
#   wget -c "https://s3.amazonaws.com/tripdata/2024${month}-citibike-tripdata.zip" \
#        -O "data/raw/citibike/2024${month}-citibike-tripdata.zip"
# done

echo "=== Download concluído ==="
ls -lh data/raw/citibike/
