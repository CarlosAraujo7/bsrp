"""
bsrp/data.py — instâncias calibradas e cenários de DIAS REAIS a partir de data/processed/citibike_2024/flows_od.npz
(gerado por scripts/12_prepare_citibike.py). Protocolo §8: perfis por tipo de dia, calibração só nas semanas de
TREINO, teste = replay de fluxos reais das semanas de TESTE.

Sem DEMAND_SCALE: λ = média observada de partidas/chegadas por intervalo de 30 min. As chegadas NÃO são
reescaladas para igualar as partidas: o sub-sistema de estações é aberto e o desequilíbrio líquido real é parte
do problema.

Escalas 4, 8 e 12 usam as estações legadas (CSVs `station_selected_<n>.csv`). Escalas MAIORES (P9, DESVIOS.md 2026-10-19) usam as 12 legadas
mais as próximas de maior volume (`extended_base`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .env import DAY_START_H, INTERVAL_MIN, N_INTERVALS, Instance, ScenarioSet, legacy_instance

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ROOT / "data" / "processed" / "citibike_2024" / "flows_od.npz"
OFFSET = DAY_START_H * 60 // INTERVAL_MIN        # 12: intervalo 0 do episódio = 06:00
LEGACY_SIZES = (4, 8, 12)


class Flows:
    def __init__(self, path=FLOWS):
        z = np.load(path, allow_pickle=False)
        self.dep, self.arr = z["dep"], z["arr"]                  # (n_st, n_dias, 48)
        self.short = [str(s) for s in z["short_name"]]
        self.dates = [str(d) for d in z["dates"]]
        self.split = np.array([str(s) for s in z["split"]])
        self.day_type = z["day_type"].astype(int)
        self.od_cnt, self.od_dur = z["od_cnt"], z["od_dur_min"]
        self.idx = {s: i for i, s in enumerate(self.short)}
        mh = json.load(open(ROOT / "manhattan_stations.json", encoding="utf-8"))
        self.name2short = {s["name"]: s["short_name"] for s in mh if s.get("short_name")}

    def rows_for(self, names):
        miss = [n for n in names if n not in self.name2short or self.name2short[n] not in self.idx]
        if miss:
            raise KeyError(f"estações sem correspondência no Citi Bike 2024: {miss}")
        return [self.idx[self.name2short[n]] for n in names]

    def days(self, split, day_type=None):
        m = self.split == split
        if day_type is not None:
            m &= self.day_type == day_type
        return np.flatnonzero(m)


def extended_base(n_stations: int, flows: Flows) -> Instance:
    """n > 12: as 12 estações legadas + as (n − 12) de MAIOR VOLUME (partidas + chegadas nos dias úteis de TREINO) ainda fora do conjunto.
    Ordem estável (volume decrescente, empate pela linha do arquivo); capacidade, lat/lon de manhattan_stations.json (idênticos aos legados nas 12);
    sem agrupamento (`cluster` = 0: só a representação tabular por cluster o usa). λ não vem daqui (ver calibrated_instance)."""
    base12 = legacy_instance(12)
    mh = {s["short_name"]: s for s in json.load(open(ROOT / "manhattan_stations.json", encoding="utf-8")) if s.get("short_name")}
    days = flows.days("train", 0)
    vol = flows.dep[:, days, :].sum(axis=(1, 2)) + flows.arr[:, days, :].sum(axis=(1, 2))
    taken = set(flows.rows_for(list(base12.station_names)))
    names, cap, lat, lon = list(base12.station_names), list(base12.cap), list(base12.lat), list(base12.lon)
    for r in np.argsort(-vol, kind="stable"):
        if len(names) >= n_stations:
            break
        s = mh.get(flows.short[r])
        if int(r) in taken or s is None or int(s["capacity"]) < 20 or flows.name2short.get(s["name"]) != flows.short[r] or s["name"] in names:
            continue
        names.append(s["name"]); cap.append(int(s["capacity"])); lat.append(float(s["lat"])); lon.append(float(s["lon"]))
    if len(names) < n_stations:
        raise ValueError(f"só {len(names)} estações elegíveis")
    n = len(names)
    z = np.zeros((N_INTERVALS, n))
    return Instance(f"ext{n_stations}", np.array(cap, np.int64), np.zeros(n, np.int64), np.array(lat), np.array(lon), z, z.copy(), tuple(names))


def _base(n_stations: int, flows: Flows) -> Instance:
    return legacy_instance(n_stations) if n_stations in LEGACY_SIZES else extended_base(n_stations, flows)


def calibrated_instance(n_stations: int, flows: Flows, day_type: int, name: str | None = None,
                        demand_scale: float = 1.0) -> Instance:
    """Estações/clusters/capacidades: legados (n_stations ∈ {4,8,12}) ou `extended_base` (n_stations > 12); λ das semanas de TREINO."""
    base = _base(n_stations, flows)
    rows = flows.rows_for(list(base.station_names))
    days = flows.days("train", day_type)
    sl = slice(OFFSET, OFFSET + N_INTERVALS)
    lam_dep = flows.dep[rows][:, days, sl].mean(axis=1).T.astype(float) * demand_scale     # (32, n)
    lam_arr = flows.arr[rows][:, days, sl].mean(axis=1).T.astype(float) * demand_scale
    return Instance(name or f"cb{n_stations}_{'we' if day_type else 'wd'}", base.cap, base.cluster, base.lat, base.lon,
                    lam_dep, lam_arr, base.station_names)


def real_scenarios(n_stations: int, flows: Flows, split: str, day_type: int, jitter: int = 2,
                   seed: int = 0) -> ScenarioSet:
    """Um cenário por DIA REAL do split/tipo de dia: demanda = contagens observadas (06h–22h); ocupação inicial
    desconhecida nos dados => 50% da capacidade ± jitter (sorteio fixado por `seed`), como nos cenários sintéticos."""
    base = _base(n_stations, flows)
    rows = flows.rows_for(list(base.station_names))
    days = flows.days(split, day_type)
    sl = slice(OFFSET, OFFSET + N_INTERVALS)
    dep = np.stack([flows.dep[rows][:, d, sl].T for d in days]).astype(np.int16)           # (E, 32, n)
    arr = np.stack([flows.arr[rows][:, d, sl].T for d in days]).astype(np.int16)
    rng = np.random.default_rng(seed)
    init = np.clip(base.cap[None] * 0.5 + rng.integers(-jitter, jitter + 1, size=(len(days), n_stations)), 0, base.cap[None]).astype(np.int64)
    return ScenarioSet(init, dep, arr, meta={"days": [flows.dates[d] for d in days], "split": split, "day_type": day_type})
