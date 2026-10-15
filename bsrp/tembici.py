"""
bsrp/tembici.py — instância e cenários de DIAS REAIS a partir da coleta GBFS da Tembici (DESVIOS.md 2026-10-15, P5).

Só `csv`, `gzip` e `numpy` (sem pandas: a máquina é compartilhada e a memória virtual é apertada). O feed só dá ESTOQUE por
estação a cada ~5 min; chegadas e partidas não são separáveis. Estimativa adotada (e declarada): chegadas = soma dos
incrementos positivos entre coletas consecutivas, partidas = soma dos decrementos, dentro do intervalo de 30 min; saltos
|Δ| >= BIG_JUMP em uma coleta são reposição/retirada operacional e ficam fora da demanda. A demanda é CENSURADA (nada se vê
de partidas não atendidas com a estação vazia), logo a demanda perdida medida é um piso.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .env import DAY_START_H, INTERVAL_MIN, N_INTERVALS, Instance, ScenarioSet

ROOT = Path(__file__).resolve().parents[1]
LOCAL = timezone(timedelta(hours=-3))                       # Brasil sem horário de verão desde 2019
HOLIDAYS = frozenset({"2026-10-12", "2026-11-02", "2026-11-15"})
MAX_GAP_S = 450                                             # 7,5 min: acima disso a coleta é considerada perdida
BIG_JUMP = 5
MIN_COVER = 0.8
OFFSET = DAY_START_H * 60 // INTERVAL_MIN                   # 12: intervalo 0 do episódio = 06:00
PER_DAY = 24 * 60 // INTERVAL_MIN                           # 48
MIN_DAYS_FOR_HOLDOUT = 4                                    # semanas completas exigidas para atribuir val/teste
G3 = {"corr_min": 0.50, "volume_rel_max": 0.25, "unmet_ratio": (0.5, 2.0)}


# ──────────────────────────────────────────────────────────────────────────────
# LEITURA
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Snapshots:
    station_ids: list          # ids únicos, na ordem de `st`
    st: np.ndarray             # (N,) índice da estação
    ts: np.ndarray             # (N,) int64, epoch UTC do feed (feed_ts)
    veh: np.ndarray            # (N,) int16, veículos disponíveis
    active: np.ndarray         # (N,) bool, is_renting e is_returning


def _open_rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def read_snapshots(city: str, root: Path = ROOT, until_ts: int | None = None) -> Snapshots:
    """Concatena data/raw/tembici/<cidade>/*.csv.gz e data/live/<cidade>/*.csv, deduplica por (estação, feed_ts)."""
    root = Path(root)
    files = sorted(glob.glob(str(root / "data" / "raw" / "tembici" / city / "*.csv.gz"))) + \
        sorted(glob.glob(str(root / "data" / "live" / city / "*.csv")))
    idx, st, ts, veh, act = {}, [], [], [], []
    for p in files:
        for r in _open_rows(p):
            try:
                t = int(r["feed_ts"] or r["t"])
                v = int(r["veh"])
            except (ValueError, KeyError):
                continue
            if until_ts is not None and t >= until_ts:
                continue
            sid = r["station_id"]
            k = idx.setdefault(sid, len(idx))
            st.append(k); ts.append(t); veh.append(v)
            act.append(r.get("rent", "1") in ("1", "True", "true") and r.get("ret", "1") in ("1", "True", "true"))
    st, ts = np.array(st, np.int64), np.array(ts, np.int64)
    order = np.lexsort((ts, st))
    st, ts, veh, act = st[order], ts[order], np.array(veh, np.int16)[order], np.array(act, bool)[order]
    keep = np.ones(len(st), bool)
    keep[1:] = (st[1:] != st[:-1]) | (ts[1:] != ts[:-1])                       # dedupe por (estação, feed_ts)
    return Snapshots(list(idx), st[keep], ts[keep], veh[keep], act[keep])


def read_station_info(city: str, root: Path = ROOT) -> dict:
    """Cadastro mais recente: {station_id: (nome, lat, lon, capacidade)}."""
    files = sorted(glob.glob(str(Path(root) / "data" / "raw" / "tembici" / "station_information" / city / "*.json.gz")))
    if not files:
        return {}
    with gzip.open(files[-1], "rt", encoding="utf-8") as f:
        stations = json.load(f)["data"]["stations"]
    out = {}
    for s in stations:
        nm = s.get("name")
        if isinstance(nm, list):
            pt = [x["text"] for x in nm if x.get("language") == "pt"]
            nm = pt[0] if pt else (nm[0]["text"] if nm else "")
        out[str(s["station_id"])] = (str(nm), float(s["lat"]), float(s["lon"]), int(s.get("capacity") or 0))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# FLUXO POR ESTAÇÃO × DIA LOCAL × MEIA HORA
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CityFlows:
    dates: list                # ["AAAA-MM-DD", ...] dias locais consecutivos
    station_ids: list
    arr: np.ndarray            # (n_st, n_days, 48) chegadas observadas (incrementos positivos)
    dep: np.ndarray            # (n_st, n_days, 48) partidas observadas
    cover: np.ndarray          # (n_st, n_days, 48) fração de 30 min com par de coletas válido
    ops: np.ndarray            # (n_st, n_days, 48) nº de saltos operacionais excluídos
    ops_vol: np.ndarray        # (n_st, n_days, 48) soma de |Δ| dos saltos excluídos (volume que NÃO entrou na demanda)
    stock06: np.ndarray        # (n_st, n_days) estoque observado às 06:00 (NaN se não há coleta a ±5 min)
    polls: np.ndarray          # (n_days, 48) nº de coletas distintas (nível cidade)
    active_frac: np.ndarray    # (n_st,) fração das linhas em 06–22h com is_renting e is_returning


def day_type(d: str) -> int:
    """0 = dia útil; 1 = fim de semana ou feriado."""
    return 1 if (date.fromisoformat(d).weekday() >= 5 or d in HOLIDAYS) else 0


def compute_flows(snap: Snapshots, big_jump: int = BIG_JUMP) -> CityFlows:
    off = 3 * 3600
    local = snap.ts - off
    day = local // 86400
    d0, d1 = int(day.min()), int(day.max())
    n_days, n_st = d1 - d0 + 1, len(snap.station_ids)
    sec = local % 86400
    interval = sec // (INTERVAL_MIN * 60)
    arr = np.zeros((n_st, n_days, PER_DAY), np.float32)
    dep = np.zeros_like(arr)
    cov = np.zeros_like(arr)
    ops = np.zeros((n_st, n_days, PER_DAY), np.int16)
    ops_vol = np.zeros_like(arr)
    same = snap.st[1:] == snap.st[:-1]
    dt = snap.ts[1:] - snap.ts[:-1]
    ok = same & (dt > 0) & (dt <= MAX_GAP_S)
    d = snap.veh[1:].astype(np.int32) - snap.veh[:-1].astype(np.int32)
    big = ok & (np.abs(d) >= big_jump)
    use = ok & ~big
    s_i, day_i, k_i = snap.st[1:], day[1:] - d0, interval[1:]
    np.add.at(arr, (s_i[use], day_i[use], k_i[use]), np.where(d[use] > 0, d[use], 0))
    np.add.at(dep, (s_i[use], day_i[use], k_i[use]), np.where(d[use] < 0, -d[use], 0))
    np.add.at(cov, (s_i[ok], day_i[ok], k_i[ok]), dt[ok] / (INTERVAL_MIN * 60.0))
    np.add.at(ops, (s_i[big], day_i[big], k_i[big]), 1)
    np.add.at(ops_vol, (s_i[big], day_i[big], k_i[big]), np.abs(d[big]).astype(np.float32))
    # estoque às 06:00 local: coleta mais próxima dentro de ±5 min
    stock06 = np.full((n_st, n_days), np.nan, np.float32)
    best = np.full((n_st, n_days), 10 ** 9, np.int64)
    near = np.abs(sec - DAY_START_H * 3600) <= 300
    for j in np.flatnonzero(near):
        a, b, dist = snap.st[j], int(day[j] - d0), abs(int(sec[j]) - DAY_START_H * 3600)
        if dist < best[a, b]:
            best[a, b], stock06[a, b] = dist, snap.veh[j]
    # nº de coletas distintas por (dia, meia hora)
    uts = np.unique(snap.ts)
    ul = uts - off
    polls = np.zeros((n_days, PER_DAY), np.int32)
    np.add.at(polls, ((ul // 86400) - d0, (ul % 86400) // (INTERVAL_MIN * 60)), 1)
    # fração do tempo ativo (06–22h)
    op_hours = (sec >= DAY_START_H * 3600) & (sec < (DAY_START_H + 16) * 3600)
    tot = np.bincount(snap.st[op_hours], minlength=n_st).astype(float)
    act = np.bincount(snap.st[op_hours & snap.active], minlength=n_st).astype(float)
    active_frac = np.divide(act, tot, out=np.zeros(n_st), where=tot > 0)
    dates = [(date(1970, 1, 1) + timedelta(days=d0 + i)).isoformat() for i in range(n_days)]
    return CityFlows(dates, snap.station_ids, arr, dep, np.minimum(cov, 1.0), ops, ops_vol, stock06, polls, active_frac)


def usable_days(cf: CityFlows, min_intervals: int = 30, min_polls: int = 5) -> np.ndarray:
    """(n_days,) bool: ≥ 30 dos 32 intervalos operacionais com ≥ 5 coletas distintas."""
    ok = cf.polls[:, OFFSET:OFFSET + N_INTERVALS] >= min_polls
    return ok.sum(axis=1) >= min_intervals


# ──────────────────────────────────────────────────────────────────────────────
# DIVISÃO POR SEMANAS (DESVIOS.md 2026-10-15)
# ──────────────────────────────────────────────────────────────────────────────
def assign_split(dates: list, usable: np.ndarray, min_weeks: int = MIN_DAYS_FOR_HOLDOUT) -> np.ndarray:
    """'dev' | 'train' | 'val' | 'test' | 'unused' por dia.

    Semana completa = seg–dom com os 7 dias dentro da janela e usáveis. Com < `min_weeks` semanas completas NADA é
    rotulado val/teste (o hold-out fica intocado): os dias usáveis são 'dev'. Com ≥ `min_weeks`: teste = última semana
    completa, val = a anterior, treino = os dias usáveis anteriores à semana de val; dias posteriores ao teste = 'unused'."""
    ds = [date.fromisoformat(d) for d in dates]
    usable = np.asarray(usable, bool)
    pos = {d: i for i, d in enumerate(ds)}
    weeks = []
    for d in ds:
        if d.weekday() == 0:
            days = [d + timedelta(days=k) for k in range(7)]
            if all(x in pos and usable[pos[x]] for x in days):
                weeks.append(d)
    out = np.full(len(ds), "unused", dtype=object)
    if len(weeks) < min_weeks:
        out[usable] = "dev"
        return out
    test_w, val_w = weeks[-1], weeks[-2]
    for i, d in enumerate(ds):
        if not usable[i]:
            continue
        if test_w <= d < test_w + timedelta(days=7):
            out[i] = "test"
        elif val_w <= d < val_w + timedelta(days=7):
            out[i] = "val"
        elif d < val_w:
            out[i] = "train"
    return out


# ──────────────────────────────────────────────────────────────────────────────
# INSTÂNCIA
# ──────────────────────────────────────────────────────────────────────────────
def _valid(cf: CityFlows) -> np.ndarray:
    return cf.cover >= MIN_COVER


def station_volume(cf: CityFlows, days: np.ndarray) -> tuple:
    """(volume médio bruto 06–22h por dia, fração média de células válidas) por estação, sobre os dias dados."""
    sl = slice(OFFSET, OFFSET + N_INTERVALS)
    v = _valid(cf)[:, days, sl]
    gross = (cf.arr[:, days, sl] + cf.dep[:, days, sl])
    frac = v.mean(axis=2)                                       # (n_st, n_days)
    per_day = np.where(v, gross, 0.0).sum(axis=2) / np.maximum(frac, 1e-9)
    good = frac >= MIN_COVER
    vol = np.where(good, per_day, np.nan)
    with np.errstate(all="ignore"):
        volume = np.nanmean(vol, axis=1)
    return np.nan_to_num(volume, nan=0.0), frac.mean(axis=1)


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def select_stations(cf: CityFlows, info: dict, train_weekdays: np.ndarray, n: int = 12, radius_km: float = 3.0,
                    min_active: float = 0.5) -> list:
    """Índices (em cf.station_ids) das n estações de maior volume nos dias úteis de treino, dentro de `radius_km` da de maior volume."""
    volume, frac = station_volume(cf, train_weekdays)
    elig = np.array([(frac[i] >= MIN_COVER) and (cf.active_frac[i] >= min_active) and (sid in info and info[sid][3] > 0)
                     for i, sid in enumerate(cf.station_ids)])
    if not elig.any():
        return []
    lat = np.array([info[s][1] if s in info else np.nan for s in cf.station_ids])
    lon = np.array([info[s][2] if s in info else np.nan for s in cf.station_ids])
    cand = np.flatnonzero(elig)
    seed = cand[np.argmax(volume[cand])]
    near = [i for i in cand if haversine_km(lat[seed], lon[seed], lat[i], lon[i]) <= radius_km]
    near.sort(key=lambda i: (-volume[i], i))
    return near[:n]


def kmeans_labels(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    """k-means determinístico (inicialização por ponto mais distante), em km locais; rótulos 0..k-1 estáveis."""
    x = np.c_[lat * 111.0, lon * 111.0 * np.cos(np.radians(lat.mean()))]
    k = min(k, len(x))
    cent = [x[0]]
    while len(cent) < k:
        d = np.min([((x - c) ** 2).sum(1) for c in cent], axis=0)
        cent.append(x[int(np.argmax(d))])
    cent = np.array(cent)
    for _ in range(50):
        lab = np.argmin(((x[:, None] - cent[None]) ** 2).sum(2), axis=1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    order = np.argsort([np.arctan2(*(c - x.mean(0))[::-1]) for c in cent])      # rótulos estáveis por ângulo
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[lab].astype(np.int64)


def build_instance(cf: CityFlows, info: dict, idx: list, train_weekdays: np.ndarray, name: str, n_clusters: int = 4) -> Instance:
    sl = slice(OFFSET, OFFSET + N_INTERVALS)
    v = _valid(cf)[idx][:, train_weekdays, sl]

    def lam(a):
        x = np.where(v, a[idx][:, train_weekdays, sl], np.nan)
        with np.errstate(all="ignore"):
            m = np.nanmean(x, axis=1)                                   # (n, 32)
        return np.nan_to_num(m, nan=0.0).T.astype(float)                # (32, n)

    ids = [cf.station_ids[i] for i in idx]
    cap = np.array([info[s][3] for s in ids], np.int64)
    lat = np.array([info[s][1] for s in ids])
    lon = np.array([info[s][2] for s in ids])
    return Instance(name, cap, kmeans_labels(lat, lon, n_clusters), lat, lon, lam(cf.dep), lam(cf.arr), tuple(info[s][0] for s in ids))


def real_scenarios(cf: CityFlows, inst: Instance, idx: list, days: np.ndarray, max_invalid: float = 0.03) -> ScenarioSet:
    """Um cenário por dia real usável: dep/arr observados (06–22h), ocupação inicial = estoque observado às 06h. Dia excluído
    se > max_invalid das células inválidas ou sem estoque às 06h; células inválidas restantes = λ arredondado."""
    sl = slice(OFFSET, OFFSET + N_INTERVALS)
    init, dep, arr, kept, dropped = [], [], [], [], []
    for d in days:
        v = _valid(cf)[idx][:, d, sl]                                   # (n, 32)
        s0 = cf.stock06[idx, d]
        if (1.0 - v.mean()) > max_invalid or np.isnan(s0).any():
            dropped.append(cf.dates[d])
            continue
        dd = np.where(v, cf.dep[idx][:, d, sl], np.rint(inst.lam_dep.T))
        aa = np.where(v, cf.arr[idx][:, d, sl], np.rint(inst.lam_arr.T))
        init.append(np.clip(s0, 0, inst.cap).astype(np.int64))
        dep.append(dd.T.astype(np.int16)); arr.append(aa.T.astype(np.int16)); kept.append(cf.dates[d])
    n = inst.n
    return ScenarioSet(np.array(init, np.int64).reshape(-1, n), np.array(dep, np.int16).reshape(-1, N_INTERVALS, n),
                       np.array(arr, np.int16).reshape(-1, N_INTERVALS, n), meta={"days": kept, "dropped": dropped})


# ──────────────────────────────────────────────────────────────────────────────
# PORTÃO G3
# ──────────────────────────────────────────────────────────────────────────────
def g3_metrics(inst: Instance, real: ScenarioSet, n_model: int = 200, seed: int = 0) -> dict:
    """(a) correlação do fluxo líquido modelo × observado por (intervalo, estação); (b) erro relativo do volume bruto diário;
    (c) razão entre a demanda perdida da política 'parada' no replay e no modelo de Poisson. Tolerâncias em G3."""
    from .env import BSRPEnv, evaluate, sample_scenarios
    from .policies import Stay

    net_obs = (real.arr.astype(float) - real.dep.astype(float)).mean(axis=0)               # (32, n)
    net_mod = inst.lam_arr - inst.lam_dep
    corr = float(np.corrcoef(net_obs.ravel(), net_mod.ravel())[0, 1]) if net_obs.std() > 0 and net_mod.std() > 0 else float("nan")
    vol_obs = float((real.arr.astype(float) + real.dep.astype(float)).sum(axis=(1, 2)).mean())
    vol_mod = float((inst.lam_arr + inst.lam_dep).sum())
    rel = abs(vol_mod - vol_obs) / vol_obs if vol_obs > 0 else float("nan")
    env = BSRPEnv(inst)
    unmet_real = float(-evaluate(env, Stay(), real).mean())
    unmet_mod = float(-evaluate(env, Stay(), sample_scenarios(inst, n_model, np.random.default_rng(seed))).mean())
    ratio = unmet_real / unmet_mod if unmet_mod > 0 else float("nan")
    lo, hi = G3["unmet_ratio"]
    ok = {"corr": corr >= G3["corr_min"], "volume": rel <= G3["volume_rel_max"], "unmet_ratio": lo <= ratio <= hi}
    return {"corr_net": corr, "volume_obs": vol_obs, "volume_model": vol_mod, "volume_rel_err": rel,
            "unmet_stay_real": unmet_real, "unmet_stay_model": unmet_mod, "unmet_ratio": ratio, "n_days": len(real),
            "pass": ok, "G3_passed": bool(all(ok.values()))}
