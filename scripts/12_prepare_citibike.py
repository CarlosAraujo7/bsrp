"""
12_prepare_citibike.py — fluxos de 30 min e matrizes O-D do Citi Bike (Manhattan), em streaming.

Lê os zips mensais (S3 tripdata) sem descompactar em disco e sem pandas. Saída em data/processed/citibike_2024/:
  flows_od.npz   dep[n_st, n_dias, 48], arr[n_st, n_dias, 48]  (int16)  — partidas/chegadas por 30 min
                 od_cnt[2, 6, n_st, n_st] (int32), od_dur_min[...] (float32) — viagens Manhattan->Manhattan por
                 (tipo de dia, bloco de 4 h da partida, origem, destino), SOMENTE semanas de TREINO
                 short_name[n_st], dates[n_dias], split[n_dias]
  stations.csv   estações de Manhattan (GBFS) com volume total
  qc.json        contagens de controle de qualidade e sha256 dos zips

Divisão treino/val/teste (fixada ANTES de olhar os dados; PROTOCOLO_RIGOR.md §8): semanas inteiras seg-dom,
índice de semana w = (dia - 2024-01-01)//7; w%5 in {0,1,2} -> treino, w%5 == 3 -> val, w%5 == 4 -> teste.
Tipo de dia: fim de semana OU feriado federal dos EUA (1/1, 15/1, 19/2, 27/5, 19/6) = 1; dia útil = 0.
Limpeza: viagens com mesma estação de origem e destino e duração < 1 min (resolução de minuto; falsa
partida) são descartadas; durações negativas ou > 24 h também (contadas no qc.json).

Uso: python scripts/12_prepare_citibike.py --zips-dir <pasta com os zips> [--limit-rows N]
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START = date(2024, 1, 1)
N_DAYS = 182                                  # 2024-01-01 .. 2024-06-30 (26 semanas)
HOLIDAYS = {date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 5, 27), date(2024, 6, 19)}
BLOCK_H = 4                                   # blocos de 4 h da partida (0-3, 4-7, ... 20-23)


def split_of(day_idx: int) -> str:
    w = (day_idx // 7) % 5
    return "train" if w in (0, 1, 2) else ("val" if w == 3 else "test")


def day_type(d: date) -> int:
    return 1 if (d.weekday() >= 5 or d in HOLIDAYS) else 0


def csv_members(zf: zipfile.ZipFile):
    for info in zf.infolist():
        n = info.filename
        if n.lower().endswith(".csv") and not n.startswith("__MACOSX") and "/." not in n:
            yield info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "citibike_2024")
    ap.add_argument("--limit-rows", type=int, default=0, help="só para teste: para após N linhas")
    ap.add_argument("--glob", default="*.zip", help="subconjunto de zips (ex.: 202401*.zip)")
    a = ap.parse_args()

    mh = json.load(open(ROOT / "manhattan_stations.json", encoding="utf-8"))
    mh = [s for s in mh if s.get("short_name")]
    short = [s["short_name"] for s in mh]
    idx = {s: i for i, s in enumerate(short)}
    n = len(short)
    dep = np.zeros((n, N_DAYS, 48), np.int16)
    arr = np.zeros((n, N_DAYS, 48), np.int16)
    od_cnt = np.zeros((2, 24 // BLOCK_H, n, n), np.int32)
    od_dur = np.zeros((2, 24 // BLOCK_H, n, n), np.float32)

    ord0 = START.toordinal()
    dcache = {}

    def day_of(s10):  # 'YYYY-MM-DD' -> índice de dia ou -1
        v = dcache.get(s10)
        if v is None:
            try:
                v = date.fromisoformat(s10).toordinal() - ord0
            except ValueError:
                v = -999999
            dcache[s10] = v
        return v

    day_split = [split_of(i) for i in range(N_DAYS)]
    day_ty = [day_type(START + timedelta(days=i)) for i in range(N_DAYS)]
    qc = {"zips": {}, "rows": 0, "no_start_id": 0, "no_end_id": 0, "bad_time": 0, "neg_or_long_dur": 0,
          "false_start_dropped": 0, "outside_period": 0, "kept_dep": 0, "kept_arr": 0, "kept_od_train": 0,
          "per_month_rows": {}}
    t0 = time.time()
    stop = False
    for zp in sorted(a.zips_dir.glob(a.glob)):
        qc["zips"][zp.name] = {"bytes": zp.stat().st_size, "sha256": sha256(zp)}
        with zipfile.ZipFile(zp) as zf:
            bad = zf.testzip()
            if bad:
                sys.exit(f"zip corrompido: {zp.name} ({bad})")
            for info in csv_members(zf):
                print(f"{time.strftime('%H:%M:%S')} {zp.name}:{info.filename} ({info.file_size/1e6:.0f} MB)", flush=True)
                with zf.open(info) as raw:
                    rd = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    h = next(rd)
                    c = {k: h.index(k) for k in ("started_at", "ended_at", "start_station_id", "end_station_id")}
                    i_s, i_e, i_so, i_eo = c["started_at"], c["ended_at"], c["start_station_id"], c["end_station_id"]
                    for r in rd:
                        qc["rows"] += 1
                        if a.limit_rows and qc["rows"] >= a.limit_rows:
                            stop = True
                            break
                        s, e, so, eo = r[i_s], r[i_e], r[i_so], r[i_eo]
                        if not so:
                            qc["no_start_id"] += 1
                        if not eo:
                            qc["no_end_id"] += 1
                        i = idx.get(so)
                        j = idx.get(eo)
                        if i is None and j is None:
                            continue
                        try:
                            ds = day_of(s[:10]); de = day_of(e[:10])
                            ms = int(s[11:13]) * 60 + int(s[14:16]); me = int(e[11:13]) * 60 + int(e[14:16])
                        except ValueError:
                            qc["bad_time"] += 1
                            continue
                        dur = (de - ds) * 1440 + (me - ms)  # minutos (resolução de minuto)
                        if dur < 0 or dur > 1440:
                            qc["neg_or_long_dur"] += 1
                            continue
                        if so == eo and dur < 1:
                            qc["false_start_dropped"] += 1
                            continue
                        mo = s[:7]
                        qc["per_month_rows"][mo] = qc["per_month_rows"].get(mo, 0) + 1
                        if i is not None:
                            if 0 <= ds < N_DAYS:
                                dep[i, ds, ms // 30] += 1
                                qc["kept_dep"] += 1
                            else:
                                qc["outside_period"] += 1
                        if j is not None:
                            if 0 <= de < N_DAYS:
                                arr[j, de, me // 30] += 1
                                qc["kept_arr"] += 1
                        if i is not None and j is not None and 0 <= ds < N_DAYS and day_split[ds] == "train":
                            b = (ms // 60) // BLOCK_H
                            od_cnt[day_ty[ds], b, i, j] += 1
                            od_dur[day_ty[ds], b, i, j] += dur
                            qc["kept_od_train"] += 1
                print(f"   linhas até aqui: {qc['rows']:,} | {time.time()-t0:.0f}s", flush=True)
        if stop:
            break

    a.out.mkdir(parents=True, exist_ok=True)
    dates = np.array([(START + timedelta(days=i)).isoformat() for i in range(N_DAYS)])
    np.savez_compressed(a.out / "flows_od.npz", dep=dep, arr=arr, od_cnt=od_cnt, od_dur_min=od_dur,
                        short_name=np.array(short), dates=dates, split=np.array(day_split),
                        day_type=np.array(day_ty, np.int8))
    tot_dep = dep.sum(axis=(1, 2)); tot_arr = arr.sum(axis=(1, 2))
    with open(a.out / "stations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["short_name", "name", "lat", "lon", "capacity_gbfs_2026", "total_dep", "total_arr"])
        for k, s in enumerate(mh):
            w.writerow([s["short_name"], s["name"], s["lat"], s["lon"], s.get("capacity", ""), int(tot_dep[k]), int(tot_arr[k])])
    qc["elapsed_s"] = round(time.time() - t0)
    qc["split_days"] = {k: day_split.count(k) for k in ("train", "val", "test")}
    json.dump(qc, open(a.out / "qc.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in qc.items() if k not in ("zips",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
