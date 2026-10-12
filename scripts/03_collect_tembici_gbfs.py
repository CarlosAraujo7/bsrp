"""
03_collect_tembici_gbfs.py  (v2 — reescrito em 2026-10-11 para GBFS v3.0)

Coleta periódica de station_status (GBFS v3.0) das estações Tembici (Bike Itaú).
Endpoints verificados em 2026-10-11: https://<cidade>.publicbikesystem.net/customer/gbfs/v3.0/
(riodejaneiro, saopaulo, brasilia, recife, salvador). Plataforma Lyft; sem termos de
uso/licença no feed -> NÃO redistribuir os dados brutos (ver data/README.md).

O que o feed dá: ESTOQUE por estação (veículos e docas), não viagens. Fluxo líquido por
intervalo = diferença de estoque entre coletas (chegadas e partidas não são separáveis, e
reposição por caminhão aparece como saltos grandes). Isso vale para o desenho do simulador.

Saída (tudo em UTC, tempos em epoch segundos):
  <live-dir>/<cidade>/<YYYY-MM-DD>.csv     arquivo do dia, append a cada coleta (à prova de crash)
  <live-dir>/_log/<YYYY-MM-DD>.csv         uma linha por (coleta, cidade): ok/erro, nº estações, ms
  <archive-dir>/<cidade>/<YYYY-MM-DD>.csv.gz   dias encerrados (gzip, verificados)
  <archive-dir>/station_information/<cidade>/<ts>.json.gz   só quando o cadastro muda
O live-dir padrão fica FORA do OneDrive (evita re-sincronizar um arquivo que cresce a cada 5 min).

Uso:
    python scripts/03_collect_tembici_gbfs.py                 # laço, 300 s, todas as cidades
    python scripts/03_collect_tembici_gbfs.py --once          # uma coleta (p/ Agendador de Tarefas)
    python scripts/03_collect_tembici_gbfs.py --cities rio,saopaulo --interval 300
    python scripts/03_collect_tembici_gbfs.py --analyze       # resumo de qualidade dos dados
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
CITIES = {  # chave curta -> subdomínio
    "rio": "riodejaneiro",
    "saopaulo": "saopaulo",
    "brasilia": "brasilia",
    "recife": "recife",
    "salvador": "salvador",
}
URL = "https://{sub}.publicbikesystem.net/customer/gbfs/v3.0/{feed}.json"
UA = "BSRP-research-collector/2.0 (academic research; 1 request/city/5min)"
TIMEOUT = 15
FIELDS = ["t", "feed_ts", "station_id", "veh", "veh_dis", "docks", "docks_dis",
          "inst", "rent", "ret", "last_rep", "types"]
LOG_FIELDS = ["t", "city", "status", "n_stations", "ms", "feed_ts", "note"]
INFO_EVERY_S = 6 * 3600


def default_live_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(root) / "bsrp_tembici" / "live"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_epoch(s):
    """RFC3339 ('...Z') -> epoch s (int); vazio se ausente/ilegível. 1970 aparece em estações sem relato."""
    if not s:
        return ""
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# REDE
# ──────────────────────────────────────────────────────────────────────────────
def new_session():
    """Sessão NOVA a cada rodada: reutilizar o keep-alive entre rodadas de 5 min pega conexão
    morta e cada 1ª tentativa trava até o timeout (observado: +16 s por cidade em 2026-10-12)."""
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def get_json(session, sub, feed, retries=2):
    """GET com 2 retentativas (2 s, 5 s). Retorna (json, ms_total, tentativas)."""
    url = URL.format(sub=sub, feed=feed)
    last = None
    t0 = time.time()
    for k, wait in enumerate([0] + [2, 5][:retries], start=1):
        if wait:
            time.sleep(wait)
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json(), int((time.time() - t0) * 1000), k
        except (requests.RequestException, ValueError) as e:
            last = e
    raise last


def status_rows(data, t_epoch):
    """station_status v3.0 -> linhas CSV compactas."""
    feed_ts = to_epoch(data.get("last_updated"))
    rows = []
    for s in data["data"]["stations"]:
        types = "|".join(f'{v["vehicle_type_id"]}:{v["count"]}'
                         for v in s.get("vehicle_types_available", []) if v.get("count"))
        rows.append([
            t_epoch, feed_ts, s["station_id"],
            s.get("num_vehicles_available", ""), s.get("num_vehicles_disabled", ""),
            s.get("num_docks_available", ""), s.get("num_docks_disabled", ""),
            int(bool(s.get("is_installed"))), int(bool(s.get("is_renting"))),
            int(bool(s.get("is_returning"))), to_epoch(s.get("last_reported")), types,
        ])
    return rows, feed_ts


# ──────────────────────────────────────────────────────────────────────────────
# ARMAZENAMENTO
# ──────────────────────────────────────────────────────────────────────────────
def append_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def archive_finished_days(live_dir: Path, archive_dir: Path, today: str):
    """Comprime em gzip os CSVs de dias < hoje, confere nº de linhas e remove o original."""
    if not live_dir.exists():
        return
    for d in sorted(p for p in live_dir.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.csv")):
            if f.stem >= today:
                continue
            out = archive_dir / d.name / (f.stem + ".csv.gz")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "rb") as src, gzip.open(out, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            with open(f, "rb") as a, gzip.open(out, "rb") as b:
                if sum(1 for _ in a) != sum(1 for _ in b):
                    print(f"  ! verificação falhou para {f}; original mantido", file=sys.stderr)
                    out.unlink(missing_ok=True)
                    continue
            f.unlink()
            print(f"  arquivado {out.relative_to(archive_dir)}")


def save_info_if_changed(session, city, sub, archive_dir: Path, live_dir: Path):
    """Guarda station_information bruto (gz) quando o conteúdo de `data` muda."""
    data, _, _ = get_json(session, sub, "station_information")
    h = hashlib.sha1(json.dumps(data["data"], sort_keys=True).encode()).hexdigest()
    hf = live_dir / city / "_info_hash.txt"
    if hf.exists() and hf.read_text().strip() == h:
        return False
    out = archive_dir / "station_information" / city / (now_utc().strftime("%Y%m%dT%H%M%SZ") + ".json.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    hf.parent.mkdir(parents=True, exist_ok=True)
    hf.write_text(h)
    print(f"  cadastro de {city}: {len(data['data']['stations'])} estações -> {out.name}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# COLETA
# ──────────────────────────────────────────────────────────────────────────────
def poll_once(cities, live_dir: Path):
    """Uma rodada. `t` = início da rodada (relógio local); o instante real da observação é `feed_ts`
    (relógio do servidor, no momento da resposta) — usar feed_ts como tempo da observação."""
    t = now_utc()
    t_epoch, day = int(t.timestamp()), t.strftime("%Y-%m-%d")
    with new_session() as session:
        for city in cities:
            try:
                data, ms, tries = get_json(session, CITIES[city], "station_status")
                rows, feed_ts = status_rows(data, t_epoch)
                append_csv(live_dir / city / f"{day}.csv", FIELDS, rows)
                log = [t_epoch, city, "ok", len(rows), ms, feed_ts, f"tries={tries}" if tries > 1 else ""]
            except Exception as e:  # uma cidade com erro não derruba as outras
                log = [t_epoch, city, "err", 0, "", "", f"{type(e).__name__}: {e}"[:200]]
            append_csv(live_dir / "_log" / f"{day}.csv", LOG_FIELDS, [log])


def acquire_lock(live_dir: Path):
    """Trava exclusiva do processo (liberada pelo SO ao sair/crashar). Retorna o handle ou None."""
    live_dir.mkdir(parents=True, exist_ok=True)
    fh = open(live_dir / ".lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        return None


def run(cities, interval, live_dir: Path, archive_dir: Path, once: bool, max_polls: int):
    lock = acquire_lock(live_dir)
    if lock is None:
        print("Outra instância do coletor já está rodando (trava em", live_dir / ".lock", ").")
        return 2
    last_info = 0.0
    n = 0
    print(f"Coletando {','.join(cities)} a cada {interval}s | live={live_dir} | arquivo={archive_dir}")
    try:
        while True:
            if time.time() - last_info > INFO_EVERY_S:
                with new_session() as session:
                    for c in cities:
                        try:
                            save_info_if_changed(session, c, CITIES[c], archive_dir, live_dir)
                        except Exception as e:
                            print(f"  ! station_information {c}: {e}", file=sys.stderr)
                last_info = time.time()
            poll_once(cities, live_dir)
            archive_finished_days(live_dir, archive_dir, now_utc().strftime("%Y-%m-%d"))
            n += 1
            if once or (max_polls and n >= max_polls):
                return 0
            # alinha ao relógio (…:00, :05, :10 …) para evitar deriva
            time.sleep(interval - (time.time() % interval))
    except KeyboardInterrupt:
        print("Interrompido.")
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# QUALIDADE DOS DADOS
# ──────────────────────────────────────────────────────────────────────────────
def load_city(city, live_dir: Path, archive_dir: Path):
    import pandas as pd
    parts = [pd.read_csv(p, compression="gzip", dtype={"station_id": str})
             for p in sorted((archive_dir / city).glob("*.csv.gz"))]
    parts += [pd.read_csv(p, dtype={"station_id": str}) for p in sorted((live_dir / city).glob("*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else None


def analyze(cities, interval, live_dir: Path, archive_dir: Path):
    import pandas as pd
    for city in cities:
        df = load_city(city, live_dir, archive_dir)
        if df is None or df.empty:
            print(f"[{city}] sem dados")
            continue
        polls = sorted(df["t"].unique())
        span = polls[-1] - polls[0]
        expected = span // interval + 1
        gaps = [b - a for a, b in zip(polls, polls[1:]) if b - a > 2 * interval]
        print(f"\n=== {city} ===")
        print(f"período: {datetime.fromtimestamp(polls[0], timezone.utc):%Y-%m-%d %H:%M} -> "
              f"{datetime.fromtimestamp(polls[-1], timezone.utc):%Y-%m-%d %H:%M} UTC | "
              f"coletas: {len(polls)} de ~{expected} esperadas ({100*len(polls)/max(expected,1):.1f}%)")
        print(f"lacunas > 2 intervalos: {len(gaps)} (maior: {max(gaps)/60:.0f} min)" if gaps else "lacunas: nenhuma")
        print(f"estações: {df['station_id'].nunique()}")
        age = (df["t"] - df["last_rep"].where(df["last_rep"] > 86400)) / 60
        print(f"relato defasado >60 min: {100*(age > 60).mean():.1f}% das linhas | sem relato (1970/vazio): "
              f"{100*(df['last_rep'].isna() | (df['last_rep'] <= 86400)).mean():.1f}%")
        df = df.drop_duplicates(["station_id", "t"]).sort_values(["station_id", "t"])
        d = df.groupby("station_id")["veh"].diff().abs().dropna()
        d = d[df.loc[d.index, "t"] - df.groupby("station_id")["t"].shift().loc[d.index] <= 1.5 * interval]
        if len(d):
            print("|Δ veículos| entre coletas consecutivas: "
                  f"0: {100*(d==0).mean():.1f}% | 1: {100*(d==1).mean():.1f}% | 2-4: {100*d.between(2,4).mean():.1f}% | "
                  f">=5 (candidato a reposição por caminhão): {100*(d>=5).mean():.2f}%")
        print(f"estações sem nenhuma variação no período: {(df.groupby('station_id')['veh'].nunique() == 1).sum()}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Coleta GBFS v3.0 Tembici")
    ap.add_argument("--cities", default=",".join(CITIES), help="lista separada por vírgula")
    ap.add_argument("--interval", type=int, default=300, help="segundos entre coletas (padrão 300)")
    ap.add_argument("--live-dir", type=Path, default=default_live_dir())
    ap.add_argument("--archive-dir", type=Path, default=BASE_DIR / "data" / "raw" / "tembici")
    ap.add_argument("--once", action="store_true", help="uma coleta e sai (Agendador de Tarefas)")
    ap.add_argument("--max-polls", type=int, default=0, help="para após N coletas (testes)")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    cities = [c.strip() for c in a.cities.split(",") if c.strip()]
    bad = [c for c in cities if c not in CITIES]
    if bad:
        sys.exit(f"cidade(s) desconhecida(s): {bad}; opções: {list(CITIES)}")
    if a.analyze:
        analyze(cities, a.interval, a.live_dir, a.archive_dir)
    else:
        sys.exit(run(cities, a.interval, a.live_dir, a.archive_dir, a.once, a.max_polls))
