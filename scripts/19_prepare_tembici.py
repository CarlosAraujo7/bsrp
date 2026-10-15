"""
19_prepare_tembici.py — coleta GBFS da Tembici -> fluxo por estação/dia/30 min, divisão por semanas, instância e portão G3.
Especificado em DESVIOS.md 2026-10-15 (P5). Só numpy/csv/gzip.

  python scripts/19_prepare_tembici.py --city rio                 # prepara (dev/treino) e relata; NÃO toca val/teste
  python scripts/19_prepare_tembici.py --city rio --g3 --confirm-holdout   # só com ≥ 4 semanas completas: avalia o portão G3

PROTEÇÃO DO HOLD-OUT: sem --g3, nenhum valor dos dias 'val'/'test' é usado para estimar ou selecionar nada (a seleção de
estações e o λ usam só dias úteis 'train' ou 'dev'). O portão G3 exige --confirm-holdout e dias de teste existentes.
Saída (fora do git, `data/` é ignorado): data/processed/tembici/<cidade>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bsrp import tembici as TB                                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="rio")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--radius-km", type=float, default=3.0)
    ap.add_argument("--big-jump", type=int, default=TB.BIG_JUMP, help="limiar |Δ| de salto operacional excluído (primário 5; sensibilidade 8)")
    ap.add_argument("--g3", action="store_true", help="avalia o portão G3 sobre os dias de TESTE (exige --confirm-holdout)")
    ap.add_argument("--confirm-holdout", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    snap = TB.read_snapshots(a.city)
    info = TB.read_station_info(a.city)
    cf = TB.compute_flows(snap, big_jump=a.big_jump)
    us = TB.usable_days(cf)
    split = TB.assign_split(cf.dates, us)
    types = np.array([TB.day_type(d) for d in cf.dates])
    print(f"[{a.city}] {len(snap.ts)} linhas, {len(cf.station_ids)} estações, janela {cf.dates[0]} .. {cf.dates[-1]}")
    print("dia           tipo  usável  coletas(06–22h)  divisão")
    for i, d in enumerate(cf.dates):
        ndist = int(cf.polls[i, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS].sum())
        print(f"{d} {'fds/feriado' if types[i] else 'útil       '}  {'sim' if us[i] else 'não'}     {ndist:5d}            {split[i]}")
    train_mask = np.isin(split, ["train", "dev"]) & (types == 0)
    days_tr = np.flatnonzero(train_mask)
    print(f"dias úteis de {'treino' if (split == 'train').any() else 'desenvolvimento'} usados na seleção/λ: {len(days_tr)}")
    if len(days_tr) == 0:
        raise SystemExit("sem dias úteis de treino/desenvolvimento usáveis")

    idx = TB.select_stations(cf, info, days_tr, n=a.n, radius_km=a.radius_km)
    if not idx:
        raise SystemExit("nenhuma estação elegível")
    inst = TB.build_instance(cf, info, idx, days_tr, f"tmb_{a.city}{len(idx)}")
    vol, frac = TB.station_volume(cf, days_tr)
    print(f"\ninstância {inst.name}: {inst.n} estações (as de maior volume bruto em dias úteis, ≤ {a.radius_km:g} km da principal)")
    print("  #  id     capacidade  volume/dia  células válidas  cluster  nome")
    for j, i in enumerate(idx):
        print(f"  {j:2d} {cf.station_ids[i]:>6s} {int(inst.cap[j]):9d} {vol[i]:11.1f} {100 * frac[i]:13.0f}%  {int(inst.cluster[j]):6d}   {inst.station_names[j]}")
    print(f"  λ bruto total por dia (soma das estações): {float((inst.lam_dep + inst.lam_arr).sum()):.1f}; "
          f"fluxo líquido absoluto médio por intervalo: {float(np.abs(inst.lam_arr - inst.lam_dep).mean()):.2f}")
    ops = cf.ops[idx][:, days_tr, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS]
    gross = float(cf.arr[idx][:, days_tr, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS].sum() + cf.dep[idx][:, days_tr, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS].sum())
    exc = float(cf.ops_vol[idx][:, days_tr, TB.OFFSET:TB.OFFSET + TB.N_INTERVALS].sum())
    print(f"  saltos |Δ| >= {a.big_jump} excluídos (dias de treino/dev): {int(ops.sum())} eventos = {100 * exc / max(gross + exc, 1e-9):.1f}% do volume observado")

    out = ROOT / "data" / "processed" / "tembici"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / (f"{a.city}.npz" if a.big_jump == TB.BIG_JUMP else f"{a.city}_jump{a.big_jump}.npz"), dates=np.array(cf.dates), split=split.astype(str), station_ids=np.array(cf.station_ids),
                        arr=cf.arr, dep=cf.dep, cover=cf.cover, ops=cf.ops, stock06=cf.stock06, polls=cf.polls,
                        sel=np.array(idx), cap=inst.cap, cluster=inst.cluster, lat=inst.lat, lon=inst.lon,
                        lam_dep=inst.lam_dep, lam_arr=inst.lam_arr, names=np.array(inst.station_names))
    print(f"\ngravado em {out}")

    if a.g3:
        test_days = np.flatnonzero((split == "test") & (types == 0))
        if not a.confirm_holdout or len(test_days) == 0:
            raise SystemExit("G3 só com --confirm-holdout e dias úteis de TESTE existentes (≥ 4 semanas completas)")
        real = TB.real_scenarios(cf, inst, idx, test_days)
        m = TB.g3_metrics(inst, real)
        print("\nG3 (dias úteis de teste):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, default=str))
        print("simulador validado em Tembici" if m["G3_passed"] else "G3 NÃO passou: Tembici vira estudo descritivo")


if __name__ == "__main__":
    main()
