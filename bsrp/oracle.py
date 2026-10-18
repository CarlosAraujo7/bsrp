"""
bsrp/oracle.py — planejador CLARIVIDENTE: conhece a demanda inteira do cenário e escolhe a sequência de ações que minimiza a demanda perdida.

Serve de referência de "quanto ainda era possível ganhar": nenhuma política NÃO antecipativa (que não vê o futuro do cenário) consegue
mais que o ótimo clarividente. Dois cuidados de leitura (DESVIOS.md 2026-10-18):
  * o ótimo clarividente NÃO é atingível: parte da demanda perdida é puro azar da amostra e só quem conhece o futuro a evita;
  * a busca em feixe de largura finita acha uma sequência VIÁVEL, logo seu valor é um limite INFERIOR do ótimo clarividente (e a fração
    do ganho capturada por um método, medida contra ele, é um limite SUPERIOR da fração verdadeira). A convergência com a largura é checada.

Dinâmica idêntica à de `bsrp.env.BSRPEnv` (verificada por tests/test_oracle.py contra programação dinâmica exaustiva sobre o próprio
BSRPEnv): ação = (destino, delta ∈ {−10, −5, 0, 5, 10} viável na estação atual); a carga/descarga ocorre na estação atual e depois o
veículo viaja max(1, ceil(T)) intervalos (1 se destino = estação atual), durante os quais a demanda do cenário acontece.
"""
from __future__ import annotations

import numpy as np

from . import env as E


def _demand(occ, cap, nd, na):
    """Um intervalo de demanda sobre M estados. occ (M, n); nd, na (n,). Retorna (occ', perdida (M,))."""
    act_dep = np.minimum(nd, occ)
    space = cap - (occ - act_dep)
    act_arr = np.minimum(na, space)
    unmet = (nd - act_dep).sum(axis=1) + (na - act_arr).sum(axis=1)
    return np.clip(occ - act_dep + act_arr, 0, cap), unmet


def _stay_future(occ, t0, cap, dep, arr):
    """Demanda perdida futura (≥ 0) se o veículo não fizer mais nada de t0 até o fim: potencial usado para ordenar o feixe."""
    acc = np.zeros(len(occ), np.int64)
    for t in range(t0, E.N_INTERVALS):
        occ, u = _demand(occ, cap, dep[t], arr[t])
        acc += u
    return acc


def _dedup(occ, pos, load, val):
    """Remove estados idênticos (pos, load, occ) mantendo o de maior valor. Chave = hash de 64 bits (produto interno com um vetor fixo,
    com estouro): uma colisão (≈ 1e-10 por camada) apenas fundiria dois estados, e o caminho mantido continua viável (o valor segue sendo
    um limite inferior)."""
    key = np.concatenate([pos[:, None], load[:, None], occ], axis=1).astype(np.int64)
    hv = np.random.default_rng([777, key.shape[1]]).integers(1, 2 ** 62, size=key.shape[1], dtype=np.int64)
    with np.errstate(over="ignore"):
        h = key @ hv
    order = np.argsort(-val, kind="stable")
    _, first = np.unique(h[order], return_index=True)
    return np.sort(order[first])


def plan_value(inst: E.Instance, sc: E.Scenario, width: int | None = 300) -> float:
    """Melhor reward (≤ 0; = −demanda perdida) encontrado por busca em feixe com conhecimento do cenário.
    width=None: sem poda (exato, só viável em instâncias minúsculas)."""
    n, cap = inst.n, inst.cap.astype(np.int64)
    tau = np.maximum(inst.travel_int(), 1)                         # (n, n); diagonal = 1 (ficar parado custa 1 intervalo)
    T = E.N_INTERVALS
    dep, arr = sc.dep.astype(np.int64), sc.arr.astype(np.int64)
    deltas = E.DELTAS
    cap_v = E.VEHICLE_CAP
    buckets = {0: [(sc.init_occ.astype(np.int64)[None, :], np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64))]}
    best = -np.inf
    for t in range(T):
        parts = buckets.pop(t, None)
        if not parts:
            continue
        occ = np.concatenate([p[0] for p in parts])
        pos = np.concatenate([p[1] for p in parts])
        load = np.concatenate([p[2] for p in parts])
        val = np.concatenate([p[3] for p in parts])
        keep = _dedup(occ, pos, load, val)
        occ, pos, load, val = occ[keep], pos[keep], load[keep], val[keep]
        if width is not None and len(pos) > width:
            score = val - _stay_future(occ, t, cap, dep, arr)       # valor acumulado + potencial (nada mais a fazer até o fim)
            top = np.argpartition(-score, width - 1)[:width]
            occ, pos, load, val = occ[top], pos[top], load[top], val[top]
        S = len(pos)
        ar = np.arange(S)
        occ_p, cap_p = occ[ar, pos], cap[pos]
        valid = np.stack([(occ_p >= 10) & (load + 10 <= cap_v), (occ_p >= 5) & (load + 5 <= cap_v), np.ones(S, bool),
                          (load >= 5) & (occ_p + 5 <= cap_p), (load >= 10) & (occ_p + 10 <= cap_p)], axis=1)
        si, ki = np.nonzero(valid)
        M = len(si)
        occ_c = occ[si].copy()
        occ_c[np.arange(M), pos[si]] += deltas[ki]
        load_c, pos_c, val_c = load[si] - deltas[ki], pos[si], val[si]
        steps_avail = T - t
        max_j = min(int(tau.max()), steps_avail)
        occ_after, unmet_cum = {}, {}
        o, cum = occ_c, np.zeros(M, np.int64)
        for j in range(1, max_j + 1):
            o, u = _demand(o, cap, dep[t + j - 1], arr[t + j - 1])
            cum = cum + u
            occ_after[j], unmet_cum[j] = o, cum
        J = np.minimum(tau[pos_c], steps_avail)                       # (M, n): intervalos efetivamente percorridos até o destino
        for j in np.unique(J):
            mi, di = np.nonzero(J == j)
            new_val = val_c[mi] - unmet_cum[int(j)][mi]
            if t + int(j) >= T:                                       # episódio termina: valor final
                best = max(best, float(new_val.max()))
            else:
                buckets.setdefault(t + int(j), []).append((occ_after[int(j)][mi], di.astype(np.int64), load_c[mi], new_val))
    return best
