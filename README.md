# BSRP — rebalanceamento de bicicletas compartilhadas com RL (alvo: ENIAC/BRACIS 2027)

Estado: **lapidação em andamento** (início 2026-10-11). Ler primeiro: [`HANDOFF.md`](HANDOFF.md) (histórico e achados),
[`PROTOCOLO_RIGOR.md`](PROTOCOLO_RIGOR.md) (v1, aprovado; tag `protocolo-v1`), [`DESVIOS.md`](DESVIOS.md).

## Mapa

| Caminho | O quê |
|---|---|
| `01_escopo.md … 03_modelagem.md` | escopo, dados e modelagem originais (v1, parcialmente superados) |
| `04_experimentos.md` | experimentos **legados** consolidados e auditados (v2) |
| `05_coleta_tembici.md` | coleta GBFS Tembici (tarefa agendada) |
| `bsrp/` | pacote novo: `env` (cenários, CRN), `policies`, `agents`, `train`, `data` |
| `tests/` | portão G0: regressão exata contra o legado, invariantes, reprodutibilidade |
| `scripts/01–10` | código **legado** (não editar; referência para regressão) |
| `scripts/11_make_tables.py` | tabelas legadas → `tables/legacy_results.md` |
| `scripts/12_prepare_citibike.py` | zips do Citi Bike → fluxos de 30 min e matrizes O–D (`data/processed/`) |
| `scripts/13_run_protocol.py` | executor: `scenarios`, `heuristics`, `learners` → `runs/*.jsonl` |
| `scripts/14_analyze_runs.py` | análise estatística do protocolo → `tables/*.md` |
| `scenarios/` | conjuntos fixos de validação (200) e teste (1000) por instância, com hash |
| `runs/` | uma linha JSON por execução (commit, sementes, hashes, métricas) |
| `data/` | **fora do git**: `live/` e `raw/tembici` (coleta), `processed/` (Citi Bike) |

## Reproduzir

```bash
python -m unittest discover -s tests -v                       # G0 (≈1 min)
python scripts/13_run_protocol.py scenarios --instances 4 8 12
python scripts/13_run_protocol.py heuristics --instances 4 8 12 --out runs/heuristics.jsonl
python scripts/13_run_protocol.py learners --instances 4 --methods tab_cluster tab_station linear dqn pg \
       --seeds 0 1 2 3 4 --budget 150000 --out runs/pilot_g1.jsonl
python scripts/14_analyze_runs.py --learners runs/pilot_g1.jsonl --heuristics runs/heuristics.jsonl --out tables/pilot_g1.md
```

## Armadilhas

- **O shell do Claude Code escreve num overlay privado de `AppData\Local`**; só a pasta do projeto é compartilhada com o
  Windows real. A coleta Tembici roda pela tarefa agendada `BSRP-Tembici-GBFS`; nunca rodar o coletor "de verdade" pelo shell da sessão.
- A máquina roda outros jobs pesados: usar poucos processos numpy e evitar `pandas` nos runners.
- Licença/termos da Tembici **não verificados**: não redistribuir dados brutos (`docs/email_tembici_rascunho.md`).
- Comparar sempre contra a heurística **calibrada** e contra a política **parada** (`stay`), nunca só contra a aleatória.
