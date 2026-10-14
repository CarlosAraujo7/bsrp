# HANDOFF — BSRP (v2, 2026-10-14) — documento autossuficiente para um novo agente Claude

> Pasta de trabalho: `C:\Users\vinic\OneDrive\Documentos\Novas pesquisas\BSRP ENIAC`. Idioma de trabalho com o usuário: **português**.
> Este arquivo **substitui** o HANDOFF v1 (arquivado em `archive/HANDOFF_v1_2026-10-11.md`, parcialmente superado).
> Leia, nesta ordem: este arquivo → `PROTOCOLO_RIGOR.md` → `DESVIOS.md` → `README.md`. Detalhes históricos: `04_experimentos.md` (legado auditado), `05_coleta_tembici.md`.
> **Atualização 2026-10-14 (tarde):** **P1 em execução** desde 09:57 (pool de 16 configs por método; ver §5) e executor da ablação (P2) pronto. Remoto `origin` = <https://github.com/CarlosAraujo7/bsrp> (**público**; push feito, com a tag `protocolo-v1`; o e-mail do autor dos commits é público, por decisão do usuário). Commits `020bdcb` (P1) e `d62b678` (P2).

## 0. Objetivo
Lapidar o BSRP (rebalanceamento de bicicletas compartilhadas com 1 veículo, RL) para **ENIAC/BRACIS 2027** (chamada ainda não publicada em 2026-10-14; ENIAC 2026 fechou em 22/jun/2026; verificar `bracis.sbc.org.br`) e **avaliar** se há caminho para periódico (condicional; ver §6). O diferencial planejado é o dado brasileiro (Tembici, só estoque, sem O–D).

## 1. Estado em uma página
- **Protocolo de rigor v1 aprovado** pelo usuário (tag git `protocolo-v1`). Toda decisão que o protocolo não cobria ou que desviou dele está em `DESVIOS.md` (com a coluna "resultados vistos antes?"). **Registre no `DESVIOS.md` ANTES de rodar qualquer coisa nova.**
- **Experimento principal concluído** em 3 instâncias calibradas em dados reais (`cb4wd`, `cb8wd`, `cb12wd`; dia útil): **10 sementes × 5 aprendizes × 500.000 passos**, hiperparâmetros ajustados em validação, comparados com heurísticas calibradas e com a política "parada", em 1000 cenários sintéticos compartilhados **e em 24 dias reais de teste**. Runs: `runs/final_<inst>.jsonl` (commit `a0c5930`, sem alterações pendentes). Análise: `tables/final_cb_wd.md`; configs: `tables/tuned_configs.md`.
- Testes: `python -m unittest discover -s tests` → 33 OK (regressão **exata** contra o simulador legado, invariantes, CRN, reprodutibilidade por semente; pool de ajuste; variantes da ablação; análise P1/P2 sobre dados sintéticos; ≈ 2 min com a máquina saturada).
- Coleta Tembici rodando por tarefa agendada (ver §7); ~3,5 dias coletados, 3 lacunas ≤ 14 min no Rio. **Instância Tembici ainda não construída.**

### Resultados (reward = −demanda não atendida por dia; maior = melhor; média de 10 sementes)
| Escala | parada | melhor heurística calibrada (por val) | tab. cluster | tab. estação | linear | policy gradient | **DQN (64-64)** |
|---|---|---|---|---|---|---|---|
| cb4wd — teste sintético | −46,9 | −24,5 | −43,0 | −40,4 | −37,6 | −37,7 | **−10,4** (IC95% −10,8; −10,1) |
| cb8wd — teste sintético | −89,8 | −52,8 | −92,8 | −94,5 | −87,8 | −101,8 | **−32,0** (−32,8; −31,2) |
| cb12wd — teste sintético | −131,3 | −80,5 | −136,1 | −137,9 | −130,0 | −124,9 | **−64,3** (−66,5; −62,1) |
| cb4wd — 24 dias reais | −32,7 | −25,4 | −35,9 | −34,2 | −33,3 | −33,8 | **−9,0** |
| cb8wd — 24 dias reais | −54,0 | −47,5 | −63,2 | −66,4 | −55,8 | −71,5 | **−19,6** |
| cb12wd — 24 dias reais | −84,5 | −67,8 | −98,1 | −95,6 | −85,0 | −87,5 | **−45,1** |

- **C1 (DQN − melhor heurística):** +14,0 / +20,8 / +16,2 (sintético) e +16,4 / +28,0 / +22,8 (dias reais); t p < 1e-4, 10/10 sementes melhores, Wilcoxon p = 0,002 (mínimo possível com n = 10). Redução da demanda perdida do DQN vs parada: 78% / 64% / 51%; vs heurística: 57% / 39% / 20% (4/8/12 est., sintético).
- **Escada de representação (C4–C6):** tabular (cluster ou estação) e linear **nenhum supera a melhor heurística calibrada**. Em relação à parada: em 4 est. (sintético) ficam entre a parada e a heurística (−37,6 a −43,0 contra −46,9 e −24,5); em 8 e 12 est. e em **todos** os dias reais ficam **≈ ou pior que a parada** (ex.: cb8wd tabular −92,8/−94,5 × parada −89,8). A rede neural domina com folga (C6 = +27 / +56 / +66). Policy gradient é instável (DP entre sementes 10–20). Não há "tabular por estação empata com o DQN" (tese do legado): **refutada**.
- **O que o texto pode e não pode afirmar:** pode: "no simulador calibrado em Citi Bike 2024, uma rede 64-64 treinada com DQN reduz a demanda perdida em 20–57% vs a melhor heurística calibrada, também em dias reais não vistos; tabular e linear não superam a heurística calibrada (e, em 8–12 est., nem a política parada)". **Não pode** (ainda): que o DQN é ótimo; que vale com roteamento/vários veículos; que vale em Tembici; que o ganho é independente do espaço de busca (ver §5 P1).

## 2. Achados que mudaram a história (resumo; detalhes em `04_experimentos.md`, `tables/real_data_checks.md`, `DESVIOS.md`)
1. **Legado (maio/2026) tinha defeitos graves:** 1 semente de treino; teste e seleção nas mesmas sementes; números de gulosa/aleatória não salvos; `unmet` sempre 0; ε não decaía sem replay; `np.random` global sem semente; Q-learning indexava ações pela posição na lista de válidas; referências erradas (3, corrigidas em `01_escopo.md`).
2. **O DQN legado estava mal treinado** (−3,49 em 4 est. ≈ política parada). Um DQN com máscara no alvo e treino estável chega a −0,3 (instância legada), verificado também **dentro do simulador legado**.
3. **Dado legado defeituoso:** chegadas ≈ 5× partidas era erro de pré-processamento (arquivo truncado); com Jan–Jun 2024 completos a razão é 0,995–1,012. A demanda legada ×3,5 valia 66–80 partidas/estação/dia contra ~**200** reais. Por isso as instâncias primárias agora são `cb*` (λ real, sem escala).
4. **Referências corretas = parada + heurística calibrada em validação.** A aleatória é barra baixa (movimentos aleatórios pioram o sistema). Calibração importa (gulosa legada → calibrada: −2,25 → −1,13 na instância legada de 4 est.).
5. **O modelo de Poisson superestima a dificuldade em 30–40%** vs dias reais (parada: sintético −47/−90/−131 × real −33/−54/−85).
6. **Subsistema aberto:** só 2,0 / 3,1 / 4,5% das partidas (4/8/12 est.) terminam dentro da instância (O–D real) → ausência de conservação de bicicletas é aproximação razoável para esses tamanhos (evidência, não suposição).
7. **Limitação estrutural do simulador:** todo deslocamento custa **exatamente 1 intervalo** (32 decisões/episódio sempre): não há roteamento nem restrição de tempo de viagem; o problema é "quanto e para onde mover a cada 30 min". Capacidade do veículo 20, deltas ∈ {−10,−5,0,5,10}.
8. Ocupação inicial dos dias reais é desconhecida (usa-se 50% ± 2).

## 3. Mapa do repositório
| Caminho | O quê |
|---|---|
| `PROTOCOLO_RIGOR.md`, `DESVIOS.md` | regras do jogo (aprovadas) e registro de desvios/especificações |
| `bsrp/env.py` | `Instance`, `Scenario(Set)` (CRN), `BSRPEnv` (dinâmica idêntica ao legado), `legacy_instance`, `sample_scenarios`, `legacy_scenarios` |
| `bsrp/data.py` | `Flows` (Citi Bike), `calibrated_instance(n, flows, daytype)`, `real_scenarios` (replay de dias reais) |
| `bsrp/policies.py` | `Stay`, `RandomPolicy`, `LegacyGreedy/Threshold` (portes exatos), `GreedyFamily` + grades 1–3, `ThresholdFamily`, `Rollout` (honesto: futuros amostrados do modelo; **implementado, não rodado em escala**) |
| `bsrp/agents.py` | `QTabular` (cluster/estação), `DQN` (`hidden=()` = linear; flags `use_replay/use_target/double` para ablação), `PolicyGradient` (REINFORCE + baseline) |
| `bsrp/train.py` | `SeedSequence` (4 fluxos independentes), treino em passos de ambiente, val periódica, `final_eval` |
| `tests/` | 28 testes: G0 (regressão, invariantes, CRN, reprodutibilidade), `test_tuning.py` (pool de 16 contém o de 8), `test_ablation.py` (variantes) |
| `scripts/20_ablation.py` | ablação do DQN (P2): `noreplay/notarget/nodouble/vanilla` sobre a config do P1, sementes 0–9 → `runs/ablation_<inst>.jsonl` (retomável) |
| `scripts/12_prepare_citibike.py` | zips Citi Bike → `data/processed/citibike_2024/flows_od.npz` (fluxos 30 min + O–D só de semanas de treino) |
| `scripts/13_run_protocol.py` | executor: `scenarios` / `heuristics` / `learners` (instâncias `4`, `legacy4x1.5`, `cb4wd`, `cb12we`, `cb4wdx1.5`…) |
| `scripts/14_analyze_runs.py` | estatística do protocolo (IC sobre sementes, Holm, TOST, dias reais) → `tables/*.md` |
| `scripts/15_real_data_checks.py`, `18_tuning_table.py`, `11_make_tables.py` | tabelas geradas |
| `scripts/16_tune.py`, `17_pipeline_final.py` | ajuste de hiperparâmetros e pipeline final **retomáveis** |
| `scripts/03_collect_tembici_gbfs.py` | coletor GBFS v3.0 (tarefa agendada) |
| `scripts/01–10` | **legado** (não editar; usados nos testes de regressão) |
| `scenarios/` | val (200) e test (1000) sintéticos por instância, com hash |
| `runs/` | `final_*.jsonl`, `tune_*.jsonl`, `tuned_*.json`, `heuristics_*.jsonl`, pilotos; P1: `tuned_ext_*.json`, `final_ext_*.jsonl`, `p1_*.out` (log, não versionado); P2: `ablation_*.jsonl` |
| `tables/` | `final_cb_wd.md` (principal), `tuned_configs.md`, `real_data_checks.md`, `pilot_*.md`, `legacy_results.md` |
| `data/` (fora do git, mas no disco/OneDrive) | `live/` + `raw/tembici/` (coleta), `processed/citibike_2024/` (fluxos). Os zips do Citi Bike (3,7 GB) estavam no scratchpad da sessão anterior e **podem não existir mais**; refaça o download se precisar (S3 `tripdata/2024MM-citibike-tripdata.zip`, licença: NYCBS Data Use Policy) |
| `archive/` | v1 do HANDOFF e de `04_experimentos.md` |

### Reproduzir
```bash
python -m unittest discover -s tests -v
python scripts/13_run_protocol.py scenarios --instances cb4wd cb8wd cb12wd
python scripts/13_run_protocol.py heuristics --instances cb4wd cb8wd cb12wd --skip-threshold --out runs/heuristics_cb_wd.jsonl
python scripts/16_tune.py --instance cb4wd --methods dqn tab_station tab_cluster linear pg --configs 8 --seeds 1000 1001 --budget 200000
python scripts/17_pipeline_final.py cb4wd         # 10 sementes x 5 métodos x 500k passos, pula o que já existe
python scripts/14_analyze_runs.py --learners runs/final_cb4wd.jsonl runs/final_cb8wd.jsonl runs/final_cb12wd.jsonl \
       --heuristics runs/heuristics_cb_wd.jsonl runs/heuristics_cb_wd_g3.jsonl --out tables/final_cb_wd.md
# P1 (pool estendido de 16 configs; as 8 originais são reaproveitadas de runs/tune_<inst>.jsonl):
python scripts/16_tune.py --instance cb4wd --space ext --configs 16 --methods dqn linear tab_station tab_cluster pg --seeds 1000 1001 --budget 200000
python scripts/17_pipeline_final.py --ext cb4wd   # reroda só os métodos cuja config escolhida difere da original -> runs/final_ext_cb4wd.jsonl
# Fatias paralelas do P1 (só com a máquina livre e autorização para > 3 processos; custo equilibrado por LPT):
python scripts/16_tune.py --instance cb4wd --space ext --configs 16 --methods dqn linear tab_station tab_cluster pg \
       --seeds 1000 1001 --budget 200000 --shard 0/4 --fill-only      # ... 1/4, 2/4, 3/4, cada fatia em seu processo
python scripts/16_tune.py --instance cb4wd --space ext --configs 16 --methods dqn linear tab_station tab_cluster pg \
       --seeds 1000 1001 --budget 200000                              # SÓ depois de todas imprimirem FILL-FIM: escolhe e grava tuned_ext
python scripts/17_pipeline_final.py --ext cb4wd --seeds 0 1 2 3 4 --methods dqn   # runs finais também em fatias (por sementes/métodos)
# P2 (ablação; precisa de runs/tuned_ext_<inst>.json com o DQN):
python scripts/20_ablation.py --instance cb12wd [--variants noreplay vanilla] [--seeds 0 1 2]
```
Use `python -X utf8` nos scripts de análise (console cp1252).
Tempos (máquina compartilhada, BLAS de 1 thread): DQN 500k passos ≈ 5–6 min por run (10 sementes ≈ 1 h por escala); linear 2–3 min; PG e tabular ≈ 1–2 min. Ajuste do DQN (8 configs × 2 sementes × 200k) ≈ 1–2 h por escala.

## 4. Regras de conduta (o que o usuário aprovou e espera)
1. **Rigor antes de resultado:** unidade de análise = semente de treino; escolha de checkpoint = último; ajuste e escolha de "melhor" só em **validação**; teste tocado uma vez; sem p-valor sobre episódios; "empata" só com TOST (δ = 0,5); reportar resultados negativos. Cada execução → uma linha em `runs/*.jsonl` com commit e flag *dirty*; **commite antes de rodar** para o hash ficar limpo.
2. **Honestidade:** declare incertezas e limitações; corrija o que errar (houve correções registradas); não escreva "DQN supera X" sem a escala, o número de sementes e o IC.
3. **Autorização:** o usuário aprovou explicitamente: tarefa agendada de coleta, download Citi Bike, protocolo. **Peça antes de**: enviar mensagens/e-mails por ele (o e-mail à Tembici é rascunho em `docs/email_tembici_rascunho.md`, **envio é do usuário**), publicar, baixar dados novos, criar tarefas persistentes, apagar dados.
4. Nunca use o e-mail do usuário em requisições externas (User-Agent, URLs); UA do coletor é neutro.
5. Não encerre processos que não sejam seus (a máquina roda jobs de outros projetos do usuário).

## 5. O que falta — prioridades e critérios de aceite
**P1 — Espaço de busca do DQN (antes de qualquer afirmação de magnitude). STATUS 14/out: EM EXECUÇÃO** (3 cadeias ocultas, uma por escala: `16_tune.py --space ext` → `17_pipeline_final.py --ext`; log em `runs/p1_<inst>.out`). Decisões do usuário: **todos os 5 métodos**, **batch ≤ 256**, 3 processos. Especificação completa, espaços novos e **regra de decisão pré-registrada** na última linha de `DESVIOS.md` com data 2026-10-14 (argmax em val sobre as 16; reroda 10 sementes × 500k só onde a escolhida difere da original; original e estendida são reportadas lado a lado). Custo medido sob CPU saturada: 1 run de ajuste do DQN (200k passos, `train_every`=1) ≈ 11 min; DQN final 500k passos ≈ 7–17 min conforme a config. *Texto original da prioridade:* A config vencedora do DQN é a mesma nas 3 escalas e tem **3 hiperparâmetros no máximo do espaço** (`target_sync=4000`, `batch=128`, `buffer=50000`; `lr=4,43e-4`, `train_every=2`). Estender (`target_sync` até 16000, `batch` 256, `buffer` 200000, `train_every` 1, `lr`) com o **mesmo orçamento de configs para todos os métodos** (registrar em DESVIOS) e rerodar o DQN final. Faça o mesmo para qualquer método cujo ótimo caia na borda. Critério: ótimo no interior do espaço ou ganho marginal documentado. Obs.: nas escalas 8/12 o ajuste foi decisivo (+18,6 e +22,0 de val); com parâmetros padrão o DQN perde para a heurística em 12 est.
**P2 — Ablação do DQN (protocolo C7–C9). STATUS 14/out: executor e testes prontos (`scripts/20_ablation.py`, `tests/test_ablation.py`); roda DEPOIS do P1**, sobre a config do DQN escolhida lá, em `cb12wd` e `cb4wd`, sementes 0–9; especificação e duas observações de desenho (double é no-op sem rede-alvo; `−replay` = lote 1 online) no `DESVIOS.md`. A análise C7–C9 (IC pareado por semente + Holm) está em `scripts/21_analyze_p1_p2.py`. *Texto original:* Variantes `use_replay=False`, `use_target=False`, `double=False` e "vanilla" (as três), **mesmo cronograma de ε** (em `train()` o ε já decai por passos para todos; o bug do legado não existe aqui), ≥ 10 sementes, na instância `cb12wd` (e `cb4wd`). Adicionar aliases de método no runner (`make_agent(..., **kw)` já aceita as flags). Só afirme "componente necessária" se o IC da diferença pareada por semente excluir 0.
**P3 — Sensibilidade (protocolo §8.1, redefinida em DESVIOS).** `cbNwdxK` com K ∈ {0,5; 0,75; 1,5} e `cb*we` (fim de semana); arm legado com escala {1,5; 2,5; 3,5; 5}. Os **pesos dos agentes não foram salvos** → é preciso retreinar. Desenho reduzido sugerido (registrar antes): DQN ajustado × 5 sementes + gulosa calibrada + parada, em `cb4wd` e `cb12wd`. Reporte onde o sinal da conclusão muda.
**P4 — Baselines faltantes.** (a) `Rollout(base=melhor heurística, h=1..2, K=4)` em todas as escalas (código pronto, `tests/test_learners.py` cobre); (b) **uma referência da literatura** adaptada e chamada "adaptação", não "reprodução": Seo et al. (2022, ADP/RL, docas) ou Liang et al. (2024, arXiv:2402.03589, single- vs dual-policy). Leia os dois e verifique o que é reproduzível. (c) Família `threshold` nas instâncias `cb*` (foi pulada com `--skip-threshold`; dominada pela gulosa nas legadas).
**P5 — Tembici (diferencial brasileiro).** Construir `scripts/19_prepare_tembici.py`: snapshots (`data/live` + `data/raw/tembici`, concatenar e deduplicar por `(station_id, t)`; **tempo da observação = `feed_ts`**) → fluxo líquido por estação e 30 min (diferença de estoque; chegadas e partidas **não são separáveis** e reposição por caminhão aparece como salto |Δ|≥5) → instância Rio/SP (escolher ~12 estações por volume em semanas de treino) + replay de fluxo líquido real. **Hold-out já definido: última semana completa seg–dom da janela; val = semana anterior; calendário marcado (12/out, 2/nov, 15/nov; 2º turno 25/out a confirmar)**. A coleta começou em 2026-10-11 (domingo); a 1ª semana completa é 12–18/out; meta ≥ 4 semanas (até ~8/nov), ideal 8–12. Portão G3: o simulador reproduz distribuição horária de fluxo líquido e ocupação em dias de hold-out dentro de tolerância **fixada antes** de ver o hold-out; senão Tembici vira estudo descritivo. **Licença/termos não verificados**: não redistribuir dados brutos; e-mail ao feed aguardando o usuário.
**P6 — Resultados e artigo.** Gerar figuras por script (curvas de aprendizado com IC sobre sementes — `curve` está em cada linha do `jsonl`; barras com IC por escala; escada de representação; sintético × dias reais). Escrever `06_resultados.md` (sem números digitados à mão) e o artigo no formato ENIAC/BRACIS 2027 (verificar a chamada). Revisar referências (item 5 de `01_escopo.md` **não verificado**: "Efficient Inventory Routing for BSS (2024)").
**P7 — Realismo para periódico (só se P1–P6 ok).** Roteamento/tempo de viagem reais, mais de um veículo, baseline da literatura reproduzido, ≥ 3 sistemas reais. Veja `PROTOCOLO_RIGOR.md` §11.

## 6. Checklist de nível de publicação (estado)
- [x] ≥ 10 sementes de treino; IC; testes sobre sementes (instâncias `cb*wd`)
- [x] Validação/teste separados; dias reais de teste
- [~] Simulador calibrado em dados e conservação justificada com O–D (**sem** roteamento/tempo de viagem)
- [x] Tabular justo (por estação) e heurísticas calibradas (grades com ótimo interior após a grade 3, salvo o que P1 revelar)
- [ ] ≥ 1 baseline da literatura de BSRP (P4)
- [~] Dados brasileiros (coleta em andamento; instância não construída) (P5)
- [~] Documentação sem números conflitantes; referências verificadas (3 corrigidas; 1 pendente)
- [ ] Ablação (P2); sensibilidade (P3)

## 7. Ambiente (armadilhas reais, todas já vividas)
- **O shell do Claude Code escreve num overlay privado de `AppData\Local`**; só a pasta do projeto e o scratchpad são compartilhados com o Windows real. Nunca rode o coletor "de verdade" nem guarde dados em AppData a partir do shell.
- **Coleta Tembici:** tarefa agendada `BSRP-Tembici-GBFS` (`pythonw scripts\03_collect_tembici_gbfs.py --once` a cada 5 min, prioridade 4, só com o usuário logado; **notebook desligado = lacuna**). Gerenciar: `Get-ScheduledTaskInfo -TaskName BSRP-Tembici-GBFS`; pausar `Disable-ScheduledTask`; remover `Unregister-ScheduledTask`. Qualidade: `python scripts/03_collect_tembici_gbfs.py --analyze`. Dados: `data/live/<cidade>/AAAA-MM-DDTHH.csv` (hora corrente) → `data/raw/tembici/` (`.csv.gz`); dias iniciais em `<cidade>/2026-10-11.csv.gz`, `2026-10-12_overlay.csv.gz`, `2026-10-12_real.csv.gz`.
- **CPU compartilhada** com jobs pesados de outros projetos do usuário (≈ 8 workers) e memória virtual apertada: use ≤ 3 processos meus; **sem pandas nos runners** (pode crashar por falta de commit); os executores já fixam `OPENBLAS_NUM_THREADS=1` (2,5× mais rápido sob disputa).
- Processos longos: `Start-Process python -ArgumentList "-I","scripts/<x>.py",... -WindowStyle Hidden -RedirectStandardOutput runs\x.out`. Morrem se o notebook desligar; `16_tune.py` e `17_pipeline_final.py` pulam o que já está gravado — relance.
- `| Out-String` bufferiza a saída até o fim do job; para acompanhar, leia `runs/*.jsonl`. `sleep` longo é bloqueado; use Monitor (expira em 30 min).
- Console Windows é cp1252: use `sys.stdout.reconfigure(errors="replace")` ou grave em arquivo UTF-8. `python -I` ignora variáveis `PYTHON*`.
- Git com remoto `origin` (GitHub **público**, <https://github.com/CarlosAraujo7/bsrp>); identidade já configurada; commits terminam com `Co-Authored-By: Claude ...`. Dados em `data/` e `*.pkl` não são versionados. **Push só com confirmação do usuário** (o de 14/out foi autorizado explicitamente). Com runs finais ativos, não edite `bsrp/` nem `scripts/` (cada run é subprocesso novo e grava a flag *dirty*); arquivos novos só com commit imediato.
- Em 14/out a CPU estava a 100% (jobs de outros projetos): meus processos pegavam 40–60% de um núcleo cada (i5-12450H, 8 núcleos/12 threads); conte com ~2× o tempo medido com a máquina livre.

## 8. Não verificado / pendente do usuário
Licença e termos da Tembici; chamada ENIAC/BRACIS 2027 (prazos); item 5 das referências; **envio do e-mail à Tembici** (`docs/email_tembici_rascunho.md`); manter o notebook ligado e logado durante a coleta.
