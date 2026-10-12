# 05 — Coleta Tembici (GBFS v3.0)

**Início:** 2026-10-11 ~10:27 (horário local) · **Meta:** ≥ 4 semanas contínuas (até ~2026-11-08) · **Script:** `scripts/03_collect_tembici_gbfs.py` (v2, reescrito em 2026-10-11/12)

## 1. Endpoints verificados (2026-10-11)

`https://<sub>.publicbikesystem.net/customer/gbfs/v3.0/{gbfs,station_status,station_information,system_information}.json` — todos respondem 200; **v2.x dá 404**. Plataforma Lyft (`feed_contact_email: mobility-data-client@lyft.com`). `ttl` do status: 5–30 s.

| Cidade | Subdomínio | Estações | Capacidade total | Observações |
|---|---|---|---|---|
| Rio | `riodejaneiro` | 403 | 6.947 | |
| São Paulo | `saopaulo` | 240 | 4.114 | |
| Brasília | `brasilia` | 70 | 833 | |
| Recife | `recife` | 90 | 1.636 | |
| Salvador | `salvador` | 75 → 73 no status | 1.109 | 2 estações sumiram do `station_status` em 12/out; 2,7% das linhas com `last_reported` em 1970 |

## 2. O que o feed dá — e o que não dá

- **Dá:** estoque por estação a cada coleta (`num_vehicles_available`, `num_vehicles_disabled`, `num_docks_available`, `num_docks_disabled`, `is_renting`, `is_returning`, `last_reported`, bicicletas por tipo incl. elétricas).
- **Não dá:** viagens. **Não há origem-destino**. O fluxo líquido por intervalo é a diferença de estoque; chegadas e partidas **não são separáveis**, e a reposição por caminhão aparece misturada (saltos |Δ| ≥ 5 em 5 min: 0,02–0,44% das observações por estação). Consequência para o artigo: Tembici serve como **estudo de caso com fluxo líquido** (e com a política operacional observada como referência), **não** como fonte de matriz O–D. Para O–D é preciso outra fonte (Citi Bike etc.). Não verifiquei se a Tembici publica dados de viagem.

## 3. Esquema dos arquivos

Tempos em epoch UTC. `t` = início da rodada (relógio local); **o instante da observação é `feed_ts`** (relógio do servidor).

| Local | Conteúdo |
|---|---|
| `%LOCALAPPDATA%\bsrp_tembici\live\<cidade>\<AAAA-MM-DD>.csv` | dia corrente, append a cada coleta (fora do OneDrive de propósito) |
| `…\live\_log\<AAAA-MM-DD>.csv` | uma linha por (coleta, cidade): `ok`/`err`, nº de estações, ms, `tries=` se houve retentativa |
| `data/raw/tembici/<cidade>/<AAAA-MM-DD>.csv.gz` | dias encerrados (UTC), gzip conferido linha a linha |
| `data/raw/tembici/station_information/<cidade>/<ts>.json.gz` | cadastro bruto, só quando muda (checa a cada 6 h) |

Colunas: `t, feed_ts, station_id, veh, veh_dis, docks, docks_dis, inst, rent, ret, last_rep, types`. Sem deduplicação (duplicatas detectáveis por `last_rep`). `data/` está no `.gitignore`.

## 4. Defeitos corrigidos

1. O script original usava campos de GBFS v2 (`num_bikes_available`; na v3.0 é `num_vehicles_available`), gravava Parquet sem `pyarrow` instalado (falharia na 1ª virada de dia) e só gravava na virada do dia.
2. **Keep-alive:** com `Session` reaproveitada entre rodadas de 5 min, cada 1ª tentativa travava até o timeout (15 s) e a retentativa funcionava; as 5 cidades eram amostradas com ~17 s de espaçamento e `feed_ts` ficava 16 s à frente de `t`. Corrigido em 2026-10-12 01:3x (sessão nova por rodada). **Os primeiros ~15 h (até 2026-10-12 ~01:33 local) têm esse espaçamento**: usar `feed_ts` como tempo da observação e não `t`.

## 5. Qualidade até 2026-10-12 01:48

186 coletas em ~15,3 h (101% do esperado), **0 lacunas** nas 5 cidades; relato defasado > 60 min: 0–1,8% das linhas; estações sem nenhuma variação: Rio 4, SP 19, Brasília 2, Recife 0, Salvador 2.

## 6. Riscos e pendências

- **Persistência:** o coletor é um processo comum (iniciado desta sessão); morre em reinício, logoff ou hibernação, e a máquina está sob carga de outros jobs (memória virtual perto do limite em 2026-10-12). Recomendado: **Agendador de Tarefas do Windows** executando `--once` a cada 5 min (a trava impede sobreposição; o script já tolera lacunas e as registra). *Não criado sem autorização.*
- **Como parar/verificar:** PID em `live\collector.pid`; `Stop-Process -Id <PID>`. Verificar vida: `python scripts/03_collect_tembici_gbfs.py --once` devolve código 2 se outra instância tem a trava.
- **Licença/termos: NÃO verificados.** O feed não declara licença nem `terms_url`; busca web não achou termos públicos. Coletar a baixa frequência para pesquisa é provavelmente aceitável, mas **redistribuir os dados brutos ou publicá-los não**, até haver resposta. Sugestão: escrever ao contato do feed pedindo autorização de uso acadêmico e de citação/publicação de agregados.
- **Calendário na janela:** 12/out (feriado), 2/nov (Finados), 15/nov (Proclamação, domingo); 2º turno das eleições gerais em 25/out *(confirmar)*. Marcar esses dias no pré-processamento; a janela é primavera, enquanto o Citi Bike é de janeiro — não comparar níveis de demanda entre as duas fontes.
- **Dias de teste (hold-out):** definir **antes** de olhar os dados (ver `PROTOCOLO_RIGOR.md` §8).
