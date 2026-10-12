# Rascunho — pedido de autorização de uso acadêmico dos feeds GBFS (Tembici / Lyft)

> **Rascunho. Não foi enviado.** Quem envia é o usuário. Preencher os campos entre [colchetes]. Destinatário: o `feed_contact_email` publicado nos feeds (`mobility-data-client@lyft.com`, em `system_information.json`). Considere copiar a Tembici (canal institucional que você preferir).
> Contexto: os feeds não declaram licença nem `terms_url`; a coleta atual é de 1 requisição por cidade a cada 5 min, com User-Agent identificado como coletor acadêmico.

---

## Português

**Assunto:** Solicitação de autorização para uso acadêmico dos feeds GBFS do Bike Itaú (Tembici)

Prezados,

Sou [nome], [cargo/curso] na [instituição], e desenvolvo uma pesquisa sobre rebalanceamento de estações em sistemas de bicicletas compartilhadas com aprendizado por reforço, a ser submetida ao [ENIAC/BRACIS 2027].

Desde 11/out/2026 coleto, para fins exclusivamente acadêmicos, o `station_status` dos feeds GBFS v3.0 públicos de Rio de Janeiro, São Paulo, Brasília, Recife e Salvador (`https://<cidade>.publicbikesystem.net/customer/gbfs/v3.0/`), a cada 5 minutos (uma requisição por cidade por rodada), por cerca de [8–12] semanas. Guardo apenas os campos do feed (estoque de bicicletas e docas por estação).

Como os feeds não indicam licença ou termos de uso, gostaria de saber:

1. Esse uso (coleta a baixa frequência para pesquisa acadêmica) é permitido?
2. Posso **publicar resultados agregados** (por exemplo, curvas de ocupação média por hora e desempenho de políticas simuladas), citando a Tembici como fonte?
3. Posso **compartilhar os dados brutos** coletados com revisores ou em um repositório de reprodutibilidade? Se não, aceitam que eu compartilhe apenas versões agregadas?
4. Existe alguma restrição de frequência de requisições que eu deva respeitar, ou um canal preferencial para pesquisadores?
5. A Tembici disponibiliza, para pesquisa, dados de viagens (origem–destino) além do estoque por estação?

Fico à disposição para detalhar a metodologia. Agradeço desde já.

Atenciosamente,
[nome] · [instituição] · [e-mail institucional]

---

## English

**Subject:** Request for permission: academic use of the Bike Itaú (Tembici) GBFS feeds

Dear team,

I am [name], [position] at [institution], working on bike-share station rebalancing with reinforcement learning, to be submitted to [ENIAC/BRACIS 2027].

Since 2026-10-11 I have been collecting, for academic purposes only, `station_status` from the public GBFS v3.0 feeds for Rio de Janeiro, São Paulo, Brasília, Recife and Salvador (`https://<city>.publicbikesystem.net/customer/gbfs/v3.0/`), once every 5 minutes (one request per city per round) for about [8–12] weeks, storing only the feed fields (bike and dock counts per station).

Because the feeds declare no license or terms of use, could you please confirm:

1. Is this use (low-frequency collection for academic research) permitted?
2. May I publish **aggregate results** (e.g., average hourly occupancy curves and performance of simulated policies), citing Tembici as the data source?
3. May I share the **raw collected data** with reviewers or in a reproducibility repository, or only aggregated versions?
4. Is there a request-rate limit I should respect, or a preferred channel for researchers?
5. Does Tembici make trip-level (origin–destination) data available for research, beyond per-station stock?

Happy to provide more detail on the methodology. Thank you.

Kind regards,
[name] · [institution] · [institutional email]
