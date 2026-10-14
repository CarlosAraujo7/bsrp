# Prompt para iniciar um novo chat que continue o BSRP

> Cole o bloco abaixo como primeira mensagem do novo chat (Claude Code com a pasta do projeto aberta). Se o novo chat **não** tiver acesso à pasta, anexe `HANDOFF.md`, `PROTOCOLO_RIGOR.md`, `DESVIOS.md` e `tables/final_cb_wd.md`.

```text
Diretório de trabalho: C:\Users\vinic\OneDrive\Documentos\Novas pesquisas\BSRP ENIAC
Idioma: português.

Você está assumindo uma pesquisa em andamento: rebalanceamento de bicicletas compartilhadas (1 veículo) com aprendizado por reforço, alvo ENIAC/BRACIS 2027, com avaliação de um possível caminho para periódico. Outro agente Claude fez o trabalho até aqui; tudo está documentado e versionado em git local.

LEIA, nesta ordem, antes de qualquer ação:
1. HANDOFF.md  (estado, resultados, mapa do repositório, prioridades P1–P7, armadilhas do ambiente)
2. PROTOCOLO_RIGOR.md e DESVIOS.md  (regras aprovadas pelo usuário; todo desvio vai no DESVIOS.md ANTES de rodar)
3. README.md e tables/final_cb_wd.md  (resultados finais com IC)
4. A memória do projeto, se estiver disponível no seu contexto.

ESTADO EM 3 LINHAS
- Experimento principal concluído (3 instâncias calibradas em Citi Bike 2024: cb4wd/cb8wd/cb12wd; 10 sementes x 5 aprendizes x 500k passos; teste em 1000 cenários sintéticos E em 24 dias reais). Resultado: DQN (rede 64-64) reduz a demanda perdida em 20–57% vs a melhor heurística calibrada; tabular e linear não superam a heurística; policy gradient é instável. Commit a0c5930; 16 testes passando.
- Coleta Tembici (GBFS, só estoque, sem O-D) roda por tarefa agendada BSRP-Tembici-GBFS; a instância Tembici AINDA NÃO foi construída; hold-out já definido (última semana completa seg-dom).
- Falta: P1 estender o espaço de busca do DQN (3 hiperparâmetros no limite), P2 ablação, P3 sensibilidade, P4 rollout + baseline da literatura, P5 instância Tembici, P6 figuras/resultados/artigo, P7 realismo para periódico.

PRIMEIROS PASSOS
a) Verifique o ambiente sem alterar nada: `git status`, `python -m unittest discover -s tests`, `Get-ScheduledTaskInfo -TaskName BSRP-Tembici-GBFS`, `python scripts/03_collect_tembici_gbfs.py --analyze`.
b) Confirme que o último commit é ddc70a9 (ou posterior) e que não há alterações pendentes. Commits terminam com a linha "Co-Authored-By: Claude ... <noreply@anthropic.com>".
c) Me apresente um plano curto para P1 e P2 (mais rápidas e que fortalecem a manchete) e espere meu OK antes de gastar horas de CPU.

REGRAS (o usuário as aprovou; siga-as)
- Rigor: unidade de análise = semente de treino; ajuste e escolha do "melhor" só em VALIDAÇÃO; teste uma vez; nada de p-valor sobre episódios; "empata" só com TOST (delta = 0,5); reporte resultados negativos; commite antes de rodar para o hash do run ficar limpo.
- Honestidade: declare limitações (simulador sem roteamento: todo deslocamento custa 1 intervalo; ocupação inicial dos dias reais desconhecida; modelo de Poisson superestima a dificuldade em 30–40% vs dias reais). Nunca escreva "DQN supera X" sem escala, nº de sementes e IC.
- Peça autorização antes de: enviar e-mails/mensagens (o e-mail à Tembici é rascunho em docs/email_tembici_rascunho.md e quem envia sou eu), publicar, baixar dados novos, criar tarefas persistentes, apagar dados.
- Nunca use meu e-mail em requisições externas (User-Agent, URLs). Não encerre processos que não sejam seus: esta máquina roda jobs pesados de outros projetos meus.

ARMADILHAS DO AMBIENTE (já vividas)
- O shell do Claude Code escreve num overlay privado de AppData\Local; só a pasta do projeto e o scratchpad são compartilhados com o Windows real. Nunca rode o coletor "de verdade" pelo shell.
- Use no máximo 3 processos meus, sem pandas nos runners; os executores já fixam OPENBLAS_NUM_THREADS=1.
- Processos longos: Start-Process python ... -WindowStyle Hidden com saída em runs\*.out; morrem se o notebook desligar; 16_tune.py e 17_pipeline_final.py são retomáveis. Não use sleep longo (use Monitor).
- Console Windows é cp1252: grave tabelas em UTF-8 ou use sys.stdout.reconfigure(errors="replace").

Pergunte-me se algo no HANDOFF estiver ambíguo antes de assumir.
```
