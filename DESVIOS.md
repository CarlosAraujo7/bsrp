# DESVIOS e ESPECIFICAÇÕES do protocolo (PROTOCOLO_RIGOR.md v1)

Cada linha: data · seção · o que mudou/foi especificado · por quê · se havia resultado à vista.
**Desvio** = mudou algo que o protocolo dizia. **Especificação** = o protocolo era silencioso e fixei antes de rodar.

| Data | Seção | Tipo | O que | Por quê | Resultados vistos antes? |
|---|---|---|---|---|---|
| 2026-10-12 | §8.3 | Especificação | Divisão dos dados Citi Bike por **semanas inteiras seg–dom**: índice `w=(dia−2024-01-01)//7`; `w%5∈{0,1,2}` treino, `3` val, `4` teste (16/5/5 semanas). Feriados federais dos EUA no período contam como fim de semana | preserva estrutura dia útil × fim de semana e evita tendência sazonal entre blocos; fixada em `scripts/12_prepare_citibike.py` antes de abrir os dados | não |
| 2026-10-12 | §3 | Especificação | Calibração das heurísticas: gulosa 18 configurações (limiares `lo`∈{.25,.35,.45}, `hi`∈{.6,.75,.9}, passo∈{5,10}); threshold 162 (`target`∈{.4,.5,.6}, `lo`, `hi`, passo, `w_abs`∈{0,.1,.3}); escolha por reward médio em **val** (200 cenários) | "grade de tamanho comparável" não definia as grades | não |
| 2026-10-12 | §5 | Especificação | Rollout: lookahead de `h` decisões com base = melhor heurística, `K` futuros **amostrados do modelo** (não da demanda real), mesmos futuros para todas as ações | o rollout legado consumia o gerador do ambiente real | não |
| 2026-10-12 | §2/§4 | Especificação | DQN/linear: alvo sincronizado a cada 1.000 **passos** (legado: 50 episódios); máscara de ações válidas também no alvo; replay em arrays circulares amostrado com gerador próprio (o legado usava `np.random` global sem semente); ε linear 1,0→0,05 em 70% do orçamento | orçamento em passos (§4) e reprodutibilidade (§2) | não |
| 2026-10-12 | §5 | Especificação | Tabulares R0/R1 usam espaço de ações **fixo de 5n com máscara** (o Q-learning legado indexava ações pela posição na lista de válidas) | o protocolo pede tabular "justo"; o legado tinha o artefato de aliasing de ação | não — mas um teste de fumaça de 6 mil passos já mostrou R0 ≈ −3,5, bem acima do legado (−7,1); registrado aqui como motivo de cautela |
