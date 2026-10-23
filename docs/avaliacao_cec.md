# Avaliação crítica do artigo para o IEEE CEC 2027 (0–100)

Escrita por quem desenvolveu o trabalho (viés a favor do projeto): o número é uma estimativa conservadora, não a de um revisor independente. Rascunho avaliado: `artigo_cec/main.tex`
(commit 085feae + referências de computação evolutiva para o reposicionamento; 7 páginas com referências, limite 8).

## Régua (peso) e nota atual

| critério | peso | nota | por quê |
|---|---|---|---|
| Relevância para o CEC | 15% | 70 | há ES, CMA-ES, RHEA e um híbrido evolução + aprendizado, mas o tema é aplicação e a maioria dos resultados evolutivos é negativa ou neutra |
| Solidez técnica e rigor | 25% | 88 | unidade de análise = semente, escolha só em validação, teste uma vez, TOST, resultados negativos, ~100 checagens de afirmações contra os dados, DQN reproduzidos exatamente |
| Originalidade | 20% | 45 | comparação controlada sem algoritmo novo; o híbrido é simples (valor congelado) e o resultado é misto; há trabalhos de EC em reposicionamento (agora citados) |
| Significância | 15% | 50 | simulador próprio, uma cidade, um veículo, sem tempo de viagem; planejadores usam o modelo verdadeiro |
| Clareza e formato | 10% | 68 | formato IEEE correto (A4, 2 colunas, anônimo, fontes embutidas); texto escrito por IA, sem revisão humana; falta divulgação de IA |
| Reprodutibilidade | 5% | 80 | tudo gerado dos logs; sem link por causa do duplo-anônimo |

**Nota global estimada: ≈ 66–68/100 após o P13; ≈ 68–70/100 após o P14; ≈ 69–71/100 após o P15 (segundo algoritmo de neuroevolução)** (antes do P12 era ≈ 50). Faixa de decisão do CEC: abaixo de ~60 tende a rejeição, 60–72 é incerto (depende dos revisores),
acima de ~75 é aceitação provável. **Meta de "trabalho satisfatório": ≥ 75.**

## Fraquezas, em ordem de ganho esperado por custo

1. **~~Faltam controles de computação evolutiva~~ (feito no P14, 23/out: a evolução ajuda o RHEA a partir de 12 est. e não ajuda a calibração de 3–5 parâmetros; o segundo algoritmo de neuroevolução (sep-CMA-ES, P15) também foi feito: a conclusão negativa vale para dois algoritmos)** (ganho ~+3, custo baixo). Um revisor de EC perguntará: o RHEA evolui de verdade ou uma busca aleatória com o mesmo número de avaliações faz o mesmo? O CMA-ES
   supera a busca aleatória com 200 avaliações? O ES é fraco por ser o OpenAI-ES, ou toda neuroevolução de pesos falha aqui? → P14: busca aleatória (mesmo orçamento) para o RHEA e para a calibração; um segundo
   algoritmo de neuroevolução (ES separável com covariância diagonal).
2. **Validade externa: uma cidade** (ganho ~+4–5, custo médio, **exige permissão para baixar dados novos**). Uma segunda cidade com dados abertos (ex.: Capital Bikeshare, Washington) pelo mesmo pipeline.
3. **Originalidade** (ganho ~+4, custo alto). Uma contribuição algorítmica mais forte que o valor congelado: p.ex. destilar o planejador no DQN (expert iteration com RHEA/rollout) ou treinar valor e busca juntos.
4. **Poder estatístico** (ganho ~+2, custo médio): 5 sementes em 20/30 estações e 3 nos orçamentos longos do ES.
5. **Clareza e divulgação de IA** (ganho ~+4, **só o usuário pode fazer**): revisão humana integral do texto, decisão sobre a divulgação de IA conforme a política do CEC.
6. Roteamento com tempo de viagem e vários veículos: fora do escopo do prazo (fica em limitações).

## Ordem de ataque

P14 (controles evolutivos) → decisão do usuário sobre a segunda cidade → sementes extras → (P15: destilação/valor conjunto, se o tempo permitir) → revisão humana. Reavaliar a nota a cada ciclo.
