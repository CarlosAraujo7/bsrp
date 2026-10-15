# Avaliação: caminho para periódico (P7) — 2026-10-15

> **Isto é uma avaliação, não um plano aprovado.** Baseada nos resultados P1–P4 (todos em `HANDOFF.md`/`06_resultados.md`). Não verifiquei escopo,
> fator de impacto nem exigências de nenhum periódico específico: tudo abaixo é inferência a partir do que li (Seo 2022; Liang 2024 §1–4.1) e do
> protocolo (`PROTOCOLO_RIGOR.md` §11). **Recomendação inalterada: ENIAC/BRACIS 2027 como alvo firme; decidir periódico depois.**

## 1. O que já temos e o que é novo

| Elemento | Estado | Valor para periódico |
|---|---|---|
| Protocolo estatístico (sementes, validação/teste, IC, TOST, desvios registrados) | completo | alto: raro na literatura de BSRP |
| Instâncias calibradas em dados reais (Citi Bike 2024), sem fator de escala, com dias reais de teste | 4/8/12 estações | médio |
| Heurística calibrada forte (STR calibrada) + rollout com o modelo + aprendizes (escada de representação) | completo | alto (comparação honesta) |
| Resultado: DQN ≫ heurística calibrada, mas **rollout com o modelo ≥ DQN em 8–12 est. (sintético)**, DQN ≥ rollout em dias reais | P4 | **o achado mais interessante** |
| Ablação pareada (só o replay é necessário) | P2 | médio |
| Sensibilidade (a vantagem do DQN não é robusta a demanda baixa em 12 est.) | P3 | médio |
| Resultados negativos (A2C publicado não competitivo; seleção com 2 sementes ruidosa; ganho de validação superestima o de teste) | P1/P4 | médio (metodológico) |
| Tembici (dado brasileiro, só estoque) | pipeline pronto, janela aberta | alto se o G3 passar |

## 2. Lacunas para um periódico (da mais barata à mais cara)

| # | Lacuna | Esforço (estimativa) | Observação |
|---|---|---|---|
| 1 | ≥ 3 sistemas reais | **baixo, depende da janela**: Citi Bike + Tembici Rio + Tembici SP (pipeline já roda nas duas cidades) | exige G3 passar; o fluxo líquido é censurado e sem O–D |
| 2 | Baseline da literatura **reproduzido** (hoje: adaptação do A2C de Seo; DPRL de Liang não implementado; RTDP/SLA/DLA não) | médio: 1–3 semanas por método; exige ler Brinkmann 2019 e o restante de Liang | uma adaptação não basta; é preciso dizer o que difere |
| 3 | Escala maior (30–50 estações) | médio-alto: o espaço de ações 5n cresce; o DPRL (ação fatorada) passa a ser necessário, não opcional | o rollout fica caro; testa se o DQN escala melhor que o planejamento |
| 4 | Tempo de viagem e roteamento no simulador | alto: reescrever o ambiente (decisão na chegada, eventos em tempo contínuo) e **refazer todos os experimentos** | é o ponto em que Liang 2024 e Seo 2022 são mais realistas que nós |
| 5 | Mais de um veículo | alto: depende de 4; ação conjunta ou decisões assíncronas | o maior salto de complexidade |
| 6 | Robustez a erro de modelo (o rollout usa o λ verdadeiro) | **médio e conceitualmente forte**: demanda sobredispersa/deslocada nos cenários de teste; pesos dos aprendizes não foram salvos → retreinar | transforma o achado P4 em contribuição |

## 3. Narrativa possível (se houver investimento)

*"Quando o aprendizado supera o planejamento com modelo no rebalanceamento de bicicletas compartilhadas?"* Eixos: escala (4 a 30–50), nível de
demanda, **conhecimento do modelo** (λ verdadeiro × estimado × com erro) e dado real (Citi Bike, Tembici). Hipótese a testar: o DQN perde com o
modelo exato no sintético, mas ganha (ou empata) quando o modelo está errado, como já sugerem os dias reais (DQN ≥ rollout). **Isso ainda não foi
testado de forma controlada**; hoje é só uma observação nos dias reais (n = 24 dias, ocupação inicial desconhecida).

## 4. Riscos

- **Novidade:** parte do mérito é metodológico (rigor e resultados negativos); revisores de periódico podem querer um método novo. Mitigação: a
  narrativa da §3 e o item 6.
- **Dados:** a licença da Tembici não está verificada; o fluxo observado mistura reposição operacional e demanda (o limiar de salto muda λ em 13–26%).
- **Escopo:** itens 4 e 5 reescrevem o simulador; a validade dos resultados atuais não se transfere automaticamente.
- **Custo:** o projeto roda numa máquina compartilhada (CPU variável); estimativas de tempo têm folga larga.

## 5. Portões de decisão sugeridos (para o usuário)

1. **Depois do aceite/rejeição do ENIAC**: se aceito, as lacunas 1, 2 e 6 viram um artigo estendido de periódico (≈ 6–10 semanas); se
   rejeitado, os pareceres dizem se falta realismo (4–5) ou novidade.
2. **Antes de qualquer reescrita do simulador (itens 4–5)**: congelar os resultados atuais (tag git) e decidir se é um artigo novo.
3. **G3 do Tembici** (≥ 4 semanas completas): se não passar, Tembici entra só como estudo descritivo e a lacuna 1 fica em 2 sistemas.
