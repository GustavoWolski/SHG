# Glossario

## SHG

`Second-Harmonic Generation`. E um fenomeno de optica nao linear em que a resposta do material gera sinal em frequencia dobrada em relacao a luz fundamental.

## Forward model

Modelo fisico direto. Recebe parametros do sistema e devolve o que se espera observar. Neste projeto, recebe parametros opticos e devolve curvas `i3` e `i1`.

## Inverse problem

Problema inverso. Faz o caminho contrario do forward model: parte das curvas observadas e tenta recuperar os parametros que as geraram.

## Fitting

Processo de ajuste numerico em que um otimizador tenta encontrar parametros que minimizem a diferenca entre curvas simuladas e curvas observadas.

## Parametro optico

Quantidade que descreve propriedades opticas do material. Neste projeto, os parametros ajustados sao `n21w`, `k21w`, `n22w` e `k22w`.

## Indice complexo

Forma de representar a resposta optica do material usando parte real e parte imaginaria. A parte real esta ligada a propagacao; a parte imaginaria, de forma simplificada, a perdas ou absorcao.

## Transmissao

Parte do sinal associada ao que atravessa o sistema. No projeto, a curva `i3` e tratada como transmissao.

## Reflexao

Parte do sinal associada ao que retorna do sistema. No projeto, a curva `i1` e tratada como reflexao.

## Dataset sintetico

Conjunto de dados gerado por simulacao, e nao medido diretamente em experimento real. Aqui ele contem curvas SHG simuladas e os parametros verdadeiros usados para gera-las.

## MLP

`Multi-Layer Perceptron`. Tipo simples de rede neural feedforward. No projeto, a MLP recebe curvas e mascara de entrada e tenta prever os quatro parametros fisicos.

## Inferencia

Etapa em que um modelo treinado e usado para fazer previsoes em novos dados. Aqui significa usar a MLP para prever parametros a partir das curvas.

## Reconstrucao fisica

Teste em que os parametros previstos sao inseridos novamente no forward model para reconstruir curvas SHG e compara-las com as curvas verdadeiras. Isso verifica coerencia fisica da predicao.

## Baseline

Metodo de referencia usado como comparacao principal. Neste projeto, o baseline do problema inverso fisico e o fitting classico com `differential_evolution`.

## Metodo hibrido

Abordagem que combina ML e fisica. No projeto, a MLP fornece uma previsao inicial e um refinamento fisico local melhora essa estimativa.

## Mascara de entrada

Vetor pequeno que informa quais curvas estao presentes na entrada do modelo. Exemplo: `[1, 1]` significa ambas; `[1, 0]`, apenas `i3`; `[0, 1]`, apenas `i1`.

## Overfitting

Situacao em que o modelo aprende muito bem os dados de treino, mas nao generaliza bem para novos dados. Em datasets sinteticos, isso pode acontecer se a avaliacao nao for separada corretamente.

## Generalizacao

Capacidade do modelo de funcionar bem em dados que ele nao viu durante o treino.

## Bounds

Limites minimo e maximo usados para cada parametro. Eles definem o espaco de busca do fitting e o espaco de amostragem do dataset sintetico.

## Differential evolution

Otimizador global usado no fitting classico. Ele testa e combina candidatos ao longo de varias iteracoes para buscar menor erro.

## L-BFGS-B

Metodo de otimizacao local com suporte a bounds. No projeto, ele e usado no refinamento da abordagem hibrida.

## Seed

Numero usado para tornar processos aleatorios reprodutiveis. Aqui aparece no sorteio de datasets, no treino da rede e no fitting classico.

## Curva SHG

Sinal SHG em funcao da espessura do filme. No projeto, as curvas principais sao `i3(d)` e `i1(d)`.

## Problema fisicamente sensivel

Situacao em que o modelo pode ficar numericamente instavel ou teoricamente delicado, por exemplo perto de denominadores muito pequenos ou regioes de ressonancia.

## Normalizacao global

Forma de normalizar curvas usando uma unica escala conjunta para transmissao e reflexao.

## Normalizacao separada

Forma de normalizar curvas tratando transmissao e reflexao com escalas independentes.
