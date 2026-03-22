# Guia de Apresentacao

## Objetivo deste material

Estas notas servem como apoio para apresentacao oral. O foco e ajudar alguem a explicar o projeto com clareza, sem exagerar o que ele faz e sem esconder as limitacoes atuais.

## Explicacao curta do problema

O problema central do projeto e um problema inverso.

Em termos simples:

- o sistema gera curvas SHG de transmissao e reflexao
- queremos descobrir quais parametros opticos do material explicam essas curvas

A dificuldade e que fazer o caminho `curva -> parametro` e muito mais dificil do que fazer `parametro -> curva`.

## Explicacao curta da solucao proposta

O projeto combina tres abordagens:

1. um modelo fisico direto para simular curvas SHG
2. um fitting classico com otimizacao global para recuperar parametros
3. uma rede neural que aprende a prever parametros rapidamente

Depois, o projeto ainda testa uma quarta ideia pratica dentro da comparacao:

- usar a rede neural como chute inicial para um refinamento fisico local

## Como resumir o projeto em 30 segundos

Uma forma curta de apresentar:

"Este projeto estuda inversao de parametros em SHG. Primeiro ele usa um modelo fisico direto para simular curvas de transmissao e reflexao. Depois compara tres estrategias para recuperar os parametros a partir dessas curvas: fitting classico com `differential_evolution`, predicao direta com uma MLP e uma abordagem hibrida em que a rede neural fornece um chute inicial para um refinamento fisico local. A comparacao considera erro nos parametros, qualidade de reconstrucao das curvas e custo computacional."

## Papel da fisica

Ponto-chave para explicar:

- a fisica nao esta apenas "enfeitando" o projeto
- ela e o nucleo do sistema

O modelo fisico direto:

- recebe parametros opticos
- simula curvas `i3` e `i1`
- permite medir se uma previsao faz sentido fisicamente

Sem esse forward model, o projeto nao conseguiria:

- fazer fitting classico
- gerar dataset sintetico
- reconstruir curvas a partir das previsoes do ML

Frase curta para apresentacao:

"A fisica fornece o mecanismo que liga parametros do material ao sinal observado."

## Papel do machine learning

O ML entra para acelerar a etapa inversa.

Em vez de testar muitos conjuntos de parametros como no fitting classico, a rede aprende um mapeamento aproximado:

```text
curvas -> parametros
```

Pontos importantes:

- a rede atual e uma MLP simples
- ela aceita dados completos ou incompletos
- a entrada usa `i3`, `i1` e uma mascara informando quais curvas estao presentes

Frase curta para apresentacao:

"O machine learning atua como um acelerador do problema inverso."

## Papel do metodo hibrido

O metodo hibrido junta velocidade e refinamento.

Fluxo:

1. a MLP gera uma previsao inicial
2. essa previsao vira chute inicial
3. um otimizador local fisico refina a resposta

Ideia principal:

- o ML encontra rapidamente uma boa regiao
- o refinamento local melhora a coerencia com o modelo fisico

Frase curta para apresentacao:

"O hibrido tenta combinar a rapidez da rede com o rigor de um ajuste fisico posterior."

## Vantagens principais

### Vantagens do projeto como plataforma de pesquisa

- integra simulacao, fitting, ML, avaliacao e comparacao no mesmo repositorio
- permite medir tanto erro parametrico quanto reconstrucao fisica
- suporta cenarios com dados incompletos
- gera dataset sintetico reprodutivel com seed e bounds configuraveis

### Vantagens do metodo classico

- baseline interpretavel
- ligado diretamente ao problema inverso formulado pela fisica

### Vantagens do ML

- inferencia muito mais rapida
- adequado para alto volume de amostras

### Vantagens do hibrido

- tende a ser mais rapido que o classico puro
- busca maior fidelidade que o ML direto

## Limitacoes principais

Estas limitacoes devem ser ditas com transparencia:

- o universo de treino e sintetico
- o arquivo experimental aceito por `fit` ainda segue um formato simples de 3 colunas, embora agora tolere faltas em `i3` ou `i1`
- o split treino/validacao/teste atual e automatico, mas continua dependente da qualidade e do tamanho do dataset sintetico informado pelo usuario
- existe um ponto teorico no modulo fisico marcado com `TODO`
- a MLP atual e propositalmente simples

Frase curta para apresentacao:

"O projeto esta forte como plataforma metodologica e computacional, mas ainda pede revisao teorica pontual e ampliacao para fluxo experimental real."

## Como explicar `i3` e `i1`

Forma simples:

- `i3` representa a curva de transmissao no contexto do projeto
- `i1` representa a curva de reflexao no contexto do projeto

Voce pode dizer:

"O projeto trabalha com duas curvas observaveis, uma associada a transmissao e outra a reflexao. Essas curvas variam com a espessura e carregam informacao sobre os parametros opticos do material."

## Como explicar a avaliacao

Nao basta dizer que a rede preve os parametros. O projeto avalia duas camadas:

1. se os parametros previstos estao proximos dos parametros reais
2. se esses parametros, ao entrar novamente no modelo fisico, reconstruem bem as curvas

Essa segunda parte e muito importante para uma banca ou professor, porque mostra preocupacao com significado fisico, e nao apenas com numeros de regressao.

## Como explicar a comparacao metodologica

Uma formulacao curta:

"A comparacao entre classico, ML e hibrido serve para medir o compromisso entre precisao, coerencia fisica e custo computacional."

O que geralmente muda entre eles:

- classico: mais lento, mas fortemente ancorado no modelo fisico
- ML: mais rapido, mas dependente do dataset sintetico
- hibrido: tenta ficar no meio do caminho

## Sequencia sugerida para apresentacao

Uma ordem simples e segura:

1. apresentar o problema inverso
2. mostrar o forward model como base
3. explicar o fitting classico como baseline
4. explicar a geracao de dataset sintetico
5. explicar a rede neural e a mascara de entrada
6. explicar a avaliacao com reconstrucao fisica
7. mostrar o metodo hibrido
8. encerrar com comparacao de erro e tempo

## Perguntas provaveis e respostas curtas

### "Por que usar dados sinteticos?"

Porque o modelo fisico direto permite gerar muitas amostras rotuladas, o que viabiliza o treino inicial da rede neural.

### "A rede substitui a fisica?"

Nao. A rede aprende a partir da fisica simulada, e o projeto ainda usa o forward model para validar reconstrucao e para o metodo hibrido.

### "Por que nao usar apenas o fitting classico?"

Porque ele e computacionalmente mais caro, principalmente quando a comparacao e feita amostra por amostra em um conjunto maior.

### "Por que nao usar apenas o ML?"

Porque velocidade nao garante coerencia fisica total. Por isso o projeto tambem mede reconstrucao das curvas e testa o refinamento hibrido.

### "O projeto ja usa dados experimentais reais?"

O CLI de fitting ja aceita um arquivo externo simples com `d_nm`, `i3` e `i1`. Isso ajuda no uso pratico, mas ainda nao representa um pipeline experimental completo com tratamento instrumental e validacao mais ampla.

### "O que a mascara de entrada faz?"

Ela informa ao modelo quais curvas estao disponiveis, permitindo operar com `i3 + i1`, apenas `i3` ou apenas `i1`.

### "O que o metodo hibrido agrega?"

Ele usa a rede para encontrar rapidamente um bom ponto inicial e depois aplica um refinamento local fisico com bounds.

### "Como voces avaliam se a predicao faz sentido fisico?"

Rodamos novamente o forward model com os parametros previstos e comparamos as curvas reconstruidas com as curvas verdadeiras.

### "Ha limitacoes teoricas reconhecidas?"

Sim. Existe um ponto no termo `0k` backward marcado como `TODO` no modulo fisico, o que precisa ser comunicado com transparencia.

## Frases prontas para fechar a apresentacao

Opcao 1:

"O principal resultado do projeto nao e apenas prever parametros, mas comparar de forma estruturada tres estrategias de inversao de SHG em termos de precisao, reconstrucao fisica e custo computacional."

Opcao 2:

"O repositorio funciona como uma plataforma integrada para estudar o problema inverso em SHG, unindo simulacao fisica, otimizacao classica, machine learning e refinamento hibrido."

Opcao 3:

"O trabalho mostra que avaliar apenas erro parametrico e insuficiente; a reconstrucao fisica das curvas e essencial para interpretar corretamente o desempenho dos metodos."
