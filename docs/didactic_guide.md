# Guia Didatico

## Para que serve este guia

Este texto foi escrito para quem precisa entender o projeto sem partir de uma base profunda em optica nao linear, simulacao numerica ou machine learning.

A ideia e responder:

- o que o projeto estuda
- por que as curvas sao importantes
- o que a simulacao faz
- o que o fitting faz
- o que a rede neural aprende
- por que existe um metodo hibrido

## O que e SHG neste projeto

SHG significa `Second-Harmonic Generation`, ou geracao de segundo harmonico.

Em termos intuitivos:

- entra luz em uma frequencia fundamental
- o material pode responder de forma nao linear
- parte da resposta aparece em uma frequencia dobrada

No projeto, essa resposta e representada por curvas simuladas ao longo da espessura do filme.

O objetivo pratico nao e explicar toda a teoria de SHG, e sim usar um modelo especifico para responder uma pergunta inversa:

"Quais parametros opticos do material poderiam ter gerado as curvas observadas?"

## O que sao `i3` e `i1`

No contexto atual do projeto:

- `i3` e tratada como curva de transmissao
- `i1` e tratada como curva de reflexao

Essas curvas variam com a espessura `d_nm`.

Em linguagem simples:

- cada ponto da curva mostra quanta resposta SHG aparece para uma certa espessura
- o formato da curva depende dos parametros opticos do material

Por isso, as curvas funcionam como uma especie de "assinatura" do sistema.

## O que significam `2k` e `0k`, de forma intuitiva

No codigo, a simulacao separa contribuicoes chamadas `2k` e `0k`.

Sem entrar em formalismo pesado, uma leitura intuitiva e:

- `2k`: termo associado a uma contribuicao oscilatoria ligada ao acoplamento com a fase da onda fundamental
- `0k`: termo associado a uma contribuicao mista que nao carrega o mesmo tipo de oscilacao espacial

Em outras palavras:

- o modelo soma efeitos diferentes que participam da geracao SHG
- a curva final vem da soma dessas partes

Importante:

- este significado aqui e didatico
- a interpretacao teorica detalhada depende do modelo fisico adotado
- o proprio codigo marca um ponto do termo `0k` backward com `TODO` para revisao teorica

## O que e um modelo fisico direto

Um modelo fisico direto, ou `forward model`, responde:

"Se eu souber os parametros do material, o que devo observar?"

No projeto:

- voce escolhe `lambda_m`
- escolhe `n21w`, `k21w`, `n22w`, `k22w`
- escolhe um vetor de espessuras `d_nm`
- roda `simulate_shg(...)`
- recebe `i3` e `i1`

Entao o forward model faz o caminho:

```text
parametros -> curvas
```

Esse e o bloco mais importante do projeto, porque todo o resto depende dele.

## O que e um problema inverso

O problema inverso faz a pergunta oposta:

"Se eu conheco as curvas, consigo recuperar os parametros?"

Aqui o caminho fica:

```text
curvas -> parametros
```

Esse caminho e mais dificil, porque:

- solucoes diferentes podem produzir curvas parecidas
- o espaco de busca tem varias dimensoes
- ha risco de ambiguidades e regioes ruins numericamente

## O que o fitting classico faz

O fitting classico testa varios conjuntos de parametros e mede quais geram curvas mais parecidas com as curvas observadas.

No projeto:

1. o otimizador sugere um vetor de parametros
2. o forward model simula `i3` e `i1`
3. a funcao objetivo calcula o erro
4. o otimizador tenta diminuir esse erro

O baseline principal e:

- `differential_evolution`

Por que usar esse metodo?

- ele faz busca global
- reduz a chance de ficar preso logo no primeiro minimo local

Desvantagem:

- costuma ser lento, especialmente quando se avaliam muitas amostras

Observacao pratica importante para laboratorio:

- o fit so e confiavel se a janela de busca dos parametros fizer sentido para o material real
- por isso o projeto agora permite ajustar bounds diretamente no CLI do `fit`
- se o melhor resultado encosta no limite minimo ou maximo de um parametro, isso costuma ser um sinal de que a faixa precisa ser revisada

## Por que gerar dados sinteticos

Para treinar um modelo de ML, normalmente sao necessarias muitas amostras.

Como o projeto ja tem um modelo fisico direto, ele pode gerar seus proprios exemplos:

1. sorteia parametros dentro de bounds
2. simula curvas
3. guarda as curvas junto com os parametros verdadeiros

Isso cria um dataset sintetico.

Vantagem:

- o projeto sabe exatamente quais parametros geraram cada curva

Cuidado:

- a rede aprende o mundo que voce simulou
- se os bounds forem estreitos ou pouco realistas, o aprendizado tambem sera limitado

## O que a rede neural aprende

A rede neural do projeto e uma MLP simples.

Ela recebe:

- curva `i3`
- curva `i1`
- uma mascara dizendo quais curvas estao presentes

E tenta prever:

- `n21w`
- `k21w`
- `n22w`
- `k22w`

Em termos intuitivos, a rede aprende um atalho:

```text
curvas -> parametros provaveis
```

Isso nao substitui a fisica. Na pratica:

- a rede aprende padroes a partir das simulacoes
- se o conjunto sintetico estiver bem construido, ela pode prever muito mais rapido que o fitting classico

## Por que o projeto suporta dados incompletos

Nem sempre as duas curvas podem estar disponiveis da mesma forma.

Por isso o projeto foi preparado para tres cenarios:

- `i3 + i1`
- apenas `i3`
- apenas `i1`

Isso e feito com uma mascara de entrada:

- `[1, 1]`: ambas disponiveis
- `[1, 0]`: apenas `i3`
- `[0, 1]`: apenas `i1`

Durante o treino, o projeto remove aleatoriamente um dos canais em parte das amostras. Assim, a rede aprende a continuar operando mesmo com informacao incompleta.

Para dado de laboratorio, isso significa:

- voce pode deixar uma celula vazia quando um ponto de `i3` ou `i1` nao foi medido
- no fitting fisico, esses pontos ausentes sao simplesmente ignorados no calculo do erro
- no `ml` e no `hybrid`, lacunas internas sao interpoladas apenas para montar a entrada da rede; o erro fisico continua sendo avaliado apenas onde ha observacao real

## Por que avaliar reconstrucao fisica alem do erro dos parametros

Um modelo pode acertar "mais ou menos" os parametros e ainda assim reconstruir bem as curvas.

Ou pode acontecer o contrario:

- os numeros previstos parecem razoaveis
- mas quando entram no forward model, as curvas reconstruidas ficam ruins

Por isso o projeto mede duas coisas:

1. erro nos parametros
2. erro de reconstrucao das curvas

Essa segunda etapa e cientificamente importante porque pergunta:

"Os parametros previstos sao apenas numericamente proximos, ou eles realmente explicam bem o comportamento fisico observado?"

## O que e o metodo hibrido

O metodo hibrido junta duas ideias:

1. a rede neural faz uma previsao rapida
2. um refinamento fisico local melhora essa previsao

No projeto, isso ocorre assim:

```text
curvas
-> MLP
-> chute inicial de parametros
-> otimizacao local com bounds
-> parametros refinados
```

Intuicao:

- o ML da velocidade
- a fisica local tenta recuperar precisao fina

Mas isso so funciona bem quando:

- o dataset sintetico foi gerado com bounds compativeis com o experimento
- a malha de espessuras do treino se parece com a malha do laboratorio
- o experimento nao esta muito fora do dominio coberto pelas simulacoes sinteticas

## Por que comparar classico, ML e hibrido

Cada abordagem tem um perfil diferente:

### Metodo classico

Ponto forte:

- mais fiel ao problema inverso formulado explicitamente

Ponto fraco:

- mais lento

### Metodo de ML

Ponto forte:

- muito rapido na inferencia

Ponto fraco:

- depende fortemente do dataset sintetico usado no treino

### Metodo hibrido

Ponto forte:

- tenta equilibrar rapidez e refinamento fisico

Ponto fraco:

- ainda depende da qualidade da previsao inicial e do refinamento local

## Como adaptar o projeto para o melhor fit possivel em dado real

Para aproximar o melhor fit experimental possivel, o usuario deve tratar o projeto como um pipeline adaptavel, nao como uma caixa-preta.

As decisoes mais importantes sao:

- formato do CSV experimental
  use um arquivo com 3 colunas nomeadas `d_nm`, `i3` e `i1`; se houver cabecalho, `i3` e `i1` podem aparecer em qualquer ordem
- qualidade da malha experimental
  quanto melhor a distribuicao de pontos nas regioes de maior variacao da curva, melhor a identificacao dos parametros
- escolha dos bounds
  bounds muito estreitos podem forcar um parametro para a borda; bounds muito largos podem aumentar ambiguidades e tempo de busca
- pesos por canal
  o projeto agora permite aumentar ou reduzir a influencia de `i3` e `i1` na funcao objetivo, o que ajuda quando um canal esta mais ruidoso que o outro
- coerencia entre treino e experimento
  se voce usar ML ou metodo hibrido, o dataset sintetico deve refletir a mesma malha `d_nm`, a mesma faixa de parametros e uma normalizacao compativel com o laboratorio

Em termos praticos, um bom procedimento e:

1. rodar primeiro o fit classico
2. ajustar bounds e pesos ate obter um erro baixo sem parametros presos nas bordas
3. gerar dataset sintetico na malha experimental
4. treinar a rede apenas depois de decidir bounds fisicamente plausiveis
5. comparar `classical`, `ml` e `hybrid` no mesmo experimento

Esse cuidado e importante porque, em problemas inversos, uma curva pode parecer bem ajustada mesmo quando os parametros recuperados ainda nao estao bem identificados.

Comparar os tres ajuda a responder uma pergunta pratica importante:

"Qual abordagem entrega o melhor compromisso entre custo computacional e qualidade fisica?"

## O que e decisao fisica, computacional e de machine learning

### Decisao fisica

E uma escolha sobre como representar o fenomeno no modelo.

Exemplos:

- como `simulate_shg(...)` foi formulado
- quais parametros opticos entram no problema
- como as contribuicoes `2k` e `0k` aparecem

### Decisao computacional

E uma escolha de implementacao numerica e de software.

Exemplos:

- usar `numpy`
- salvar datasets em `.npz`
- colocar validacoes numericas para evitar `NaN` e `inf`
- usar `argparse` no CLI

### Decisao de machine learning

E uma escolha sobre representacao de entrada, modelo e treino.

Exemplos:

- usar MLP simples
- usar mascara para dados incompletos
- fazer augmentacao removendo canais
- avaliar cenarios `i3_only` e `i1_only`

## Limitacoes e cuidados ao interpretar resultados

Este projeto e util para pesquisa e estudo, mas os resultados precisam ser lidos com cuidado.

### 1. O universo de treino e sintetico

Isso significa:

- a rede aprende a partir do proprio modelo
- ela ainda depende fortemente de quao representativo e o dataset sintetico usado
- aceitar um arquivo externo no `fit` nao significa que o projeto ja resolva toda a variabilidade de um experimento real

### 2. O fitting aceita arquivo externo, mas o fluxo experimental ainda e simples

Hoje o comando `fit`:

- pode carregar um arquivo externo com 3 colunas numericas: `d_nm`, `i3`, `i1`
- aceita que uma linha tenha apenas `i3`, apenas `i1` ou ambos, desde que `d_nm` exista
- usa um pequeno conjunto interno de exemplo apenas como fallback, quando nenhum arquivo e informado

Isso facilita o uso pratico, mas ainda nao equivale a um pipeline experimental completo com pre-processamento, ruído instrumental, calibracao e validacao metrologica.

### 3. Existe um ponto teorico marcado para revisao

No modulo fisico ha um `TODO` sobre um termo `0k` backward.

Isso nao invalida o projeto como estrutura computacional, mas exige transparencia ao apresentar resultados cientificos.

### 4. Bounds influenciam fortemente o problema

Os bounds usados para gerar dataset e para fitting definem o espaco de busca.

Se esses limites forem ruins:

- o fitting pode procurar no lugar errado
- a rede vai aprender um problema mal definido

### 5. Erro pequeno de parametro nao e a unica medida importante

Sempre vale conferir tambem:

- reconstrucao de `i3`
- reconstrucao de `i1`
- robustez quando falta um canal
- custo computacional

## Resumo intuitivo

Se fosse explicar o projeto em poucas frases:

- o forward model diz que curvas esperar para certos parametros
- o fitting classico tenta recuperar esses parametros testando varias possibilidades
- a rede neural aprende a fazer isso de forma muito mais rapida
- o metodo hibrido usa a velocidade da rede e o refinamento da fisica
- a avaliacao completa compara nao apenas os parametros previstos, mas tambem se as curvas reconstruidas fazem sentido
