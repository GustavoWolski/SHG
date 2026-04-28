# Arquitetura do Projeto

## Objetivo deste documento

Este arquivo explica como o projeto esta organizado hoje, com base nos modulos reais existentes no codigo. A ideia e mostrar:

- qual e o papel de cada modulo
- como os dados circulam
- onde entra a fisica
- onde entra o problema inverso
- onde entra o machine learning
- como a comparacao metodologica e montada

Este documento descreve o estado atual do repositorio. Quando algo ainda estiver incompleto ou simplificado, isso sera dito explicitamente.

## Visao geral da arquitetura

O projeto pode ser lido em 5 blocos principais:

1. `physics`
   - contem o modelo fisico direto de SHG
2. `inverse`
   - contem a funcao objetivo e os otimizadores para recuperar parametros
3. `data`
   - gera e carrega datasets sinteticos
4. `ml`
   - prepara dados, define a MLP, treina, avalia e compara metodos
5. `main` e `utils`
   - fazem a integracao por CLI, graficos e funcoes auxiliares

Em forma de fluxo:

```text
Parametros fisicos
-> forward model SHG
-> curvas i3 e i1
-> uso em fitting classico ou geracao de dataset
-> treino e avaliacao de ML
-> comparacao entre classico, natural, ML e hibrido
```

## Estrutura real dos modulos

### Camada fisica

#### `src/physics/constants.py`

Guarda constantes fisicas usadas pelo projeto.

Papel:

- centralizar valores constantes
- evitar repeticao no codigo

#### `src/physics/optics.py`

Contem funcoes opticas auxiliares.

Papel:

- calcular coeficientes de Fresnel
- fornecer o indice do vidro (`nlimeglass`)

Essas funcoes sao usadas pelo forward model.

#### `src/physics/shg_model.py`

E o nucleo fisico do projeto.

Papel:

- definir a dataclass `SHGParams`
- executar a simulacao direta com `simulate_shg(...)`
- validar entradas e estabilidade numerica
- expor `default_shg_params()`
- expor `validate_default_simulation(...)`

Saida principal:

- `i3`: curva tratada no projeto como transmissao
- `i1`: curva tratada no projeto como reflexao

Observacao importante:

- o modulo contem comentarios sobre pontos fisicamente sensiveis
- existe um `TODO` explicito pedindo revisao teorica de um termo `0k` backward

## Camada do problema inverso

#### `src/inverse/objective.py`

Implementa a ligacao entre simulacao fisica e otimizacao.

Papel:

- transformar um vetor numerico de parametros em `SHGParams`
- normalizar curvas experimentais e simuladas
- calcular o erro entre experimento e simulacao com `error_function(...)`

Decisao importante:

- o projeto permite duas estrategias de normalizacao:
  - `global`
  - `separate`

Em termos conceituais:

```text
parametros candidatos
-> forward model
-> curvas simuladas
-> comparacao com curvas experimentais
-> valor de erro
```

#### `src/inverse/fitters.py`

Implementa os otimizadores usados na inversao.

Papel:

- rodar o baseline classico com `run_fit(...)`
- encapsular o resultado em `FitResult`
- simular curvas a partir do melhor ajuste
- fazer refinamento local com `refine_fit_locally(...)`

Otimizadores usados hoje:

- global: `scipy.optimize.differential_evolution`
- global por computacao natural: `scipy.optimize.dual_annealing`
- local: `scipy.optimize.minimize(..., method="L-BFGS-B")`

#### `src/inverse/methods.py`

Orquestra os metodos inversos sobre uma unica curva experimental.

Papel:

- rodar o metodo classico em uma amostra experimental
- rodar o metodo natural em uma amostra experimental
- rodar a predição direta da MLP
- rodar o metodo hibrido com refinamento local
- comparar os metodos e escolher o melhor pelo erro observado

Importante:

- esse modulo nao altera o forward model
- quando o experimento tem faltas em pontos isolados de `i3` ou `i1`, a entrada da MLP e completada por interpolacao linear
- a saida direta do metodo `ml` e recortada aos bounds fisicos do projeto
- o erro fisico continua sendo calculado somente sobre os pontos realmente medidos

## Camada de dados

#### `src/data/synthetic_generator.py`

Gera amostras sinteticas usando o modelo fisico direto.

Papel:

- sortear parametros dentro de bounds
- rodar `simulate_shg(...)`
- opcionalmente normalizar curvas
- organizar arrays em formato adequado para ML
- salvar em `.npz`

Estrutura principal do dataset sintetico:

- `d_nm`
- `i3`
- `i1`
- `curves`
- `parameters`
- `lambda_m`
- `bounds`
- `normalization`
- `seed`

#### `src/data/loaders.py`

Carrega dados a partir de arquivo.

Papel:

- carregar colunas numericas de texto com `load_columns(...)`
- carregar dataset sintetico salvo com `load_synthetic_dataset(...)`
- carregar um arquivo experimental simples com `load_experimental_shg_data(...)`

Importante:

- o loader experimental espera 3 colunas: `d_nm`, `i3`, `i1`
- `i3` e `i1` podem ter amostras faltantes; o fitting usa mascaras para ignorar os pontos ausentes
- `src/main.py` usa esse loader quando `fit` recebe `--data-path`
- se `--data-path` nao for informado, o CLI ainda tem um pequeno conjunto interno de fallback

## Camada de machine learning

#### `src/ml/datasets.py`

Converte o dataset sintetico para o formato usado pela rede.

Papel:

- definir `SHGDataset`
- construir mascaras de observacao
- concatenar entrada como `[i3, i1, mascara]`
- criar cenarios com apenas `i3`, apenas `i1` ou ambos
- dividir o dataset em treino, validacao e teste com utilitarios de split
- salvar o resumo do split para rastreabilidade

Essa camada e a base para o suporte a dados incompletos.

#### `src/ml/models.py`

Define o modelo de regressao.

Papel:

- definir `ModelConfig`
- definir `MLPRegressor`
- criar a rede com `build_model(...)`
- salvar modelo em `.npz`
- carregar modelo de `.npz`

Arquitetura atual:

- MLP simples
- implementada em `numpy`
- entrada: curvas concatenadas + mascara
- saida: quatro parametros fisicos

#### `src/ml/train.py`

Executa o treino da MLP.

Papel:

- definir configuracoes de treino
- aplicar augmentacao com mascaras aleatorias
- normalizar entradas e alvos
- treinar com mini-batches
- usar Adam e gradient clipping
- acompanhar perda de validacao
- restaurar o melhor conjunto de pesos observado no treino
- salvar resumo do treino em JSON

Importante:

- o treino pode ser chamado via CLI (`train-ml`) ou via API Python
- o CLI atual faz o split automatico em treino, validacao e teste antes do treino
- ao final, o pipeline roda avaliacao automatica nos subconjuntos de validacao e teste

#### `src/ml/evaluate.py`

Avalia o modelo treinado.

Papel:

- prever parametros em tres cenarios de entrada
- calcular metricas por parametro
- reconstruir curvas usando o forward model
- calcular metricas fisicas de reconstrucao
- salvar figuras e resumo em JSON

Cenarios avaliados:

- `i3_i1`
- `i3_only`
- `i1_only`

#### `src/ml/compare.py`

Compara quatro abordagens de inversao:

1. classica
2. natural
3. ML direta
4. hibrida

Papel:

- medir erro parametrico
- medir erro de reconstrucao fisica
- medir tempo computacional
- salvar figuras comparativas
- salvar resumo em JSON e CSV

## Camada de interface e utilitarios

#### `src/utils/plotting.py`

Gera graficos para simulacao e fitting classico.

Papel:

- plotar curvas SHG
- plotar comparacao experimento vs simulacao
- plotar mapa de erro

Observacao:

- esses graficos sao exibidos na tela
- eles nao sao salvos automaticamente por `simulate` e `fit`

#### `src/utils/io.py`

Funcoes pequenas de I/O.

Papel:

- garantir que diretorios existam
- salvar arrays em colunas

#### `src/main.py`

E a porta de entrada principal do projeto.

Papel:

- registrar subcomandos do CLI
- despachar o fluxo correto
- ligar entrada do usuario aos modulos internos
- carregar dados experimentais externos para `fit`
- coordenar o pipeline de split, treino e avaliacao para `train-ml`

Subcomandos reais hoje:

- `simulate`
- `fit`
- `generate-dataset`
- `train-ml`
- `evaluate-ml`
- `compare-methods`

#### `main.py`

Wrapper fino para manter um ponto de execucao simples na raiz do projeto.

## Fluxo de dados

### 1. Fluxo do forward model

```text
SHGParams + d_nm
-> simulate_shg(...)
-> i3 e i1
-> plots, fitting, dataset sintetico ou reconstrucao
```

### 2. Fluxo do fitting classico

```text
curvas experimentais
-> error_function(...)
-> differential_evolution
-> melhor vetor de parametros
-> FitResult
-> curvas simuladas do melhor ajuste
```

### 3. Fluxo de geracao de dataset

```text
bounds fisicos + seed + d_nm
-> sorteio de parametros
-> forward model
-> amostras sinteticas
-> arquivo .npz
```

### 4. Fluxo de ML

```text
dataset sintetico
-> SHGDataset
-> [i3, i1, mascara]
-> MLP
-> parametros previstos
```

### 4a. Fluxo do treinamento via CLI

```text
dataset sintetico em .npz
-> train-ml
-> load_synthetic_dataset(...)
-> from_synthetic_dataset(...)
-> ModelConfig + TrainingConfig
-> train_model(...)
-> modelo .npz + resumo JSON
```

### 5. Fluxo de avaliacao

```text
modelo treinado + dataset de teste
-> previsao de parametros
-> metricas parametricas
-> forward model com parametros previstos
-> curvas reconstruidas
-> metricas fisicas + figuras
```

### 6. Fluxo da comparacao metodologica

```text
mesmo conjunto de teste
-> metodo classico
-> metodo ML
-> metodo hibrido
-> comparacao de erro + reconstrucao + tempo
```

## Papel do forward model

O forward model responde:

"Se os parametros do material forem estes, que curvas SHG eu espero observar?"

Ele e a base de quase todo o projeto:

- gera curvas para estudo direto
- alimenta o fitting classico
- gera datasets sinteticos
- reconstrui curvas a partir das predicoes da rede
- permite comparar fidelidade fisica entre metodos

Sem esse modelo, o projeto perderia a ponte entre parametro fisico e curva observada.

## Papel do problema inverso

O problema inverso faz a pergunta oposta:

"Se eu observo as curvas, quais parametros podem ter gerado essas curvas?"

No projeto, isso aparece de duas formas:

- explicitamente no fitting classico com otimizador
- implicitamente no ML, que aprende um atalho estatistico entre curva e parametro

## Onde entra o machine learning

O ML entra depois que o projeto consegue gerar muitas simulacoes sinteticas.

Papel do ML:

- aprender a mapear curva -> parametro
- acelerar a inferencia
- tolerar cenarios com dados incompletos via mascara

Limite importante:

- o modelo aprende a partir do universo sintetico que o proprio projeto gerou
- se os bounds ou a formulacao fisica forem inadequados, a rede aprende esse universo limitado

## Onde entra a comparacao metodologica

A comparacao metodologica existe para responder, com numeros:

- qual metodo erra menos nos parametros
- qual metodo reconstrui melhor as curvas
- qual metodo e mais rapido

Ela usa os modulos:

- `src/inverse/fitters.py`
- `src/ml/models.py`
- `src/ml/evaluate.py`
- `src/ml/compare.py`

## Como a abordagem hibrida funciona

No projeto atual, o metodo hibrido funciona assim:

```text
curvas de entrada
-> MLP faz uma previsao inicial
-> essa previsao vira chute inicial
-> um refinamento local fisico ajusta os parametros
-> resultado final fica dentro de bounds validos
```

Intuicao:

- o ML reduz o tempo de busca
- a etapa fisica local tenta recuperar rigor perto de uma boa regiao do espaco de parametros

## Decisoes fisicas, computacionais e de ML

### Decisoes fisicas

Sao escolhas ligadas ao modelo de SHG e aos parametros opticos.

Exemplos no projeto:

- formulacao do `simulate_shg(...)`
- uso de `n21w`, `k21w`, `n22w`, `k22w`
- tratamento de `i3` e `i1`
- termos `2k` e `0k`
- uso de indice do vidro via `nlimeglass(...)`

### Decisoes computacionais

Sao escolhas de implementacao numerica e organizacao de software.

Exemplos:

- guardas contra divisao instavel
- penalizacao grande em casos numericamente invalidos
- salvamento em `.npz`
- uso de `argparse`
- calculo e salvamento de metricas em JSON e CSV

### Decisoes de machine learning

Sao escolhas sobre representacao de dados, treino e avaliacao do modelo.

Exemplos:

- entrada concatenada com mascara
- MLP simples
- augmentacao removendo canais
- avaliacao em tres cenarios de observacao
- comparacao entre predicao direta e metodo hibrido

## Limitacoes arquiteturais atuais

O projeto esta funcional, mas ainda tem limites claros:

- a MLP e simples e implementada manualmente em `numpy`
- nao existe camada de experimento com ruido instrumental real
- o arquivo experimental aceito por `fit` ainda e um formato simples de 3 colunas, nao um pipeline laboratorial completo
- parte da formulacao fisica ainda pede revisao teorica em um ponto marcado como `TODO`

## Resumo

Em uma frase:

este projeto usa um modelo fisico direto de SHG como base para simulacao, fitting classico, geracao de dados sinteticos, treino de uma MLP e comparacao entre abordagens classica, de ML e hibrida.
