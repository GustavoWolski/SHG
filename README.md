# SHG Inverse Project

Projeto em Python para estudar inversao de parametros de SHG (Second-Harmonic Generation) por tres caminhos complementares:

1. simulacao fisica direta
2. fitting classico com otimizacao global
3. predicao por machine learning e refinamento hibrido

O objetivo do projeto e gerar curvas SHG simuladas, ajustar parametros opticos a partir dessas curvas e comparar abordagens classicas e baseadas em redes neurais.

## Visao Geral

Em termos simples, o projeto tenta responder esta pergunta:

"Se eu observo as curvas de transmissao e reflexao de SHG, quais parametros opticos do material provavelmente geraram essas curvas?"

Para isso, o repositorio combina:

- um modelo fisico direto que simula SHG
- um problema inverso classico resolvido com `differential_evolution`
- uma MLP simples para prever parametros diretamente
- uma abordagem hibrida que usa a rede neural como chute inicial e depois refina com fisica

## Objetivo Cientifico

Do ponto de vista cientifico, o projeto explora como recuperar os parametros:

- `n21w`
- `k21w`
- `n22w`
- `k22w`

a partir de curvas SHG:

- `i3`: componente tratada no projeto como curva de transmissao
- `i1`: componente tratada no projeto como curva de reflexao

A ideia central e comparar custo computacional, precisao parametrica e qualidade de reconstrucao fisica entre:

- fitting classico
- machine learning
- metodo hibrido

## Estado Atual do Projeto

O projeto ja possui:

- simulacao SHG via CLI
- fitting classico via CLI
- geracao de dataset sintetico via CLI
- treino de MLP via CLI
- avaliacao de modelo ML via CLI
- comparacao de metodos via CLI
- treino de MLP via modulo Python

O projeto ainda nao possui:

- fitting via arquivo experimental externo no CLI
- pipeline automatico de split treino/validacao/teste
- empacotamento formal com `pyproject.toml` ou `requirements.txt`

## Estrutura do Projeto

```text
SHG/
|- main.py                      # Wrapper simples para src.main
|- README.md
|- data/
|  |- shg_dataset.npz           # Exemplo de dataset presente no repositorio
|- docs/
|  |- architecture.md
|  |- didactic_guide.md
|  |- glossary.md
|  |- how_to_run.md
|  |- presentation_notes.md
|- src/
|  |- main.py                   # CLI principal
|  |- physics/
|  |  |- constants.py
|  |  |- optics.py
|  |  |- shg_model.py
|  |- inverse/
|  |  |- objective.py
|  |  |- fitters.py
|  |- data/
|  |  |- synthetic_generator.py
|  |  |- loaders.py
|  |- ml/
|  |  |- datasets.py
|  |  |- models.py
|  |  |- train.py
|  |  |- evaluate.py
|  |  |- compare.py
|  |- utils/
|     |- io.py
|     |- plotting.py
|- tests/
```

## Dependencias e Instalacao

O codigo usa principalmente:

- Python 3.10+ (o ambiente atual do projeto usa Python 3.12)
- `numpy`
- `scipy`
- `matplotlib`

Exemplo de instalacao em Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy matplotlib
```

Se voce ja vai usar o ambiente que esta dentro do repositorio:

```powershell
.venv\Scripts\Activate.ps1
```

## Validacao Rapida

Para uma checagem automatizada e barata do estado do projeto:

```powershell
python -m unittest discover -s tests
```

Esse conjunto atual cobre:

- validacao basica do modelo fisico
- geracao e recarga de dataset sintetico
- treino curto, persistencia e avaliacao do pipeline de ML

## Subcomandos CLI Disponiveis

Os subcomandos reais hoje sao:

- `simulate`
- `fit`
- `generate-dataset`
- `train-ml`
- `evaluate-ml`
- `compare-methods`

## Como Rodar Cada Subcomando

### 1. Simular curvas SHG

```powershell
python main.py simulate --lambda-nm 1560 --n21w 5.6428 --k21w 0.0849 --n22w 2.8698 --k22w 0.4492 --d-max-nm 600 --d-step-nm 1
```

O comando abre graficos de transmissao e reflexao normalizadas.

### 2. Rodar fitting classico

```powershell
python main.py fit --normalization global --seed 7
```

Importante:

- no estado atual, `fit` usa dados experimentais de exemplo definidos no proprio codigo
- ele nao le um arquivo experimental externo pelo CLI
- ele abre grafico do melhor ajuste e mapa de erro

### 3. Gerar dataset sintetico

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --seed 42 --normalization global
```

Esse comando salva um `.npz` contendo:

- `d_nm`
- `i3`
- `i1`
- `curves`
- `parameters`
- metadados como `lambda`, bounds e seed

### 4. Treinar um modelo de ML

```powershell
python main.py train-ml --dataset-path data/shg_synthetic_dataset.npz --model-path models/shg_mlp.npz --summary-path outputs/train_ml/training_summary.json --hidden-dims 256 128 --epochs 300 --batch-size 64 --learning-rate 1e-3 --seed 42
```

Esse comando:

- carrega o dataset sintetico salvo em `.npz`
- monta a MLP com as camadas ocultas informadas
- executa o treino com augmentacao de mascaras
- salva o modelo treinado
- salva um resumo JSON com historico de loss e hiperparametros

### 5. Avaliar um modelo de ML treinado

```powershell
python main.py evaluate-ml --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/evaluate_ml
```

Esse comando:

- carrega o modelo salvo
- carrega o dataset de teste
- avalia tres cenarios de entrada:
  - `i3_i1`
  - `i3_only`
  - `i1_only`
- salva metricas e figuras em `outputs/evaluate_ml` por padrao

### 6. Comparar metodos

```powershell
python main.py compare-methods --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/compare_methods --normalization global --local-bounds neighborhood --neighborhood-fraction 0.1 --max-samples 10 --classical-seed 3
```

Esse comando compara:

- fitting classico
- ML direto
- metodo hibrido

e salva:

- `comparison_summary.json`
- `comparison_summary.csv`
- figuras de comparacao

## Treino Via Modulo Python

O treino tambem continua disponivel como API Python. Exemplo:

```powershell
@'
from src.data.loaders import load_synthetic_dataset
from src.ml.datasets import from_synthetic_dataset
from src.ml.models import ModelConfig, save_model
from src.ml.train import TrainingConfig, train_model

dataset = from_synthetic_dataset(load_synthetic_dataset("data/shg_synthetic_dataset.npz"))
model_config = ModelConfig(
    input_dim=dataset.input_dim,
    output_dim=dataset.output_dim,
    hidden_dims=(256, 128),
)
training_config = TrainingConfig(
    epochs=300,
    batch_size=64,
    learning_rate=1e-3,
    seed=42,
    verbose=True,
)
training_result = train_model(dataset, model_config, training_config)
save_model(training_result.model, "models/shg_mlp.npz")
print(training_result.train_loss_history[-1])
'@ | python -
```

Observacao importante:

- o projeto nao cria automaticamente conjuntos de treino, validacao e teste
- voce precisa separar os dados manualmente para avaliacao cientifica correta

## Onde Ficam Entradas, Modelos, Resultados e Figuras

### Entradas

- datasets sinteticos gerados por CLI:
  - por padrao em `data/shg_synthetic_dataset.npz`
- datasets de teste:
  - caminho livre, informado em `--dataset-path`
- dados experimentais do fitting CLI:
  - hoje ficam hardcoded em `src/main.py`

### Modelos treinados

- nao existe pasta fixa obrigatoria
- o modelo e salvo no caminho escolhido pelo usuario ao chamar `train-ml` ou `save_model(...)`
- exemplos desta documentacao usam `models/shg_mlp.npz`

### Resultados e figuras

- treino ML:
  - resumo por padrao em `outputs/train_ml/training_summary.json`
- avaliacao ML:
  - por padrao em `outputs/evaluate_ml`
- comparacao de metodos:
  - por padrao em `outputs/compare_methods`
- `simulate` e `fit`:
  - hoje exibem figuras na tela, sem salvar automaticamente

## Fluxo Completo do Projeto

```text
1. Escolher parametros fisicos e vetor de espessura
2. Rodar o forward model de SHG
3. Gerar curvas i3 e i1
4. Usar essas curvas de duas formas:
   a) fitting classico para recuperar parametros
   b) gerar dataset sintetico para treinar a MLP
5. Treinar o modelo de ML
6. Avaliar o modelo em previsao de parametros e reconstrucao fisica
7. Comparar metodo classico, ML e hibrido
```

Fluxo em uma linha:

```text
simulacao -> fitting -> dataset sintetico -> treino -> avaliacao -> comparacao
```

Observacao importante:

- o treino pode ser feito por CLI ou por API Python
- o split entre treino, validacao e teste continua sendo responsabilidade do usuario

## Troubleshooting

### "ModuleNotFoundError: scipy"

Causa:

- `fit`, refinamento hibrido e comparacao de metodos usam `scipy`

Solucao:

```powershell
pip install scipy
```

### "ModuleNotFoundError: matplotlib"

Causa:

- simulacao, fitting, avaliacao e comparacao geram figuras com `matplotlib`

Solucao:

```powershell
pip install matplotlib
```

### "Estou rodando em ambiente sem interface grafica"

Se voce estiver em servidor, CI ou validacao automatizada, rode os comandos com:

```powershell
$env:MPLBACKEND='Agg'
```

Assim o `matplotlib` usa backend nao interativo e os comandos com graficos nao tentam abrir janela.

### "fit esta rodando muito devagar"

Causa:

- `differential_evolution` e um otimizador global
- a comparacao metodologica roda fitting classico amostra por amostra

Solucao:

- use `--max-samples` em `compare-methods`
- teste primeiro com poucas amostras
- use a abordagem ML ou hibrida quando quiser agilidade

### "evaluate-ml nao funciona"

Causa comum:

- o modelo ainda nao foi salvo em `.npz`
- o dataset passado nao esta no formato sintetico esperado

Confira:

- `save_model(...)` para gerar o modelo
- `generate-dataset` ou `save_synthetic_dataset(...)` para gerar o dataset

### "train-ml falha"

Causas comuns:

- `--dataset-path` aponta para um arquivo que nao e um dataset sintetico valido
- os hiperparametros passados sao invalidos, por exemplo dimensoes ocultas nao positivas

Confira:

- se o arquivo foi gerado por `generate-dataset`
- se `--hidden-dims`, `--epochs`, `--batch-size` e `--learning-rate` sao coerentes

### "fit nao usa meus dados experimentais"

Isso tambem e esperado no estado atual.

- o comando `fit` usa um conjunto pequeno de exemplo definido em `src/main.py`
- ainda nao existe um subcomando para carregar dados experimentais externos

## O Que Este Projeto Nao Faz

Para evitar interpretacoes erradas, hoje o projeto nao faz automaticamente:

- leitura de dados experimentais reais para o fitting CLI
- split automatico entre treino, validacao e teste
- busca automatica de hiperparametros da rede
- garantia de validade fisica fora dos bounds escolhidos
- calibracao com dados instrumentais reais, ruido real ou incerteza experimental
- reproducao exata de um experimento de laboratorio sem adaptacao

Tambem e importante dizer:

- o modelo fisico implementado e uma formulacao especifica de SHG, nao "a fisica inteira de SHG"
- existe um `TODO` explicito no modulo fisico sobre a revisao teorica de um termo `0k` backward
- os resultados de ML dependem fortemente dos bounds, da qualidade do dataset sintetico e da forma como os dados sao separados

## Documentacao Complementar

- [Arquitetura](docs/architecture.md)
- [Guia Didatico](docs/didactic_guide.md)
- [Guia de Execucao](docs/how_to_run.md)
- [Notas para Apresentacao](docs/presentation_notes.md)
- [Glossario](docs/glossary.md)
