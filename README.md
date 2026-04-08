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
- fitting classico com arquivo experimental externo via CLI
- geracao de dataset sintetico via CLI
- treino de MLP com split automatico treino/validacao/teste via CLI
- avaliacao de modelo ML via CLI
- comparacao de metodos via CLI
- treino de MLP via modulo Python

O projeto ainda nao possui:

- empacotamento formal com `pyproject.toml`

## Estrutura do Projeto

```text
SHG/
|- main.py                      # Wrapper simples para src.main
|- README.md
|- requirements.txt
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
|  |  |- methods.py
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
pip install -r requirements.txt
```

Se voce ja vai usar o ambiente que esta dentro do repositorio:

```powershell
.venv\Scripts\Activate.ps1
```

Se quiser instalar manualmente em vez de usar o arquivo:

```powershell
pip install numpy scipy matplotlib
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

### 2. Rodar fitting experimental

O subcomando `fit` agora pode operar em quatro modos:

- `--method classical`: fitting classico com `differential_evolution`
- `--method ml`: predicao direta pela rede treinada
- `--method hybrid`: rede + refinamento fisico local
- `--method compare`: roda os tres e indica o melhor pelo erro observado

Exemplo classico:

```powershell
python main.py fit --method classical --normalization global --seed 7
```

Importante:

- se voce nao passar `--data-path`, `fit` usa dados experimentais de exemplo definidos no proprio codigo
- o modo `classical` abre comparacao visual e mapa de erro
- os modos `ml`, `hybrid` e `compare` exigem `--model-path`

Exemplo `ml`:

```powershell
python main.py fit --method ml --model-path models/shg_mlp.npz --data-path data/experimental_fit.csv --lambda-nm 1560 --delimiter "," --normalization global
```

Exemplo `hybrid`:

```powershell
python main.py fit --method hybrid --model-path models/shg_mlp.npz --data-path data/experimental_fit.csv --lambda-nm 1560 --delimiter "," --normalization global --local-bounds neighborhood --neighborhood-fraction 0.1
```

Exemplo `compare`:

```powershell
python main.py fit --method compare --model-path models/shg_mlp.npz --data-path data/experimental_fit.csv --lambda-nm 1560 --delimiter "," --normalization global --seed 7 --output-dir outputs/fit_compare
```

Formato esperado do arquivo externo:

- exatamente 3 colunas numericas
- se houver cabecalho com os nomes `d_nm`, `i3` e `i1`, o loader usa os nomes e aceita `d_nm, i3, i1` ou `d_nm, i1, i3`
- se nao houver cabecalho, o projeto assume a ordem posicional `d_nm, i3, i1`
- `i3` e `i1` podem ficar vazios em uma linha especifica; o fitting ignora esses pontos usando mascara
- `--skiprows` deve ser usado apenas quando houver linhas extras antes do cabecalho ou antes da primeira linha numerica

Novos controles importantes no `fit`:

- bounds fisicos por parametro: `--n21w-min`, `--n21w-max`, `--k21w-min`, `--k21w-max`, `--n22w-min`, `--n22w-max`, `--k22w-min`, `--k22w-max`
- peso por canal na funcao objetivo: `--i3-weight` e `--i1-weight`

Exemplo classico com bounds e pesos ajustados para dado de laboratorio:

```powershell
python main.py fit --method classical --data-path data/experimental_fit.csv --lambda-nm 1560 --normalization global --n21w-min 1.5 --n21w-max 6.5 --n22w-min 1.0 --n22w-max 6.0 --i3-weight 1.5 --i1-weight 0.7 --seed 7 --output-dir outputs/fit_classical_lab
```

Como interpretar esses novos controles:

- use bounds mais largos quando voce ainda nao conhece bem o material
- estreite os bounds quando ja houver informacao fisica externa, porque isso ajuda a reduzir ambiguidades do problema inverso
- aumente `--i3-weight` se a transmissao for mais confiavel experimentalmente
- aumente `--i1-weight` se a reflexao tiver melhor relacao sinal-ruido
- reduza o peso do canal mais ruidoso em vez de forcar o fit a seguir ruido instrumental

Observacao importante para `ml` e `hybrid`:

- a mascara da rede e por canal, nao por ponto individual
- se faltarem alguns pontos dentro de um canal disponivel, o projeto preenche esses pontos por interpolacao linear apenas para a entrada da MLP
- se o numero de pontos do experimento nao bater com o usado no treino, a curva e reamostrada para a malha esperada pela rede
- no modo `ml`, a saida direta da rede e limitada aos bounds fisicos do projeto antes de ser apresentada
- o erro fisico continua sendo calculado apenas sobre os pontos realmente observados

Quando `--output-dir` e informado no `fit`, o projeto salva:

- resumo JSON do metodo executado
- figura PNG com os pontos experimentais e as curvas reconstruidas
- figura PNG com o forward model rodado com os melhores parametros obtidos e os pontos experimentais
- no modo `classical`, tambem salva o mapa de erro

### Estrategia recomendada para o melhor fit em dado de laboratorio

Para dado experimental real, o melhor resultado normalmente nao vem de um unico comando. O fluxo recomendado e:

1. prepare o CSV com 3 colunas nomeadas `d_nm`, `i3`, `i1`; se a ordem das duas ultimas colunas variar, o loader corrige isso pelo cabecalho
2. preserve valores faltantes apenas onde o ponto realmente nao foi medido; nao substitua por zero
3. rode primeiro o metodo `classical` com bounds fisicos relativamente amplos e salve o resultado com `--output-dir`
4. compare `--normalization global` e `--normalization separate`; a melhor escolha depende de como o laboratorio normalizou os canais
5. ajuste `--i3-weight` e `--i1-weight` conforme a confianca em cada canal
6. observe se o melhor ajuste encosta nos limites de algum parametro; se isso acontecer, amplie os bounds e rode novamente
7. se voce pretende usar `ml` ou `hybrid`, gere um dataset sintetico com a mesma malha `d_nm` do experimento usando `--experimental-grid-path`
8. treine a rede com bounds coerentes com o seu material; a rede nao consegue extrapolar bem para fora do espaco sintetico em que foi treinada
9. use `compare` para decidir entre `classical`, `ml` e `hybrid` pelo erro observado no seu experimento

Sinais de que voce ainda precisa adaptar melhor o pipeline:

- o melhor parametro aparece repetidamente colado no minimo ou no maximo do bound
- pequenas mudancas em `seed`, normalizacao ou pesos mudam demais os parametros recuperados
- a curva ajustada parece boa, mas os parametros fisicos resultantes nao fazem sentido para o material
- a malha experimental tem poucos pontos ou ignora justamente a regiao em que a curva varia mais rapido

### 3. Gerar dataset sintetico

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --seed 42 --normalization global
```

Exemplo usando diretamente a malha de espessuras do experimento:

```powershell
python main.py generate-dataset --num-samples 5000 --output data/shg_dataset_expgrid.npz --lambda-nm 1560 --experimental-grid-path data/experimental_fit.csv --grid-delimiter "," --seed 42 --normalization global
```

Esse comando salva um `.npz` contendo:

- `d_nm`
- `i3`
- `i1`
- `curves`
- `parameters`
- metadados como `lambda`, bounds e seed

Se `--experimental-grid-path` for informado:

- o projeto le a coluna `d_nm` do arquivo experimental
- essa malha passa a ser a malha usada para todas as simulacoes sinteticas
- `--d-max-nm` e `--d-step-nm` deixam de ser a fonte da grade

### 4. Treinar um modelo de ML

```powershell
python main.py train-ml --dataset-path data/shg_synthetic_dataset.npz --model-path models/shg_mlp.npz --summary-path outputs/train_ml/training_summary.json --hidden-dims 256 128 --epochs 300 --batch-size 64 --learning-rate 1e-3 --seed 42
```

Esse comando:

- carrega o dataset sintetico salvo em `.npz`
- faz o split automatico em treino, validacao e teste
- monta a MLP com as camadas ocultas informadas
- executa o treino com augmentacao de mascaras e selecao do melhor modelo por validacao
- salva o modelo treinado
- salva um resumo JSON com historico de loss e hiperparametros
- salva `dataset_split.json` com os indices usados no split
- executa avaliacao automatica nos conjuntos de validacao e teste

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

- o CLI `train-ml` ja faz split automatico em treino, validacao e teste
- pela API Python, voce ainda pode controlar esse fluxo manualmente se quiser

## Onde Ficam Entradas, Modelos, Resultados e Figuras

### Entradas

- datasets sinteticos gerados por CLI:
  - por padrao em `data/shg_synthetic_dataset.npz`
- datasets de teste:
  - caminho livre, informado em `--dataset-path`
- dados experimentais do fitting CLI:
  - podem vir de arquivo externo com `--data-path`
  - se nenhum arquivo for informado, o CLI usa a amostra definida em `src/main.py`

### Modelos treinados

- nao existe pasta fixa obrigatoria
- o modelo e salvo no caminho escolhido pelo usuario ao chamar `train-ml` ou `save_model(...)`
- exemplos desta documentacao usam `models/shg_mlp.npz`

### Resultados e figuras

- treino ML:
  - resumo por padrao em `outputs/train_ml/training_summary.json`
  - split salvo por padrao em `outputs/train_ml/dataset_split.json`
  - avaliacao automatica em `outputs/train_ml/validation` e `outputs/train_ml/test`
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
- no CLI, o split treino/validacao/teste ja e automatico
- na API Python, o usuario ainda pode assumir o controle manual do split

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
- o dataset tem poucas amostras para um split treino/validacao/teste nao vazio
- os hiperparametros passados sao invalidos, por exemplo dimensoes ocultas nao positivas

Confira:

- se o arquivo foi gerado por `generate-dataset`
- se `--hidden-dims`, `--epochs`, `--batch-size` e `--learning-rate` sao coerentes
- se o dataset tem amostras suficientes para treino, validacao e teste

### "fit nao usa meus dados experimentais"

Agora o CLI aceita seus dados, desde que o arquivo esteja no formato esperado.

Use:

```powershell
python main.py fit --data-path caminho/do/arquivo.csv --lambda-nm 1560 --delimiter ","
```

Lembre:

- se houver cabecalho com `d_nm`, `i3` e `i1`, o loader aceita `d_nm, i3, i1` ou `d_nm, i1, i3`
- sem cabecalho, o arquivo deve continuar na ordem posicional `d_nm, i3, i1`
- valores faltantes em `i3` ou `i1` sao aceitos no CSV externo
- `--skiprows` so e necessario quando existem linhas extras antes do cabecalho ou antes da primeira linha numerica
- se voce nao passar `--data-path`, o CLI cai no conjunto interno de exemplo

## O Que Este Projeto Nao Faz

Para evitar interpretacoes erradas, hoje o projeto nao faz automaticamente:

- split automatico entre treino, validacao e teste
- busca automatica de hiperparametros da rede
- garantia de validade fisica fora dos bounds escolhidos
- calibracao com dados instrumentais reais, ruido real ou incerteza experimental
- reproducao exata de um experimento de laboratorio sem adaptacao

Tambem e importante dizer:

- o modelo fisico implementado e uma formulacao especifica de SHG, nao "a fisica inteira de SHG"
- existe um `TODO` explicito no modulo fisico sobre a revisao teorica de um termo `0k` backward
- os resultados de ML dependem fortemente dos bounds, da qualidade do dataset sintetico e da forma como os dados sao separados
- em dado de laboratorio, um fit aparentemente bom pode continuar mal identificado se houver poucos pontos, ruido alto ou bounds muito largos

## Documentacao Complementar

- [Arquitetura](docs/architecture.md)
- [Guia Didatico](docs/didactic_guide.md)
- [Guia de Execucao](docs/how_to_run.md)
- [Notas para Apresentacao](docs/presentation_notes.md)
- [Glossario](docs/glossary.md)
