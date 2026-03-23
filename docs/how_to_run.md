# Guia de Execucao

## Objetivo

Este guia mostra como executar o projeto na pratica, usando os comandos e modulos que existem hoje no repositorio.

Importante:

- os comandos abaixo assumem que voce esta na raiz do projeto
- o CLI principal atual e `python main.py ...`
- `fit` pode usar um arquivo experimental externo simples ou, se nenhum arquivo for passado, cair no exemplo interno do projeto

## 1. Instalar dependencias

O projeto depende principalmente de:

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`

Exemplo em Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se voce ja estiver usando o ambiente do proprio projeto:

```powershell
.venv\Scripts\Activate.ps1
```

Se preferir instalar manualmente:

```powershell
pip install numpy scipy matplotlib
```

Se estiver em ambiente sem interface grafica e quiser evitar abertura de janelas do `matplotlib`:

```powershell
$env:MPLBACKEND='Agg'
```

## 2. Validar o modelo fisico

O projeto possui uma funcao de auto-validacao no modulo fisico:

- `validate_default_simulation(...)`

Ela verifica:

- se as saidas tem o mesmo tamanho de `d_nm`
- se nao ha `NaN` ou `inf`
- se as intensidades reais sao nao negativas dentro da tolerancia numerica

Exemplo:

```powershell
@'
from src.physics.shg_model import validate_default_simulation

i3, i1 = validate_default_simulation()
print(i3.shape, i1.shape)
print(float(i3.min()), float(i1.min()))
'@ | python -
```

Se tudo estiver correto, o script deve terminar sem erro.

## 3. Simular curvas SHG

Use o subcomando `simulate` para rodar o forward model e abrir as curvas na tela.

Exemplo:

```powershell
python main.py simulate --lambda-nm 1560 --n21w 5.6428 --k21w 0.0849 --n22w 2.8698 --k22w 0.4492 --d-max-nm 600 --d-step-nm 1
```

O que esse comando faz:

- monta o vetor de espessuras `d_nm`
- cria `SHGParams`
- roda `simulate_shg(...)`
- abre um grafico com as curvas normalizadas

Observacao:

- esse comando exibe figura, mas nao salva arquivo automaticamente

## 4. Rodar fitting experimental

Use o subcomando `fit` para executar a inversao experimental em um de quatro modos:

- `classical`: fitting classico com `differential_evolution`
- `ml`: predicao direta da rede treinada
- `hybrid`: rede neural seguida de refinamento fisico local
- `compare`: executa os tres modos acima e aponta o melhor pelo erro observado

Exemplo usando o conjunto interno de fallback:

```powershell
python main.py fit --method classical --normalization global --seed 7
```

Exemplo usando arquivo experimental externo:

```powershell
python main.py fit --method classical --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --skiprows 1 --normalization global --seed 7
```

Exemplo em modo `ml`:

```powershell
python main.py fit --method ml --model-path models/shg_mlp.npz --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --skiprows 1 --normalization global
```

Exemplo em modo `hybrid`:

```powershell
python main.py fit --method hybrid --model-path models/shg_mlp.npz --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --skiprows 1 --normalization global --local-bounds neighborhood --neighborhood-fraction 0.1
```

Exemplo em modo `compare`:

```powershell
python main.py fit --method compare --model-path models/shg_mlp.npz --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --skiprows 1 --normalization global --seed 7 --output-dir outputs/fit_compare
```

Opcoes principais hoje:

- `--method {classical,ml,hybrid,compare}`
- `--data-path`
- `--lambda-nm`
- `--delimiter`
- `--skiprows`
- `--normalization global`
- `--normalization separate`
- `--seed`
- `--model-path`
- `--local-bounds`
- `--neighborhood-fraction`
- `--output-dir`

O que esse comando faz:

- carrega o arquivo experimental informado em `--data-path`
- ou, se `--data-path` nao for informado, usa os dados de exemplo definidos em `src/main.py`
- executa o metodo escolhido
- mostra os parametros previstos, o erro observado e o tempo
- abre a comparacao entre curvas experimentais e reconstruidas
- no modo `classical`, tambem abre o mapa de erro em funcao de `n21w` e `k21w`
- no modo `compare`, imprime os tres resultados e indica o melhor pelo erro observado

Importante:

- o arquivo experimental precisa ter exatamente 3 colunas: `d_nm`, `i3`, `i1`
- `i3` e `i1` podem ter valores faltantes em linhas especificas; esses pontos sao ignorados no erro do fitting
- `ml`, `hybrid` e `compare` exigem `--model-path`
- no `ml` e no `hybrid`, lacunas internas de um canal disponivel sao preenchidas por interpolacao linear apenas para montar a entrada da MLP
- no `ml`, a previsao final e recortada aos bounds fisicos do projeto antes de ser exibida
- se `scipy` nao estiver instalado, o fitting nao roda

## 5. Gerar dataset sintetico

Use `generate-dataset` para criar amostras simuladas em `.npz`.

Exemplo simples:

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --seed 42 --normalization global
```

Exemplo com bounds explicitamente definidos:

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --n21w-min 3.0 --n21w-max 7.0 --k21w-min 0.0 --k21w-max 1.0 --n22w-min 2.0 --n22w-max 5.0 --k22w-min 0.0 --k22w-max 1.0 --seed 42 --normalization separate
```

Opcoes importantes:

- `--num-samples`
- `--output`
- `--lambda-nm`
- `--d-max-nm`
- `--d-step-nm`
- limites min e max de cada parametro
- `--normalization {none,global,separate}`
- `--seed`
- `--no-progress`

Saida salva:

- arquivo `.npz`
- arrays de curvas
- parametros reais
- metadados como `lambda_m`, bounds, normalizacao e seed

## 6. Treinar o modelo

Use `train-ml` para transformar um dataset sintetico em um modelo `.npz`.

```powershell
python main.py train-ml --dataset-path data/shg_synthetic_dataset.npz --model-path models/shg_mlp.npz --output-dir outputs/train_ml --summary-path outputs/train_ml/training_summary.json --hidden-dims 256 128 --epochs 300 --batch-size 64 --learning-rate 1e-3 --weight-decay 1e-5 --gradient-clip 5.0 --train-fraction 0.7 --validation-fraction 0.15 --test-fraction 0.15 --seed 42 --split-seed 42 --verbose
```

Observacoes importantes:

- o comando faz split automatico em treino, validacao e teste
- o melhor modelo e selecionado pela menor loss de validacao observada
- o modelo salvo e um arquivo `.npz`
- o comando salva `dataset_split.json`, um resumo JSON do treino e avaliacoes automaticas em validacao e teste

Opcoes principais do `train-ml`:

- `--dataset-path`
- `--model-path`
- `--output-dir`
- `--summary-path`
- `--hidden-dims`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--weight-decay`
- `--gradient-clip`
- `--seed`
- `--split-seed`
- `--train-fraction`
- `--validation-fraction`
- `--test-fraction`
- `--examples-per-group`
- `--no-figures`
- `--verbose`

Se voce quiser continuar usando a API Python, o modulo `src/ml/train.py` segue disponivel.

## 7. Avaliar o modelo

Use `evaluate-ml` para avaliar parametros previstos e reconstrucao fisica.

Exemplo:

```powershell
python main.py evaluate-ml --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/evaluate_ml
```

Opcoes reais:

- `--model-path`
- `--dataset-path`
- `--output-dir`
- `--examples-per-group`
- `--no-figures`

O que esse comando faz:

- carrega o modelo treinado
- carrega o dataset de teste
- avalia tres cenarios:
  - `i3_i1`
  - `i3_only`
  - `i1_only`
- calcula metricas parametricas
- reconstrui curvas com o forward model
- salva `evaluation_summary.json`
- salva figuras por cenario, se `matplotlib` estiver disponivel

Estrutura tipica de saida:

```text
outputs/evaluate_ml/
|- evaluation_summary.json
|- i3_i1/
|  |- predicted_vs_true.png
|  |- parameter_error_histograms.png
|  |- reconstruction_examples.png
|- i3_only/
|- i1_only/
```

## 8. Comparar metodos

Use `compare-methods` para comparar:

- fitting classico
- ML direto
- metodo hibrido

Exemplo:

```powershell
python main.py compare-methods --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/compare_methods --normalization global --local-bounds neighborhood --neighborhood-fraction 0.1 --max-samples 10 --classical-seed 3
```

Opcoes reais:

- `--model-path`
- `--dataset-path`
- `--output-dir`
- `--normalization {global,separate}`
- `--local-bounds {global,neighborhood}`
- `--neighborhood-fraction`
- `--classical-seed`
- `--max-samples`
- `--examples-per-group`
- `--no-figures`
- `--no-progress`

O que esse comando faz:

- executa o metodo classico amostra por amostra
- executa a predicao direta por MLP
- executa o refinamento hibrido a partir da MLP
- mede erro parametrico, reconstrucao fisica e tempo
- salva resumo estruturado e figuras comparativas

Estrutura tipica de saida:

```text
outputs/compare_methods/
|- comparison_summary.json
|- comparison_summary.csv
|- timing_comparison.png
|- parameter_rmse_comparison.png
|- reconstruction_error_comparison.png
|- method_reconstruction_examples.png
```

## 9. Consultar ajuda do CLI

Para listar o menu principal:

```powershell
python main.py --help
```

Para ajuda de um subcomando:

```powershell
python main.py simulate --help
python main.py fit --help
python main.py generate-dataset --help
python main.py evaluate-ml --help
python main.py compare-methods --help
```

## 10. Fluxo pratico recomendado

Se voce quer rodar tudo de forma organizada, uma sequencia pratica e:

### Passo A. Validar a simulacao

```powershell
@'
from src.physics.shg_model import validate_default_simulation
validate_default_simulation()
print("validacao ok")
'@ | python -
```

### Passo B. Gerar dataset sintetico

```powershell
python main.py generate-dataset --num-samples 1000 --output data/shg_train.npz --seed 42 --normalization global
```

### Passo C. Treinar a MLP via CLI

```powershell
python main.py train-ml --dataset-path data/shg_train.npz --model-path models/shg_mlp.npz --output-dir outputs/train_ml --summary-path outputs/train_ml/training_summary.json --seed 42
```

### Passo D. Avaliar no conjunto de teste

O `train-ml` ja gera avaliacao automatica nos subconjuntos de validacao e teste criados no split. Se quiser rodar uma avaliacao separada sobre outro `.npz`, use:

```powershell
python main.py evaluate-ml --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/evaluate_ml
```

### Passo E. Comparar os tres metodos

```powershell
python main.py compare-methods --model-path models/shg_mlp.npz --dataset-path data/shg_test.npz --output-dir outputs/compare_methods --max-samples 10
```

## 11. Problemas comuns

### Erro: `ModuleNotFoundError: scipy`

Instale:

```powershell
pip install scipy
```

### Erro: `ModuleNotFoundError: matplotlib`

Instale:

```powershell
pip install matplotlib
```

### O comando `train-ml` falha

Verifique:

- se `--dataset-path` aponta para um `.npz` valido gerado pelo projeto
- se o dataset tem amostras suficientes para os fracionamentos escolhidos
- se `--train-fraction`, `--validation-fraction` e `--test-fraction` sao positivos e formam um split viavel
- se `--hidden-dims` contem apenas inteiros positivos
- se `--epochs`, `--batch-size` e `--learning-rate` sao positivos

### O comando `fit` nao usa meus dados reais

Confira:

- se voce passou `--data-path`
- se o arquivo tem exatamente 3 colunas no formato `d_nm, i3, i1`
- se os vazios aparecem apenas em `i3` e/ou `i1` e nao em `d_nm`
- se `--lambda-nm` corresponde ao experimento que voce quer ajustar
- se voce passou `--model-path` ao usar `--method ml`, `--method hybrid` ou `--method compare`

### A comparacao esta lenta

O gargalo normalmente e o fitting classico. Tente:

- usar `--max-samples`
- comecar com poucas amostras
- desabilitar figuras com `--no-figures`

## 12. O que ainda exige cuidado

Mesmo que o pipeline rode, ainda e importante lembrar:

- ha um ponto teorico no modulo fisico marcado como `TODO`
- o treino e feito sobre dados sinteticos, nao sobre experimento real
- a qualidade dos resultados depende fortemente dos bounds escolhidos

## 13. Smoke tests do repositorio

O projeto agora possui um conjunto pequeno de testes de smoke em `tests/test_smoke.py`.

Para rodar:

```powershell
python -m unittest discover -s tests
```

Esses testes nao cobrem todo o projeto, mas validam o essencial:

- consistencia numerica da simulacao padrao
- geracao e recarga de dataset sintetico
- treino curto, salvamento, recarga e avaliacao do pipeline de ML
