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
python main.py fit --method classical --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --normalization global --seed 7
```

Exemplo em modo `ml`:

```powershell
python main.py fit --method ml --model-path models/shg_mlp.npz --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --normalization global
```

Exemplo em modo `hybrid`:

```powershell
python main.py fit --method hybrid --model-path models/shg_mlp.npz --data-path data/experimental_shg.csv --lambda-nm 1560 --delimiter ',' --normalization global --local-bounds neighborhood --neighborhood-fraction 0.1
```

Exemplo em modo `compare`:

```powershell
python main.py fit --method compare --model-path models/shg_mlp_expgrid.npz --data-path data/experimental_fit.csv --lambda-nm 1560 --delimiter "," --normalization separate --seed 42 --output-dir outputs/fit_compare 
```

Exemplo com bounds fisicos customizados e pesos por canal:

```powershell
python main.py fit --method classical --data-path data/experimental_fit.csv --lambda-nm 1560 --normalization global --n21w-min 1.0 --n21w-max 6.0 --k21w-min 0.0 --k21w-max 1.0 --n22w-min 1.0 --n22w-max 6.0 --k22w-min 0.0 --k22w-max 1.0 --i3-weight 1.3 --i1-weight 0.8 --seed 42 --output-dir outputs/fit_classical_lab
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
- `--n21w-min`, `--n21w-max`, `--k21w-min`, `--k21w-max`, `--n22w-min`, `--n22w-max`, `--k22w-min`, `--k22w-max`
- `--i3-weight`, `--i1-weight`

O que esse comando faz:

- carrega o arquivo experimental informado em `--data-path`
- ou, se `--data-path` nao for informado, usa os dados de exemplo definidos em `src/main.py`
- executa o metodo escolhido
- mostra os parametros previstos, o erro observado e o tempo
- abre a comparacao entre curvas experimentais e reconstruidas
- no modo `classical`, tambem abre o mapa de erro em funcao de `n21w` e `k21w`
- no modo `compare`, imprime os tres resultados e indica o melhor pelo erro observado
- se `--output-dir` for informado, salva resumo JSON e graficos PNG

Importante:

- se houver cabecalho com os nomes `d_nm`, `i3` e `i1`, o loader aceita as colunas `i3` e `i1` em qualquer ordem
- sem cabecalho, o arquivo experimental continua precisando estar na ordem posicional `d_nm`, `i3`, `i1`
- `i3` e `i1` podem ter valores faltantes em linhas especificas; esses pontos sao ignorados no erro do fitting
- `--skiprows` so e necessario quando ha linhas extras antes do cabecalho ou antes dos dados numericos
- `ml`, `hybrid` e `compare` exigem `--model-path`
- no `ml` e no `hybrid`, lacunas internas de um canal disponivel sao preenchidas por interpolacao linear apenas para montar a entrada da MLP
- se o experimento vier com outro numero de pontos, a curva e reamostrada para a malha esperada pela rede
- no `ml`, a previsao final e recortada aos bounds fisicos do projeto antes de ser exibida
- se `scipy` nao estiver instalado, o fitting nao roda

Como usar bounds e pesos para melhorar o fit:

- comece com bounds amplos, mas fisicamente plausiveis para o material
- se o melhor ajuste encostar no limite minimo ou maximo, amplie a faixa e teste novamente
- se a transmissao `i3` for mais estavel que `i1`, aumente `--i3-weight`
- se a reflexao `i1` estiver mais limpa experimentalmente, aumente `--i1-weight`
- se um canal estiver muito contaminado por ruido, reduza seu peso em vez de descartar todo o canal

Fluxo recomendado para dado de laboratorio:

1. monte um CSV com `d_nm`, `i3` e `i1`; deixe celulas vazias onde nao houve medicao
2. rode `fit --method classical` com `--output-dir` para inspecionar curvas e mapa de erro
3. teste `--normalization global` e `--normalization separate`
4. ajuste bounds e pesos ate obter um ajuste que nao fique colado nas bordas do espaco de busca
5. use `generate-dataset --experimental-grid-path` para criar um dataset sintetico na mesma malha do laboratorio
6. treine a MLP com bounds coerentes com seu experimento
7. rode `fit --method compare` para decidir entre classico, ML e hibrido pelo erro observado

Arquivos tipicos salvos pelo `fit` com `--output-dir`:

- `fit_classical_summary.json`
- `fit_classical_curves.png`
- `fit_classical_simulation.png`
- `fit_classical_error_map.png`
- `fit_ml_summary.json`
- `fit_ml_curves.png`
- `fit_ml_simulation.png`
- `fit_hybrid_summary.json`
- `fit_hybrid_curves.png`
- `fit_hybrid_simulation.png`
- `fit_compare_summary.json`
- `fit_compare_curves.png`
- `fit_compare_simulation.png`

## 5. Gerar dataset sintetico

Use `generate-dataset` para criar amostras simuladas em `.npz`.

Exemplo simples:

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --seed 42 --normalization global
```

Exemplo reutilizando as espessuras de um arquivo experimental:

```powershell
python main.py generate-dataset --num-samples 5000 --output data/shg_dataset_expgrid.npz --lambda-nm 1560 --experimental-grid-path data/experimental_fit.csv --grid-delimiter ',' --seed 42 --normalization global
```

Exemplo com bounds explicitamente definidos:

```powershell
python main.py generate-dataset --num-samples 500 --output data/shg_synthetic_dataset.npz --lambda-nm 1560 --d-max-nm 600 --d-step-nm 1 --n21w-min 1.0 --n21w-max 6.0 --k21w-min 0.0 --k21w-max 1.0 --n22w-min 1.0 --n22w-max 6.0 --k22w-min 0.0 --k22w-max 1.0 --seed 42 --normalization separate
```

Recomendacao importante para adaptacao experimental:

- use os mesmos bounds no `generate-dataset` e no `fit` quando voce treinar uma MLP para um material especifico
- se os bounds do treino forem muito diferentes dos bounds usados no fitting experimental, o metodo `ml` ou `hybrid` tende a perder confianca e estabilidade

Opcoes importantes:

- `--num-samples`
- `--output`
- `--lambda-nm`
- `--d-max-nm`
- `--d-step-nm`
- `--experimental-grid-path`
- `--grid-delimiter`
- `--grid-skiprows`
- limites min e max de cada parametro
- `--normalization {none,global,separate}`
- `--seed`
- `--no-progress`

Saida salva:

- arquivo `.npz`
- arrays de curvas
- parametros reais
- metadados como `lambda_m`, bounds, normalizacao e seed

Observacao:

- se voce passar `--experimental-grid-path`, o dataset sintetico usa exatamente a coluna `d_nm` do arquivo experimental
- isso e o caminho recomendado quando voce quer treinar a rede na mesma malha do laboratorio

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
- se o arquivo tem exatamente 3 colunas numericas de interesse
- se houver cabecalho com `d_nm`, `i3` e `i1`, o loader aceita `d_nm, i3, i1` ou `d_nm, i1, i3`
- se nao houver cabecalho, mantenha a ordem posicional `d_nm, i3, i1`
- se os vazios aparecem apenas em `i3` e/ou `i1` e nao em `d_nm`
- se `--skiprows` esta sendo usado apenas para pular linhas extras antes do cabecalho ou antes dos dados numericos
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
- pesos por canal e normalizacao tambem influenciam bastante o resultado em dado de laboratorio

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
