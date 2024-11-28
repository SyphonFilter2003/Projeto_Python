# Instalar dependências: 
```pip install flask pandas scikit-learn matplotlib joblib```

# Análise de Dados com Modelos de Machine Learning

Este projeto é uma aplicação web desenvolvida com Flask para carregar um arquivo CSV, treinar um modelo de Machine Learning baseado nos dados carregados e realizar análises dos dados. O usuário pode fazer upload de um arquivo CSV, configurar as variáveis do modelo, treinar o modelo e visualizar os resultados e métricas de desempenho do modelo.

Objetivo
O objetivo deste projeto é fornecer uma interface web simples para permitir que o usuário faça upload de um arquivo CSV contendo dados, configure um modelo de aprendizado de máquina, treine o modelo, visualize a precisão e as métricas de classificação, e gere gráficos de análises. O modelo pode ser re-treinado e os resultados podem ser atualizados dinamicamente.

Tecnologias Utilizadas
Flask: Framework web para criação de rotas e renderização de templates.
Pandas: Manipulação e análise de dados.
Scikit-learn: Implementação dos modelos de Machine Learning e métricas de avaliação.
Matplotlib: Geração de gráficos de visualização.
Joblib: Salvamento e carregamento de modelos treinados.
HTML/CSS: Interface de usuário.
Funcionalidades
1. Upload de Arquivo CSV
O usuário pode fazer upload de um arquivo CSV contendo os dados que deseja analisar. O arquivo é processado e as variáveis são extraídas para treinamento de modelos de aprendizado de máquina.

2. Configuração do Modelo
Após o upload, o usuário pode selecionar as variáveis (features) e a variável alvo (target) para treinar o modelo. Ele também pode escolher entre três tipos de modelos: Random Forest, Regressão Logística e SVM.

3. Treinamento do Modelo
O modelo é treinado com os dados selecionados e o relatório de classificação, incluindo precisão, recall e F1-score, é gerado. O usuário também pode visualizar gráficos como a importância das variáveis e a matriz de confusão.

4. Re-treinamento do Modelo
O modelo pode ser re-treinado com novas configurações ou dados sem precisar reiniciar o aplicativo.

Endpoints
A aplicação possui as seguintes rotas (endpoints):

1. / - Rota de Upload de Arquivo CSV
Método: GET e POST
Descrição: Esta rota permite que o usuário faça o upload de um arquivo CSV. Quando o arquivo é enviado, ele é armazenado e lido usando o Pandas.
Fluxo:
GET: Exibe a página de upload.
POST: Processa o arquivo enviado, lê o CSV e exibe uma mensagem de sucesso ou erro.
Parâmetros de entrada:

Arquivo CSV enviado pelo formulário (campo file).
2. /analyze - Rota de Análise dos Dados
Método: GET
Descrição: Exibe uma análise estatística básica dos dados carregados (média, desvio padrão, etc.), separando colunas numéricas e categóricas. Também gera gráficos para as variáveis numéricas.
Fluxo:
GET: Exibe as estatísticas e gráficos para as colunas numéricas.
Saídas:

Tabela com estatísticas descritivas dos dados.
Gráficos de distribuição para cada variável numérica.
3. /configure - Rota de Configuração do Modelo de Machine Learning
Método: GET e POST
Descrição: Esta rota permite que o usuário selecione as variáveis independentes (features) e a variável alvo (target) para treinar o modelo, além de escolher o tipo de modelo. O usuário também pode definir parâmetros como o número de árvores para o modelo Random Forest.
Fluxo:
GET: Exibe o formulário para selecionar as variáveis e o tipo de modelo.
POST: Treina o modelo e exibe os resultados, incluindo a acurácia e um gráfico da importância das variáveis (para Random Forest).
Parâmetros de entrada:

features: Lista das variáveis independentes selecionadas.
target: Variável dependente (alvo).
model: Tipo de modelo escolhido (Random Forest, Regressão Logística ou SVM).
n_estimators: Número de árvores no modelo Random Forest (opcional).
Saídas:

Relatório de classificação (precisão, recall, F1-score).
Gráfico de importância das variáveis (para Random Forest).
4. /retrain - Rota de Re-treinamento do Modelo
Método: POST
Descrição: Esta rota permite re-treinar o modelo com novas configurações ou dados. O usuário pode modificar as variáveis ou o tipo de modelo e o sistema irá re-treinar o modelo com as novas opções.
Fluxo:
POST: Re-treina o modelo e exibe os resultados com o novo modelo treinado.
Parâmetros de entrada:

features: Lista das variáveis independentes.
target: Variável alvo.
model: Tipo de modelo (Random Forest, Regressão Logística ou SVM).
n_estimators: Número de árvores para Random Forest.
Saídas:

Relatório de classificação do novo modelo.
Gráfico de importância das variáveis (se Random Forest).
5. /result - Rota de Resultados
Método: GET
Descrição: Exibe os resultados do modelo treinado, incluindo o relatório de classificação, a acurácia, e outros gráficos gerados (como matriz de confusão e curva ROC).
Fluxo:
GET: Exibe os resultados após o treinamento ou re-treinamento do modelo.


