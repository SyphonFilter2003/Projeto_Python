[mental_health_diagnosis_treatment_.csv](https://github.com/user-attachments/files/18000486/mental_health_diagnosis_treatment_.csv)# Instalar dependências: 
```Primeiro, instale as dependências necessárias com o comando:```
```pip install flask pandas scikit-learn matplotlib joblib```

# OBJETIVO DO TRABALHO

Este projeto foi desenvolvido para criar uma aplicação web simples usando Flask, que permite carregar dados de um arquivo CSV, treinar um modelo de aprendizado de máquina com esses dados e visualizar os resultados de forma interativa. Através dessa aplicação, você pode fazer upload de dados, configurar o modelo, treinar e ainda visualizar gráficos e métricas de desempenho.

Objetivo
O objetivo deste projeto é fornecer uma interface web simples para permitir que o usuário faça upload de um arquivo CSV contendo dados, configure um modelo de aprendizado de máquina, treine o modelo, visualize a precisão e as métricas de classificação, e gere gráficos de análises. O modelo pode ser re-treinado e os resultados podem ser atualizados dinamicamente.

Funcionalidades
# 1. Upload de Arquivo CSV
Você pode enviar um arquivo CSV com dados que deseja analisar. A aplicação irá processar esse arquivo e exibir informações sobre os dados.

# 2. Configuração do Modelo
Após carregar os dados, você pode escolher as variáveis (features) e a variável de previsão (target) para treinar o modelo. A aplicação oferece três opções de modelos de aprendizado de máquina: Random Forest, Regressão Logística e SVM.

# 3. Treinamento do Modelo
Após configurar o modelo, você pode treiná-lo com os dados carregados. O sistema irá exibir métricas de desempenho, como a precisão do modelo, recall e F1-score, além de gráficos, como a importância das variáveis e a matriz de confusão.

Endpoints
A aplicação possui as seguintes rotas (endpoints):

# 1. / - Rota de Upload de Arquivo CSV
Permite fazer o upload de um arquivo CSV. Após o envio, o arquivo é processado e lido para exibir os dados.GET: Exibe a página de upload.

Método: GET (para mostrar o formulário) e POST (para processar o arquivo)
Parâmetros de entrada: Arquivo CSV

# 2. /analyze - Rota de Análise dos Dados
Exibe uma análise estatística básica do arquivo CSV, com média, desvio padrão e gráficos das variáveis numéricas.

Método: GET
Saídas: Gráficos das variáveis numéricas

# 3. /configure - Rota de Configuração do Modelo de Machine Learning
Aqui você escolhe as variáveis que o modelo vai usar e o tipo de modelo (Random Forest, Regressão Logística ou SVM). Também é possível definir parâmetros como o número de árvores para o modelo Random Forest.

Método: GET (para mostrar o formulário) e POST (para treinar o modelo)

Parâmetros de entrada:

features: Variáveis independentes
target: Variável de previsão
model: Tipo de modelo (Random Forest, Regressão Logística ou SVM)
n_estimators: Número de árvores (apenas para Random Forest)
Saídas: Relatório de precisão, recall e F1-score, além de gráficos de importância das variáveis (se for Random Forest).

# 4. /retrain - Rota de Re-treinamento do Modelo(FUNÇÃO QUE ERA PRA SER FEITA PELO LUCAS E PELO VINÍCIUS, PORÉM, SE RECUSARAM A FAZER E A FUNÇÃO NÃO FUNCIONA COM DEVERIA...)
Permite re-treinar o modelo com novas configurações ou dados. A aplicação irá usar as novas variáveis ou tipo de modelo escolhidos.

Método: POST
Parâmetros de entrada: Mesmos parâmetros de /configure
Saídas: Relatório de classificação e gráficos atualizados.

# 5. /result - Rota de Resultados
Exibe os resultados do modelo treinado ou re-treinado, como o relatório de classificação, acurácia e gráficos de matriz de confusão e curva ROC.

Método: GET
Saídas: Relatório de resultados e gráficos.
Descrição: Exibe os resultados do modelo treinado, incluindo o relatório de classificação, a acurácia, e outros gráficos gerados (como matriz de confusão e curva ROC).
Fluxo:
GET: Exibe os resultados após o treinamento ou re-treinamento do modelo.

# Baixar arquivo CSV para testes

[Baixar arquivo CSV](https://drive.google.com/file/d/1VI8WLyAgjcaQWruOVo_7RSWcrojnyJeA/view?usp=sharing)
