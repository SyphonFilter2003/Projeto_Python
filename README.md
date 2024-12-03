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

ARQUIVO E TEMA DO NOSSO TRABALHO:

[Uploading mental_health_diagnPatient ID,Age,Gender,Diagnosis,Symptom Severity (1-10),Mood Score (1-10),Sleep Quality (1-10),Physical Activity (hrs/week),Medication,Therapy Type,Treatment Start Date,Treatment Duration (weeks),Stress Level (1-10),Outcome,Treatment Progress (1-10),AI-Detected Emotional State,Adherence to Treatment (%)
1,43,Female,Major Depressive Disorder,10,5,8,5,Mood Stabilizers,Interpersonal Therapy,2024-01-25,11,9,Deteriorated,7,Anxious,66
2,40,Female,Major Depressive Disorder,9,5,4,7,Antipsychotics,Interpersonal Therapy,2024-02-27,11,7,No Change,7,Neutral,78
3,55,Female,Major Depressive Disorder,6,3,4,3,SSRIs,Mindfulness-Based Therapy,2024-03-20,14,7,Deteriorated,5,Happy,62
4,34,Female,Major Depressive Disorder,6,3,6,5,SSRIs,Mindfulness-Based Therapy,2024-03-29,8,8,Deteriorated,10,Excited,72
5,52,Male,Panic Disorder,7,6,6,8,Anxiolytics,Interpersonal Therapy,2024-03-18,12,5,Deteriorated,6,Excited,63
6,28,Male,Panic Disorder,8,7,6,4,SSRIs,Cognitive Behavioral Therapy,2024-01-11,13,9,No Change,7,Stressed,82
7,59,Male,Generalized Anxiety,6,6,5,3,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-21,13,8,Deteriorated,5,Anxious,79
8,32,Female,Major Depressive Disorder,5,3,5,2,Antidepressants,Cognitive Behavioral Therapy,2024-02-03,12,5,No Change,6,Stressed,74
9,21,Female,Major Depressive Disorder,6,7,7,10,Antidepressants,Mindfulness-Based Therapy,2024-01-16,15,5,No Change,6,Neutral,67
10,21,Female,Panic Disorder,9,5,9,7,Antidepressants,Cognitive Behavioral Therapy,2024-01-03,11,8,No Change,8,Depressed,89
11,42,Female,Panic Disorder,6,6,7,2,Antidepressants,Mindfulness-Based Therapy,2024-02-01,11,5,Deteriorated,8,Stressed,72
12,54,Female,Panic Disorder,8,8,5,5,Antidepressants,Mindfulness-Based Therapy,2024-02-19,10,8,Deteriorated,10,Happy,82
13,46,Female,Generalized Anxiety,7,4,9,4,Anxiolytics,Interpersonal Therapy,2024-01-16,8,9,No Change,7,Neutral,74
14,19,Male,Bipolar Disorder,6,7,9,10,Antipsychotics,Dialectical Behavioral Therapy,2024-01-24,9,10,Deteriorated,10,Excited,75
15,23,Female,Panic Disorder,9,8,6,1,Antipsychotics,Dialectical Behavioral Therapy,2024-04-01,10,7,Deteriorated,6,Stressed,72
16,38,Female,Generalized Anxiety,10,3,8,2,Mood Stabilizers,Interpersonal Therapy,2024-03-23,15,5,Improved,7,Depressed,69
17,30,Male,Bipolar Disorder,10,8,5,5,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-19,9,9,No Change,6,Depressed,86
18,28,Female,Bipolar Disorder,8,4,7,6,SSRIs,Mindfulness-Based Therapy,2024-02-09,9,8,No Change,6,Depressed,78
19,52,Female,Generalized Anxiety,7,3,4,9,Anxiolytics,Mindfulness-Based Therapy,2024-03-23,12,6,No Change,5,Happy,84
20,37,Male,Bipolar Disorder,5,5,4,5,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-31,15,9,Deteriorated,6,Anxious,67
21,19,Male,Major Depressive Disorder,8,4,5,7,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-01,14,5,Improved,5,Depressed,67
22,41,Female,Bipolar Disorder,5,7,9,7,Antidepressants,Interpersonal Therapy,2024-01-30,9,9,Improved,5,Anxious,65
23,50,Male,Generalized Anxiety,7,4,5,1,Anxiolytics,Interpersonal Therapy,2024-01-02,16,10,Deteriorated,10,Stressed,89
24,53,Female,Panic Disorder,9,5,9,10,Anxiolytics,Mindfulness-Based Therapy,2024-01-25,16,8,Deteriorated,9,Neutral,66
25,60,Male,Major Depressive Disorder,9,6,5,8,Anxiolytics,Interpersonal Therapy,2024-01-24,13,9,No Change,7,Anxious,74
26,19,Female,Bipolar Disorder,9,7,9,6,Antipsychotics,Dialectical Behavioral Therapy,2024-02-07,15,8,Deteriorated,6,Excited,73
27,23,Male,Bipolar Disorder,7,5,7,6,Benzodiazepines,Interpersonal Therapy,2024-03-25,15,5,No Change,10,Happy,77
28,56,Male,Generalized Anxiety,10,5,8,1,Benzodiazepines,Mindfulness-Based Therapy,2024-04-09,10,7,Deteriorated,8,Happy,83
29,58,Male,Major Depressive Disorder,8,6,7,6,Antidepressants,Cognitive Behavioral Therapy,2024-02-08,8,6,Improved,6,Neutral,89
30,41,Female,Major Depressive Disorder,7,6,6,9,SSRIs,Cognitive Behavioral Therapy,2024-01-12,14,8,Improved,9,Depressed,83
31,44,Male,Major Depressive Disorder,6,3,5,3,Antipsychotics,Mindfulness-Based Therapy,2024-02-17,15,7,Improved,6,Happy,74
32,47,Male,Bipolar Disorder,7,6,7,5,Benzodiazepines,Mindfulness-Based Therapy,2024-03-14,9,9,Deteriorated,8,Depressed,87
33,39,Female,Generalized Anxiety,10,6,4,3,SSRIs,Interpersonal Therapy,2024-01-22,14,6,Improved,6,Stressed,63
34,27,Female,Major Depressive Disorder,7,7,4,9,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-10,15,8,Improved,6,Depressed,71
35,53,Male,Generalized Anxiety,6,5,6,10,SSRIs,Interpersonal Therapy,2024-02-04,10,6,Deteriorated,9,Excited,84
36,53,Male,Bipolar Disorder,10,6,5,7,Mood Stabilizers,Cognitive Behavioral Therapy,2024-01-08,11,9,Deteriorated,7,Depressed,68
37,45,Female,Bipolar Disorder,5,5,6,6,Benzodiazepines,Mindfulness-Based Therapy,2024-01-18,14,9,Deteriorated,6,Neutral,88
38,33,Male,Generalized Anxiety,5,6,6,2,Antidepressants,Dialectical Behavioral Therapy,2024-02-28,10,8,Deteriorated,5,Neutral,86
39,31,Male,Major Depressive Disorder,6,8,4,10,Antidepressants,Dialectical Behavioral Therapy,2024-01-02,12,6,Deteriorated,9,Neutral,90
40,41,Female,Bipolar Disorder,8,7,7,2,Anxiolytics,Interpersonal Therapy,2024-01-08,14,7,Deteriorated,8,Stressed,87
41,46,Female,Generalized Anxiety,9,8,8,1,Antipsychotics,Dialectical Behavioral Therapy,2024-02-25,13,7,Improved,8,Excited,65
42,56,Male,Panic Disorder,5,7,6,4,Antipsychotics,Dialectical Behavioral Therapy,2024-04-05,11,9,Improved,10,Happy,60
43,31,Female,Panic Disorder,6,5,9,10,SSRIs,Dialectical Behavioral Therapy,2024-03-27,13,7,Improved,10,Anxious,73
44,50,Female,Bipolar Disorder,8,8,7,2,SSRIs,Cognitive Behavioral Therapy,2024-02-17,14,9,Improved,6,Neutral,61
45,40,Female,Panic Disorder,10,8,7,8,Antipsychotics,Cognitive Behavioral Therapy,2024-03-18,12,5,Deteriorated,10,Excited,72
46,43,Female,Bipolar Disorder,9,6,5,10,Antidepressants,Dialectical Behavioral Therapy,2024-03-29,9,10,No Change,9,Depressed,83
47,48,Male,Generalized Anxiety,10,5,5,9,Antidepressants,Cognitive Behavioral Therapy,2024-01-30,16,9,Improved,6,Excited,61
48,26,Male,Generalized Anxiety,6,3,5,6,Benzodiazepines,Interpersonal Therapy,2024-02-01,16,9,No Change,6,Neutral,90
49,41,Female,Major Depressive Disorder,7,6,9,10,Antipsychotics,Cognitive Behavioral Therapy,2024-01-28,11,10,No Change,10,Depressed,79
50,46,Female,Generalized Anxiety,10,5,4,2,SSRIs,Interpersonal Therapy,2024-03-08,13,9,No Change,7,Depressed,72
51,40,Male,Generalized Anxiety,6,5,7,2,Anxiolytics,Cognitive Behavioral Therapy,2024-01-01,14,7,No Change,10,Stressed,81
52,31,Female,Panic Disorder,8,6,9,5,Antipsychotics,Dialectical Behavioral Therapy,2024-01-20,9,8,Deteriorated,6,Depressed,77
53,38,Female,Bipolar Disorder,5,8,6,8,Benzodiazepines,Interpersonal Therapy,2024-03-02,12,7,Deteriorated,6,Neutral,77
54,18,Male,Panic Disorder,7,4,4,1,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-17,13,10,No Change,7,Anxious,69
55,28,Female,Panic Disorder,10,7,6,4,Antipsychotics,Mindfulness-Based Therapy,2024-01-28,16,9,Improved,10,Excited,69
56,29,Female,Bipolar Disorder,10,4,9,3,Antidepressants,Mindfulness-Based Therapy,2024-04-02,16,6,Improved,7,Depressed,68
57,43,Male,Panic Disorder,8,4,5,8,Antipsychotics,Interpersonal Therapy,2024-01-14,16,6,No Change,8,Anxious,83
58,31,Male,Major Depressive Disorder,6,5,4,5,Antidepressants,Interpersonal Therapy,2024-02-11,10,7,No Change,7,Stressed,83
59,35,Female,Panic Disorder,10,3,9,3,Antidepressants,Cognitive Behavioral Therapy,2024-02-21,15,8,Deteriorated,10,Excited,64
60,47,Male,Bipolar Disorder,10,4,6,4,SSRIs,Interpersonal Therapy,2024-04-01,12,7,Improved,8,Anxious,60
61,56,Female,Generalized Anxiety,9,8,7,1,Benzodiazepines,Interpersonal Therapy,2024-03-13,10,6,Improved,9,Depressed,87
62,41,Female,Generalized Anxiety,6,4,7,6,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-25,12,9,Deteriorated,5,Neutral,61
63,30,Male,Bipolar Disorder,10,6,6,3,Mood Stabilizers,Interpersonal Therapy,2024-02-06,13,6,No Change,8,Depressed,82
64,37,Male,Major Depressive Disorder,8,5,8,1,Anxiolytics,Dialectical Behavioral Therapy,2024-04-06,9,9,No Change,9,Anxious,67
65,23,Female,Generalized Anxiety,7,5,7,8,SSRIs,Cognitive Behavioral Therapy,2024-01-20,14,7,No Change,9,Excited,69
66,30,Male,Bipolar Disorder,5,4,4,9,Antipsychotics,Mindfulness-Based Therapy,2024-04-04,11,6,Deteriorated,8,Happy,68
67,28,Female,Major Depressive Disorder,10,6,4,1,Benzodiazepines,Mindfulness-Based Therapy,2024-02-24,15,7,Deteriorated,7,Depressed,71
68,35,Female,Major Depressive Disorder,7,3,5,9,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-18,14,8,Improved,10,Happy,75
69,39,Female,Major Depressive Disorder,10,7,4,3,Anxiolytics,Cognitive Behavioral Therapy,2024-01-31,15,10,No Change,10,Excited,62
70,52,Female,Bipolar Disorder,9,4,6,10,SSRIs,Interpersonal Therapy,2024-01-20,12,8,Improved,8,Anxious,68
71,51,Female,Generalized Anxiety,9,7,9,8,Benzodiazepines,Interpersonal Therapy,2024-03-23,9,5,Improved,5,Happy,86
72,57,Male,Major Depressive Disorder,5,4,5,3,Antipsychotics,Mindfulness-Based Therapy,2024-02-25,13,5,No Change,10,Excited,61
73,60,Female,Major Depressive Disorder,9,6,8,8,Benzodiazepines,Mindfulness-Based Therapy,2024-01-28,13,8,Deteriorated,9,Anxious,83
74,57,Male,Panic Disorder,5,5,8,9,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-02,13,7,Improved,5,Anxious,87
75,24,Male,Major Depressive Disorder,5,8,6,5,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-15,14,8,Deteriorated,7,Anxious,67
76,45,Female,Panic Disorder,9,3,6,1,Antidepressants,Mindfulness-Based Therapy,2024-02-12,14,10,Deteriorated,9,Depressed,82
77,21,Male,Generalized Anxiety,10,5,5,5,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-03,15,8,Improved,5,Happy,84
78,32,Female,Bipolar Disorder,5,8,6,1,Anxiolytics,Interpersonal Therapy,2024-01-14,11,5,No Change,10,Neutral,85
79,29,Male,Bipolar Disorder,10,5,7,2,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-29,14,5,Improved,7,Excited,80
80,29,Male,Generalized Anxiety,5,3,6,10,SSRIs,Cognitive Behavioral Therapy,2024-02-12,10,5,Improved,7,Excited,90
81,21,Male,Major Depressive Disorder,8,3,7,1,SSRIs,Interpersonal Therapy,2024-02-29,11,5,Deteriorated,10,Happy,83
82,23,Male,Major Depressive Disorder,6,6,6,6,Benzodiazepines,Interpersonal Therapy,2024-02-21,15,7,No Change,8,Excited,86
83,33,Male,Panic Disorder,8,6,7,8,SSRIs,Mindfulness-Based Therapy,2024-01-18,10,5,Improved,8,Depressed,63
84,31,Female,Generalized Anxiety,7,8,8,6,Antidepressants,Cognitive Behavioral Therapy,2024-03-25,14,9,Deteriorated,9,Excited,62
85,57,Male,Generalized Anxiety,10,6,7,1,SSRIs,Mindfulness-Based Therapy,2024-01-12,12,6,Deteriorated,6,Anxious,87
86,19,Male,Panic Disorder,7,3,5,5,Mood Stabilizers,Interpersonal Therapy,2024-03-04,11,5,No Change,6,Depressed,60
87,22,Male,Generalized Anxiety,6,6,6,6,Antidepressants,Mindfulness-Based Therapy,2024-01-19,15,6,Improved,5,Neutral,70
88,50,Male,Major Depressive Disorder,5,3,7,1,Antidepressants,Dialectical Behavioral Therapy,2024-03-18,12,9,No Change,9,Neutral,64
89,56,Female,Panic Disorder,9,7,9,10,SSRIs,Cognitive Behavioral Therapy,2024-03-18,10,7,No Change,10,Excited,79
90,22,Female,Major Depressive Disorder,9,7,7,4,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-13,16,6,Improved,6,Excited,61
91,55,Female,Generalized Anxiety,9,6,8,5,SSRIs,Dialectical Behavioral Therapy,2024-02-04,16,9,Improved,7,Stressed,87
92,43,Female,Bipolar Disorder,9,8,7,10,Anxiolytics,Dialectical Behavioral Therapy,2024-03-20,8,7,Deteriorated,9,Depressed,75
93,49,Female,Panic Disorder,9,6,8,5,Anxiolytics,Cognitive Behavioral Therapy,2024-01-20,8,9,Improved,7,Neutral,79
94,22,Female,Bipolar Disorder,8,3,8,5,Antidepressants,Mindfulness-Based Therapy,2024-03-02,14,6,No Change,6,Stressed,75
95,49,Male,Bipolar Disorder,6,8,8,6,Mood Stabilizers,Cognitive Behavioral Therapy,2024-01-26,16,10,No Change,10,Stressed,60
96,40,Female,Major Depressive Disorder,5,8,4,2,Mood Stabilizers,Interpersonal Therapy,2024-01-27,9,9,No Change,10,Stressed,72
97,51,Male,Major Depressive Disorder,6,5,6,8,Anxiolytics,Dialectical Behavioral Therapy,2024-03-09,14,6,Improved,8,Happy,85
98,46,Male,Major Depressive Disorder,10,8,9,3,Antidepressants,Dialectical Behavioral Therapy,2024-03-29,16,6,No Change,10,Excited,74
99,43,Female,Panic Disorder,6,5,8,4,Antipsychotics,Dialectical Behavioral Therapy,2024-02-12,11,5,Deteriorated,9,Neutral,78
100,24,Male,Generalized Anxiety,5,7,5,5,Antipsychotics,Cognitive Behavioral Therapy,2024-03-11,14,9,Improved,7,Excited,86
101,56,Male,Generalized Anxiety,7,7,6,5,SSRIs,Interpersonal Therapy,2024-03-12,13,8,No Change,9,Excited,79
102,23,Male,Bipolar Disorder,7,5,8,3,Anxiolytics,Cognitive Behavioral Therapy,2024-03-12,10,8,Deteriorated,5,Anxious,90
103,31,Male,Bipolar Disorder,6,4,6,5,Antidepressants,Mindfulness-Based Therapy,2024-01-16,12,7,Deteriorated,6,Anxious,83
104,36,Male,Major Depressive Disorder,9,6,4,3,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-19,10,5,No Change,10,Excited,85
105,44,Female,Generalized Anxiety,10,7,7,6,Antipsychotics,Dialectical Behavioral Therapy,2024-03-23,12,8,No Change,8,Stressed,66
106,19,Female,Panic Disorder,8,8,9,9,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-27,10,7,Improved,9,Happy,73
107,40,Male,Major Depressive Disorder,7,3,6,1,Anxiolytics,Cognitive Behavioral Therapy,2024-02-28,10,7,Improved,9,Excited,81
108,30,Male,Generalized Anxiety,9,7,7,8,Mood Stabilizers,Interpersonal Therapy,2024-01-27,13,6,Improved,7,Neutral,79
109,23,Male,Panic Disorder,7,3,4,6,Anxiolytics,Mindfulness-Based Therapy,2024-01-19,14,10,No Change,8,Depressed,78
110,30,Female,Bipolar Disorder,5,4,5,4,Antipsychotics,Dialectical Behavioral Therapy,2024-03-30,15,10,No Change,8,Happy,75
111,41,Male,Generalized Anxiety,5,8,8,2,Anxiolytics,Cognitive Behavioral Therapy,2024-01-03,14,9,No Change,5,Depressed,65
112,58,Female,Generalized Anxiety,10,8,6,3,Benzodiazepines,Mindfulness-Based Therapy,2024-02-23,11,8,Deteriorated,6,Depressed,76
113,37,Male,Bipolar Disorder,5,8,7,2,Anxiolytics,Mindfulness-Based Therapy,2024-03-05,9,6,Improved,8,Excited,82
114,20,Female,Major Depressive Disorder,5,8,4,2,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-17,10,5,Improved,5,Depressed,65
115,28,Male,Generalized Anxiety,7,7,8,3,Anxiolytics,Mindfulness-Based Therapy,2024-02-03,15,6,Improved,10,Neutral,89
116,33,Male,Bipolar Disorder,8,5,4,4,Antipsychotics,Mindfulness-Based Therapy,2024-03-30,12,8,Improved,8,Excited,60
117,43,Male,Bipolar Disorder,5,4,6,3,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-13,12,5,Improved,9,Depressed,85
118,31,Male,Bipolar Disorder,5,7,8,3,Antidepressants,Interpersonal Therapy,2024-03-27,8,5,Improved,9,Depressed,87
119,29,Male,Bipolar Disorder,9,6,6,10,Anxiolytics,Cognitive Behavioral Therapy,2024-03-15,16,9,No Change,6,Anxious,82
120,45,Female,Bipolar Disorder,8,6,9,3,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-16,13,8,Improved,10,Neutral,66
121,50,Male,Major Depressive Disorder,5,4,8,3,Anxiolytics,Dialectical Behavioral Therapy,2024-02-22,12,7,Deteriorated,10,Depressed,81
122,19,Male,Panic Disorder,8,4,6,3,Anxiolytics,Cognitive Behavioral Therapy,2024-01-11,16,7,Improved,8,Stressed,80
123,35,Female,Generalized Anxiety,10,6,6,5,SSRIs,Dialectical Behavioral Therapy,2024-02-03,12,8,Deteriorated,5,Stressed,85
124,50,Male,Major Depressive Disorder,8,4,5,5,SSRIs,Interpersonal Therapy,2024-01-14,9,7,No Change,5,Happy,84
125,53,Male,Generalized Anxiety,7,5,9,7,Benzodiazepines,Mindfulness-Based Therapy,2024-01-27,9,6,Deteriorated,7,Anxious,64
126,43,Male,Generalized Anxiety,7,3,9,8,Benzodiazepines,Cognitive Behavioral Therapy,2024-02-04,14,10,No Change,8,Anxious,77
127,47,Female,Bipolar Disorder,10,4,9,6,Antipsychotics,Interpersonal Therapy,2024-01-28,11,5,No Change,5,Neutral,65
128,53,Male,Major Depressive Disorder,5,3,8,5,Antipsychotics,Cognitive Behavioral Therapy,2024-04-03,9,8,No Change,7,Depressed,71
129,31,Male,Bipolar Disorder,6,3,4,9,Mood Stabilizers,Cognitive Behavioral Therapy,2024-01-24,16,10,Improved,8,Anxious,82
130,28,Male,Generalized Anxiety,9,4,5,7,Antipsychotics,Interpersonal Therapy,2024-02-07,13,10,Deteriorated,5,Excited,61
131,35,Male,Generalized Anxiety,9,5,6,9,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-07,10,5,Improved,8,Stressed,64
132,50,Male,Bipolar Disorder,9,5,7,4,Anxiolytics,Mindfulness-Based Therapy,2024-03-25,8,8,No Change,5,Happy,76
133,47,Female,Panic Disorder,5,6,9,6,Benzodiazepines,Mindfulness-Based Therapy,2024-03-28,13,8,No Change,10,Excited,67
134,41,Female,Major Depressive Disorder,10,4,9,3,Anxiolytics,Mindfulness-Based Therapy,2024-01-01,15,8,Deteriorated,8,Stressed,68
135,37,Female,Bipolar Disorder,9,3,6,3,Antidepressants,Mindfulness-Based Therapy,2024-01-06,15,8,No Change,6,Depressed,77
136,42,Female,Major Depressive Disorder,5,6,7,4,Anxiolytics,Cognitive Behavioral Therapy,2024-03-24,12,9,Improved,10,Happy,72
137,32,Male,Panic Disorder,7,7,8,7,Antidepressants,Interpersonal Therapy,2024-02-02,12,9,No Change,8,Happy,68
138,53,Male,Generalized Anxiety,9,8,5,6,SSRIs,Interpersonal Therapy,2024-01-15,9,7,No Change,9,Stressed,73
139,57,Female,Major Depressive Disorder,10,6,5,6,Antidepressants,Cognitive Behavioral Therapy,2024-03-20,15,9,Deteriorated,5,Neutral,63
140,57,Male,Generalized Anxiety,8,5,5,7,SSRIs,Interpersonal Therapy,2024-01-04,9,9,Improved,6,Depressed,85
141,41,Female,Bipolar Disorder,7,5,5,8,Anxiolytics,Dialectical Behavioral Therapy,2024-02-08,12,6,Deteriorated,5,Neutral,79
142,55,Male,Generalized Anxiety,10,3,9,3,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-20,8,10,Improved,5,Anxious,73
143,50,Male,Generalized Anxiety,6,5,5,5,Antipsychotics,Dialectical Behavioral Therapy,2024-03-11,14,9,Improved,5,Stressed,87
144,21,Male,Bipolar Disorder,9,5,7,3,Benzodiazepines,Mindfulness-Based Therapy,2024-01-17,9,6,No Change,6,Stressed,90
145,21,Female,Major Depressive Disorder,8,8,7,1,Antipsychotics,Interpersonal Therapy,2024-02-19,12,5,No Change,7,Happy,76
146,29,Male,Panic Disorder,10,6,4,3,SSRIs,Cognitive Behavioral Therapy,2024-03-28,13,8,Deteriorated,10,Anxious,78
147,48,Male,Major Depressive Disorder,8,5,9,9,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-17,11,9,Improved,9,Excited,84
148,45,Female,Generalized Anxiety,6,3,4,3,Mood Stabilizers,Interpersonal Therapy,2024-03-29,13,5,No Change,6,Anxious,65
149,46,Male,Generalized Anxiety,5,3,7,10,Anxiolytics,Mindfulness-Based Therapy,2024-01-08,12,8,Deteriorated,6,Stressed,70
150,47,Male,Bipolar Disorder,10,6,5,7,Antidepressants,Dialectical Behavioral Therapy,2024-04-06,10,5,Improved,7,Happy,73
151,22,Female,Panic Disorder,10,4,8,3,SSRIs,Cognitive Behavioral Therapy,2024-03-06,12,10,Improved,8,Stressed,65
152,19,Male,Panic Disorder,7,4,5,10,SSRIs,Dialectical Behavioral Therapy,2024-01-11,14,5,Improved,6,Neutral,83
153,58,Male,Generalized Anxiety,5,5,5,10,Antidepressants,Interpersonal Therapy,2024-02-17,8,8,Improved,8,Anxious,68
154,21,Female,Panic Disorder,9,8,8,1,Mood Stabilizers,Interpersonal Therapy,2024-03-21,11,9,No Change,10,Anxious,87
155,26,Female,Major Depressive Disorder,5,4,5,4,Benzodiazepines,Dialectical Behavioral Therapy,2024-01-24,9,6,Deteriorated,8,Excited,65
156,27,Female,Generalized Anxiety,7,3,7,5,Antidepressants,Mindfulness-Based Therapy,2024-03-24,12,10,No Change,7,Excited,66
157,44,Male,Panic Disorder,6,8,7,6,Antipsychotics,Dialectical Behavioral Therapy,2024-03-26,14,8,Improved,10,Happy,78
158,60,Male,Generalized Anxiety,10,6,6,10,Anxiolytics,Dialectical Behavioral Therapy,2024-01-06,9,5,Deteriorated,8,Anxious,76
159,55,Female,Major Depressive Disorder,5,5,7,3,SSRIs,Cognitive Behavioral Therapy,2024-01-24,12,9,No Change,7,Stressed,84
160,26,Female,Generalized Anxiety,5,3,7,4,Benzodiazepines,Mindfulness-Based Therapy,2024-02-08,15,8,Improved,10,Excited,67
161,18,Male,Panic Disorder,7,6,4,4,SSRIs,Interpersonal Therapy,2024-03-25,16,10,No Change,5,Anxious,85
162,22,Male,Generalized Anxiety,5,3,6,3,Benzodiazepines,Mindfulness-Based Therapy,2024-02-04,16,5,No Change,9,Anxious,73
163,21,Female,Panic Disorder,8,6,5,6,Antipsychotics,Dialectical Behavioral Therapy,2024-02-08,13,7,Deteriorated,5,Anxious,66
164,35,Male,Major Depressive Disorder,6,6,4,2,Benzodiazepines,Mindfulness-Based Therapy,2024-01-27,10,10,No Change,7,Anxious,75
165,37,Male,Bipolar Disorder,8,6,8,3,Anxiolytics,Mindfulness-Based Therapy,2024-01-13,8,7,Improved,10,Anxious,65
166,52,Female,Generalized Anxiety,5,6,6,1,Benzodiazepines,Mindfulness-Based Therapy,2024-03-25,15,8,Improved,5,Happy,90
167,19,Male,Major Depressive Disorder,8,8,9,2,Antipsychotics,Dialectical Behavioral Therapy,2024-03-12,12,5,No Change,9,Depressed,74
168,23,Male,Major Depressive Disorder,10,3,7,7,SSRIs,Interpersonal Therapy,2024-02-13,16,10,No Change,10,Excited,90
169,28,Male,Panic Disorder,6,3,9,7,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-02,16,10,Improved,10,Stressed,72
170,56,Female,Generalized Anxiety,8,6,5,4,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-24,11,6,Improved,5,Happy,75
171,24,Male,Major Depressive Disorder,5,4,4,8,Anxiolytics,Dialectical Behavioral Therapy,2024-01-12,14,10,Deteriorated,7,Depressed,75
172,47,Female,Bipolar Disorder,7,7,6,6,Antidepressants,Cognitive Behavioral Therapy,2024-03-11,16,9,Deteriorated,6,Neutral,63
173,19,Male,Panic Disorder,8,6,8,5,Benzodiazepines,Mindfulness-Based Therapy,2024-02-08,15,6,Improved,9,Stressed,71
174,55,Female,Bipolar Disorder,6,3,5,8,Benzodiazepines,Mindfulness-Based Therapy,2024-03-25,12,5,Improved,7,Neutral,61
175,51,Male,Major Depressive Disorder,9,8,5,7,SSRIs,Interpersonal Therapy,2024-01-07,13,6,No Change,6,Neutral,85
176,24,Male,Panic Disorder,8,6,6,4,Benzodiazepines,Interpersonal Therapy,2024-03-12,9,10,Improved,9,Stressed,76
177,31,Female,Major Depressive Disorder,8,7,5,7,Antipsychotics,Interpersonal Therapy,2024-02-08,11,5,Deteriorated,9,Happy,86
178,56,Male,Major Depressive Disorder,6,4,4,3,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-16,12,6,Deteriorated,7,Anxious,76
179,41,Male,Panic Disorder,6,5,7,3,Benzodiazepines,Cognitive Behavioral Therapy,2024-02-09,15,9,Deteriorated,9,Depressed,62
180,19,Male,Generalized Anxiety,7,7,7,10,Anxiolytics,Dialectical Behavioral Therapy,2024-01-12,13,7,No Change,6,Excited,73
181,23,Male,Generalized Anxiety,10,8,8,1,Antidepressants,Mindfulness-Based Therapy,2024-03-22,14,9,Improved,10,Anxious,89
182,28,Male,Major Depressive Disorder,6,5,6,6,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-01,12,7,Improved,5,Neutral,60
183,18,Female,Bipolar Disorder,10,7,8,10,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-10,16,8,No Change,7,Depressed,89
184,59,Female,Major Depressive Disorder,8,4,7,5,Antidepressants,Mindfulness-Based Therapy,2024-02-05,9,9,Deteriorated,8,Neutral,62
185,40,Female,Panic Disorder,8,7,6,1,Anxiolytics,Interpersonal Therapy,2024-02-16,9,10,No Change,10,Excited,76
186,54,Female,Generalized Anxiety,10,6,5,9,Mood Stabilizers,Interpersonal Therapy,2024-01-14,12,7,Deteriorated,5,Neutral,68
187,25,Male,Generalized Anxiety,7,8,9,4,Mood Stabilizers,Cognitive Behavioral Therapy,2024-01-04,9,7,Deteriorated,10,Neutral,66
188,36,Male,Major Depressive Disorder,8,8,5,8,Antipsychotics,Interpersonal Therapy,2024-03-22,16,7,No Change,5,Happy,67
189,60,Female,Generalized Anxiety,10,7,5,7,Antipsychotics,Interpersonal Therapy,2024-01-06,8,5,No Change,8,Anxious,69
190,35,Male,Major Depressive Disorder,7,8,8,10,SSRIs,Cognitive Behavioral Therapy,2024-02-23,13,7,No Change,6,Excited,65
191,56,Female,Panic Disorder,7,8,4,3,Antidepressants,Interpersonal Therapy,2024-01-16,9,5,Improved,7,Excited,85
192,58,Female,Bipolar Disorder,6,6,6,1,Benzodiazepines,Interpersonal Therapy,2024-04-02,9,5,No Change,10,Anxious,78
193,47,Male,Bipolar Disorder,10,3,7,8,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-28,14,9,Improved,8,Depressed,64
194,53,Female,Panic Disorder,6,6,8,6,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-01,11,10,Improved,7,Neutral,82
195,27,Male,Panic Disorder,8,3,6,8,SSRIs,Interpersonal Therapy,2024-01-13,13,8,Deteriorated,9,Anxious,73
196,27,Female,Bipolar Disorder,5,8,4,4,SSRIs,Dialectical Behavioral Therapy,2024-02-13,10,6,No Change,7,Excited,90
197,50,Male,Generalized Anxiety,8,7,8,10,Anxiolytics,Dialectical Behavioral Therapy,2024-04-08,14,7,No Change,7,Happy,70
198,31,Male,Panic Disorder,6,7,8,5,Anxiolytics,Cognitive Behavioral Therapy,2024-01-17,9,6,Deteriorated,8,Depressed,85
199,24,Female,Major Depressive Disorder,10,6,9,4,Benzodiazepines,Mindfulness-Based Therapy,2024-01-22,14,8,No Change,6,Happy,80
200,54,Female,Bipolar Disorder,9,7,9,5,Benzodiazepines,Mindfulness-Based Therapy,2024-01-03,15,9,Deteriorated,5,Excited,61
201,24,Female,Panic Disorder,7,7,9,4,SSRIs,Dialectical Behavioral Therapy,2024-01-16,15,9,No Change,8,Neutral,87
202,50,Male,Generalized Anxiety,7,3,9,5,Antipsychotics,Dialectical Behavioral Therapy,2024-04-01,11,9,Deteriorated,5,Anxious,78
203,30,Male,Major Depressive Disorder,7,6,6,6,SSRIs,Mindfulness-Based Therapy,2024-02-23,11,7,No Change,8,Neutral,79
204,48,Male,Major Depressive Disorder,9,6,4,6,SSRIs,Dialectical Behavioral Therapy,2024-02-18,14,6,No Change,8,Stressed,60
205,26,Male,Generalized Anxiety,7,4,6,6,Anxiolytics,Dialectical Behavioral Therapy,2024-02-09,11,9,No Change,7,Happy,81
206,31,Male,Panic Disorder,7,3,4,3,Antidepressants,Dialectical Behavioral Therapy,2024-03-03,13,9,No Change,9,Happy,70
207,43,Female,Bipolar Disorder,5,5,6,9,Anxiolytics,Mindfulness-Based Therapy,2024-03-23,12,10,Improved,6,Neutral,65
208,59,Female,Panic Disorder,8,7,4,5,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-29,13,6,No Change,6,Neutral,68
209,26,Male,Generalized Anxiety,10,8,8,5,Antidepressants,Interpersonal Therapy,2024-01-26,13,6,Improved,8,Anxious,66
210,50,Male,Generalized Anxiety,6,5,9,3,Mood Stabilizers,Dialectical Behavioral Therapy,2024-04-09,12,7,Improved,9,Anxious,76
211,46,Female,Bipolar Disorder,6,5,7,8,SSRIs,Cognitive Behavioral Therapy,2024-03-24,14,9,No Change,8,Anxious,82
212,40,Female,Major Depressive Disorder,7,4,8,8,Antidepressants,Dialectical Behavioral Therapy,2024-02-29,10,10,Deteriorated,10,Anxious,81
213,46,Male,Bipolar Disorder,7,5,4,1,Benzodiazepines,Mindfulness-Based Therapy,2024-01-17,9,10,Deteriorated,6,Anxious,86
214,51,Female,Major Depressive Disorder,8,3,4,7,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-21,11,5,Improved,6,Depressed,82
215,39,Male,Panic Disorder,5,6,4,4,SSRIs,Interpersonal Therapy,2024-01-08,8,9,Deteriorated,7,Depressed,74
216,21,Female,Generalized Anxiety,7,4,7,8,Anxiolytics,Cognitive Behavioral Therapy,2024-01-13,8,8,No Change,8,Stressed,89
217,49,Female,Panic Disorder,7,7,8,3,SSRIs,Cognitive Behavioral Therapy,2024-02-02,14,9,Improved,7,Excited,63
218,34,Female,Major Depressive Disorder,10,3,5,8,Benzodiazepines,Interpersonal Therapy,2024-02-16,11,7,No Change,7,Happy,70
219,20,Female,Bipolar Disorder,10,4,6,8,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-08,14,8,No Change,10,Happy,62
220,44,Female,Major Depressive Disorder,10,4,6,9,Anxiolytics,Interpersonal Therapy,2024-03-15,11,5,Improved,5,Excited,81
221,40,Male,Generalized Anxiety,5,3,7,4,Mood Stabilizers,Interpersonal Therapy,2024-01-21,9,8,Improved,9,Anxious,65
222,49,Female,Major Depressive Disorder,8,6,7,1,Antidepressants,Dialectical Behavioral Therapy,2024-03-12,10,9,No Change,8,Neutral,85
223,60,Male,Panic Disorder,9,5,8,4,Antipsychotics,Interpersonal Therapy,2024-02-14,13,5,No Change,6,Neutral,65
224,47,Male,Bipolar Disorder,5,5,6,5,Antipsychotics,Interpersonal Therapy,2024-02-06,15,9,No Change,10,Anxious,63
225,51,Female,Panic Disorder,8,5,8,10,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-20,8,10,Improved,8,Stressed,83
226,21,Male,Generalized Anxiety,8,8,4,2,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-22,15,7,No Change,5,Neutral,88
227,33,Male,Major Depressive Disorder,6,6,7,1,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-12,9,6,Improved,10,Excited,70
228,22,Male,Generalized Anxiety,6,3,9,1,Antipsychotics,Mindfulness-Based Therapy,2024-02-21,16,6,No Change,8,Anxious,77
229,41,Female,Generalized Anxiety,8,6,5,7,Anxiolytics,Interpersonal Therapy,2024-02-13,11,5,Deteriorated,8,Excited,85
230,40,Female,Generalized Anxiety,5,5,7,1,Antipsychotics,Dialectical Behavioral Therapy,2024-03-10,16,6,No Change,6,Stressed,90
231,22,Female,Panic Disorder,5,8,9,1,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-14,12,6,No Change,5,Excited,74
232,34,Male,Generalized Anxiety,9,6,4,1,Antipsychotics,Interpersonal Therapy,2024-03-09,14,7,No Change,10,Happy,63
233,55,Female,Bipolar Disorder,10,6,7,5,SSRIs,Dialectical Behavioral Therapy,2024-04-03,10,7,Deteriorated,9,Neutral,80
234,18,Female,Bipolar Disorder,8,6,7,7,SSRIs,Dialectical Behavioral Therapy,2024-01-26,11,9,No Change,7,Anxious,87
235,29,Male,Generalized Anxiety,9,5,9,10,SSRIs,Interpersonal Therapy,2024-03-19,10,10,Improved,5,Neutral,60
236,27,Male,Generalized Anxiety,6,6,6,2,Anxiolytics,Dialectical Behavioral Therapy,2024-01-01,12,10,Improved,10,Anxious,77
237,50,Male,Bipolar Disorder,8,8,7,5,Antipsychotics,Mindfulness-Based Therapy,2024-01-12,12,7,Deteriorated,7,Happy,90
238,18,Male,Bipolar Disorder,9,4,4,7,SSRIs,Interpersonal Therapy,2024-03-14,12,6,Improved,7,Happy,83
239,54,Female,Major Depressive Disorder,6,4,8,3,Mood Stabilizers,Interpersonal Therapy,2024-01-18,9,10,Deteriorated,9,Anxious,78
240,60,Male,Generalized Anxiety,8,8,8,1,SSRIs,Dialectical Behavioral Therapy,2024-01-22,14,8,Deteriorated,5,Stressed,82
241,20,Female,Bipolar Disorder,10,4,6,7,Antipsychotics,Cognitive Behavioral Therapy,2024-03-30,9,5,Deteriorated,6,Neutral,87
242,38,Female,Generalized Anxiety,9,3,5,8,Mood Stabilizers,Mindfulness-Based Therapy,2024-04-10,16,8,No Change,9,Anxious,66
243,18,Male,Panic Disorder,9,8,5,1,Anxiolytics,Dialectical Behavioral Therapy,2024-02-01,15,8,No Change,8,Stressed,84
244,24,Female,Generalized Anxiety,7,5,9,10,Antipsychotics,Interpersonal Therapy,2024-01-05,9,10,Improved,7,Anxious,89
245,20,Female,Panic Disorder,8,6,4,4,Antipsychotics,Dialectical Behavioral Therapy,2024-02-19,11,10,Deteriorated,5,Depressed,89
246,26,Female,Generalized Anxiety,10,6,8,10,SSRIs,Dialectical Behavioral Therapy,2024-02-21,8,9,No Change,8,Stressed,82
247,54,Male,Panic Disorder,9,4,8,3,Antidepressants,Mindfulness-Based Therapy,2024-03-20,12,10,Improved,9,Neutral,90
248,44,Female,Major Depressive Disorder,6,7,4,3,Antidepressants,Cognitive Behavioral Therapy,2024-03-19,13,8,Deteriorated,5,Stressed,87
249,47,Male,Panic Disorder,7,6,4,5,Antidepressants,Interpersonal Therapy,2024-03-29,13,6,Deteriorated,8,Stressed,66
250,32,Male,Major Depressive Disorder,8,5,5,4,Benzodiazepines,Mindfulness-Based Therapy,2024-04-06,9,6,No Change,10,Anxious,82
251,56,Male,Bipolar Disorder,7,8,9,10,Antipsychotics,Dialectical Behavioral Therapy,2024-02-01,12,10,Improved,10,Depressed,84
252,30,Female,Bipolar Disorder,6,4,8,2,Antipsychotics,Mindfulness-Based Therapy,2024-02-01,15,6,No Change,5,Anxious,87
253,31,Male,Panic Disorder,8,7,6,9,SSRIs,Cognitive Behavioral Therapy,2024-03-23,12,5,Improved,5,Anxious,90
254,38,Male,Bipolar Disorder,5,7,4,6,Anxiolytics,Interpersonal Therapy,2024-02-22,14,9,Improved,6,Stressed,88
255,55,Female,Panic Disorder,7,5,9,3,SSRIs,Interpersonal Therapy,2024-01-08,12,6,Improved,8,Stressed,85
256,26,Male,Major Depressive Disorder,8,6,8,6,Antidepressants,Dialectical Behavioral Therapy,2024-04-06,8,10,No Change,8,Excited,75
257,26,Male,Generalized Anxiety,10,8,8,7,Antidepressants,Dialectical Behavioral Therapy,2024-03-11,14,8,Deteriorated,10,Neutral,72
258,22,Female,Panic Disorder,8,6,7,10,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-02,10,6,Deteriorated,7,Excited,63
259,31,Female,Panic Disorder,10,3,7,1,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-16,14,9,Improved,7,Excited,68
260,44,Male,Generalized Anxiety,6,6,5,6,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-21,13,10,Improved,5,Excited,88
261,20,Male,Generalized Anxiety,6,4,6,2,Anxiolytics,Dialectical Behavioral Therapy,2024-01-14,13,6,Deteriorated,10,Neutral,88
262,51,Female,Major Depressive Disorder,6,4,6,3,Antidepressants,Cognitive Behavioral Therapy,2024-02-21,16,7,Deteriorated,9,Happy,66
263,37,Female,Major Depressive Disorder,8,8,9,6,Anxiolytics,Cognitive Behavioral Therapy,2024-02-16,12,8,Deteriorated,10,Excited,62
264,35,Female,Generalized Anxiety,7,5,9,1,Benzodiazepines,Dialectical Behavioral Therapy,2024-01-24,13,5,No Change,5,Neutral,82
265,26,Female,Bipolar Disorder,7,7,8,3,Benzodiazepines,Mindfulness-Based Therapy,2024-03-18,11,5,Deteriorated,5,Excited,68
266,31,Female,Major Depressive Disorder,7,6,8,4,SSRIs,Mindfulness-Based Therapy,2024-03-15,12,7,Improved,8,Neutral,82
267,23,Male,Major Depressive Disorder,10,8,8,7,Anxiolytics,Interpersonal Therapy,2024-03-10,10,7,No Change,8,Anxious,69
268,52,Female,Panic Disorder,8,6,4,3,Anxiolytics,Cognitive Behavioral Therapy,2024-04-09,13,10,Improved,9,Happy,77
269,35,Female,Major Depressive Disorder,8,4,6,3,Benzodiazepines,Interpersonal Therapy,2024-02-01,8,6,No Change,10,Stressed,82
270,36,Male,Panic Disorder,9,3,8,8,Benzodiazepines,Mindfulness-Based Therapy,2024-01-05,11,9,Deteriorated,8,Neutral,67
271,30,Male,Bipolar Disorder,8,5,4,2,Antipsychotics,Mindfulness-Based Therapy,2024-02-07,12,10,Improved,5,Neutral,79
272,51,Male,Bipolar Disorder,10,7,6,1,SSRIs,Dialectical Behavioral Therapy,2024-01-06,16,5,Deteriorated,7,Depressed,70
273,56,Female,Generalized Anxiety,5,6,7,8,Anxiolytics,Cognitive Behavioral Therapy,2024-02-21,16,10,No Change,7,Happy,67
274,33,Female,Major Depressive Disorder,10,6,8,5,Anxiolytics,Mindfulness-Based Therapy,2024-04-03,8,10,Improved,7,Stressed,78
275,46,Male,Generalized Anxiety,9,7,9,8,Antipsychotics,Dialectical Behavioral Therapy,2024-01-09,14,8,No Change,10,Depressed,84
276,45,Male,Bipolar Disorder,6,4,8,10,Antipsychotics,Mindfulness-Based Therapy,2024-03-09,9,10,Deteriorated,9,Anxious,76
277,24,Male,Panic Disorder,5,3,6,4,Antipsychotics,Cognitive Behavioral Therapy,2024-02-15,11,6,No Change,7,Excited,76
278,49,Female,Generalized Anxiety,8,3,7,1,Benzodiazepines,Mindfulness-Based Therapy,2024-02-03,10,8,Improved,8,Neutral,73
279,59,Female,Panic Disorder,6,5,7,5,SSRIs,Dialectical Behavioral Therapy,2024-01-24,8,6,No Change,9,Stressed,73
280,32,Male,Panic Disorder,6,7,9,2,Anxiolytics,Dialectical Behavioral Therapy,2024-03-28,9,10,No Change,6,Excited,71
281,48,Male,Panic Disorder,6,5,5,5,Mood Stabilizers,Mindfulness-Based Therapy,2024-02-24,15,6,Deteriorated,7,Happy,80
282,35,Male,Major Depressive Disorder,8,5,5,8,Anxiolytics,Interpersonal Therapy,2024-03-12,14,9,Deteriorated,5,Anxious,61
283,52,Male,Major Depressive Disorder,8,3,5,1,SSRIs,Dialectical Behavioral Therapy,2024-02-14,13,9,Deteriorated,9,Stressed,81
284,44,Female,Bipolar Disorder,7,8,8,8,Mood Stabilizers,Interpersonal Therapy,2024-02-04,9,5,Improved,8,Stressed,72
285,28,Female,Generalized Anxiety,10,5,9,5,Anxiolytics,Interpersonal Therapy,2024-03-31,14,9,No Change,8,Excited,84
286,35,Male,Generalized Anxiety,9,5,4,1,Antipsychotics,Mindfulness-Based Therapy,2024-02-11,16,7,Deteriorated,7,Stressed,72
287,41,Male,Bipolar Disorder,8,6,5,7,Antidepressants,Interpersonal Therapy,2024-03-25,10,5,Improved,8,Excited,62
288,49,Male,Major Depressive Disorder,6,6,7,9,Benzodiazepines,Interpersonal Therapy,2024-01-04,11,7,No Change,5,Anxious,83
289,51,Female,Bipolar Disorder,7,7,9,4,Mood Stabilizers,Interpersonal Therapy,2024-03-14,14,6,No Change,5,Neutral,77
290,26,Male,Bipolar Disorder,5,7,5,4,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-16,11,10,Improved,8,Neutral,83
291,54,Male,Panic Disorder,7,6,7,3,Anxiolytics,Cognitive Behavioral Therapy,2024-02-08,14,5,No Change,10,Excited,62
292,30,Female,Panic Disorder,9,7,5,8,Anxiolytics,Interpersonal Therapy,2024-02-28,11,5,Improved,9,Neutral,86
293,51,Female,Generalized Anxiety,6,4,9,3,Antidepressants,Mindfulness-Based Therapy,2024-02-02,16,6,Deteriorated,10,Stressed,84
294,51,Female,Bipolar Disorder,7,8,6,4,Anxiolytics,Mindfulness-Based Therapy,2024-01-07,9,9,Improved,10,Neutral,81
295,38,Male,Bipolar Disorder,8,4,5,8,Benzodiazepines,Dialectical Behavioral Therapy,2024-01-26,14,7,Deteriorated,9,Happy,77
296,26,Male,Panic Disorder,6,4,7,7,Antipsychotics,Interpersonal Therapy,2024-03-02,16,7,Improved,8,Anxious,90
297,58,Female,Generalized Anxiety,7,3,6,4,Benzodiazepines,Interpersonal Therapy,2024-03-04,9,6,Deteriorated,9,Stressed,86
298,56,Male,Generalized Anxiety,6,3,4,10,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-16,16,8,No Change,8,Happy,85
299,60,Female,Generalized Anxiety,7,6,4,8,Antipsychotics,Cognitive Behavioral Therapy,2024-01-09,13,6,Improved,8,Neutral,72
300,32,Male,Major Depressive Disorder,5,8,4,3,Mood Stabilizers,Interpersonal Therapy,2024-01-24,11,7,No Change,7,Stressed,62
301,54,Female,Generalized Anxiety,9,4,4,6,Benzodiazepines,Mindfulness-Based Therapy,2024-01-11,10,8,Improved,8,Stressed,81
302,55,Female,Generalized Anxiety,7,5,7,6,Antidepressants,Mindfulness-Based Therapy,2024-02-16,14,5,Deteriorated,5,Anxious,73
303,32,Male,Generalized Anxiety,6,5,9,2,Antipsychotics,Interpersonal Therapy,2024-01-24,13,5,Improved,9,Anxious,77
304,37,Female,Major Depressive Disorder,5,7,6,4,Benzodiazepines,Interpersonal Therapy,2024-01-05,13,5,Improved,9,Stressed,64
305,21,Female,Major Depressive Disorder,6,4,5,1,SSRIs,Interpersonal Therapy,2024-02-17,14,10,No Change,10,Happy,74
306,27,Male,Bipolar Disorder,6,4,6,4,Antipsychotics,Cognitive Behavioral Therapy,2024-02-01,13,9,Deteriorated,6,Excited,83
307,51,Female,Bipolar Disorder,10,3,9,1,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-10,8,8,Deteriorated,6,Happy,80
308,60,Male,Panic Disorder,6,6,6,10,SSRIs,Dialectical Behavioral Therapy,2024-02-12,14,8,Improved,10,Depressed,81
309,24,Male,Panic Disorder,8,7,7,6,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-23,10,8,No Change,7,Neutral,82
310,47,Male,Panic Disorder,6,8,6,9,SSRIs,Cognitive Behavioral Therapy,2024-02-20,11,8,Deteriorated,5,Anxious,73
311,30,Female,Panic Disorder,9,6,8,3,Antipsychotics,Cognitive Behavioral Therapy,2024-03-21,10,10,Improved,8,Anxious,64
312,32,Male,Panic Disorder,9,4,9,7,Anxiolytics,Mindfulness-Based Therapy,2024-02-02,11,6,Deteriorated,8,Depressed,78
313,20,Male,Bipolar Disorder,10,4,6,6,Benzodiazepines,Interpersonal Therapy,2024-03-11,16,7,No Change,5,Depressed,69
314,28,Male,Panic Disorder,9,5,6,4,SSRIs,Interpersonal Therapy,2024-01-17,15,7,Improved,10,Depressed,66
315,20,Female,Panic Disorder,5,8,9,10,Anxiolytics,Cognitive Behavioral Therapy,2024-03-16,15,6,No Change,10,Anxious,64
316,28,Female,Generalized Anxiety,10,4,6,6,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-07,14,6,No Change,6,Stressed,87
317,60,Female,Major Depressive Disorder,10,3,9,2,Benzodiazepines,Mindfulness-Based Therapy,2024-03-19,9,6,No Change,6,Neutral,77
318,37,Male,Major Depressive Disorder,10,5,5,2,Benzodiazepines,Mindfulness-Based Therapy,2024-02-29,9,6,No Change,7,Anxious,63
319,45,Male,Bipolar Disorder,8,4,9,3,Benzodiazepines,Mindfulness-Based Therapy,2024-01-10,12,10,Deteriorated,5,Anxious,69
320,49,Male,Generalized Anxiety,7,3,6,1,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-21,12,10,Improved,10,Happy,85
321,44,Male,Bipolar Disorder,10,5,8,7,Anxiolytics,Mindfulness-Based Therapy,2024-03-25,16,9,Deteriorated,7,Neutral,87
322,18,Male,Major Depressive Disorder,5,3,6,5,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-01,12,10,Improved,9,Excited,63
323,50,Female,Major Depressive Disorder,6,6,4,4,Benzodiazepines,Interpersonal Therapy,2024-04-06,9,7,No Change,7,Stressed,60
324,33,Male,Panic Disorder,6,8,8,1,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-14,9,8,Improved,5,Depressed,85
325,33,Male,Panic Disorder,9,7,6,9,SSRIs,Mindfulness-Based Therapy,2024-02-02,12,6,Improved,8,Happy,65
326,36,Male,Bipolar Disorder,9,7,9,5,Antipsychotics,Dialectical Behavioral Therapy,2024-02-26,12,10,No Change,5,Anxious,89
327,37,Male,Bipolar Disorder,7,6,5,6,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-11,13,5,Improved,8,Excited,62
328,42,Female,Major Depressive Disorder,8,3,5,3,Anxiolytics,Dialectical Behavioral Therapy,2024-01-11,16,9,No Change,10,Stressed,85
329,43,Male,Panic Disorder,5,3,5,4,Mood Stabilizers,Mindfulness-Based Therapy,2024-04-08,9,8,No Change,8,Anxious,88
330,34,Male,Generalized Anxiety,8,3,6,1,Antidepressants,Interpersonal Therapy,2024-01-06,11,9,Improved,6,Stressed,67
331,50,Male,Major Depressive Disorder,7,8,9,4,SSRIs,Mindfulness-Based Therapy,2024-01-06,8,5,No Change,5,Neutral,69
332,46,Male,Major Depressive Disorder,7,7,5,8,Antipsychotics,Cognitive Behavioral Therapy,2024-03-01,9,10,Deteriorated,8,Happy,72
333,32,Female,Bipolar Disorder,7,7,6,6,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-28,15,9,No Change,9,Anxious,74
334,55,Male,Bipolar Disorder,5,3,9,2,Antipsychotics,Cognitive Behavioral Therapy,2024-03-01,14,6,No Change,10,Excited,67
335,43,Female,Panic Disorder,8,8,6,6,Antipsychotics,Interpersonal Therapy,2024-01-18,9,9,Deteriorated,5,Excited,76
336,34,Male,Bipolar Disorder,8,5,9,7,Antidepressants,Cognitive Behavioral Therapy,2024-02-15,16,5,Deteriorated,10,Neutral,86
337,60,Male,Generalized Anxiety,10,8,9,2,Mood Stabilizers,Interpersonal Therapy,2024-01-21,15,5,No Change,8,Happy,71
338,51,Female,Bipolar Disorder,8,8,5,8,SSRIs,Cognitive Behavioral Therapy,2024-03-20,11,6,Deteriorated,9,Excited,89
339,48,Male,Panic Disorder,7,7,8,3,Antidepressants,Mindfulness-Based Therapy,2024-01-07,15,10,Deteriorated,5,Excited,76
340,42,Female,Generalized Anxiety,7,5,4,8,Antidepressants,Mindfulness-Based Therapy,2024-01-10,8,9,No Change,5,Neutral,88
341,21,Male,Generalized Anxiety,9,3,4,6,Antipsychotics,Dialectical Behavioral Therapy,2024-02-25,11,7,Improved,8,Happy,62
342,25,Female,Panic Disorder,6,4,4,2,Antipsychotics,Interpersonal Therapy,2024-02-08,8,9,Deteriorated,6,Neutral,75
343,24,Female,Bipolar Disorder,7,8,9,9,Mood Stabilizers,Interpersonal Therapy,2024-02-25,8,10,Deteriorated,9,Excited,66
344,35,Male,Generalized Anxiety,9,3,5,9,Anxiolytics,Mindfulness-Based Therapy,2024-04-10,15,8,Improved,10,Neutral,76
345,20,Female,Major Depressive Disorder,5,8,4,7,Benzodiazepines,Mindfulness-Based Therapy,2024-01-15,15,6,No Change,8,Excited,85
346,56,Male,Generalized Anxiety,6,7,8,7,Antidepressants,Interpersonal Therapy,2024-01-19,9,10,Deteriorated,7,Neutral,66
347,32,Female,Panic Disorder,6,5,8,2,Mood Stabilizers,Cognitive Behavioral Therapy,2024-01-13,12,5,Improved,5,Stressed,86
348,35,Female,Bipolar Disorder,6,8,4,1,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-30,14,9,Improved,6,Neutral,83
349,34,Female,Major Depressive Disorder,9,7,4,5,SSRIs,Cognitive Behavioral Therapy,2024-01-12,9,8,No Change,5,Stressed,85
350,47,Male,Major Depressive Disorder,10,8,8,1,Antipsychotics,Mindfulness-Based Therapy,2024-02-16,9,5,Deteriorated,6,Anxious,62
351,33,Male,Panic Disorder,5,5,5,6,SSRIs,Cognitive Behavioral Therapy,2024-03-21,10,6,Deteriorated,10,Anxious,80
352,39,Male,Bipolar Disorder,9,7,8,3,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-20,12,9,No Change,9,Happy,63
353,57,Female,Panic Disorder,10,8,4,9,Antidepressants,Cognitive Behavioral Therapy,2024-01-14,12,7,Deteriorated,9,Depressed,77
354,34,Male,Generalized Anxiety,9,8,5,8,Antipsychotics,Cognitive Behavioral Therapy,2024-03-23,11,5,Improved,6,Excited,85
355,27,Male,Panic Disorder,8,5,6,6,Benzodiazepines,Interpersonal Therapy,2024-03-09,11,8,Deteriorated,8,Excited,75
356,40,Male,Major Depressive Disorder,5,5,5,5,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-23,9,10,No Change,6,Anxious,90
357,60,Male,Panic Disorder,5,5,5,4,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-15,9,8,Deteriorated,7,Neutral,85
358,55,Male,Major Depressive Disorder,10,5,6,10,Antipsychotics,Cognitive Behavioral Therapy,2024-02-03,10,5,Deteriorated,7,Anxious,78
359,49,Female,Bipolar Disorder,5,3,8,6,Antidepressants,Mindfulness-Based Therapy,2024-02-22,12,6,Deteriorated,7,Happy,87
360,46,Female,Generalized Anxiety,6,8,9,10,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-21,9,7,Deteriorated,9,Depressed,84
361,22,Female,Major Depressive Disorder,10,3,9,4,Anxiolytics,Dialectical Behavioral Therapy,2024-03-24,15,7,Deteriorated,5,Happy,67
362,56,Male,Panic Disorder,5,7,7,7,SSRIs,Mindfulness-Based Therapy,2024-01-21,10,10,Deteriorated,6,Stressed,69
363,33,Female,Major Depressive Disorder,5,7,5,1,Antipsychotics,Mindfulness-Based Therapy,2024-01-13,8,7,Improved,5,Anxious,76
364,53,Male,Major Depressive Disorder,8,6,6,9,Mood Stabilizers,Interpersonal Therapy,2024-01-16,12,10,Improved,9,Anxious,85
365,50,Female,Major Depressive Disorder,6,7,8,4,Benzodiazepines,Dialectical Behavioral Therapy,2024-01-23,16,10,Improved,7,Excited,61
366,40,Male,Major Depressive Disorder,6,5,8,8,Antidepressants,Cognitive Behavioral Therapy,2024-03-24,12,10,Deteriorated,6,Anxious,90
367,31,Female,Generalized Anxiety,7,3,9,1,Benzodiazepines,Interpersonal Therapy,2024-01-07,11,9,Improved,7,Happy,79
368,29,Female,Generalized Anxiety,8,7,8,8,Benzodiazepines,Mindfulness-Based Therapy,2024-02-13,9,7,Improved,10,Neutral,75
369,24,Female,Major Depressive Disorder,10,4,7,2,Anxiolytics,Mindfulness-Based Therapy,2024-03-23,16,7,Deteriorated,5,Stressed,76
370,31,Male,Panic Disorder,5,4,5,7,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-07,9,5,No Change,5,Anxious,76
371,60,Female,Bipolar Disorder,8,8,6,3,Benzodiazepines,Cognitive Behavioral Therapy,2024-02-05,15,9,Improved,9,Stressed,68
372,57,Male,Bipolar Disorder,6,6,4,6,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-16,11,6,Improved,9,Excited,62
373,45,Female,Generalized Anxiety,5,4,7,5,SSRIs,Cognitive Behavioral Therapy,2024-02-13,15,10,No Change,9,Neutral,80
374,19,Female,Major Depressive Disorder,7,3,5,7,SSRIs,Interpersonal Therapy,2024-03-22,14,7,Deteriorated,6,Happy,65
375,34,Male,Generalized Anxiety,8,4,7,6,Mood Stabilizers,Interpersonal Therapy,2024-03-09,14,6,Deteriorated,5,Anxious,75
376,51,Female,Generalized Anxiety,10,7,7,9,SSRIs,Cognitive Behavioral Therapy,2024-02-13,15,6,No Change,8,Excited,76
377,57,Female,Generalized Anxiety,9,6,6,2,SSRIs,Dialectical Behavioral Therapy,2024-03-03,11,9,No Change,6,Anxious,79
378,27,Female,Panic Disorder,9,8,6,1,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-28,16,10,Deteriorated,7,Neutral,68
379,33,Male,Panic Disorder,5,4,6,7,SSRIs,Dialectical Behavioral Therapy,2024-03-23,12,10,Improved,6,Stressed,60
380,21,Female,Major Depressive Disorder,5,5,7,3,SSRIs,Interpersonal Therapy,2024-02-15,10,9,No Change,9,Stressed,88
381,27,Female,Panic Disorder,7,6,8,5,SSRIs,Mindfulness-Based Therapy,2024-03-25,8,6,Deteriorated,5,Stressed,68
382,32,Male,Generalized Anxiety,9,8,4,6,Antipsychotics,Cognitive Behavioral Therapy,2024-01-26,15,7,Improved,10,Happy,61
383,54,Female,Generalized Anxiety,10,8,4,1,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-20,16,6,No Change,8,Depressed,89
384,39,Female,Generalized Anxiety,8,7,8,5,SSRIs,Dialectical Behavioral Therapy,2024-02-01,9,10,Deteriorated,5,Depressed,74
385,44,Female,Generalized Anxiety,6,8,8,1,Benzodiazepines,Mindfulness-Based Therapy,2024-01-26,12,5,Deteriorated,7,Happy,90
386,37,Male,Major Depressive Disorder,9,6,7,3,Antipsychotics,Cognitive Behavioral Therapy,2024-02-06,12,10,Improved,6,Depressed,80
387,19,Female,Generalized Anxiety,7,7,8,6,Antipsychotics,Mindfulness-Based Therapy,2024-02-03,13,9,Improved,10,Stressed,66
388,45,Male,Panic Disorder,7,6,6,6,SSRIs,Dialectical Behavioral Therapy,2024-02-26,15,6,No Change,8,Anxious,67
389,36,Female,Bipolar Disorder,9,7,6,7,Antidepressants,Dialectical Behavioral Therapy,2024-01-13,14,5,Improved,7,Anxious,83
390,19,Male,Major Depressive Disorder,10,3,4,9,Antidepressants,Mindfulness-Based Therapy,2024-02-10,10,9,Deteriorated,6,Excited,88
391,25,Male,Generalized Anxiety,9,6,4,9,Mood Stabilizers,Interpersonal Therapy,2024-03-04,13,5,No Change,9,Anxious,65
392,34,Female,Bipolar Disorder,9,3,9,10,Benzodiazepines,Cognitive Behavioral Therapy,2024-01-20,13,10,Deteriorated,5,Depressed,61
393,19,Female,Major Depressive Disorder,6,6,8,4,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-27,15,7,Deteriorated,7,Happy,90
394,46,Female,Panic Disorder,6,3,9,1,Antidepressants,Cognitive Behavioral Therapy,2024-02-04,15,10,No Change,7,Depressed,86
395,38,Male,Panic Disorder,9,4,4,4,Antipsychotics,Interpersonal Therapy,2024-04-06,14,9,Deteriorated,6,Depressed,77
396,19,Male,Panic Disorder,5,5,7,9,Anxiolytics,Mindfulness-Based Therapy,2024-02-20,13,10,Deteriorated,6,Depressed,67
397,56,Female,Bipolar Disorder,8,3,7,3,Antipsychotics,Dialectical Behavioral Therapy,2024-04-03,15,9,Improved,5,Depressed,64
398,23,Male,Major Depressive Disorder,9,3,5,5,Antipsychotics,Interpersonal Therapy,2024-03-25,10,10,Improved,10,Stressed,67
399,32,Female,Major Depressive Disorder,7,4,7,3,Benzodiazepines,Dialectical Behavioral Therapy,2024-01-22,15,6,Deteriorated,8,Depressed,75
400,33,Male,Generalized Anxiety,8,4,4,5,Antidepressants,Mindfulness-Based Therapy,2024-01-07,8,7,Improved,10,Happy,69
401,49,Male,Generalized Anxiety,6,3,9,2,Antidepressants,Dialectical Behavioral Therapy,2024-01-16,15,8,No Change,8,Anxious,62
402,59,Male,Panic Disorder,8,7,5,10,Antipsychotics,Interpersonal Therapy,2024-03-26,12,7,Deteriorated,10,Stressed,62
403,18,Female,Bipolar Disorder,5,8,4,10,Antipsychotics,Mindfulness-Based Therapy,2024-04-01,13,5,No Change,5,Neutral,68
404,43,Male,Generalized Anxiety,5,4,9,1,SSRIs,Mindfulness-Based Therapy,2024-03-31,9,5,Improved,6,Anxious,88
405,44,Female,Major Depressive Disorder,8,8,6,2,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-25,13,8,Improved,10,Neutral,67
406,52,Female,Panic Disorder,9,5,6,6,SSRIs,Dialectical Behavioral Therapy,2024-01-13,16,5,Improved,6,Neutral,74
407,25,Female,Bipolar Disorder,5,4,9,4,Anxiolytics,Interpersonal Therapy,2024-02-29,14,5,Deteriorated,5,Depressed,87
408,50,Male,Generalized Anxiety,9,6,4,6,Mood Stabilizers,Mindfulness-Based Therapy,2024-03-28,10,5,Improved,7,Anxious,80
409,58,Female,Major Depressive Disorder,9,5,4,2,Antipsychotics,Dialectical Behavioral Therapy,2024-04-04,13,10,Deteriorated,8,Anxious,60
410,44,Male,Panic Disorder,7,8,6,9,Mood Stabilizers,Interpersonal Therapy,2024-01-05,10,6,Improved,9,Neutral,69
411,51,Female,Major Depressive Disorder,9,5,8,1,Antidepressants,Cognitive Behavioral Therapy,2024-03-27,10,10,Deteriorated,6,Depressed,66
412,31,Female,Generalized Anxiety,5,7,5,9,Antipsychotics,Dialectical Behavioral Therapy,2024-02-23,10,9,No Change,6,Stressed,69
413,32,Female,Generalized Anxiety,10,7,9,10,Antipsychotics,Mindfulness-Based Therapy,2024-01-30,10,9,Improved,9,Happy,77
414,20,Male,Bipolar Disorder,8,8,9,10,Anxiolytics,Mindfulness-Based Therapy,2024-03-18,15,8,Improved,10,Excited,86
415,33,Female,Bipolar Disorder,9,7,7,3,SSRIs,Dialectical Behavioral Therapy,2024-01-22,11,7,Improved,8,Depressed,71
416,59,Male,Bipolar Disorder,8,7,5,5,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-17,13,9,Improved,8,Happy,63
417,35,Female,Bipolar Disorder,10,5,4,6,SSRIs,Interpersonal Therapy,2024-01-21,8,8,Improved,8,Excited,86
418,31,Male,Bipolar Disorder,6,5,9,10,Antidepressants,Interpersonal Therapy,2024-01-26,12,9,No Change,5,Stressed,72
419,45,Female,Generalized Anxiety,7,7,6,2,Antidepressants,Mindfulness-Based Therapy,2024-01-15,12,8,Deteriorated,10,Stressed,83
420,55,Female,Generalized Anxiety,8,3,6,10,Mood Stabilizers,Mindfulness-Based Therapy,2024-02-16,14,7,Deteriorated,6,Anxious,63
421,28,Male,Bipolar Disorder,6,3,9,9,Antipsychotics,Dialectical Behavioral Therapy,2024-02-15,12,7,Improved,6,Happy,65
422,20,Female,Major Depressive Disorder,10,8,4,4,Antipsychotics,Interpersonal Therapy,2024-03-09,8,7,No Change,8,Anxious,83
423,50,Male,Major Depressive Disorder,5,4,5,3,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-16,13,8,Deteriorated,9,Happy,69
424,41,Female,Generalized Anxiety,8,7,8,7,Mood Stabilizers,Mindfulness-Based Therapy,2024-02-24,12,6,Deteriorated,6,Neutral,88
425,46,Male,Bipolar Disorder,7,4,8,9,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-28,11,9,Deteriorated,6,Happy,63
426,26,Male,Generalized Anxiety,7,3,5,7,SSRIs,Cognitive Behavioral Therapy,2024-01-10,11,9,Improved,10,Happy,64
427,39,Male,Panic Disorder,9,8,4,10,Benzodiazepines,Mindfulness-Based Therapy,2024-01-22,8,8,Deteriorated,5,Happy,71
428,52,Female,Panic Disorder,5,8,7,5,SSRIs,Dialectical Behavioral Therapy,2024-01-21,10,8,Improved,7,Anxious,82
429,35,Female,Bipolar Disorder,5,5,4,8,Antidepressants,Mindfulness-Based Therapy,2024-04-03,12,9,Deteriorated,5,Happy,85
430,57,Female,Panic Disorder,5,6,6,2,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-28,11,5,No Change,10,Happy,85
431,20,Female,Major Depressive Disorder,7,3,6,5,Antidepressants,Cognitive Behavioral Therapy,2024-01-10,12,8,Improved,8,Stressed,70
432,51,Female,Bipolar Disorder,7,3,5,4,Benzodiazepines,Dialectical Behavioral Therapy,2024-04-09,11,7,Improved,5,Neutral,67
433,57,Male,Panic Disorder,10,6,4,1,Antipsychotics,Dialectical Behavioral Therapy,2024-02-09,9,5,Deteriorated,6,Depressed,70
434,52,Female,Bipolar Disorder,8,8,5,3,Antidepressants,Interpersonal Therapy,2024-01-28,9,7,No Change,5,Excited,87
435,37,Female,Bipolar Disorder,8,3,6,3,SSRIs,Interpersonal Therapy,2024-01-26,8,5,Deteriorated,6,Happy,72
436,35,Female,Panic Disorder,7,8,7,8,Benzodiazepines,Mindfulness-Based Therapy,2024-03-31,10,7,Deteriorated,10,Depressed,60
437,24,Female,Panic Disorder,9,5,4,8,SSRIs,Dialectical Behavioral Therapy,2024-01-23,11,9,No Change,9,Happy,80
438,49,Male,Bipolar Disorder,10,8,6,1,Antidepressants,Interpersonal Therapy,2024-02-05,16,8,Improved,5,Neutral,88
439,48,Female,Major Depressive Disorder,10,6,9,6,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-31,9,5,No Change,10,Anxious,74
440,36,Female,Panic Disorder,5,5,4,8,Antidepressants,Cognitive Behavioral Therapy,2024-02-07,14,8,Improved,5,Excited,64
441,40,Male,Panic Disorder,9,6,8,2,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-16,10,6,No Change,7,Excited,90
442,37,Male,Major Depressive Disorder,10,7,9,1,Benzodiazepines,Interpersonal Therapy,2024-02-02,16,5,Improved,7,Excited,86
443,53,Female,Generalized Anxiety,7,3,4,10,SSRIs,Mindfulness-Based Therapy,2024-01-23,12,8,No Change,5,Anxious,61
444,27,Male,Panic Disorder,9,3,4,5,SSRIs,Mindfulness-Based Therapy,2024-03-07,12,10,Deteriorated,10,Neutral,70
445,59,Male,Major Depressive Disorder,6,7,9,9,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-28,14,6,Deteriorated,6,Neutral,62
446,35,Female,Bipolar Disorder,6,8,8,1,Antipsychotics,Interpersonal Therapy,2024-02-21,16,10,Deteriorated,7,Stressed,87
447,39,Female,Panic Disorder,5,3,9,10,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-29,16,9,Improved,8,Depressed,89
448,33,Male,Bipolar Disorder,8,5,7,1,Antidepressants,Dialectical Behavioral Therapy,2024-02-25,11,5,No Change,9,Neutral,73
449,52,Female,Generalized Anxiety,10,7,8,8,Benzodiazepines,Cognitive Behavioral Therapy,2024-03-13,9,5,Deteriorated,8,Stressed,81
450,25,Male,Bipolar Disorder,9,3,7,7,Antidepressants,Interpersonal Therapy,2024-04-01,15,7,Deteriorated,8,Stressed,82
451,20,Female,Generalized Anxiety,9,7,7,5,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-05,10,10,Improved,6,Happy,85
452,52,Male,Major Depressive Disorder,7,8,5,9,Mood Stabilizers,Mindfulness-Based Therapy,2024-01-31,14,8,Deteriorated,9,Anxious,74
453,58,Male,Bipolar Disorder,10,5,5,2,Antipsychotics,Interpersonal Therapy,2024-03-07,11,7,No Change,10,Stressed,63
454,44,Male,Major Depressive Disorder,5,8,5,10,Anxiolytics,Dialectical Behavioral Therapy,2024-01-24,14,10,Deteriorated,10,Depressed,80
455,18,Male,Generalized Anxiety,9,4,7,4,SSRIs,Mindfulness-Based Therapy,2024-02-20,12,8,Improved,5,Neutral,72
456,21,Female,Generalized Anxiety,9,4,4,4,SSRIs,Dialectical Behavioral Therapy,2024-01-26,11,10,Deteriorated,5,Excited,66
457,18,Male,Panic Disorder,9,3,7,5,Antipsychotics,Cognitive Behavioral Therapy,2024-01-14,13,5,Deteriorated,8,Anxious,63
458,30,Male,Major Depressive Disorder,6,5,8,2,Antidepressants,Mindfulness-Based Therapy,2024-01-19,16,10,Improved,5,Stressed,85
459,28,Male,Generalized Anxiety,5,8,9,6,SSRIs,Mindfulness-Based Therapy,2024-04-07,8,7,Deteriorated,6,Depressed,76
460,18,Female,Panic Disorder,8,6,7,2,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-17,16,7,Improved,5,Excited,88
461,40,Male,Major Depressive Disorder,8,7,6,3,Antipsychotics,Interpersonal Therapy,2024-01-22,15,9,Deteriorated,5,Anxious,61
462,25,Female,Bipolar Disorder,8,8,8,4,Antipsychotics,Mindfulness-Based Therapy,2024-04-06,13,7,No Change,8,Excited,83
463,31,Female,Bipolar Disorder,6,4,6,2,Antidepressants,Cognitive Behavioral Therapy,2024-04-02,10,7,Improved,7,Excited,74
464,37,Male,Panic Disorder,5,5,4,8,Anxiolytics,Interpersonal Therapy,2024-03-15,12,8,Deteriorated,8,Anxious,69
465,24,Female,Major Depressive Disorder,10,4,9,2,Mood Stabilizers,Dialectical Behavioral Therapy,2024-01-18,13,10,No Change,6,Happy,75
466,56,Male,Generalized Anxiety,6,6,7,6,Antipsychotics,Interpersonal Therapy,2024-03-06,9,5,No Change,8,Anxious,77
467,45,Female,Bipolar Disorder,5,4,9,6,Benzodiazepines,Dialectical Behavioral Therapy,2024-03-26,15,8,Deteriorated,5,Anxious,77
468,57,Female,Generalized Anxiety,8,4,6,8,Benzodiazepines,Interpersonal Therapy,2024-03-23,8,8,Deteriorated,10,Neutral,89
469,50,Male,Bipolar Disorder,8,4,6,3,Antidepressants,Mindfulness-Based Therapy,2024-03-04,8,9,Deteriorated,9,Excited,68
470,55,Male,Panic Disorder,6,4,7,6,Mood Stabilizers,Dialectical Behavioral Therapy,2024-03-20,14,8,Deteriorated,7,Excited,72
471,60,Male,Major Depressive Disorder,8,4,4,8,Anxiolytics,Cognitive Behavioral Therapy,2024-01-11,14,5,Improved,6,Neutral,75
472,38,Male,Bipolar Disorder,9,5,7,3,Anxiolytics,Interpersonal Therapy,2024-03-08,15,6,Deteriorated,8,Neutral,65
473,59,Male,Bipolar Disorder,8,8,6,4,Antipsychotics,Mindfulness-Based Therapy,2024-03-20,9,6,Deteriorated,7,Happy,84
474,49,Female,Major Depressive Disorder,5,3,7,4,Antidepressants,Dialectical Behavioral Therapy,2024-03-14,12,9,Deteriorated,5,Excited,79
475,20,Male,Generalized Anxiety,7,7,9,9,SSRIs,Cognitive Behavioral Therapy,2024-01-20,15,8,No Change,6,Stressed,62
476,39,Female,Generalized Anxiety,5,3,8,2,Mood Stabilizers,Interpersonal Therapy,2024-03-19,16,5,No Change,10,Stressed,81
477,40,Male,Panic Disorder,5,5,4,7,Benzodiazepines,Cognitive Behavioral Therapy,2024-02-12,8,8,Improved,6,Happy,67
478,60,Female,Bipolar Disorder,8,3,7,9,Mood Stabilizers,Dialectical Behavioral Therapy,2024-02-14,10,9,Deteriorated,7,Anxious,89
479,29,Male,Bipolar Disorder,10,5,5,6,Antipsychotics,Mindfulness-Based Therapy,2024-02-25,14,10,Improved,5,Stressed,78
480,55,Male,Major Depressive Disorder,6,4,6,2,Anxiolytics,Mindfulness-Based Therapy,2024-03-25,13,6,Deteriorated,8,Neutral,85
481,58,Female,Generalized Anxiety,8,5,7,7,Mood Stabilizers,Interpersonal Therapy,2024-03-06,15,10,Improved,5,Stressed,72
482,36,Female,Generalized Anxiety,5,3,5,10,Antipsychotics,Cognitive Behavioral Therapy,2024-02-22,14,6,Improved,5,Stressed,77
483,60,Male,Major Depressive Disorder,5,8,7,4,Benzodiazepines,Cognitive Behavioral Therapy,2024-04-04,13,6,Deteriorated,9,Happy,90
484,60,Male,Major Depressive Disorder,6,5,4,1,SSRIs,Cognitive Behavioral Therapy,2024-01-29,11,10,No Change,6,Anxious,72
485,32,Male,Bipolar Disorder,8,8,6,3,SSRIs,Cognitive Behavioral Therapy,2024-03-09,13,10,Improved,7,Neutral,72
486,58,Male,Generalized Anxiety,5,3,7,6,Antidepressants,Dialectical Behavioral Therapy,2024-02-26,16,5,Improved,8,Depressed,63
487,24,Female,Generalized Anxiety,5,8,9,7,Antidepressants,Dialectical Behavioral Therapy,2024-03-26,13,10,No Change,9,Excited,61
488,56,Male,Major Depressive Disorder,5,8,9,10,Antipsychotics,Interpersonal Therapy,2024-01-28,11,5,Improved,10,Happy,90
489,27,Female,Bipolar Disorder,8,4,6,6,Mood Stabilizers,Cognitive Behavioral Therapy,2024-03-12,13,8,Deteriorated,6,Excited,86
490,60,Male,Bipolar Disorder,8,3,7,6,Benzodiazepines,Dialectical Behavioral Therapy,2024-02-08,10,9,Improved,8,Anxious,87
491,33,Female,Bipolar Disorder,10,3,7,1,Mood Stabilizers,Interpersonal Therapy,2024-03-02,16,10,Deteriorated,8,Happy,83
492,42,Male,Panic Disorder,10,3,7,10,Mood Stabilizers,Cognitive Behavioral Therapy,2024-02-14,13,8,Improved,6,Anxious,85
493,18,Male,Bipolar Disorder,10,5,8,7,Antipsychotics,Mindfulness-Based Therapy,2024-01-28,11,8,Improved,9,Neutral,71
494,23,Male,Panic Disorder,9,4,8,2,Anxiolytics,Interpersonal Therapy,2024-01-04,9,8,Improved,6,Happy,60
495,19,Male,Major Depressive Disorder,8,5,5,2,SSRIs,Dialectical Behavioral Therapy,2024-02-05,14,7,Improved,10,Stressed,70
496,24,Male,Generalized Anxiety,10,4,8,6,Mood Stabilizers,Dialectical Behavioral Therapy,2024-04-09,8,9,Improved,10,Depressed,73
497,22,Male,Panic Disorder,5,6,6,7,Benzodiazepines,Mindfulness-Based Therapy,2024-02-05,13,6,Deteriorated,8,Happy,86
498,23,Male,Major Depressive Disorder,7,3,4,2,Antidepressants,Cognitive Behavioral Therapy,2024-03-24,10,5,Deteriorated,5,Neutral,87
499,48,Male,Bipolar Disorder,9,4,6,9,Antidepressants,Mindfulness-Based Therapy,2024-03-22,10,6,Improved,7,Anxious,73
500,58,Male,Bipolar Disorder,5,7,6,2,SSRIs,Cognitive Behavioral Therapy,2024-03-24,10,6,Deteriorated,5,Happy,70
osis_treatment_.csv…]()
