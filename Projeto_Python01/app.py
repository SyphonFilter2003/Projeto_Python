import os
import io
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib  # Para salvar o modelo treinado
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Pasta onde os arquivos serão salvos
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Cria a pasta caso não exista

# Variáveis globais
uploaded_data = None  # Dados carregados do CSV
label_encoders = {}   # Dicionário para armazenar os LabelEncoders
model = None          # Modelo treinado
target_encoder = None # Para armazenar o encoder da variável alvo (se necessário)

# Função para carregar o modelo treinado (se houver)
def load_model():
    global model
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')  # Carrega o modelo salvo em disco

# Função para treinar o modelo com os dados fornecidos
def train_model(data, features, target, model_choice, n_estimators=100):
    X = data[features]
    y = data[target]

    # Divisão dos dados em treino e teste (30% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escolher o modelo baseado na escolha
    if model_choice == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "logistic_regression":
        model = LogisticRegression(max_iter=1000)  # Aumenta o número de iterações para a convergência
    elif model_choice == "svm":
        model = SVC()

    # Treinamento do modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # Fazendo previsões

    # Salvar o modelo treinado com joblib para reuso
    joblib.dump(model, 'model.pkl')

    # Gerar o relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = model.score(X_test, y_test)  # Acurácia do modelo

    # Gerar o gráfico de importância das variáveis (somente para RandomForest)
    feature_importances = None
    if model_choice == "random_forest":
        feature_importances = model.feature_importances_

        # Gerar o gráfico da importância das variáveis
        plt.figure()
        plt.barh(features, feature_importances, color="skyblue")
        plt.xlabel("Importância")
        plt.title("Importância das Variáveis - Random Forest")
        
        # Salvar o gráfico em base64 para exibição no template
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        importance_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

    return report, accuracy, importance_graph_base64  # Agora retornamos o gráfico de importância também

# Rota de upload de arquivos CSV
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Verificar se o arquivo foi enviado corretamente
        if 'file' not in request.files:
            return render_template('upload.html', upload_success=False, error_message="Nenhum arquivo enviado.")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', upload_success=False, error_message="Arquivo sem nome.")
        if file and file.filename.endswith('.csv'):  # Verifica se é um arquivo CSV
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Salva o arquivo no diretório de uploads
            global uploaded_data
            uploaded_data = pd.read_csv(filepath)  # Carrega os dados CSV para o pandas
            
            # Exibe pop-up de sucesso e depois redireciona para a análise
            return render_template('upload.html', upload_success=True)
        else:
            return render_template('upload.html', upload_success=False, error_message="Por favor, envie um arquivo CSV.")
    
    return render_template("upload.html", upload_success=False)

# Rota para análise dos dados (estatísticas e gráficos)
@app.route("/analyze", methods=["GET", "POST"])
def analyze_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))  # Se não houver dados carregados, redireciona para o upload

    # Geração de estatísticas descritivas (média, desvio padrão, etc.)
    summary = uploaded_data.describe(include='all').transpose()

    # Listando colunas numéricas e categóricas
    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = uploaded_data.select_dtypes(exclude=['number']).columns.tolist()

    # Inicializando o dicionário para armazenar os gráficos
    plots = {}

    # Gerar gráficos para as colunas numéricas
    for column in numeric_columns:
        plt.figure()
        uploaded_data[column].hist(bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Distribuição de {column}")
        plt.xlabel(column)
        plt.ylabel("Frequência")

        # Salvar o gráfico em formato base64 para exibição no HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()  # Fecha a figura para liberar recursos
        plots[column] = image_base64  # Armazenando o gráfico

    # Passa os dados para o template renderizado
    return render_template(
        "analyze.html",
        summary=summary.to_html(classes="table"),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        plots=plots  # Passando os gráficos gerados para o template
    )

# Rota para configuração do modelo (escolher variáveis e modelo)
@app.route("/configure", methods=["GET", "POST"])
def configure_prediction():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))  # Se não houver dados, redireciona para o upload

    columns = uploaded_data.columns  # Obtém as colunas do DataFrame
    if request.method == "POST":
        # Pega as variáveis do formulário
        features = request.form.getlist("features")
        target = request.form.get("target")
        model_choice = request.form.get("model")
        n_estimators = int(request.form.get("n_estimators", 100))  # Valor padrão de 100 árvores no Random Forest

        # Validação das variáveis
        if not features or not target:
            return "Por favor, selecione as variáveis.", 400
        if target not in uploaded_data.columns or any(f not in uploaded_data.columns for f in features):
            return "Seleção inválida de variáveis.", 400

        # Codificando variáveis categóricas com LabelEncoder
        for col in uploaded_data.columns:
            if uploaded_data[col].dtype == 'object':  # Se a coluna for categórica
                le = LabelEncoder()
                uploaded_data[col] = le.fit_transform(uploaded_data[col])  # Codifica a coluna
                label_encoders[col] = le  # Salva o encoder

        # Treina o modelo com as variáveis selecionadas
        report, accuracy, importance_graph_base64 = train_model(uploaded_data, features, target, model_choice, n_estimators)

        # Renderiza a página de resultados com o modelo treinado e o gráfico de importância
        return render_template(
            "result.html", 
            report=report, 
            accuracy=accuracy,
            importance_graph=importance_graph_base64  # Passa o gráfico para o template
        )

    return render_template("configure.html", columns=columns)  # Exibe a página de configuração com as colunas


# Rota para re-treinamento do modelo (usando os mesmos dados)
@app.route("/retrain", methods=["POST"])
def retrain_model():
    global uploaded_data, model

    if uploaded_data is None:
        return redirect(url_for('upload_file'))  # Se não houver dados, redireciona para o upload

    # Coleta os dados do formulário
    features = request.form.getlist("features")
    target = request.form.get("target")
    model_choice = request.form.get("model")
    n_estimators = int(request.form.get("n_estimators", 100))  # Caso esteja utilizando Random Forest

    # Adiciona prints para depuração
    print("Features:", features)
    print("Target:", target)
    print("Modelo escolhido:", model_choice)

    # Valida se todas as variáveis foram passadas corretamente
    if not features or not target or not model_choice:
        return "Por favor, forneça todas as variáveis necessárias", 400

    # Codifica as variáveis categóricas (se necessário)
    for col in uploaded_data.columns:
        if uploaded_data[col].dtype == 'object':
            le = LabelEncoder()
            uploaded_data[col] = le.fit_transform(uploaded_data[col])
            label_encoders[col] = le

    # Re-treina o modelo com os dados e variáveis atualizados
    report, accuracy = train_model(uploaded_data, features, target, model_choice, n_estimators)

    # Renderiza a página de resultados com o novo modelo treinado
    return render_template("result.html", report=report, accuracy=accuracy)

# Rota para a página de resultados do modelo treinado
@app.route("/result")
def result_page():
    return render_template('result.html')

if __name__ == "__main__":
    # Tenta carregar o modelo existente, se houver
    load_model()
    app.run(debug=True)
