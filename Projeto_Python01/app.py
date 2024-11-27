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
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

uploaded_data = None
label_encoders = {}
model = None
target_encoder = None

# Função para carregar o modelo
def load_model():
    global model
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')

# Função para treinar o modelo
def train_model(data, features, target, model_choice, n_estimators=100):
    X = data[features]
    y = data[target]

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escolher o modelo baseado na escolha
    if model_choice == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "svm":
        model = SVC()

    # Treinamento do modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Gerar o relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = model.score(X_test, y_test)

    # Salvar o modelo treinado com joblib (se desejado)
    # joblib.dump(model, 'model.pkl')

    return report, accuracy

# Rota de upload de arquivos
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template('upload.html', upload_success=False, error_message="Nenhum arquivo enviado.")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', upload_success=False, error_message="Arquivo sem nome.")
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            global uploaded_data
            uploaded_data = pd.read_csv(filepath)
            
            # Exibe pop-up de sucesso e depois redireciona
            return render_template('upload.html', upload_success=True)
        else:
            return render_template('upload.html', upload_success=False, error_message="Por favor, envie um arquivo CSV.")
    
    return render_template("upload.html", upload_success=False)

@app.route("/analyze", methods=["GET", "POST"])
def analyze_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))

    # Geração de estatísticas descritivas
    summary = uploaded_data.describe(include='all').transpose()

    # Colunas numéricas e categóricas
    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = uploaded_data.select_dtypes(exclude=['number']).columns.tolist()

    # Inicializando a variável de gráficos
    plots = {}

    # Gerar gráficos para colunas numéricas
    for column in numeric_columns:
        plt.figure()
        uploaded_data[column].hist(bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Distribuição de {column}")
        plt.xlabel(column)
        plt.ylabel("Frequência")

        # Salvar o gráfico em base64 para exibição no HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        plots[column] = image_base64  # Armazenando o gráfico gerado na variável `plots`

    # Passar informações para o template
    return render_template(
        "analyze.html",
        summary=summary.to_html(classes="table"),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        plots=plots  # Passando a variável `plots` para o template
    )

@app.route("/configure", methods=["GET", "POST"])
def configure_prediction():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))

    columns = uploaded_data.columns
    if request.method == "POST":
        features = request.form.getlist("features")
        target = request.form.get("target")
        model_choice = request.form.get("model")
        n_estimators = int(request.form.get("n_estimators", 100))  # Número de árvores no Random Forest

        # Validação das variáveis
        if not features or not target:
            return "Por favor, selecione as variáveis.", 400
        if target not in uploaded_data.columns or any(f not in uploaded_data.columns for f in features):
            return "Seleção inválida de variáveis.", 400

        # Codificando as variáveis categóricas com LabelEncoder
        for col in uploaded_data.columns:
            if uploaded_data[col].dtype == 'object':
                le = LabelEncoder()
                uploaded_data[col] = le.fit_transform(uploaded_data[col])
                label_encoders[col] = le

        # Treinando o modelo
        report, accuracy = train_model(uploaded_data, features, target, model_choice, n_estimators)

        # Renderiza a página de resultados
        return render_template("result.html", report=report, accuracy=accuracy)

    return render_template("configure.html", columns=columns)

# Rota para re-treinamento do modelo
@app.route("/retrain", methods=["POST"])
def retrain_model():
    global uploaded_data, model

    # Verifica se há dados carregados
    if uploaded_data is None:
        return redirect(url_for('upload_file'))

    # Obtenção dos dados do formulário
    features = request.form.getlist("features")
    target = request.form.get("target")
    
    # Verificar se 'features' e 'target' são válidos
    if not features or not target:
        return render_template('result.html', error_message="Por favor, selecione as variáveis corretamente.", upload_success=False)

    model_choice = request.form.get("model")
    n_estimators = int(request.form.get("n_estimators", 100))

    # Re-treinando o modelo
    report, accuracy = train_model(uploaded_data, features, target, model_choice, n_estimators)

    # Verificando se o modelo foi treinado e se as variáveis existem
    if report is None or accuracy is None:
        return render_template('result.html', error_message="Erro ao treinar o modelo.", upload_success=False)

    # Renderiza a página de resultados com o novo modelo treinado
    return render_template("result.html", report=report, accuracy=accuracy)

@app.route("/result")
def result_page():
    return render_template('result.html')

if __name__ == "__main__":
    # Tenta carregar o modelo existente, se houver
    load_model()
    app.run(debug=True)
