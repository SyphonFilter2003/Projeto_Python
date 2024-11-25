import os
import io
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variáveis globais para dados e modelos
uploaded_data = None
label_encoders = {}
model = None
target_encoder = None

# Página inicial: Upload de arquivos
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "Nenhum arquivo enviado", 400
        file = request.files['file']
        if file.filename == '':
            return "Arquivo sem nome", 400
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            global uploaded_data
            uploaded_data = pd.read_csv(filepath)

            # Limpeza dos dados
            uploaded_data = uploaded_data.dropna()

            print(f"Quantidade de linhas no dataset: {len(uploaded_data)}")
            return redirect(url_for('analyze_data'))
    return render_template("upload.html")

# Página de análise de dados
@app.route("/analyze", methods=["GET", "POST"])
def analyze_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))
    
    # Exibir a distribuição do target no console
    print("Distribuição da variável-alvo (Outcome):")
    if 'Outcome' in uploaded_data.columns:
        print(uploaded_data['Outcome'].value_counts())

    # Resumo estatístico
    summary = uploaded_data.describe(include='all').transpose()

    # Identificar tipos de colunas
    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = uploaded_data.select_dtypes(exclude=['number']).columns.tolist()

    return render_template(
        "analyze.html",
        summary=summary.to_html(classes="table"),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )

# Visualização de dados
@app.route("/visualize", methods=["POST"])
def visualize_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))

    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    plots = {}

    for column in numeric_columns:
        plt.figure()
        uploaded_data[column].hist(bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Distribuição de {column}")
        plt.xlabel(column)
        plt.ylabel("Frequência")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        plots[column] = image_base64

    return render_template("visualize.html", plots=plots)

# Configurar e treinar o modelo
@app.route("/configure", methods=["GET", "POST"])
def configure_prediction():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for("upload_file"))

    columns = uploaded_data.columns

    if request.method == "POST":
        features = request.form.getlist("features")
        target = request.form.get("target")
        model_choice = request.form.get("model")

        X = uploaded_data[features]
        y = uploaded_data[target]

        global label_encoders, target_encoder, model
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        if y.dtype == 'object' or y.nunique() < 20:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        else:
            target_encoder = None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Escolher o modelo
        if model_choice == "RandomForestClassifier":
            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        elif model_choice == "LogisticRegression":
            model = LogisticRegression(solver='liblinear', random_state=42)
        elif model_choice == "XGBClassifier":
            model = XGBClassifier(random_state=42)
        elif model_choice == "SVM":
            model = SVC(random_state=42, probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        return render_template("result.html", accuracy=accuracy, features=features, report=report)

    return render_template(
        "configure.html",
        columns=columns,
        model_choices=["RandomForestClassifier", "LogisticRegression", "XGBClassifier", "SVM"]
    )

# Página de predição
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global model, label_encoders, target_encoder
    if model is None:
        return redirect(url_for("configure_prediction"))

    if request.method == "POST":
        if 'file' not in request.files:
            return "Nenhum arquivo enviado", 400
        file = request.files['file']
        if file.filename == '':
            return "Arquivo sem nome", 400
        if file and file.filename.endswith('.csv'):
            new_data = pd.read_csv(file)
            original_columns = list(model.feature_names_in_)
            if not set(original_columns).issubset(set(new_data.columns)):
                return "Arquivo inválido: colunas não correspondem ao modelo.", 400

            X_new = new_data[original_columns].copy()
            for col, le in label_encoders.items():
                if col in X_new.columns:
                    X_new[col] = le.transform(X_new[col])
            
            predictions = model.predict(X_new)
            if target_encoder:
                predictions = target_encoder.inverse_transform(predictions)
            new_data["Predictions"] = predictions

            return render_template("prediction.html", predictions=new_data.to_html(classes="table"))
    return render_template("prediction.html")

# Página de download do modelo treinado
@app.route("/download_model")
def download_model():
    if model is None:
        return redirect(url_for("configure_prediction"))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "trained_model.pkl")
    joblib.dump(model, filepath)
    return redirect(f"/uploads/trained_model.pkl")

if __name__ == "__main__":
    app.run(debug=True)
