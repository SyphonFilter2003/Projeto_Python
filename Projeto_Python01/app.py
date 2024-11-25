import os
import io
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report   
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data and models
uploaded_data = None
label_encoders = {}
model = None
target_encoder = None

# Home page: File upload
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
            return redirect(url_for('analyze_data'))
    return render_template("upload.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))
    
    # Geração de estatísticas descritivas
    summary = uploaded_data.describe(include='all').transpose()

    # Previsão de gráficos com base nos tipos de dados
    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = uploaded_data.select_dtypes(exclude=['number']).columns.tolist()

    # Passar informações para o template
    return render_template(
        "analyze.html",
        summary=summary.to_html(classes="table"),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )

@app.route("/visualize", methods=["POST"])
def visualize_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))

    # Gerar gráficos para colunas numéricas
    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    plots = {}

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
        plots[column] = image_base64

    return render_template("visualize.html", plots=plots)

# Configure prediction page
@app.route("/configure", methods=["GET", "POST"])
def configure_prediction():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))
    
    # Select columns for analysis
    columns = uploaded_data.columns
    if request.method == "POST":
        features = request.form.getlist("features")
        target = request.form.get("target")
        if not features or not target:
            return "Por favor, selecione as variáveis.", 400
        if target not in uploaded_data.columns or any(f not in uploaded_data.columns for f in features):
            return "Seleção inválida de variáveis.", 400
        
        # Prepare data for training
        X = uploaded_data[features].copy()
        y = uploaded_data[target]
        
        # Encode categorical variables
        global label_encoders, target_encoder, model
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Handle target variable encoding (only for categorical targets)
        if y.dtype == 'object' or y.nunique() < 20:  # Categórico se menos de 20 classes
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            target_names = target_encoder.classes_
        else:
            target_encoder = None
            target_names = y.unique().astype(str)  # Valores numéricos como strings para exibição

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Generate report
        try:
            if target_encoder:  # Para variáveis categóricas
                report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                report_string = classification_report(y_test, y_pred, target_names=target_names)
            else:  # Para variáveis numéricas
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_string = classification_report(y_test, y_pred)
        except Exception as e:
            report_dict = {"Erro": f"Erro ao gerar relatório de classificação: {str(e)}"}
            report_string = str(report_dict)

        # Passar o relatório e outras informações para o template
        return render_template(
            "result.html",
            report=report_dict,
            report_string=report_string,
            features=list(X.columns),
            accuracy=model.score(X_test, y_test)
        )
    return render_template("configure.html", columns=columns)

if __name__ == "__main__":
    app.run(debug=True)
