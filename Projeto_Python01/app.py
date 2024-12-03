import os
import io
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib 
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

uploaded_data = None  
label_encoders = {}   
model = None          
target_encoder = None

def load_model():
    global model
    if os.path.exists('model.pkl') and os.path.getsize('model.pkl') > 0:
        try:
            model = joblib.load('model.pkl')  
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            model = None  
    else:
        print("Nenhum modelo salvo encontrado ou o arquivo está vazio.")
        model = None 

def train_model(data, features, target, model_choice, n_estimators=100):
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_choice == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "logistic_regression":
        model = LogisticRegression(max_iter=1000)  
    elif model_choice == "svm":
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  

    try:
        joblib.dump(model, 'model.pkl')
        print("Modelo salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = model.score(X_test, y_test) 

    importance_graph_base64 = None  # Inicializa a variável

    if model_choice == "random_forest":
        feature_importances = model.feature_importances_

        plt.figure()
        plt.barh(features, feature_importances, color="skyblue")
        plt.xlabel("Importância")
        plt.title("Importância das Variáveis - Random Forest")
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        importance_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
    
    else:
        importance_graph_base64 = None

    return report, accuracy, importance_graph_base64

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
            
            return render_template('upload.html', upload_success=True)
        else:
            return render_template('upload.html', upload_success=False, error_message="Por favor, envie um arquivo CSV.")
    
    return render_template("upload.html", upload_success=False)

@app.route("/analyze", methods=["GET", "POST"])
def analyze_data():
    global uploaded_data
    if uploaded_data is None:
        return redirect(url_for('upload_file'))  

    summary = uploaded_data.describe(include='all').transpose()

    numeric_columns = uploaded_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = uploaded_data.select_dtypes(exclude=['number']).columns.tolist()

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

    return render_template(
        "analyze.html",
        summary=summary.to_html(classes="table"),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        plots=plots  
    )

@app.route("/configure", methods=["GET", "POST"])
def configure_prediction():
    global uploaded_data, target_column
    if uploaded_data is None:
        return redirect(url_for('upload_file')) 

    columns = uploaded_data.columns  
    if request.method == "POST":
        features = request.form.getlist("features")
        target = request.form.get("target")
        target_column = target 
        model_choice = request.form.get("model")
        n_estimators = int(request.form.get("n_estimators", 100)) 

        if not features or not target:
            return "Por favor, selecione as variáveis.", 400
        if target not in uploaded_data.columns or any(f not in uploaded_data.columns for f in features):
            return "Seleção inválida de variáveis.", 400

        for col in uploaded_data.columns:
            if uploaded_data[col].dtype == 'object': 
                le = LabelEncoder()
                uploaded_data[col] = le.fit_transform(uploaded_data[col])  
                label_encoders[col] = le  

        report, accuracy, importance_graph_base64 = train_model(uploaded_data, features, target, model_choice, n_estimators)

        return render_template(
            "result.html", 
            report=report, 
            accuracy=accuracy,
            importance_graph=importance_graph_base64,  
            accuracy_graph_base64=generate_accuracy_graph(accuracy)  
        )

    return render_template("configure.html", columns=columns)  

@app.route("/retrain", methods=["POST"])
def retrain_model():
    global uploaded_data, model

    if uploaded_data is None:
        return redirect(url_for('upload_file')) 

    features = request.form.getlist("features")
    target = request.form.get("target")
    model_choice = request.form.get("model")

    if not features or not target:
        return "Por favor, selecione as variáveis.", 400
    if target not in uploaded_data.columns or any(f not in uploaded_data.columns for f in features):
        return "Seleção inválida de variáveis.", 400

    for col in uploaded_data.columns:
        if uploaded_data[col].dtype == 'object': 
            le = LabelEncoder()
            uploaded_data[col] = le.fit_transform(uploaded_data[col])  
            label_encoders[col] = le 

    report, accuracy, importance_graph_base64 = train_model(uploaded_data, features, target, model_choice)

    model = joblib.load('model.pkl') 

    return render_template(
        "result.html", 
        report=report, 
        accuracy=accuracy,
        importance_graph=importance_graph_base64
    )

def generate_accuracy_graph(accuracy):
    fig, ax = plt.subplots(figsize=(5, 3))
    
    ax.bar(['Precisão'], [accuracy], color='blue')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Precisão')
    ax.set_title(f'Precisão do Modelo: {accuracy:.2f}')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


@app.route('/results')
def results():
    global model

    if model is None:
        return redirect(url_for('configure_prediction')) 

    X_test = uploaded_data.drop(columns=[target_column])  
    y_test = uploaded_data[target_column]  
    accuracy = model.score(X_test, y_test)  
    accuracy_graph_base64 = generate_accuracy_graph(accuracy)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return render_template(
        'results.html',
        accuracy=accuracy,
        report=report,
        accuracy_graph_base64=accuracy_graph_base64 
    )

load_model()

if __name__ == "__main__":
    app.run(debug=True)
