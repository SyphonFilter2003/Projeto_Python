import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import base64

app = Flask(__name__)
app.secret_key = 'chave_secreta'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Garantir que a pasta de upload existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variável global para armazenar o caminho do último arquivo CSV enviado
uploaded_csv_path = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global uploaded_csv_path  # Para atualizar o caminho global do arquivo

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo selecionado')
            return redirect(request.url)

        file = request.files['file']
        
        if file.filename == '':
            flash('Nenhum arquivo selecionado')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            uploaded_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_csv_path)
            flash('Arquivo enviado com sucesso!')
            return redirect(url_for('analysis'))
        else:
            flash('Formato de arquivo inválido. Apenas arquivos .csv são permitidos.')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/analysis')
def analysis():
    global uploaded_csv_path

    if not uploaded_csv_path:
        flash('Nenhum arquivo disponível para análise.')
        return redirect(url_for('upload_file'))

    data = pd.read_csv(uploaded_csv_path)
    
    # Gerar gráficos
    img_b64_hist = create_base64_histogram(data) if not data.empty else None
    img_b64_scatter = create_base64_scatter(data) if not data.empty else None
    img_b64_boxplot = create_base64_boxplot(data) if not data.empty else None
    img_b64_line = create_base64_line_plot(data) if not data.empty else None
    img_b64_bar = create_base64_bar_chart(data) if not data.empty else None

    # Somente gráficos válidos (não None) serão enviados ao template
    return render_template('analysis.html', 
                           img_b64_hist=img_b64_hist, 
                           img_b64_scatter=img_b64_scatter,
                           img_b64_boxplot=img_b64_boxplot,
                           img_b64_line=img_b64_line,
                           img_b64_bar=img_b64_bar)

def create_base64_histogram(data):
    """Gera um histograma das colunas numéricas em base64."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    numeric_data.hist(bins=20)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    
    return img_b64

def create_base64_scatter(data):
    """Gera um gráfico de dispersão usando as duas primeiras colunas numéricas, em base64."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        return None  # Verifique se há pelo menos duas colunas numéricas
    
    plt.figure(figsize=(8, 5))
    plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], alpha=0.5)
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    return img_b64

def create_base64_boxplot(data):
    """Gera um boxplot das colunas numéricas em base64."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    numeric_data.plot(kind='box', vert=False)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    
    return img_b64

def create_base64_line_plot(data):
    """Gera um gráfico de linha para as colunas numéricas em base64."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    numeric_data.plot()
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    
    return img_b64

def create_base64_bar_chart(data):
    """Gera um gráfico de barras com a média das colunas numéricas em base64."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    numeric_data.mean().plot(kind='bar')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    
    return img_b64

def encode_categorical_data(data, columns):
    """Codifica as colunas categóricas usando LabelEncoder."""
    le = LabelEncoder()
    for column in columns:
        if data[column].dtype == 'object':  # Verifique se a coluna é categórica
            data[column] = le.fit_transform(data[column])
    return data

# Adicione uma nova rota para predição
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global uploaded_csv_path

    if not uploaded_csv_path:
        flash('Nenhum arquivo disponível para análise.')
        return redirect(url_for('upload_file'))

    data = pd.read_csv(uploaded_csv_path)

    if request.method == 'POST':
        model_type = request.form.get('model')
        target_variable = request.form.get('target')
        feature_columns = request.form.getlist('features')

        if target_variable and feature_columns:
            # Codifique as variáveis categóricas
            data_encoded = encode_categorical_data(data, feature_columns)

            # Dividir dados
            X = data_encoded[feature_columns]
            y = data[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Escolher o modelo
            if model_type == 'LogisticRegression':
                model = LogisticRegression()
            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier()
            elif model_type == 'RandomForest':
                model = RandomForestClassifier()
            else:
                flash("Modelo inválido")
                return redirect(request.url)

            # Treinar e avaliar o modelo
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            return render_template('results.html', report=report)

        flash("Por favor, selecione um alvo e variáveis de entrada")
        return redirect(request.url)

    return render_template('prediction.html', columns=data.columns)

if __name__ == '__main__':
    app.run(debug=True)
