from flask import Flask, request, render_template
import joblib
import numpy as np

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo e o scaler salvos
try:
    model = joblib.load('modelo_nivel_rio.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    print("Arquivos de modelo ou scaler não encontrados. Execute o notebook primeiro.")
    model = None
    scaler = None

# Define os níveis de alerta para Rio do Sul (em cm)
# Estes valores podem ser ajustados conforme a realidade da cidade
NIVEL_ATENCAO = 400  # 4 metros
NIVEL_ALERTA = 600   # 6 metros

def get_status_e_cor(nivel_previsto):
    """
    Retorna o status e a cor correspondente com base no nível previsto.
    """
    if nivel_previsto >= NIVEL_ALERTA:
        status = "ALERTA DE ENCHENTE!"
        cor = "red"  # Vermelho para alerta máximo
    elif nivel_previsto >= NIVEL_ATENCAO:
        status = "Situação de ATENÇÃO!"
        cor = "orange" # Laranja/Amarelo para atenção
    else:
        status = "Situação NORMAL"
        cor = "green" # Verde para normal
    return status, cor

# Define a rota principal que renderiza a página inicial
@app.route('/')
def home():
    return render_template('index.html')

# Define a rota para realizar a predição
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Erro: Modelo não carregado.')

    try:
        # Pega os dados enviados no formulário
        input_features = [float(x) for x in request.form.values()]
        
        # Converte para o formato que o modelo espera (array 2D)
        features_array = np.array(input_features).reshape(1, -1)
        
        # Padroniza os dados de entrada usando o mesmo scaler do treinamento
        scaled_features = scaler.transform(features_array)
        
        # Faz a predição
        prediction = model.predict(scaled_features)
        
        # Formata o resultado
        output = round(prediction[0], 2)
        
        # Obtém o status e a cor com base na previsão
        status_texto, status_cor = get_status_e_cor(output)
        
        # Retorna a página com o resultado, o status e a cor
        return render_template('index.html', 
            prediction_text=f'Nível previsto do rio: {output} cm',
            status_text=status_texto,
            status_color=status_cor)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Erro ao processar a predição: {e}')

# Executa a aplicação
if __name__ == "__main__":
    app.run(debug=True)