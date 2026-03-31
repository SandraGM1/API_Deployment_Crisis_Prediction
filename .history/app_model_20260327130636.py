from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from model import construir_target

os.chdir(os.path.dirname(__file__)) #cambio de directorio
#para poder mostrar depsués la última predicción
last_prediction = None

app = Flask(__name__) #instancia aplicacion de flask

# Carga el modelo
def load_model():
    with open("modelo_xgb.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()
#threshold definido en la API, lo ponemos aquí porque guardamos el modelo sin definir el threshold y además así podemos cambiarlo.
THRESHOLD = 0.45


# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def home():
    return """
    <h2>API ML_Crisis_Prediction</h2>
    <p>Usa el endpoint <b>/metrics</b> con método GET para obtener información sobre la métricas.</p>
    <p>Usa el endpoint <b>/predict</b> con método POST para obtener una predicción. (Tip: envía un JSON con las variables de entrada del modelo en Postman!)</p>
    <p>usa el endpoint <b>/prediction</b> con método GET para visualizar la predicción del modelo.</p>
    <p>También puedes reentrenar el modelo con el endpoint <b>/retrain</b> y el método GET .</p>
    """

# Enruta la funcion al endpoint /predict ocn el método POST en el BODY (no header, no args, sino json)
@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se recibió JSON"}), 400

        df = pd.DataFrame([data])
        missing = [col for col in df.columns if df[col].isna().any()]

        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= THRESHOLD)

        result = {
            "prediction": pred,
            "probability": proba,
            "threshold_used": THRESHOLD
        }
        if missing:
            result["warning"] = f"Valores faltantes imputados: {', '.join(missing)}" #
        
        last_prediction = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#devuelve la ultima predicción guardada
@app.route("/prediction", methods=["GET"])
def get_last_prediction():
    if last_prediction is None:
        return {
            "status": "error",
            "message": "No hay predicciones todavía. Usa POST /predict primero."
        }, 400

    return jsonify({
        "status": "ok",
        "last_prediction": last_prediction
    })


# Enruta la funcion al endpoint /retrain del modelo entero.
@app.route("/retrain", methods=["GET"])
def retrain():

    try:
        from model import train_model

        # Reentrenar modelo
        train_model()

        # Recargar modelo actualizado
        global model
        model = load_model()

        return jsonify({
            "status": "ok",
            "message": "Modelo reentrenado y recargado correctamente"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

#ver las metricas: si se hace el primer paso del modelo original, si se hace despues de reentrenar, del modelo reentrenado
@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        # 1. Cargar datos originales
        df = pd.read_excel("./src/data_sample/Datos_paises_despivotados.xlsx")
        target = pd.read_excel("./src/data_sample/TARGET.xlsx")

        # 2. Construir target igual que en train_model()
        df = construir_target(df, target)

        # 3. X e y (sin columnas manuales)
        X = df.drop(columns=["crisis_target"])
        y = df["crisis_target"]

        # 4. Predicciones del modelo ACTUAL (original o reentrenado)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # 5. Métricas
        results = {
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__': #esto es un control, activa el modo de depuración en modo local. Cuando esté en el servidor este bucle no se ejecuta con debug a true
    app.run(debug=True)
