import pandas as pd
import pickle
from functools import partial
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,balanced_accuracy_score
from src.utils.funciones_pipeline import pipeline,cols_nulos_wrapper,relleno_nulos_wrapper
import os

os.chdir(os.path.dirname(__file__)) #cambio de directorio

#target
def construir_target(df, target):

    import re

    t = target[target["Country"].notna()].copy()
    t = t[~t["Country"].astype(str).str.fullmatch(r"\d{4}")]
    t = t[t["Country"].str.lower() != "total"]
    t = t[~t["Country"].astype(str).str.startswith("1/")]

    def extract_years(x):
        if pd.isna(x):
            return []
        return [int(y) for y in re.findall(r"\d{4}", str(x))]

    crisis_cols = {
        "banking_start": "Systemic Banking Crisis (starting date)",
        "currency": "Currency Crisis",
        "sovereign_debt": "Sovereign Debt Crisis (year)",
        "debt_restruct": "Sovereign Debt Restructuring (year)"
    }

    rows = []
    for crisis_type, col in crisis_cols.items():
        tmp = t[["Country", col]].copy()
        tmp["year"] = tmp[col].apply(extract_years)
        tmp = tmp.explode("year").dropna(subset=["year"])
        tmp["crisis_type"] = crisis_type
        rows.append(tmp[["Country", "crisis_type", "year"]])

    events_long = pd.concat(rows, ignore_index=True).drop_duplicates()

    events_wide = (
        events_long.assign(value=1)
        .pivot_table(index=["Country", "year"],
                     columns="crisis_type",
                     values="value",
                     aggfunc="max",
                     fill_value=0)
        .reset_index()
    )

    cols = ["banking_start", "currency", "debt_restruct", "sovereign_debt"]
    events_wide["Crisis"] = (events_wide[cols].sum(axis=1) > 0).astype(int)

    crisis_start = events_wide.loc[events_wide.Crisis > 0, ["Country", "year"]]
    crisis_start["year_pred"] = crisis_start["year"] - 1
    crisis_start["crisis_target"] = 1

    keys = crisis_start[["Country", "year_pred"]].drop_duplicates()
    keys = keys.rename(columns={"Country": "Country Name", "year_pred": "year"})

    df["crisis_target"] = 0
    df = df.merge(keys.assign(crisis_target=1),
                  on=["Country Name", "year"],
                  how="left",
                  suffixes=("", "_y"))

    df["crisis_target"] = df["crisis_target_y"].fillna(df["crisis_target"]).astype(int)
    df = df.drop(columns=["crisis_target_y"])

    return df


#train

THRESHOLD = 0.45

COLS_FINAL = [
    'Deposit interest rate (%)',
    'Broad money (% of GDP)',
    'Exports of goods and services (current US$)',
    'Imports of goods and services (current US$)',
    'External debt stocks (% of GNI)',
    'Total debt service (% of exports of goods, services and primary income)',
    'GDP growth (annual %)',
    'GDP per capita growth (annual %)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Inflation, consumer prices (annual %)'
]


def train_model():

    # 1. Importo los nuevos datos:
    df = pd.read_excel("./src/data_sample/Datos_paises_despivotados.xlsx")
    target = pd.read_excel("./src/data_sample/TARGET.xlsx")

    # 2. Tratamiento del target:
    df = construir_target(df, target)

    # 2. Preprocesado [Tratamiento de nulos]
    p = pipeline(cols_nulos_wrapper)
    df = p(df)

    p = pipeline(partial(relleno_nulos_wrapper, how="mean"))
    df = p(df)

    # 3. X e y:
    X = df[COLS_FINAL].copy()
    y = df["crisis_target"]

    # 4. Muestras para el entreno del modelo:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    
    # 5. Preprocesado
    # preprocessing = ColumnTransformer(
    #     transformers=[
    #     ("impute_num", SimpleImputer(strategy="median"), COLS_FINAL)
    # ],
    # remainder="passthrough"
    # )

    # 6. Cargar el modelo:
    with open('model_xgb.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    # 8. Pipeline
    pipe = Pipeline([
        # Activar antes paso 5 - 
        # ("preprocess", preprocessing), 
        # Commiteo para ver si la imputacion de nulos anterior funciona.
        ("model", trained_model)
    ])

    # # scale_pos_weight
    # n_pos = (y == 1).sum()
    # n_neg = (y == 0).sum()
    # spw = n_neg / n_pos

    # #XGBoost
    # model = XGBClassifier(
    # colsample_bytree=0.7,
    # learning_rate=0.01,
    # max_depth=3,
    # n_estimators=300,
    # scale_pos_weight=spw,
    # eval_metric='aucpr',
    # random_state=42
    # )

    
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=4,
                                scoring="balanced_accuracy")
    print("Balanced Accuracy CV:", cv_scores.mean())
    #entrenar train
    pipe.fit(X_train, y_train)


    #preds
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    #metricas
    print("Métricas de ML_CRISIS_PREDICTION")
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    #entrenar con todo
    pipe.fit(X, y)


    #gyardar modelo y pipeline completo
    with open("modelo_xgb.pkl", "wb") as f:
        pickle.dump(pipe, f)
        
    print("Modelo guardado correctamente")




def train_new_model():

    # 1. Importo los nuevos datos:
    df = pd.read_excel("./src/new_data/Datos_paises_despivotados.xlsx")

    # 2. Tratamiento del target:
    # df = construir_target(df, target)

    # 2. Preprocesado [Tratamiento de nulos]
    p = pipeline(cols_nulos_wrapper)
    df = p(df)

    p = pipeline(partial(relleno_nulos_wrapper, how="mean"))
    df = p(df)

    # 3. X e y:
    X = df[COLS_FINAL].copy()
    y = df["target"]

    # 4. Muestras para el entreno del modelo:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # 5. Preprocesado
    # preprocessing = ColumnTransformer(
    #     transformers=[
    #     ("impute_num", SimpleImputer(strategy="median"), COLS_FINAL)
    # ],
    # remainder="passthrough"
    # )

    # 6. Scale_pos_weight
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos

    # 7. XGBoost (new_model)
    new_model = XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=300,
    scale_pos_weight=spw,
    eval_metric='aucpr',
    random_state=42
    )

    # 8. Pipeline
    pipe = Pipeline([
        # Activar antes paso 5 - 
        # ("preprocess", preprocessing), 
        # Commiteo para ver si la imputacion de nulos anterior funciona.
        ("model", new_model)
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=4,
                                scoring="balanced_accuracy")
    print("Balanced Accuracy CV:", cv_scores.mean())

    # 5. Entrenar train:
    pipe.fit(X_train, y_train)

    # 6. Predicciones y probabilidades:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # 7. Métricas:
    print("Métricas de ML_CRISIS_PREDICTION")
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Entrenar con todo:
    pipe.fit(X, y)

    # 9. Guardar modelo y pipeline completo
    with open("modelo_xgb.pkl", "wb") as f:
        pickle.dump(pipe, f)

    # 10. Devolver informacion útil:
    return {
        "status" : "Ok",
        "message" : "Modelo guardado correctamente",
        "Balanced Accuracy" : balanced_accuracy_score(y_test, y_pred),
        "ROC-AUC" : roc_auc_score(y_test, y_proba),
        "Class.Report" : classification_report(y_test, y_pred, output_dict=True),
        "Confusion_Matrix" : confusion_matrix(y_test, y_pred)
    }    
    



def predict_new_file():
    # 1. Leer archivo nuevo:
    df = pd.read_excel("./src/new_data/Datos_paises_new.xlsx")

    # 2. Preprocesado [Tratamiento de nulos]
    p = pipeline(cols_nulos_wrapper)
    df = p(df)

    p = pipeline(partial(relleno_nulos_wrapper, how="mean"))
    df = p(df)

    # 3. Cols finales:
    X = df[COLS_FINAL].copy()

    # 4. Cargar el modelo:
    with open('model_xgb.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    # 5. Predecir probabilidades y clases:
    y_proba = trained_model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # 6. Añadir resultados al dataframe original:
    df['probability'] = y_proba
    df['prediction'] = y_pred

    # 7. Guardar resultado en un nuevo Excel:
    output_path = "./src/new_data/Datos_paises_new.xlsx"
    df.to_excel(output_path,index=False)

    # 8. Devolver informacion útil:
    return {
        "rows_predicted": int(len(df)),
        "output_file": output_path,
        "preview": df[["probability", "prediction"]].head(10).to_dict(orient="records")
    }