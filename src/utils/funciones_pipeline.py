# PIPELINE

def pipeline(*steps):
    # Crea una función que aplica varios pasos de procesamiento en orden.
    # La salida de cada paso se usa como entrada del siguiente.
    def wrapper(inputs):
        # Función interna del pipeline.
        # Recorre todos los steps definidos y transforma progresivamente el input.  
        for step in steps:
            inputs = step(inputs)
        return inputs
    return wrapper

# Contar nulos originales

def step2(df):
    # Genera una lista con las columnas que queremos analizar/trabajar.
    # Excluye columnas base como identificadores, año y target.
    lista_cols = []
    cols_base = ['Country Name','Country Code','year', 'count_null', 'crisis_target']
    for col in df.columns:
        if col not in cols_base:
            lista_cols.append(col)
    return lista_cols

def cols_nulos(df,lista_cols):
    # Cuenta cuántos valores nulos tiene cada fila en las columnas seleccionadas.
    # También crea una variable binaria ('some_null') que indica si una fila tiene al menos un nulo.
    df = df.copy()
    df["count_null"] = df[lista_cols].isna().sum(axis=1)
    df["some_null"]  = (df["count_null"] > 0).astype(int)
    return df


def cols_nulos_wrapper(df):
    # Función puente para usar cols_nulos dentro del pipeline.
    # Primero obtiene las columnas relevantes con step2 y luego calcula count_null y some_null.
    lista_cols = step2(df)
    return cols_nulos(df,lista_cols)


# Filtrado de países con >50% de nulos

def step1(df):
    # Genera una lista de los paises de la columna Country Name.
    lista_paises = df['Country Name'].drop_duplicates().to_list()
    return lista_paises

def step2(df):
    # Genera una lista con las columnas que queremos analizar/trabajar.
    # Excluye columnas base como identificadores, año y target.
    lista_cols = []
    cols_base = ['Country Name','Country Code','year', 'count_null', 'crisis_pred']
    for col in df.columns:
        if col not in cols_base:
            lista_cols.append(col)
    return lista_cols

def contador_nulos(df, lista_paises, lista_cols,valor):
    # Calcula, para cada país y cada indicador, el porcentaje de valores nulos.
    # Devuelve solo aquellos pares país-indicador cuyo porcentaje de nulos supera el umbral indicado en 'valor'.

    df = df.copy()
    rows = []
    for pais in lista_paises:
        for col in lista_cols:
            porc_nulos_col = round(df.loc[df['Country Name'] == pais,col].isna().mean(),2)
            if porc_nulos_col > valor:
                rows.append((pais,col,porc_nulos_col))

    
                        
    return pd.DataFrame(rows,columns=['country','indicator','media_null'])

def contador_nulos_wrapper(df,valor=0.70):
    # Función auxiliar para automatizar el filtrado de nulos dentro del pipeline.
    # Obtiene los países y columnas relevantes, calcula los nulos por país e indicador
    # y devuelve tanto el dataframe resumen como la lista de países afectados.
    lista_paises = step1(df)
    lista_cols = step2(df)
    df_null = contador_nulos(df, lista_paises, lista_cols,valor)
    paises_null = df_null['country'].drop_duplicates().to_list()
    return df_null,paises_null



# Relleno de nulos:

def step1(df):
    # Genera una lista de los paises de la columna Country Name.
    lista_paises = df['Country Name'].drop_duplicates().to_list()
    return lista_paises

def step2(df):
    # Genera una lista con las columnas que queremos analizar/trabajar.
    # Excluye columnas base como identificadores, año y target.
    lista_cols = []
    cols_base = ['Country Name','Country Code','year', 'count_null', 'crisis_pred']
    for col in df.columns:
        if col not in cols_base:
            lista_cols.append(col)
    return lista_cols



def relleno_nulos_media_pais(df,lista_paises,lista_cols, how = 'mean'):
    # Rellena los valores nulos de cada indicador usando estadísticas calculadas por país.
    # Si 'how' = 'mean', rellena con la media del país.
    # Si 'how' = 'median', rellena con la mediana del país.
    # Si un país no tiene datos para un indicador, usa como fallback la mediana global de esa columna.
    import numpy as np
    from pandas.api.types import is_numeric_dtype
    df = df.copy()
    if how == 'mean':   
        for pais in lista_paises:
            for col in lista_cols:
                if is_numeric_dtype(df[col]):
                    media_col_x_pais = round(df.loc[(df['Country Name'] == pais),col].mean(),6)
                    if np.isnan(media_col_x_pais):
                        media_col_x_pais = round(df[col].median(),6)
                    filtro = df['Country Name'] == pais
                    df.loc[filtro, col] = df.loc[filtro, col].fillna(media_col_x_pais)
                else:
                    continue
        print(df.isna().mean())
        return df
    elif how == 'median':
        for pais in lista_paises:
            for col in lista_cols:
                if is_numeric_dtype(df[col]):
                    media_col_x_pais = round(df.loc[(df['Country Name'] == pais),col].median(),6)
                    if np.isnan(media_col_x_pais):
                        media_col_x_pais = round(df[col].median(),6)
                    filtro = df['Country Name'] == pais
                    df.loc[filtro, col] = df.loc[filtro, col].fillna(media_col_x_pais)
                else:
                    continue
        print(df.isna().mean())
        return df
    
def relleno_nulos_wrapper(df,how = 'mean'):
    # Función auxiliar para aplicar el relleno de nulos de forma automática.
    # Primero identifica países y columnas relevantes y después aplica la imputación
    # usando media o mediana por país, según se indique en 'how'.
    lista_paises = step1(df)
    lista_cols = step2(df)
    return relleno_nulos_media_pais(df,lista_paises,lista_cols, how)
