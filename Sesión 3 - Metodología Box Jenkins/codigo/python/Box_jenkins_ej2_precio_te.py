# %% =========================
# 0.1 Importación de paquetes
# ============================

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

#TODO: Tal vez todo ésto se pueda hacer mejor con programación funcional! Explorar en el futuro!

# %% =========================
# 0.2 Cargar bases de datos en python usando rutas relativas
# ============================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[2]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Rutas de las bases de datos 
ruta_te = DATA_DIR / "PTEAUSDM2005-202506.csv" # Base de datos del precio del té

# %% =========================
# 2. SEGUNDA SERIE: PRECIO INTERNACIONAL DEL TÉ
# ============================

# Base de datos con la serie importada a python
te_base = pd.read_csv(
    ruta_te,
    header=None,
    names=["precio_te"],
    decimal=",",
    sep="\t"
)

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(te_base))

# Ver primeras y últimas observaciones de la base de datos
print(te_base.head()) # Primeras observaciones
print(te_base.tail()) # Últimas observaciones 

# %% Creación del índice temporal de las series de tiempo

# Creación del índice temporal
fechas_te_base = pd.date_range(
    start="2005-01-01",
    periods=len(te_base),
    freq="MS"
)

# Agregar el índice temporal a la base de datos del precio del té
te_base.index = fechas_te_base

# El tipo de objeto de la base de datos sigue siendo Pandas.DataFrame
print(type(te_base))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(te_base.head()) # Primeras observaciones
print(te_base.tail()) # Últimas observaciones 


# %% Creación de la serie de tiempo del "precio del té"

te_serie = te_base["precio_te"].copy()
te_serie = pd.to_numeric(te_serie, errors="coerce")
te_serie = te_serie.dropna()

# Ver el principio y final de la serie de tiempo
print(te_serie.head())
print(te_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(te_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(te_serie.describe())
print(f"Media muestral precio del té: {te_serie.mean():.3f}")


# %% =========================
# 2.1 Paso 1: Identificación 
# ============================

# Gráfica de la serie de tiempo "precio del té"
plt.figure(figsize=(10, 5))
plt.plot(te_serie)
plt.title("Precio internacional del té, 2005-2025")
plt.xlabel("Fecha")
plt.ylabel("Precio té (USD)")
plt.grid(True)
plt.show()


# %% FAC y FACP del precio del té (serie original)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    te_serie,
    lags=15,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC del precio del té")

plot_pacf(
    te_serie,
    lags=15,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP del precio del té")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(te_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(te_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie. 

# %% Logaritmo del precio del té (en niveles)

te_serie_log = np.log(te_serie)

plt.figure(figsize=(10, 5))
plt.plot(te_serie_log)
plt.title("Logaritmo del precio internacional del té")
plt.xlabel("Fecha")
plt.ylabel("Log(Precio té)")
plt.grid(True)
plt.show()

# %% FAC y FACP del precio del té (serie en logaritmos)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    te_serie_log,
    lags=15,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC del logaritmo precio del té")

plot_pacf(
    te_serie_log,
    lags=15,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP del logaritmo precio del té")

plt.tight_layout()
plt.show()

# %% Identificación del modelo usando FAC y FACP

# En éste caso no es tan sencillo determinar el orden p y q del modelo ARIMA de la 
# FAC y la FACP. 
# Algunos modelos sugeridos por la FAC y la FACP son: ARMA(1,0), ARMA(2,0) y ARMA(1,1)

# La FACP no decae tan rápidamente, pero si está decayendo.

# P.d. También se usaran criterios de información para la 
#      Selección de los ordenes p y q del modelo ARIMA. 

# %% =========================
# Paso 2.2 Estimación 
# ============================

# Se estimaran 3 modelos en éste caso, un ARIMA(1,0,0), un ARIMA(2,0,0) y un ARIMA(1,0,1)

# Se crea un diccionari de python especificando los ordenes (p,q) de cada uno de los modelos
# que se estimarán
modelos_serie_te = {
    "ARMA(1,0)": (1, 0, 0),
    "ARMA(2,0)": (2, 0, 0),
    "ARMA(1,1)": (1, 0, 1),
}

nombres_modelos = list(modelos_serie_te.keys())

# Diccionario que almacenara las estimaciones de cada uno de los modelos. Los "keys" del 
# diccionario son los nombres de los modelos, y los "values" son las estimaciones de los modelos
estimaciones_te_serie = {}

# Función para calcular la media de largo plazo del modelo, a partir de los coeficientes
# estimados en los modelos
def media_incondicional_sarimax(resultado):
    """Calcula E[y_t] para un modelo ARMA(p,q)"""
    params = resultado.params
    intercepto = params.get("intercept", 0)
    suma_ar = sum(
        valor
        for parametro, valor in params.items()
        if parametro.startswith("ar.L")
    )
    denominador = 1 - suma_ar

    if np.isclose(denominador, 0):
        return np.nan

    return intercepto / denominador


def se_media_incondicional_sarimax(resultado):
    """Calcula el error estándar de la media incondicional usando método delta."""
    params = resultado.params

    if "intercept" not in params.index:
        return np.nan

    parametros_ar = [
        parametro
        for parametro in params.index
        if parametro.startswith("ar.L")
    ]
    intercepto = params["intercept"]
    denominador = 1 - sum(params[parametro] for parametro in parametros_ar)

    if np.isclose(denominador, 0):
        return np.nan

    cov_params = resultado.cov_params()
    gradiente = pd.Series(0.0, index=params.index)
    gradiente["intercept"] = 1 / denominador

    for parametro in parametros_ar:
        gradiente[parametro] = intercepto / denominador**2

    varianza_mu = float(gradiente @ cov_params @ gradiente)

    if varianza_mu < 0:
        return np.nan

    return np.sqrt(varianza_mu)

# El loop estima los 3 modelos ARMA propusetos en "modelos_serie_te"

# Se itera sobre el diccionario "modelos_serie_te". "nombre" corresponde al nombre del modelo
# y "orden" indica el orden (p,d,q) del modelo ARIMA. En éste caso se itera 3 veces, porque
# el diccionario tiene 3 modelos. 
for nombre, orden in modelos_serie_te.items():
    
    # Especicación de cada uno de los modelos
    modelo = SARIMAX(
        te_serie,
        order=orden,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ) 
    
    # Nota: Se especifican en éste ejemplo particular, cada uno de los modelos en nivel

    # Se estiman los modelos, dada la especificación dada
    estimacion_modelo = modelo.fit(disp=False)
    
    # Nota: El método de estimación es máxima verosimilitud 
    #       sobre la representación del modelo en un espacio de estados
    
    # Se llena el diccionario de modelos estimados. Los "keys" del diccionario serán los nombres
    # de los modelos y los "values" del diccionario son las estimaciones de los modelos
    estimaciones_te_serie[nombre] = estimacion_modelo

    print("\n", nombre)
    print(estimacion_modelo.summary())
    print("\n")
    print(
        "Media incondicional implícita en SARIMAX "
        f"c / (1 - suma AR): {media_incondicional_sarimax(estimacion_modelo):.3f}"
    )

# %% Tabla resumen de modelos estimados

# Inicialmente la tabla de modelos es una lista de python
tabla_modelos = []

# Se itera sobre los keys y values del diccionario "estimaciones_te_serie" 
for nombre, resultado in estimaciones_te_serie.items():
    # nombre: es la variable que itera en el nombre de los modelos
    # resultado: es la variable que itera en la estimación de los modelos
    params = resultado.params # Da los estimaciones de los parámetros del modelo
    errores = resultado.bse # Da los errores estándar de las estimaciones de los parámetros del modelo

    # Estimacioón del intercepto del modelo y su error estándar
    intercepto_sarimax = params.get("intercept", np.nan)
    se_intercepto_sarimax = errores.get("intercept", np.nan)

    # Parámetros de la parte AR y MA
    ar1 = params.get("ar.L1", np.nan)
    ar2 = params.get("ar.L2", np.nan)
    ma1 = params.get("ma.L1", np.nan)

    # Error estándar de la parte AR y MA
    se_ar1 = errores.get("ar.L1", np.nan)
    se_ar2 = errores.get("ar.L2", np.nan)
    se_ma1 = errores.get("ma.L1", np.nan)

    # Media de largo plazo para los modelos AR
    ar1_mu = params.get("ar.L1", 0)
    ar2_mu = params.get("ar.L2", 0)

    # Cálculo de la media dependiendo del tipo de modelo 
    if "ar.L1" in params.index or "ar.L2" in params.index:
        mu = intercepto_sarimax / (1 - ar1_mu - ar2_mu)
    else:
        mu = intercepto_sarimax
    se_mu = se_media_incondicional_sarimax(resultado)

    # Se llena una a una la lista, con los resultados principales de las estimaciones
    # Dichos resultados quedan como un diccionario, es decir se obtiene una lista de diccionarios. 
    tabla_modelos.append({
        "Modelo": nombre,
        "intercepto_sarimax": intercepto_sarimax,
        "se_intercepto_sarimax": se_intercepto_sarimax,
        "media_incondicional": mu,
        "se_media_incondicional": se_mu,
        "a1": ar1,
        "se_a1": se_ar1,
        "a2": ar2,
        "se_a2": se_ar2,
        "b1": ma1,
        "se_b1": se_ma1,
        "AIC": resultado.aic,
        "BIC": resultado.bic,
    })

# Transformar la lista en un pandas DataFrame
tabla_modelos_te = pd.DataFrame(tabla_modelos)

# Nota: Hasta acá ya tenemos una tabla funcional. 

# Especificar los valores de las estimaciones en la tabla
def formato_estimacion(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"{valor:.{decimales}f}"

# Especificar los valores de los errores estándar en la tabla
def formato_error_estandar(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"({valor:.{decimales}f})"

# Acá se generá la tabla final de publicación. 

filas_tabla_publicacion = [
    ("a1", "a1", "se_a1"),
    ("", "se_a1", None),
    ("a2", "a2", "se_a2"),
    ("", "se_a2", None),
    ("b1", "b1", "se_b1"),
    ("", "se_b1", None),
    ("intercepto SARIMAX", "intercepto_sarimax", "se_intercepto_sarimax"),
    ("", "se_intercepto_sarimax", None),
    ("media incondicional", "media_incondicional", "se_media_incondicional"),
    ("", "se_media_incondicional", None),
    ("AIC", "AIC", None),
    ("BIC", "BIC", None),
]

# Primero la tabla final es una lista
tabla_publicacion = []

# loop que llena la lista para la tabla final 
for etiqueta, columna_valor, columna_error in filas_tabla_publicacion:
    fila = {"Parámetro": etiqueta}

    for nombre in nombres_modelos:
        modelo = tabla_modelos_te.loc[
            tabla_modelos_te["Modelo"] == nombre
        ].iloc[0]

        if columna_error is None and columna_valor.startswith("se_"):
            fila[nombre] = formato_error_estandar(modelo[columna_valor])
        elif columna_valor in ["AIC", "BIC"]:
            fila[nombre] = formato_estimacion(modelo[columna_valor], decimales=1)
        else:
            fila[nombre] = formato_estimacion(modelo[columna_valor])

    tabla_publicacion.append(fila)

# Genera la tabla final como un Pandas Dataframe
tabla_modelos_te_publicacion = pd.DataFrame(tabla_publicacion)

# Imprimir la tabla final en formato de texto
print("\nTabla resumen de modelos estimados")
print(tabla_modelos_te_publicacion.to_string(index=False))
print("\nErrores estándar entre paréntesis.")


# %% =========================
# Paso 2.3: Validación de Supuestos
# ============================

# Gráfica de los residuales, FAC de los residuales y FAC de los residuales al cuadrado

# Genera una grilla de 3x3 en matplotlib 
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

# Itero en cada uno de los nombres de los modelos
for i, nombre in enumerate(nombres_modelos):
    # Cuando uno itera en un objeto enumerate, 
    # La primera variable de iteración (i) es un índice
    # La segunda variable de iteración (nombre) es el nombre del modelo
    
    # Nota: Las iteraciones se dan modelo por modelo, una iteración por cada modelo 
    
    # Estimaciones de cada uno de los modelos
    resultado = estimaciones_te_serie[nombre]

    #  Encontrar el orden p, q del modelo. 
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    # Generar los residuales y los residuales al cuadrado para efectos de graficación y FACs
    residuos = resultado.resid.dropna().iloc[n_inicial:]
    residuos_cuadrado = residuos**2

    # Gráfica de los residuales
    axes[i, 0].plot(
        residuos,
        color="black",
        linewidth=1
    )

    # labels de las gráficas de residuales
    axes[i, 0].set_title(f"Residuales {nombre}")
    axes[i, 0].set_xlabel("Fecha")
    axes[i, 0].set_ylabel("Residuales")

    # ACF de los residuales
    plot_acf(
        residuos,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 1]
    )

    # labels de las ACF de residuales
    axes[i, 1].set_title(f"FAC residuos {nombre}")
    axes[i, 1].set_xlabel("Rezago")
    axes[i, 1].set_ylabel("ACF")

    # PACF de los residuales
    plot_acf(
        residuos_cuadrado,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 2]
    )

    # labels de las gráficas de residuales al cuadrado
    axes[i, 2].set_title(f"FAC residuos² {nombre}")
    axes[i, 2].set_xlabel("Rezago")
    axes[i, 2].set_ylabel("ACF")

plt.tight_layout()
plt.show()

# %% Gráfica residuales Q-Q plot

# Especificar una grilla 1x3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, nombre in enumerate(nombres_modelos):
    # Estimaciones de cada uno de los modelos
    resultado = estimaciones_te_serie[nombre]

    # Encontrar el orden p, q del modelo
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    # Se obtienen los residuales
    residuos = resultado.resid.dropna().iloc[n_inicial:]

    # Q-Q plot de los residuales
    probplot(
        residuos.iloc[1:-1],
        dist="norm",
        plot=axes[i]
    )

    axes[i].set_title(f"Q-Q plot residuos {nombre}")
    axes[i].grid(True)

plt.tight_layout()
plt.show()


# %% Tabla con los principales resultados de las prubas de validación de supuestos

# La tabla con los resultados de las pruebas de validación de supuestos inicialmente será una lista 
tabla_diagnostico = []

# Itera a través de los nombres de los modelos y genero los resultados de cada test de validación
# por modelo
for nombre in nombres_modelos:
    
    # Nota: Cada iteración se da por modelo
    
    # Resultado de las estimaciones de cada modelo
    resultado = estimaciones_te_serie[nombre]

    # Orden del modelo ARMA que se estimo 
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    # Se obtienen los residuales 
    residuos = resultado.resid.dropna().iloc[n_inicial:]

    # Prueba de Jarque Bera para los residuales
    jb_pvalue = jarque_bera(residuos).pvalue

    # Prueba Arch para los residuales
    arch_1 = het_arch(residuos, nlags=1)[1]
    arch_2 = het_arch(residuos, nlags=2)[1]
    arch_5 = het_arch(residuos, nlags=5)[1]

    # Prueba Ljung-box para los residuales
    ljung_box = acorr_ljungbox(
        residuos,
        lags=[5, 10, 20],
        return_df=True
    )


    tabla_diagnostico.append({
        "Modelo": nombre,
        "JB": jb_pvalue,
        "A(1)": arch_1,
        "A(2)": arch_2,
        "A(5)": arch_5,
        "LB(5)": ljung_box.loc[5, "lb_pvalue"],
        "LB(10)": ljung_box.loc[10, "lb_pvalue"],
        "LB(20)": ljung_box.loc[20, "lb_pvalue"],
    })

# Luego la tabla se transformará en un Pandas Dataframe
tabla_diagnostico = pd.DataFrame(tabla_diagnostico)

print(tabla_diagnostico.round(3))

# %% =========================
# Paso 2.4: Pronóstico
# ============================

# Se calculan pronósticos 12 pasos adelante para cada uno de los modelos estimados
horizonte_pronostico = 12

# Diccionario para almacenar los pronósticos que se usarán en la gráfica
pronosticos_te = {}

# Colores para que coincidan con la descripción de la gráfica:
# ARMA(1,0), ARMA(2,0) y ARMA(1,1) en verde, azul y rojo, respectivamente.
colores_pronostico = {
    "ARMA(1,0)": "green",
    "ARMA(2,0)": "blue",
    "ARMA(1,1)": "red",
}


def calcular_pronostico_modelo(nombre):
    """Genera la tabla de pronóstico de un modelo estimado."""
    
    # Se almacena las estimaciones, de cada uno de los modelos
    estimacion = estimaciones_te_serie[nombre]

    # Pronóstico 12 pasos adelante
    pronostico_modelo = estimacion.get_forecast(steps=horizonte_pronostico)

    # Pronóstico puntual e intervalos de predicción
    pronostico_puntual = pronostico_modelo.predicted_mean
    intervalos = pronostico_modelo.conf_int()

    # Guardar resultados en un diccionario para posterior uso en la grafica
    # TODO: Ésto genera side effects porque está acciendo una variable global. Corregir en el futuro.
    pronosticos_te[nombre] = {
        "pronostico": pronostico_puntual,
        "intervalos": intervalos,
    }

    # Tabla de pronósticos para cada modelo
    return (
        pd.DataFrame({
            "pronostico": pronostico_puntual,
            "limite_inferior": intervalos.iloc[:, 0],
            "limite_superior": intervalos.iloc[:, 1],
        })
        .rename_axis("Fecha")
        .reset_index()
    )


# Tabla 1: Pronóstico ARMA(1,0)
tabla_pronostico_arma10 = calcular_pronostico_modelo("ARMA(1,0)")

# Tabla 2: Pronóstico ARMA(2,0)
tabla_pronostico_arma20 = calcular_pronostico_modelo("ARMA(2,0)")

# Tabla 3: Pronóstico ARMA(1,1)
tabla_pronostico_arma11 = calcular_pronostico_modelo("ARMA(1,1)")

# Imprimir las 3 tablas de pronósticos con los doce pasos adelante
print("\n" + "=" * 60)
print("Pronósticos 12 pasos adelante - ARMA(1,0)")
print("=" * 60)
print(tabla_pronostico_arma10.round(3).to_string(index=False))

print("\n" + "=" * 60)
print("Pronósticos 12 pasos adelante - ARMA(2,0)")
print("=" * 60)
print(tabla_pronostico_arma20.round(3).to_string(index=False))

print("\n" + "=" * 60)
print("Pronósticos 12 pasos adelante - ARMA(1,1)")
print("=" * 60)
print(tabla_pronostico_arma11.round(3).to_string(index=False))

# Gráfica del histórico
plt.figure(figsize=(10, 5))
plt.plot(
    te_serie,
    label="Datos históricos",
    color="black",
    linewidth=1
)

# Gráfica conjunta de los pronósticos de los 3 modelos
for nombre in nombres_modelos:
    plt.plot(
        pronosticos_te[nombre]["pronostico"],
        label=nombre,
        color=colores_pronostico[nombre],
        linewidth=2
    )

plt.title("Pronósticos del precio internacional del té")
plt.xlabel("Fecha")
plt.ylabel("Precio del té")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Nota: Revise la validación de supuestos, la FAC y la FACP de la serie original, y los criterios
#       de información y seleccione el modelo que considere mejor para modelar la serie! 