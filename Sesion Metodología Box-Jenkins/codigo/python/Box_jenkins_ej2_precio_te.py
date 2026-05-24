# %% =========================
# 0.1 Importación de paquetes
# ============================

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


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
    
    # Estimación de cada uno de los modelos
    modelo = SARIMAX(
        te_serie,
        order=orden,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ) 
    
    # Nota: Se estima cada uno de los modelos en nivel

    # Se estima el modelo 
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

tabla_modelos = []

# Itera sobre 
for nombre, resultado in estimaciones_te_serie.items():
    params = resultado.params
    errores = resultado.bse

    intercepto_sarimax = params.get("intercept", np.nan)
    se_intercepto_sarimax = errores.get("intercept", np.nan)

    ar1 = params.get("ar.L1", np.nan)
    ar2 = params.get("ar.L2", np.nan)
    ma1 = params.get("ma.L1", np.nan)

    se_ar1 = errores.get("ar.L1", np.nan)
    se_ar2 = errores.get("ar.L2", np.nan)
    se_ma1 = errores.get("ma.L1", np.nan)

    ar1_mu = params.get("ar.L1", 0)
    ar2_mu = params.get("ar.L2", 0)

    if "ar.L1" in params.index or "ar.L2" in params.index:
        mu = intercepto_sarimax / (1 - ar1_mu - ar2_mu)
    else:
        mu = intercepto_sarimax
    se_mu = se_media_incondicional_sarimax(resultado)

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

tabla_modelos_te = pd.DataFrame(tabla_modelos)


def formato_estimacion(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"{valor:.{decimales}f}"


def formato_error_estandar(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"({valor:.{decimales}f})"


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

tabla_publicacion = []

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

tabla_modelos_te_publicacion = pd.DataFrame(tabla_publicacion)

print("\nTabla resumen de modelos estimados")
print(tabla_modelos_te_publicacion.to_string(index=False))
print("\nErrores estándar entre paréntesis.")


# %% =========================
# GRÁFICAS
# ============================

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for i, nombre in enumerate(nombres_modelos):
    resultado = estimaciones_te_serie[nombre]

    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    residuos = resultado.resid.dropna().iloc[n_inicial:]
    residuos_cuadrado = residuos**2

    axes[i, 0].plot(
        residuos,
        color="black",
        linewidth=1
    )

    axes[i, 0].set_title(f"Residuos {nombre}")
    axes[i, 0].set_xlabel("Fecha")
    axes[i, 0].set_ylabel("Residuo")

    plot_acf(
        residuos,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 1]
    )

    axes[i, 1].set_title(f"FAC residuos {nombre}")
    axes[i, 1].set_xlabel("Rezago")
    axes[i, 1].set_ylabel("ACF")

    plot_acf(
        residuos_cuadrado,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 2]
    )

    axes[i, 2].set_title(f"FAC residuos² {nombre}")
    axes[i, 2].set_xlabel("Rezago")
    axes[i, 2].set_ylabel("ACF")

plt.tight_layout()
plt.show()


# %% =========================
# VALIDACIÓN SUPUESTOS
# ============================

tabla_diagnostico = []

for nombre in nombres_modelos:
    resultado = estimaciones_te_serie[nombre]

    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    residuos = resultado.resid.dropna().iloc[n_inicial:]

    jb_pvalue = jarque_bera(residuos).pvalue

    arch_1 = het_arch(residuos, nlags=1)[1]
    arch_2 = het_arch(residuos, nlags=2)[1]
    arch_5 = het_arch(residuos, nlags=5)[1]

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

tabla_diagnostico = pd.DataFrame(tabla_diagnostico)

print(tabla_diagnostico.round(3))

# %%
