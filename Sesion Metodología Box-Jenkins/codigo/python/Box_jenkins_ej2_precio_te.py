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
print(type(expo_base))

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
print(type(expo_base))

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


# %% FAC y FACP del precio del té

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
    method="ywm",
    ax=axes[1]
)

axes[1].set_title("FACP del precio del té")

plt.tight_layout()
plt.show()


# %% Logaritmo del precio del té

y_te_log = np.log(y_te)

plt.figure(figsize=(10, 5))
plt.plot(y_te_log)
plt.title("Logaritmo del precio internacional del té")
plt.xlabel("Fecha")
plt.ylabel("Log(Precio té)")
plt.grid(True)
plt.show()


# %% =========================
# ESTIMACIÓN
# ============================

modelos_te = {
    "ARMA(1,0)": (1, 0, 0),
    "ARMA(2,0)": (2, 0, 0),
    "ARMA(1,1)": (1, 0, 1),
}

resultados_te = {}

for nombre, orden in modelos_te.items():
    modelo = SARIMAX(
        y_te,
        order=orden,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    resultado = modelo.fit(disp=False)
    resultados_te[nombre] = resultado

    print("\n", nombre)
    print(resultado.summary())


# %% =========================
# Tabla resumen de modelos estimados
# ============================

tabla_modelos = []

for nombre, resultado in resultados_te.items():
    params = resultado.params
    errores = resultado.bse

    intercepto = params.get("intercept", np.nan)

    ar1 = params.get("ar.L1", np.nan)
    ar2 = params.get("ar.L2", np.nan)
    ma1 = params.get("ma.L1", np.nan)

    se_ar1 = errores.get("ar.L1", np.nan)
    se_ar2 = errores.get("ar.L2", np.nan)
    se_ma1 = errores.get("ma.L1", np.nan)

    ar1_mu = params.get("ar.L1", 0)
    ar2_mu = params.get("ar.L2", 0)

    if "ar.L1" in params.index or "ar.L2" in params.index:
        mu = intercepto / (1 - ar1_mu - ar2_mu)
    else:
        mu = intercepto

    tabla_modelos.append({
        "Modelo": nombre,
        "a1": ar1,
        "se_a1": se_ar1,
        "a2": ar2,
        "se_a2": se_ar2,
        "b1": ma1,
        "se_b1": se_ma1,
        "mu": mu,
        "AIC": resultado.aic,
        "BIC": resultado.bic,
    })

tabla_modelos_te = pd.DataFrame(tabla_modelos)

print(tabla_modelos_te.round(3))


# %% =========================
# GRÁFICAS
# ============================

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

nombres_modelos = ["ARMA(1,0)", "ARMA(2,0)", "ARMA(1,1)"]

for i, nombre in enumerate(nombres_modelos):
    resultado = resultados_te[nombre]

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
    resultado = resultados_te[nombre]

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
