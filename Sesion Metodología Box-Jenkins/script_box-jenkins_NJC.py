# %% =========================
# Importar paquetes
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

# %% =========================
# Cargar datos
# ============================

BASE_DIR = Path(__file__).resolve().parent

ruta = BASE_DIR / "Expotradicionales1990-2017.csv"

ruta_te = BASE_DIR / "PTEAUSDM2005-202506.csv"

expotradicionales = pd.read_csv(
    ruta,
    header=None,
    names=["expo_tradicionales"]
)

print(expotradicionales.head())


# %% =========================
# Crear índice temporal
# ============================

fechas = pd.date_range(
    start="1990-01-01",
    periods=len(expotradicionales),
    freq="MS"
)

expotradicionales.index = fechas

print(expotradicionales.head())
print(expotradicionales.tail())


# %% =========================
# Crear serie de tiempo
# ============================

y = expotradicionales["expo_tradicionales"].copy()
y = pd.to_numeric(y, errors="coerce")
y = y.dropna()

print(y.head())
print(y.tail())
print(y.describe())


# %% =========================
# PASO 1: IDENTIFICACIÓN
# ============================

plt.figure(figsize=(10, 5))
plt.plot(y)
plt.title("Exportaciones tradicionales, 1990-2017")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()


# %% FAC de la serie original

plot_acf(
    y,
    lags=24,
    alpha=0.05,
    bartlett_confint=False
)

plt.title("Función de autocorrelación (FAC)")
plt.show()


# %% Serie diferenciada

y_diff = y.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(y_diff)
plt.title("Serie diferenciada")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()


# %% Diferencia del logaritmo

y_log_diff = np.log(y).diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(y_log_diff)
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()


# %% FAC y FACP de la diferencia del logaritmo

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    y_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(
    y_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la diferencia del logaritmo")

plt.tight_layout()
plt.show()


# %% Interpretación preliminar

# La FAC muestra que solo la primera autocorrelación es significativa.
# La FACP decae rápidamente.
# Por tanto, se propone inicialmente un modelo MA(1) sobre la serie
# transformada, equivalente a un ARIMA(0, 1, 1) sobre log(y).


# %% =========================
# PASO 2: ESTIMACIÓN
# ============================

modelo_ma1 = SARIMAX(
    y_log_diff,
    order=(0, 0, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado_ma1 = modelo_ma1.fit(disp=False)

print(resultado_ma1.summary())


# %% =========================
# PASO 3: VALIDACIÓN DEL MODELO
# ============================

residuos = resultado_ma1.resid.dropna()

print(residuos.describe())


# %% FAC de los residuos

plot_acf(
    residuos,
    lags=24,
    alpha=0.05,
    bartlett_confint=False
)

plt.title("FAC de los residuos")
plt.show()


# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuos,
    lags=[6, 12, 18, 24],
    return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.


# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuos, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.


# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuos)

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos siguen una distribución normal.
# Si p-value > 0.05, no se rechaza H0.

# %% Q-Q plot de los residuos
import scipy.stats as stats
plt.figure(figsize=(6, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()

# %% =========================
# PASO 4: PRONÓSTICO
# ============================

modelo_pronostico = SARIMAX(
    np.log(y),
    order=(0, 1, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado_pronostico = modelo_pronostico.fit(disp=False)

pronostico_log = resultado_pronostico.get_forecast(steps=12)

media_log = pronostico_log.predicted_mean
intervalos_log = pronostico_log.conf_int()

pronostico_nivel = np.exp(media_log)
intervalos_nivel = np.exp(intervalos_log)

tabla_pronostico = pd.DataFrame({
    "pronostico": pronostico_nivel,
    "limite_inferior": intervalos_nivel.iloc[:, 0],
    "limite_superior": intervalos_nivel.iloc[:, 1]
})

print(tabla_pronostico)
# %%
plt.figure(figsize=(10, 5))
plt.plot(y, label="Datos históricos")
plt.plot(pronostico_nivel, label="Pronóstico", color="orange")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza"
)
# %% =========================
# SEGUNDA SERIE: PRECIO INTERNACIONAL DEL TÉ
# ============================


te = pd.read_csv(
    ruta_te,
    header=None,
    names=["precio_te"],
    decimal=",",
    sep="\t"
)

print(te.head())
print(te.tail())


# %% =========================
# Crear índice temporal
# ============================

fechas_te = pd.date_range(
    start="2005-01-01",
    periods=len(te),
    freq="MS"
)

te.index = fechas_te

print(te.head())
print(te.tail())


# %% =========================
# Crear serie de tiempo
# ============================

y_te = te["precio_te"].copy()
y_te = pd.to_numeric(y_te, errors="coerce")
y_te = y_te.dropna()

print(y_te.head())
print(y_te.tail())
print(y_te.describe())


# %% =========================
# IDENTIFICACIÓN
# ============================

plt.figure(figsize=(10, 5))
plt.plot(y_te)
plt.title("Precio internacional del té, 2005-2025")
plt.xlabel("Fecha")
plt.ylabel("Precio té (USD)")
plt.grid(True)
plt.show()


# %% FAC y FACP del precio del té

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    y_te,
    lags=15,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC del precio del té")

plot_pacf(
    y_te,
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
