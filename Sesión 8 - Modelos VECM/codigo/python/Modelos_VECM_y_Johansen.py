#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           UNIVERSIDAD NACIONAL DE COLOMBIA
#                            Facultad de Ciencias Económicas 
#                               Econometría II | Monitoría 
#
#                                        Sesión 8   
#                           Cointegración y metodologia Johansen
#                                  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Importamos las librerias necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================
# 1. Introducción a metodología Johansen
# ============================================

# Aspectos generales de la metodología de Johansen ----

# Consiste en un procedimiento en 4 etapas: 

## Etapa 1: Verificación preliminar de las variables a trabajar (Orden de 
#            integración y gráficas) e Identificación del número de rezagos del 
#            VECM mediante criterios de información sobre el VAR en niveles 
## Etapa 2: Determinación del rango de la matriz Pi (es decir del número de 
#            relaciones de cointegración) y estimación del VECM
## Etapa 3: Análisis de la matriz beta (matriz que contiene el vector de 
#            cointegración) y matriz alpha (matriz que contiene los parámetros 
#            de velocidad de ajuste)
## Etapa 4: Validación de supuestos y usos del modelo 


# Ejemplo 1: Precio de referencia Brent y WTI

# Vamos a utilizar una serie del precio spot del petróleo de referencia Brent y 
# una serie del precio spot del petróleo de referencia WTI. Las series tienen 
# frecuencia mensual y comprenden el periodo de Enero del 2000 a Diciembre de 2020.



#Vamos a cargas las series
data = pd.read_excel("Petróleo.xlsx")
data.head()

#Creamos las series de tiempo

data.index = pd.date_range(start="2000-01-01", periods=len(data), freq="MS")

Brent = data["Brent"]
Brent.head()

WTI = data["WTI"]
WTI.head()

#Graficamos las series

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Brent.index, Brent, label="Brent", color="lightblue", linewidth=1.5)
ax.plot(WTI.index,   WTI,   label="WTI",   color="coral",     linewidth=1.5)
ax.set_title("Precios spot del petróleo\n(Petróleo Brent & WTI)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("USD por barril")
ax.legend(loc="lower center", ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
 
# =========================================================
# 1. Identificación del orden de integración de las series
# =========================================================

# Procedemos a hacer las pruebas de raíz unitaria (Para identificar el orden de 
# integración de las dos series)

#'ct' → tendencia + constante  (equivale a type='trend' en R)
#'c'  → solo constante          (equivale a type='drift' en R)
#'n'  → sin constante           (equivale a type='none'  en R)

#~~~~ BRENT ~~~~#

adf_Brent = adfuller(Brent,regression="n", maxlag=12)

print("ADF statistic:", adf_Brent[0])
print("Critical values:")
for key, value in adf_Brent[4].items():
    print(f"{key}: {value}");print("p-value:", adf_Brent[1])


#~~~~ WTI ~~~~#

adf_WTI = adfuller(WTI,regression="n", maxlag=12)

print("ADF statistic:", adf_WTI[0])
print("Critical values:")
for key, value in adf_WTI[4].items():
    print(f"{key}: {value}");print("p-value:", adf_WTI[1])


#Las series no son estacionarias, por lo que aplicamos diferenciación

Brent_diff = Brent.diff().dropna()
WTI_diff = WTI.diff().dropna()



#Volvemos a aplicar la prueba ADF sobre las series diferenciadas

#~~~~ Brent diferenciada ~~~#

adf_Brent_diff = adfuller(Brent_diff,regression="n", maxlag=12)

print("ADF statistic:", adf_Brent_diff[0])
print("Critical values:")
for key, value in adf_Brent_diff[4].items():
    print(f"{key}: {value}");print("p-value:", adf_Brent_diff[1])


#~~~~ WTI diferenciada ~~~#

adf_WTI_diff = adfuller(WTI_diff,regression="n", maxlag=12)

print("ADF statistic:", adf_WTI_diff[0])
print("Critical values:")
for key, value in adf_WTI_diff[4].items():
    print(f"{key}: {value}");print("p-value:", adf_WTI_diff[1])


#Ambas series al diferenciarlas se vuelven estacionarias, 
# por lo que ambas series son I(1)


# ===========================
# 1. 2 Modelo VAR en niveles
# ===========================

# Posteriormente, estimaremos un VAR en niveles para determinar el número rezagos del VECM

# Ojo: Se analizaran los criterios de información sobre el VAR en niveles 

# Se va a construir la matriz con las series


Y = pd.DataFrame({"Brent": Brent,"WTI": WTI})
Y.head()

#Guardamos el modelo VAR para luego determinar el número de rezagos del VECM 
modelo = VAR(Y)

#Modelo VAR con tendencia y constante
lag_order_ct = modelo.select_order(6, trend="ct");print(lag_order_ct.summary())
var_both = modelo.fit(2, trend="ct");print(var_both.summary())


#Modelo VAR con constante
lag_order_c = modelo.select_order(6, trend="c");print(lag_order_c.summary())
var_const = modelo.fit(2, trend="c");print(var_const.summary())


#Modelo VAR sin tendencia ni constante
lag_order_n = modelo.select_order(6, trend="n");print(lag_order_n.summary())
var_none = modelo.fit(2, trend="n");print(var_none.summary())



# Elegimos VAR(2) en niveles 

 # c = constante
VAR2 = modelo.fit(2, trend="c")
print(VAR2.summary())


#Analicemos los residuales del VAR(2)

residuales = VAR2.resid

# Dada que es una serie mensual, 
# analicemos su comportamiento en puntos críticos.


#Veamos la prueba Lyung_box 
acorr_ljungbox(residuales["Brent"], lags=[12], return_df=True)
acorr_ljungbox(residuales["WTI"], lags=[12], return_df=True)

acorr_ljungbox(residuales["Brent"], lags=[24], return_df=True)
acorr_ljungbox(residuales["WTI"], lags=[24], return_df=True)

acorr_ljungbox(residuales["Brent"], lags=[36], return_df=True)
acorr_ljungbox(residuales["WTI"], lags=[36], return_df=True)

#veamos la prueba Portmaneu

P_12=VAR2.test_whiteness(nlags=12);print(P_12.summary())
P_24=VAR2.test_whiteness(nlags=24);print(P_24.summary())
P_36=VAR2.test_whiteness(nlags=36);print(P_36.summary())

# A medida que se alejan los periodos, se cumple el supuesto.
# Efecto desvanecimiento. Es normal que ocurra esto, por lo que en general, 
# validaremos el cumplimiento del supuesto.


#~~~ Graficamos los residuales ~~~#

for col in residuales.columns:
    fig, axes = plt.subplots(3, 2, figsize=(12, 6))
    fig.suptitle(f"Diagnóstico de residuales — {col}", fontsize=13)
    residuales[col].plot(ax=axes[0, 0], title="Residuales", color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0, 0].spines[["top", "right"]].set_visible(False)
    axes[0, 1].hist(residuales[col], bins=30, color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Distribución")
    axes[0, 1].spines[["top", "right"]].set_visible(False)
    plot_acf(residuales[col],  lags=20, ax=axes[1, 0], title="ACF residuales")
    plot_pacf(residuales[col], lags=20, ax=axes[1, 1], title="PACF residuales")
    plot_acf(residuales[col]**2,  lags=20, ax=axes[2, 0], title="ACF residuales al cuadrado")
    plot_pacf(residuales[col]**2, lags=20, ax=axes[2, 1], title="PACF residuales al cuadrado")
    plt.tight_layout()
    plt.show()

#Se ven bien comportados, excepto por heterocedasticidad

# ===========================================
# 2. Determinación del rango de la matriz Pi
# ===========================================

#Lo haremos mediante la funcion coint_johansen de la libreria statsmodels,
#  la cual nos permite obtener los estadísticos de la traza y 
# del valor propio máximo, así como sus respectivos valores críticos.

# ======================================
# 2.1 Test de Johansen - Sin intercepto 
# ======================================

#ecdet="none"-> det_order=-1
#ecdet="const"-> det_order=0
#ecdet="trend"-> det_order=1

#k_ar_diff = p - 1

johansen_test_none = coint_johansen(Y, det_order=-1, k_ar_diff=1)

#~~ Criterio del valor propio máximo ~~#

# Generalmente es la prueba preferida y la  más robusta. 
# El procedimiento que se analiza es:
# H0: r=0 vs H1: r=1, luego H0: r=1 vs H1: r=2, y así sucesivamente.

eigen_table_none = pd.DataFrame({
    "r": ["r = 0", "r ≤ 1"],
    "Eigen Statistic": johansen_test_none.lr2,
    "CV 90%": johansen_test_none.cvm[:,0],
    "CV 95%": johansen_test_none.cvm[:,1],
    "CV 99%": johansen_test_none.cvm[:,2]
})

print(eigen_table_none)


#~~ Criterio de la traza ~~#

# Es un procedimiento secuencial en donde se contrasta
# H0: r=0 vs H1: r>=1, luego H0: r<=1 vs H1: r>1, y así sucesivamente. 
trace_table_none = pd.DataFrame({
    "r": ["r = 0", "r ≤ 1"],
    "Trace Statistic": johansen_test_none.lr1,
    "CV 90%": johansen_test_none.cvt[:,0],
    "CV 95%": johansen_test_none.cvt[:,1],
    "CV 99%": johansen_test_none.cvt[:,2]
})

print(trace_table_none)



# ======================================
# 2.2 Test de Johansen - Con intercepto 
# ======================================

johansen_test_const = coint_johansen(Y, det_order=0, k_ar_diff=1)


#~~ Criterio del valor propio máximo ~~#

eigen_table_const = pd.DataFrame({
    "r": ["r = 0", "r ≤ 1"],
    "Eigen Statistic": johansen_test_const.lr2,
    "CV 90%": johansen_test_const.cvm[:,0],
    "CV 95%": johansen_test_const.cvm[:,1],
    "CV 99%": johansen_test_const.cvm[:,2]
})

print(eigen_table_const)


#~~ Criterio de la traza ~~#

trace_table_const = pd.DataFrame({
    "r": ["r = 0", "r ≤ 1"],
    "Trace Statistic": johansen_test.lr1,
    "CV 90%": johansen_test.cvt[:,0],
    "CV 95%": johansen_test_const.cvt[:,1],
    "CV 99%": johansen_test_const.cvt[:,2]
})

print(trace_table_const)


# ======================================
# 3 Estimaciones de los modelos vistos
# ======================================

#Especificaciones:
#"n" - no deterministic terms
#"co" - constant outside the cointegration relation
#"ci" - constant within the cointegration relation
#"lo" - linear trend outside the cointegration relation
#"li" - linear trend within the cointegration relation


#~~Sin constante~~~#

vecm_none = VECM(Y, k_ar_diff=1, coint_rank=1, deterministic="n")
vecm_none_fit = vecm_none.fit()
print(vecm_none_fit.summary())

#Vector de cointegración (beta) y velocidad de ajuste (alpha)

#Beta
print("\nVector de cointegración (beta):");print(vecm_none_fit.beta)

#Alpha
print("\nVelocidades de ajuste (alpha):");print(vecm_none_fit.alpha)



#~~~ Con constante~~~#

vecm_const = VECM(Y, k_ar_diff=1, coint_rank=1, deterministic="ci")
vecm_const_fit = vecm_const.fit()
print(vecm_const_fit.summary())


#Beta
print("\nVector de cointegración (beta):");print(vecm_const_fit.beta)

#Constante de cointegración
print(vecm_const_fit.det_coef_coint)

#Alpha
print("\nVelocidades de ajuste (alpha):");print(vecm_const_fit.alpha)


#Veamos el modelo con tendencia lineal

vecm_trend = VECM(Y, k_ar_diff=1, coint_rank=1, deterministic="lo")

vecm_trend_fit = vecm_trend.fit()

print(vecm_trend_fit.summary())

#Vemos que no es significativo el término de tendencia lineal, 
# por lo que nos quedamos con el modelo con constante.


# =============================================
# 3 Validación de supuestos y usos del modelo
# =============================================

#~~~Autocorrelacion- Portamanteu Test~~~#

P_12=vecm_const_fit.test_whiteness(nlags=12);print(P_12.summary())

P_24=vecm_const_fit.test_whiteness(nlags=24); print(P_24.summary())

P_36=vecm_const_fit.test_whiteness(nlags=36); print(P_36.summary())



#Homocedasticidad
resid = vecm_const_fit.resid

print("===== ARCH TEST (12 lags) =====")
for i in range(resid.shape[1]):
    arch = het_arch(resid[:, i], nlags=12)
    print(f"Ecuación {i+1}")
    print("LM stat:", arch[0])
    print("p-value:", arch[1])
    print()

print("===== ARCH TEST (24 lags) =====")
for i in range(resid.shape[1]):
    arch = het_arch(resid[:, i], nlags=24)
    print(f"Ecuación {i+1}")
    print("LM stat:", arch[0])
    print("p-value:", arch[1])
    print()


#Normalidad
vecm_none_fit.test_normality().summary()


# =============================================
# 4.1 Pronostico del VECM
# =============================================

forecast = vecm_const_fit.predict(steps=12)
print(forecast)

forecast_df = pd.DataFrame(forecast, columns=Y.columns)
forecast_df.head()


#~~~ Grafica del pronóstico ~~~#

# Últimos datos observados
Y_values = Y.values
n_obs = Y_values.shape[0]

# Pronóstico
steps = 12
forecast, lower, upper = vecm_const_fit.predict(steps=steps, alpha=0.05)

# Eje de tiempo
t = np.arange(n_obs)
t_forecast = np.arange(n_obs, n_obs + steps)
nombres = ["Petróleo Brent", "Petróleo WTI"]
# Graficar cada variable
for i in range(Y_values.shape[1]): 
    plt.figure()
    # Histórico
    plt.plot(t, Y_values[:, i], label="Histórico")
    # Pronóstico
    plt.plot(t_forecast, forecast[:, i], linestyle="--", label="Pronóstico")
    # Intervalos
    plt.fill_between(t_forecast, lower[:, i], upper[:, i], alpha=0.5)
    plt.title(f"Pronóstico VECM - {nombres[i]}")
    plt.xlabel("Tiempo")
    plt.ylabel("Precio")
    plt.legend()
    plt.show()


# =============================================
# 4.2 Funciones impulso respuesta
# =============================================

irf = vecm_none_fit.irf(12)
irf.plot(orth=True)
plt.show()

# =============================================
# 5. Un comportamiento llamativo
# =============================================

Y2 = data[["WTI", "Brent"]]
Y2.head()

modelo2 = VAR(Y2)
VAR_2 = modelo2.fit(2)
print(VAR2.summary())



 #Reparametrización del modelo VECM 
vecm_const2 = VECM(Y2, k_ar_diff=1, coint_rank=1, deterministic="co")
vecm2_fit = vecm_const2.fit()
print(vecm_fit.summary())


#Veamos las funciones impulso respuesta

irf = vecm2_fit.irf(12)
irf.plot(orth=True)
plt.show()

# Los impuslo respuesta cambian significativamente. ¿La noción de esto? Depende
# del contexto económico.

# Lo que muestra la conclusión teórica es que la variable que se coloca primero 
# es la más exogena de las dos. Por lo que, cuando construyan su VAR, coloquen 
# la más exogena arriba.


#============================================
#~~~~~~~~~FIN DEL CODIGO ;) ~~~~~~~~~~~~~~~~
#============================================