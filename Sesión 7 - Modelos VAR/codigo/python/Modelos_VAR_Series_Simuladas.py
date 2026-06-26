"""
Universidad Nacional de Colombia
Facultad de Ciencias Económicas

Econometría II | Monitoría
Sesión 7: Modelos de vectores autorregresivos

Semestre: 2026-1
"""

#%reset -f Por si queremos limpiar el entorno de trabajo antes de ejecutar el código

#~~~~~~~~~~~~~~~~~~~~~#
# Tabla de contenidos #
#~~~~~~~~~~~~~~~~~~~~~#
#
#  Instalación de paquetes
# 1. Creación de serie simulada 
# 2. Metodologia Box-Jenkins para series multivariadas
#  2.1. Identificación
#  2.2. Estimación
#  2.3. Validación de supuestos
#  2.4. Pronóstico y funciones Impulso respuesta 
# 3. Ejemplo de Enders: VAR con 3 variables


#Importamos las librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

# ============================================
# 1.1 Creación de series simuladas
# ============================================

# Fijar semilla
np.random.seed(82901)

# Tamaño de muestra de 300 observaciones
T = 300

# y_t es una matriz de 2 variables (una variable por columna), En tanto hay dos 
# variables, habrá 2 residuales.
y_t = np.zeros((T,2))


# Residuales
u_t = np.random.normal(0, 1, size=(T, 2))

# Matriz de coeficientes VAR
A1 = np.array([[0.3,0.2],[0.5,0.6]])

A0 = np.array([0.5,0.5])

# ============================================
# Función para simular VAR(1)
# ============================================

def sim(y_t, A1, u_t, T):
    
    for i in range(1,T):
        y_t[i,:] = A0 + A1 @ y_t[i-1,:] + u_t[i,:]
    
    return y_t


y_t = sim(y_t, A1, u_t, T) # La función sim lo que busca es llenar la matriz 
                            # de ceros y_t con valores

# ============================================
# Convertir a DataFrame tipo serie temporal
# ============================================

#QE= Quarterly end
#QS= Quarterly start
#ME, MS
#6MS
#YE,YS

fechas = pd.date_range(start="1900-01-01", periods=T, freq="QS")
df = pd.DataFrame(y_t, index=fechas, columns=["y1","y2"])
df.head()
# ============================================
# 1.2 Grafica de las series
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["y1"].plot(ax=axes[0], color="lightblue", linewidth=1)
axes[0].set_title("Variable y_1")
axes[0].set_xlabel("")
df["y2"].plot(ax=axes[1], color="royalblue", linewidth=1)
axes[1].set_title("Variable y_2")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()
 
# ============================================
# 1.3 PRUEBA ADF (Dickey-Fuller Aumentada)
# ============================================

# Recuerden que los modelos VAR requieren de series 
# estacionarias. 

#Recordemos la especificación de la prueba ADF en Python:

# n-> Sin constante ni tendencia
# c-> Con constante"
# ct-> Con constante y tendencia"

#Prueba de raíz unitaria ADF para y1
adf1 = adfuller(df["y1"],regression="c",autolag="AIC")

print("ADF statistic:", adf1[0])
print("Critical values:")
for key, value in adf1[4].items():
    print(f"{key}: {value}");print("p-value:", adf1[1])


#Prueba de raíz unitaria ADF para y2
adf2 = adfuller(df["y2"],regression="c",autolag="AIC")

print("ADF statistic:", adf2[0])
print("Critical values:")
for key, value in adf2[4].items():
    print(f"{key}: {value}");print("p-value:", adf2[1])



# =====================================================
# 2. METODOLÓGIA BOX-JENKINS PARA SERIES MULTIVARIADAS
# =====================================================

# ============================================
# 2.1 Identificación
# ============================================

#Primero planteamos el modelo
modelo = VAR(df)

#Seleccion de rezagos de un Modelo VAR con tendencia y constante
lag_order_ct = modelo.select_order(6, trend="ct");print(lag_order_ct.summary()) 

#Seleccion de rezagos de un Modelo VAR con constante
lag_order_c = modelo.select_order(6, trend="c");print(lag_order_c.summary())

#Seleccion de rezagos de un Modelo VAR sin tendencia ni constante
lag_order_n = modelo.select_order(6, trend="n");print(lag_order_n.summary())

# Todos los criterios recomiendan un VAR(1)

# ============================================
# 2.2 Estimación
# ============================================

#Estimación del modelo VAR con tendencia y constante
V_tr = modelo.fit(1, trend="ct");print(V_tr.summary())
#La tendencia no es significativa, veamos solo con constante 

#Estimación del modelo VAR con constante
V_c = modelo.fit(1, trend="c");print(V_c.summary())
#El termino de deriva es significativo

#Estimación del modelo VAR sin tendencia ni constante
V_n = modelo.fit(1, trend="n");print(V_n.summary())


#Verificamos la estabilidad del proceso

roots = V_n.roots

print("\nRaíces del proceso (deben ser > |1|):");print(np.abs(V_n.roots))

## Ahora veamos cada uno de los coeficientes estimados. 
print(V_n.coefs)

#Veamos la matriz de varianzas y covarianzas de los residuales
Sigma_est = V_n.sigma_u;print(Sigma_est)


# ============================================
# 2.3 Validacion de los supuestos del modelo
# ============================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# No autocorrelación serial
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Guardamos los residuales del modelo VAR
residuales = V_n.resid

#~~~~~~~ Prueba Lyung-box~~~~~~~~~~~

# Serie 1
lb_y1 = acorr_ljungbox(residuales.iloc[:,0], lags=[10,20,30,75], return_df=True);print(lb_y1)

# Serie 2
lb_y2 = acorr_ljungbox(residuales.iloc[:,1], lags=[10,20,30,75], return_df=True);print(lb_y2)

#~~~ Prueba Portmanteu ~~~#
P_12=V_n.test_whiteness(nlags=12);print(P_12.summary())
P_24=V_n.test_whiteness(nlags=24);print(P_24.summary())
P_36=V_n.test_whiteness(nlags=36);print(P_36.summary())


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Grafica de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

for nombre, serie in residuales.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"Diagnóstico de residuales – {nombre}", fontsize=13)
    serie.plot(ax=axes[0, 0], title="Residuales", color="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 1].hist(serie, bins=30, color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Distribución")
    plot_acf(serie, ax=axes[1, 0], lags=20, title="ACF residuales")
    plot_pacf(serie, ax=axes[1, 1], lags=20, title="PACF residuales")
    plt.tight_layout()
    plt.show()
 

res1 = residuales.iloc[:,0]
res2 = residuales.iloc[:,1]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Homocedasticidad de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Test tipo ARCH 

# Ho: No hay heterocedasticidad ARCH
# Ha: Hay heterocedasticidad ARCH

arch_res_1 = het_arch(res1, nlags=12);print("ARCH test p-value:", arch_res_1[1])

arch_res_2 = het_arch(res2, nlags=12);print("ARCH test p-value:", arch_res_2[1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Normalidad de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Ho: Los residuales siguen una distribución normal
#Ha: Los residuales no siguen una distribución normal

print("Jarque-Bera Series 1 p-value:", jarque_bera(res1).pvalue)
print("Jarque-Bera Series 2 p-value:", jarque_bera(res2).pvalue)

# ============================================
# 2.4 Pronóstico
# ============================================

#~~~Pronostico puntual~~~#
steps = 12
forecast_V_n = V_n.forecast(y=V_n.endog, steps=steps)
print(forecast_V_n)

freq = df.index.freq or pd.tseries.frequencies.to_offset(pd.infer_freq(df.index))
future_idx = pd.date_range(start=df.index[-1], periods=steps + 1, freq=freq)[1:]

#Volvemos el pronostico a un DataFrame
forecast_df = pd.DataFrame(
    forecast_V_n,
    index=future_idx,
    columns=df.columns); print(forecast_df)


#~~~Pronostico por intervalos de confianza~~~#

forecast, lower, upper = V_n.forecast_interval(
    y=V_n.endog, steps=steps, alpha=0.05
)

lower_df = pd.DataFrame(lower, index=future_idx, columns=df.columns)
upper_df = pd.DataFrame(upper, index=future_idx, columns=df.columns)

for col in df.columns:
    print(f"\nVariable: {col}")
    print(pd.DataFrame({
        "forecast": forecast_df[col],
        "lower": lower_df[col],
        "upper": upper_df[col]
    }))


#~~~Grafica del pronostico~~~#

V_n.plot_forecast(12)
fig = plt.gcf()
for ax in fig.axes:
    ax.legend(loc="upper left")

fig.set_size_inches(15, 10)
plt.show()



# ============================================
# 3.  Funciones impulso - respuesta
# ============================================

#~~~IRF "Sencillas"~~~#
irf = V_n.irf(18)
irf.plot()
plt.show()

#Representacion de las IRF sencillas 
print(irf.irfs)


#~~~IRF ortogonalizadas ~~~#
# Coeficientes de las IRF ortogonales
irf = V_n.irf(18)
irf.plot(orth=True)

#Representacion de las IRF ortogonalizadas
print(irf.orth_irfs)


