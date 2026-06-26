# %%
"""
Universidad Nacional de Colombia
Facultad de Ciencias Economicas

Econometria II | Monitoria
Sesion 7: Modelos de vectores autorregresivos - Series Simuladas

Semestre: 2026-1
"""

# ===
# Tabla de contenidos
# ===

# 1. Simulacion de un proceso VAR(1) con 3 variables
#  1.1 Especificacion de las condiciones de la simulacion
#  1.2 Simulacion de los errores en forma reducida " u_t "
#    1.2.1 Construccion de los errores " u_t " usando descomposicion de Cholesky
#    1.2.2 Propiedades de los errores en forma reducida " u_t "
#  1.3 Simulacion del VAR(1) de 3 variables
# 2. Metodologia Box-Jenkins para series multivariadas
#  2.1. Identificacion
#  2.2. Estimacion
#  2.3. Validacion de supuestos
#  2.4. Uso del modelo: pronostico y funciones Impulso respuesta (IRF)


# Nota: Tips practicos en Python
## Para limpiar el entorno en IPython/Jupyter se puede correr: "%reset -f"
## Para cerrar todas las graficas actualmente abiertas: "plt.close('all')"
## En VS Code o Spyder, los bloques marcados con "# %%" se ejecutan por celdas.


# %%
# Importacion de paquetes ----

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


try:
    RUTA_SCRIPT = Path(__file__).resolve().parent
except NameError:
    RUTA_SCRIPT = Path.cwd()
    if not (RUTA_SCRIPT / "funciones_auxiliares_graficacion_VAR.py").exists():
        RUTA_SCRIPT = Path.cwd() / "codigo" / "python"

if str(RUTA_SCRIPT) not in sys.path:
    sys.path.append(str(RUTA_SCRIPT))

from funciones_auxiliares_graficacion_VAR import (  # noqa: E402
    graficar_diagnostico_errores,
    graficar_diagnostico_residuales_var,
    graficar_fevd_var,
    graficar_grilla_irf,
    graficar_pronostico_var,
    graficar_ts,
)


pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
plt.close("all")


def mostrar_graficas():
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


# %%
# ===
# 1. Simulacion de un proceso VAR(1) con 3 variables ====
# ===

# 1.1 Especificacion de las condiciones de la simulacion ----

# Fijamos la semilla para que siempre de el mismo resultado
semilla_simulacion = 82901
generador = np.random.default_rng(semilla_simulacion)

# Determinamos un tamano de muestra de 5000 observaciones
T = 5000  # Nota: Entre mas muestra, mejor se dara la convergencia
          #       de los resultados simulados a los teoricos

# Nombre de las variables y los errores asociados
variables = ["y_1", "y_2", "y_3"]
errores = ["u_1", "u_2", "u_3"]

# Se va a simular un modelo VAR(1) cuya ecuacion esta dada por:
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: Y_t es una matriz de 3 variables (una variable por columna).
#       u_t tambien es una matriz de 3 variables, donde cada columna de la
#       matriz es el error asociado a cada variable de la matriz Y_t.

# Y_t se crea como una matriz de 0s. Luego se llena con valores reales, cuando
# ocurra la simulacion del VAR(1)
Y_t = np.zeros((T, len(variables)))
print(pd.DataFrame(Y_t, columns=variables).head())


# %%
# 1.2 Simulacion de los errores en forma reducida " u_t " ----

# ===
# Nota: Tenga presente que esta es la parte mas importante de toda la simulacion.
#       Dado que la forma en la que se simulan dichos errores determinara
#       todas las caracteristicas de las series simuladas Y_t. Esto ocurre porque
#       a partir de u_t, usando la formula Y_t = A_0 + A_1 Y_{t-1} + u_t, es que
#       se construye la serie Y_t. Por tanto, lo crucial de la simulacion es
#       simular de manera correcta la distribucion de los errores en forma
#       reducida "u_t".
# ===

# En esta simulacion se quiere que los errores esten correlacionados y que,
# ademas, tengan desviaciones estandar diferentes. Para ello simulamos:
#
#   u_t ~ N_3(0, Sigma_u) ; Distribucion Normal Trivariada
#
# La matriz P_chol es triangular inferior. Esto permite construir una matriz de
# varianzas y covarianzas:
#
#   Sigma_u = P_chol P_chol' ; Donde P_chol es la matriz de la descomposicion de Cholesky
#
# Esta construccion es coherente con una identificacion recursiva tipo Cholesky:
# y_1 es contemporaneamente mas exogena que y_2 y y_3, mientras que y_2 es mas
# exogena que y_3. Los errores reducidos u_t estaran correlacionados, pero los
# errores estructurales eps_t que los generan son ortogonales.


# 1.2.1 Construccion de los errores " u_t " usando descomposicion de Cholesky ----

P_chol = np.array(
    [
        [0.70, 0.00, 0.00],
        [0.35, 1.10, 0.00],
        [0.25, 0.55, 1.60],
    ]
)
P_chol_df = pd.DataFrame(P_chol, index=errores, columns=["eps_1", "eps_2", "eps_3"])
print(P_chol_df)

# Matriz de varianzas-covarianzas teorica de la distribucion normal multivariada
Sigma_u_teorica = P_chol @ P_chol.T
Sigma_u_teorica_df = pd.DataFrame(Sigma_u_teorica, index=errores, columns=errores)
print(Sigma_u_teorica_df)

# Matriz de correlaciones teorica
desv_u_teoricas = np.sqrt(np.diag(Sigma_u_teorica))
cor_u_teorica = Sigma_u_teorica / np.outer(desv_u_teoricas, desv_u_teoricas)
cor_u_teorica_df = pd.DataFrame(cor_u_teorica, index=errores, columns=errores)
print(cor_u_teorica_df)

# Desviaciones estandar teoricas de los errores en forma reducida
desv_u_teoricas_s = pd.Series(desv_u_teoricas, index=errores, name="desv_teorica")
print(desv_u_teoricas_s)

# La media de los errores en forma reducida sera el vector de ceros
media_u = np.zeros(len(errores))

# Errores estructurales ortogonales eps_t
eps_t = generador.normal(loc=0, scale=1, size=(T, len(errores)))

# Errores en forma reducida " u_t " simulados de una normal trivariada usando
# la matriz de Cholesky. Con esta construccion:
#
#   Cov(u_t) = P_chol P_chol'
#
u_t = eps_t @ P_chol.T
u_t = pd.DataFrame(u_t, columns=errores)


# %%
# 1.2.2 Propiedades de los errores en forma reducida " u_t " ----

resumen_errores = pd.DataFrame(
    {
        "error": errores,
        "media": u_t.mean().to_numpy(),
        "desviacion_estandar": u_t.std(ddof=1).to_numpy(),
        "varianza": u_t.var(ddof=1).to_numpy(),
    }
)
print(resumen_errores)

# Matriz muestral de varianzas y covarianzas de los errores simulados.
Sigma_u_muestral = u_t.cov()
print(Sigma_u_muestral)

# Matriz muestral de correlaciones de los errores simulados.
cor_u_muestral = u_t.corr()
print(cor_u_muestral)

# Verificacion grafica de la normalidad de los errores " u_t " simulados
graficos_errores = graficar_diagnostico_errores(
    u_t=u_t,
    errores=errores,
    cor_u_muestral=cor_u_muestral,
)
mostrar_graficas()


# %%
# 1.3 Simulacion del VAR(1) de 3 variables ----

# Nota: Recuerde que se va a simular un modelo VAR(1) cuya ecuacion esta dada por:
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Definimos el vector constante A_0
A_0 = np.array([0.5, 0.2, -0.1])
print(pd.Series(A_0, index=variables, name="A_0"))

# Definimos la matriz de coeficientes autorregresivos.
A_1 = np.array(
    [
        [0.35, 0.08, 0.04],
        [0.25, 0.30, 0.06],
        [0.15, 0.20, 0.25],
    ]
)
A_1_df = pd.DataFrame(A_1, index=variables, columns=[f"L1.{v}" for v in variables])
print(A_1_df)

# La matriz A_1 no es triangular inferior. Por tanto, la simulacion permite
# efectos rezagados cruzados entre las tres variables. Esto separa claramente
# la dinamica del VAR de la identificacion contemporanea de Cholesky: el orden
# recursivo y_1, y_2, y_3 se mantiene por el orden de las columnas de Y_t y por
# la estructura triangular de P_chol, no porque A_1 sea triangular.


def sim_VAR1(Y_t, A_0, A_1, u_t, T):
    for i in range(1, T):
        # Se usa la formula de un VAR(1): Y_t = A_0 + A_1 Y_{t-1} + u_t
        # Para llenar cada una de las filas de Y_t
        Y_t[i, :] = A_0 + A_1 @ Y_t[i - 1, :] + u_t.iloc[i, :].to_numpy()
    return Y_t


Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T)

# Convertimos la serie en un DataFrame con indice de serie de tiempo trimestral.
# En R, time(ts(..., start = c(1900, 1), frequency = 4)) produce un indice
# numerico. Usamos la misma idea porque 5000 trimestres desde 1900 exceden el
# rango de fechas tipo Timestamp de pandas.
tiempo = pd.Index(1900 + np.arange(T) / 4, name="tiempo")
Y_t = pd.DataFrame(Y_t, index=tiempo, columns=variables)

# Graficas de las series simuladas
fig_series, axes_series = plt.subplots(1, 3, figsize=(15, 4))
graficar_ts(Y_t["y_1"], titulo="Variable y_1", color="lightblue", ax=axes_series[0])
graficar_ts(Y_t["y_2"], titulo="Variable y_2", color="royalblue", ax=axes_series[1])
graficar_ts(Y_t["y_3"], titulo="Variable y_3", color="darkorange", ax=axes_series[2])
fig_series.tight_layout()
mostrar_graficas()


# %%
# Nota: Recuerden que los modelos VAR requieren de series estacionarias. Por tanto,
#       empleamos Test ADF para verificar la estacionariedad de las series.


def imprimir_adf(resultado, nombre_variable):
    print(f"\nADF para {nombre_variable}")
    print(f"Estadistico ADF: {resultado[0]:.6f}")
    print(f"p-valor: {resultado[1]:.6f}")
    print(f"Rezagos usados: {resultado[2]}")
    print("Valores criticos:")
    for clave, valor in resultado[4].items():
        print(f"  {clave}: {valor:.6f}")


adf1 = adfuller(Y_t["y_1"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf1, "y_1")

adf2 = adfuller(Y_t["y_2"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf2, "y_2")

adf3 = adfuller(Y_t["y_3"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf3, "y_3")


# %%
# ===
# 2. Metodologia Box-Jenkins para series multivariadas ====
# ===

# ===
# 2.1. Identificacion ====
# ===

# Ya tenemos las series simuladas en la matriz Y_t, por lo que ya es posible
# aplicar la metodologia Box-Jenkins en las series simuladas.

# Para la estimacion usamos un RangeIndex. La escala temporal trimestral se
# conserva en Y_t para graficar, pero statsmodels prefiere indices soportados
# cuando se instancian modelos de series de tiempo.
Y_t_modelo = Y_t.reset_index(drop=True)
modelo = VAR(Y_t_modelo)


def imprimir_seleccion_rezagos(resultado, titulo, incluye_lag_cero=True):
    print(f"\n{titulo}")
    try:
        print(resultado.summary())
    except IndexError:
        # statsmodels 0.14 puede fallar al imprimir summary() cuando trend="n".
        # El resultado si existe; por eso imprimimos manualmente los criterios.
        n_filas = len(next(iter(resultado.ics.values())))
        indice = range(0, n_filas) if incluye_lag_cero else range(1, n_filas + 1)
        tabla = pd.DataFrame(resultado.ics, index=indice)
        columnas = [col for col in ["aic", "bic", "fpe", "hqic"] if col in tabla]
        print(tabla[columnas])
        print("Rezagos seleccionados:")
        print(pd.Series(resultado.selected_orders))


# Seleccion de rezagos para un VAR con tendencia e intercepto.
lag_order_ct = modelo.select_order(maxlags=6, trend="ct")
imprimir_seleccion_rezagos(
    lag_order_ct,
    "Seleccion de rezagos para un VAR con tendencia e intercepto",
)

# Seleccion de rezagos para un VAR con solo intercepto.
lag_order_c = modelo.select_order(maxlags=6, trend="c")
imprimir_seleccion_rezagos(
    lag_order_c,
    "Seleccion de rezagos para un VAR con solo intercepto",
)

# Seleccion de rezagos para un VAR sin terminos deterministicos.
lag_order_n = modelo.select_order(maxlags=6, trend="n")
imprimir_seleccion_rezagos(
    lag_order_n,
    "Seleccion de rezagos para un VAR sin terminos deterministicos",
    incluye_lag_cero=False,
)

# Como el proceso generador de datos es un VAR(1), esperamos que los criterios
# de informacion favorezcan rezagos bajos, especialmente p = 1.


# %%
# ===
# 2.2. Estimacion ====
# ===

# Para seleccionar el VAR(1), verificamos si tiene intercepto y deriva:

# VAR con tendencia e intercepto
V_tr = modelo.fit(1, trend="ct")
print(V_tr.summary())

# VAR con intercepto.
V_dr = modelo.fit(1, trend="c")
print(V_dr.summary())

# VAR sin terminos deterministicos.
V_no = modelo.fit(1, trend="n")
print(V_no.summary())

# Elegimos el modelo con constante, pues se ha visto que tiene constante
# significativa en buena parte del sistema.

# Estabilidad del VAR(1):
# En statsmodels, las raices reportadas por roots deben quedar fuera del circulo
# unitario. Complementamos con is_stable(), que revisa la estabilidad del VAR.
raices = pd.DataFrame(
    {
        "raiz": V_dr.roots,
        "modulo": np.abs(V_dr.roots),
    }
)
print(raices)
print("El proceso es estable:", V_dr.is_stable(verbose=True))

# Coeficientes estimados del VAR(1):
coeficientes_A1 = pd.DataFrame(
    V_dr.coefs[0],
    index=variables,
    columns=[f"L1.{v}" for v in variables],
)
print(coeficientes_A1)

# Matriz teorica "A1" vs matriz estimada por statsmodels
print(A_1_df)
print(coeficientes_A1)

# Vector constante estimado
print(pd.Series(V_dr.intercept, index=variables, name="constante_estimada"))

# Matriz de varianzas y covarianzas de los residuales teorica vs estimada
print(Sigma_u_teorica_df)
Sigma_est = V_dr.sigma_u
print(Sigma_est)


# %%
# ===
# 2.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ===

# Portmanteau multivariado. En statsmodels usamos test_whiteness().
P_75 = V_dr.test_whiteness(nlags=75, adjusted=False)
print(P_75.summary())

P_30 = V_dr.test_whiteness(nlags=30, adjusted=False)
print(P_30.summary())

P_20 = V_dr.test_whiteness(nlags=20, adjusted=False)
print(P_20.summary())

P_10 = V_dr.test_whiteness(nlags=10, adjusted=False)
print(P_10.summary())

# Graficamos diagnosticos de residuales: serie, distribucion, ACF y PACF.
residuales = pd.DataFrame(
    np.asarray(V_dr.resid),
    index=Y_t.index[V_dr.k_ar :],
    columns=variables,
)
figuras_residuales = graficar_diagnostico_residuales_var(residuales, lags=20)
mostrar_graficas()

# Homocedasticidad ===

# statsmodels no tiene un equivalente directo a arch.test() multivariado de vars.
# Como aproximacion docente, aplicamos pruebas ARCH univariadas por ecuacion.
for lags in [24, 12]:
    print(f"\nPruebas ARCH univariadas con {lags} rezagos")
    for variable in variables:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuales[variable], nlags=lags)
        print(
            f"{variable}: LM p-value = {lm_pvalue:.6f}; "
            f"F p-value = {f_pvalue:.6f}"
        )

# Normalidad ===

# Jarque-Bera multivariado de statsmodels para los residuales del VAR.
normalidad = V_dr.test_normality()
print(normalidad.summary())

# Tambien miramos Jarque-Bera univariado para cada ecuacion.
for variable in variables:
    jb = jarque_bera(residuales[variable])
    print(f"Jarque-Bera {variable} p-value: {jb.pvalue:.6f}")


# %%
# ===
# 2.4. Uso del modelo: pronostico y funciones Impulso respuesta (IRF) ====
# ===

# Pronostico ===

pasos_pronostico = 12
ultimos_valores = V_dr.endog[-V_dr.k_ar :]

pronostico = V_dr.forecast(y=ultimos_valores, steps=pasos_pronostico)

fechas_futuras = pd.Index(
    Y_t.index[-1] + np.arange(1, pasos_pronostico + 1) / 4,
    name="tiempo",
)

pronostico_df = pd.DataFrame(pronostico, index=fechas_futuras, columns=variables)
print(pronostico_df)

# Pronostico por intervalos de confianza
pronostico_int, inferior, superior = V_dr.forecast_interval(
    y=ultimos_valores,
    steps=pasos_pronostico,
    alpha=0.05,
)

inferior_df = pd.DataFrame(inferior, index=fechas_futuras, columns=variables)
superior_df = pd.DataFrame(superior, index=fechas_futuras, columns=variables)

for variable in variables:
    print(f"\nVariable: {variable}")
    print(
        pd.DataFrame(
            {
                "pronostico": pronostico_df[variable],
                "inferior": inferior_df[variable],
                "superior": superior_df[variable],
            }
        )
    )

fig_pronostico, axes_pronostico = graficar_pronostico_var(
    pronostico_df,
    inferior_df,
    superior_df,
)
mostrar_graficas()


# %%
# Funciones de impulso respuesta no ortogonalizadas ===

# Nota: Recuerde que para poder calcular las IRF de un modelo VAR
#       este debe tener su representacion como VMA(infinito).
#       Es decir, pasamos del VAR(1) --> VMA(infinito)

# IRFs no ortogonalizadas:
print(V_dr.irf(10).irfs)

# Graficacion de las IRFs
pasos_adelante = np.arange(0, 19)
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_montecarlo_irf = 100

# La funcion graficar_grilla_irf() se importa desde el script
# "funciones_auxiliares_graficacion_VAR.py". Calcula el objeto irf() una sola
# vez y luego usa programacion funcional para crear cada panel de la grilla.
irf_no_ortog = graficar_grilla_irf(
    V_dr,
    variables,
    pasos_adelante,
    ortog=False,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso",
    semilla=semilla_irf,
    runs=repeticiones_montecarlo_irf,
)

print(irf_no_ortog["objeto_irf"].irfs)
mostrar_graficas()


# %%
# Funciones de impulso respuesta ortogonalizadas ===

# Cuando ortog = True, statsmodels usa una descomposicion de Cholesky de la
# matriz de varianzas y covarianzas de los residuales. En este script el orden
# de las variables es y_1, y_2, y_3; por tanto, la identificacion recursiva
# interpreta a y_1 como la variable contemporaneamente mas exogena, luego y_2 y
# finalmente y_3 como la mas endogena.

# IRFs ortogonalizadas:
print(V_dr.irf(10).orth_irfs)

irf_ortog = graficar_grilla_irf(
    V_dr,
    variables,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_montecarlo_irf,
)

print(irf_ortog["objeto_irf"].orth_irfs)
assert len(irf_no_ortog["graficas"]) == len(variables) ** 2
assert len(irf_ortog["graficas"]) == len(variables) ** 2
print("VALIDACION_IRF_OK")
mostrar_graficas()


# %%
# Descomposicion de varianza del error de pronostico ===

# La descomposicion de varianza del error de pronostico (FEVD) da la proporcion
# de la varianza de error de pronostico de cada variable explicada por las
# variables dentro del sistema.

fevd_var = V_dr.fevd(periods=18)
fevd_var.summary()
fig_fevd, axes_fevd = graficar_fevd_var(fevd_var)
mostrar_graficas()
