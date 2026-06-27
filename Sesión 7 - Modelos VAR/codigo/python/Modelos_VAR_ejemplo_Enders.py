# %%
"""
Universidad Nacional de Colombia
Facultad de Ciencias Economicas

Econometria II | Monitoria
Sesion 7: Modelos de vectores autorregresivos - Ejemplo de Enders

Semestre: 2026-1
"""

# ===
# Tabla de contenidos
# ===

# 1. Importacion de paquetes, rutas y funciones auxiliares
# 2. Carga y preparacion de los datos
# 3. Analisis grafico y pruebas de estacionariedad
# 4. Metodologia Box-Jenkins para series multivariadas
#  4.1. Identificacion
#  4.2. Estimacion
#  4.3. Validacion de supuestos
#  4.4. Pronostico y funciones impulso-respuesta
#  4.5. Descomposicion de varianza del error de pronostico


# Nota: Tips practicos en Python
## Para limpiar el entorno en IPython/Jupyter se puede correr: "%reset -f"
## Para cerrar todas las graficas actualmente abiertas: "plt.close('all')"
## En VS Code o Spyder, los bloques marcados con "# %%" se ejecutan por celdas.


# %%
# ===
# 1. Importacion de paquetes, rutas y funciones auxiliares ====
# ===

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


# Cargar bases de datos en Python usando rutas relativas ----

try:
    ruta_script = Path(__file__).resolve().parent
except NameError:
    ruta_script = Path.cwd()
    if not (ruta_script / "funciones_auxiliares_graficacion_VAR.py").exists():
        ruta_script = Path.cwd() / "codigo" / "python"

directorio_codigo_python = ruta_script
directorio_sesion_var = directorio_codigo_python.parent.parent
directorio_datos = directorio_sesion_var / "datos"

# Ruta donde se encuentra base de datos del Enders (con las variables de interes)
ruta_enders = directorio_datos / "ENDERS.xlsx"


# Importacion de funciones auxiliares de graficacion

# Ruta con las funciones auxiliares de graficacion
ruta_funciones_auxiliares_var = (
    directorio_codigo_python / "funciones_auxiliares_graficacion_VAR.py"
)

if str(directorio_codigo_python) not in sys.path:
    sys.path.append(str(directorio_codigo_python))

from funciones_auxiliares_graficacion_VAR import (  # noqa: E402
    graficar_diagnostico_residuales_var,
    graficar_fevd_var,
    graficar_grilla_irf,
    graficar_pronostico_var,
    graficar_ts,
)


pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
plt.close("all")


def mostrar_graficas():
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


def imprimir_adf(resultado, nombre_variable):
    print(f"\nADF para {nombre_variable}")
    print(f"Estadistico ADF: {resultado[0]:.6f}")
    print(f"p-valor: {resultado[1]:.6f}")
    print(f"Rezagos usados: {resultado[2]}")
    print("Valores criticos:")
    for clave, valor in resultado[4].items():
        print(f"  {clave}: {valor:.6f}")


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


def imprimir_matrices_acof(var_resultados, variables):
    for lag, matriz in enumerate(var_resultados.coefs, start=1):
        matriz_lag = pd.DataFrame(
            matriz,
            index=variables,
            columns=[f"L{lag}.{variable}" for variable in variables],
        )
        print(f"\nMatriz A_{lag}")
        print(matriz_lag)


def pronostico_bootstrap_var(var_resultados, pasos, nboot=1000, semilla=None):
    """Pronostico bootstrap residual condicional a los ultimos p valores."""
    rng = np.random.default_rng(semilla)
    endog = np.asarray(var_resultados.endog)
    residuales = np.asarray(var_resultados.resid)
    coefs = np.asarray(var_resultados.coefs)
    intercepto = np.asarray(var_resultados.intercept)
    p = var_resultados.k_ar
    n_variables = endog.shape[1]

    ultimos_valores = endog[-p:, :]
    trayectorias = np.empty((nboot, pasos, n_variables))

    for b in range(nboot):
        y_boot = np.vstack([ultimos_valores.copy(), np.zeros((pasos, n_variables))])
        indices_residuales = rng.integers(0, residuales.shape[0], size=pasos)

        for h in range(p, p + pasos):
            prediccion = intercepto.copy()
            for lag in range(p):
                prediccion = prediccion + coefs[lag] @ y_boot[h - lag - 1, :]
            y_boot[h, :] = prediccion + residuales[indices_residuales[h - p], :]

        trayectorias[b, :, :] = y_boot[p:, :]

    alpha = 0.05
    return {
        "trayectorias": trayectorias,
        "pronostico": trayectorias.mean(axis=0),
        "inferior": np.quantile(trayectorias, alpha / 2, axis=0),
        "superior": np.quantile(trayectorias, 1 - alpha / 2, axis=0),
    }


# %%
# ===
# 2. Carga y preparacion de los datos ====
# ===

# La base de datos de Enders contiene series trimestrales de Estados Unidos
# para 1960T1-2012T4:
  # IPI  = indice de produccion industrial
  # CPI  = indice de precios al consumidor
  # Unem = tasa de desempleo
Base = pd.read_excel(ruta_enders)
Base.info()
print(Base.head())

# Series en niveles.
tiempo_niveles = pd.Index(1960 + np.arange(len(Base)) / 4, name="tiempo")
IPI = pd.Series(Base["IPI"].to_numpy(), index=tiempo_niveles, name="IPI")
CPI = pd.Series(Base["CPI"].to_numpy(), index=tiempo_niveles, name="CPI")
UNEM = pd.Series(Base["Unem"].to_numpy(), index=tiempo_niveles, name="Unem")

# Transformaciones usadas por Enders:
  # dl.IPI: aproxima la tasa de crecimiento del indice de produccion industrial.
  # dl.CPI: aproxima la inflacion trimestral.
  # Unem: se conserva en niveles porque es una tasa.
dl_IPI = np.log(IPI).diff().dropna()
dl_CPI = np.log(CPI).diff().dropna()

# Al tomar diferencias logaritmicas se pierde la primera observacion. Por ello,
# el desempleo se alinea desde 1960T2 hasta 2012T4, que es el periodo comun de
# las tres variables transformadas.
Unem = UNEM.loc[dl_IPI.index]

# Nota: Se ordenan las variables del modelo VAR. Este orden es importante, porque
# determina el orden de exogenidad de las variables, que sera muy importante a la
# hora de construir las IRF ortogonalizadas. La identificacion de Cholesky usa el
# orden de las columnas de la matriz Y para determinar cuales son las variables
# mas exogenas. La variable en la primera columna de Y sera la mas exogena,
# mientras que la variable en la ultima columna sera la mas endogena.

# Variables que se modelaran mediante el VAR
variables = ["dl.IPI", "Unem", "dl.CPI"]

# Se construye la matriz Y, que contiene las series de tiempo del modelo VAR
Y = pd.concat([dl_IPI, Unem, dl_CPI], axis=1)

# Se nombran las columnas de la matriz, con las series de tiempo
Y.columns = variables

# Algunas caracteristicas de las series de tiempo del modelo VAR
print("Inicio de Y:", Y.index[0])
print("Fin de Y:", Y.index[-1])
print(Y.head())
print(Y.tail())


# %%
# ===
# 3. Analisis grafico y pruebas de estacionariedad ====
# ===

# Graficas de las series transformadas que entraran al VAR.
fig_series, axes_series = plt.subplots(1, 3, figsize=(15, 4))
graficar_ts(
    Y["dl.IPI"],
    titulo="Crecimiento logaritmico del IPI",
    color="lightblue",
    ax=axes_series[0],
)
graficar_ts(
    Y["Unem"],
    titulo="Tasa de desempleo",
    color="mediumpurple",
    ax=axes_series[1],
)
graficar_ts(
    Y["dl.CPI"],
    titulo="Inflacion logaritmica del CPI",
    color="sienna",
    ax=axes_series[2],
)
fig_series.tight_layout()
mostrar_graficas()


# %%
# Pruebas ADF en niveles ----

# Nota: Para aplicar un modelo VAR en niveles, todas las variables tienen que ser
#       estacionarias, entonces se verifica que en efecto las variables sean
#       estacionarias. En caso de tener variables no estacionarias, toca tratar
#       las series, ya sea diferenciandolas o haciendo pruebas de cointegracion.

adf_ipi_nivel = adfuller(IPI, maxlag=6, autolag="AIC", regression="ct")
imprimir_adf(adf_ipi_nivel, "IPI en niveles")

adf_cpi_nivel = adfuller(CPI, maxlag=6, autolag="AIC", regression="ct")
imprimir_adf(adf_cpi_nivel, "CPI en niveles")

adf_unem_nivel = adfuller(UNEM, maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_unem_nivel, "Unem en niveles")


# %%
# Pruebas ADF sobre las variables que entran al VAR ----

# En el VAR se usan la tasa de crecimiento del IPI, la tasa de desempleo y la
# inflacion. Se verifica que estas variables sean estacionarias.

adf_dl_ipi = adfuller(Y["dl.IPI"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_dl_ipi, "dl.IPI")

adf_dl_cpi = adfuller(Y["dl.CPI"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_dl_cpi, "dl.CPI")

adf_unem = adfuller(Y["Unem"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_unem, "Unem")


# %%
# ===
# 4. Metodologia Box-Jenkins para series multivariadas ====
# ===

# El sistema a estimar es un VAR(p) sobre:
#   Y_t = (dl.IPI_t, Unem_t, dl.CPI_t)'
#
# La metodologia sigue cuatro pasos:
  # 1. Identificacion,
  # 2. Estimacion,
  # 3. Validacion y
  # 4. Uso del modelo para pronostico y funciones impulso-respuesta.


# %%
# ===
# 4.1. Identificacion ====
# ===

# Para la estimacion usamos un RangeIndex. La escala temporal trimestral se
# conserva en Y para graficar, pero statsmodels prefiere indices soportados
# cuando se instancian modelos de series de tiempo.
Y_modelo = Y.reset_index(drop=True)
modelo_enders = VAR(Y_modelo)

# Seleccion de rezagos para un VAR con tendencia e intercepto.
seleccion_rezagos_both = modelo_enders.select_order(maxlags=6, trend="ct")
imprimir_seleccion_rezagos(
    seleccion_rezagos_both,
    "Seleccion de rezagos para un VAR con tendencia e intercepto",
)

# Seleccion de rezagos para un VAR con solo intercepto.
seleccion_rezagos_const = modelo_enders.select_order(maxlags=6, trend="c")
imprimir_seleccion_rezagos(
    seleccion_rezagos_const,
    "Seleccion de rezagos para un VAR con solo intercepto",
)

# Seleccion de rezagos para un VAR sin terminos deterministicos.
seleccion_rezagos_none = modelo_enders.select_order(maxlags=6, trend="n")
imprimir_seleccion_rezagos(
    seleccion_rezagos_none,
    "Seleccion de rezagos para un VAR sin terminos deterministicos",
    incluye_lag_cero=False,
)

# En el ejemplo de Enders se trabaja con p = 3. A la hora de seleccionar el
# numero de rezagos, los criterios de informacion y la inspeccion de los
# residuales deben usarse conjuntamente: un VAR muy corto puede dejar
# autocorrelacion, mientras que un VAR excesivamente largo consume grados de
# libertad.
p_var = 3


# %%
# ===
# 4.2. Estimacion ====
# ===

# VAR con tendencia e intercepto.
V_tr_1 = modelo_enders.fit(p_var, trend="ct")
print(V_tr_1.summary())

# VAR con intercepto.
V_dr_1 = modelo_enders.fit(p_var, trend="c")
print(V_dr_1.summary())

# VAR sin terminos deterministas.
V_no_1 = modelo_enders.fit(p_var, trend="n")
print(V_no_1.summary())

# Se trabajara con un VAR(3) con constante. La constante es razonable porque las
# variables transformadas pueden tener medias distintas de cero.
VAR_enders = V_dr_1

# Estabilidad del VAR(3):

# Nota: En statsmodels las raices reportadas por roots deben quedar por fuera
#       del circulo unitario. Adicionalmente, is_stable() revisa que los valores
#       propios de la matriz de compania queden dentro del circulo unitario.
raices_var = pd.DataFrame(
    {
        "raiz": VAR_enders.roots,
        "modulo": np.abs(VAR_enders.roots),
    }
)
print(raices_var)
print("El proceso es estable:", VAR_enders.is_stable(verbose=True))

# Coeficientes estimados

# Coeficientes estimados por ecuacion. En R, Acoef() separa las matrices A_1,
# A_2 y A_3; aqui usamos var_resultados.coefs con el mismo objetivo.
imprimir_matrices_acof(VAR_enders, variables)

# Matriz de varianzas y covarianzas

# Matriz de varianzas y covarianzas estimada de los residuales en forma reducida.
Sigma_e = VAR_enders.sigma_u
print(Sigma_e)

# Analisis de todo el modelo VAR(3) estimado
print(VAR_enders.summary())


# %%
# ===
# 4.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ----

# En R se usa serial.test(). En statsmodels usamos el Portmanteau multivariado
# test_whiteness(); adjusted=False emula la version asintotica.
P_50 = VAR_enders.test_whiteness(nlags=50, adjusted=False)
print(P_50.summary())

P_30 = VAR_enders.test_whiteness(nlags=30, adjusted=False)
print(P_30.summary())

P_20 = VAR_enders.test_whiteness(nlags=20, adjusted=False)
print(P_20.summary())

P_10 = VAR_enders.test_whiteness(nlags=10, adjusted=False)
print(P_10.summary())

# Graficamos los diagnosticos de residuales: serie, distribucion, ACF y PACF.
residuales_enders = pd.DataFrame(
    np.asarray(VAR_enders.resid),
    index=Y.index[VAR_enders.k_ar :],
    columns=variables,
)

figuras_residuales = graficar_diagnostico_residuales_var(
    residuales_enders,
    lags=20,
)
mostrar_graficas()

# Nota: La decision se toma revisando los p-valores del Portmanteau para los
#       distintos horizontes de rezagos.


# %%
# Homocedasticidad ----

# statsmodels no tiene un equivalente directo a arch.test() multivariado de
# vars. Como aproximacion docente, aplicamos pruebas ARCH univariadas por
# ecuacion con 24 y 12 rezagos.
for lags in [24, 12]:
    print(f"\nPruebas ARCH univariadas con {lags} rezagos")
    for variable in variables:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
            residuales_enders[variable],
            nlags=lags,
        )
        print(
            f"{variable}: LM p-value = {lm_pvalue:.6f}; "
            f"F p-value = {f_pvalue:.6f}"
        )

# Nota: La decision se toma revisando los p-valores de las pruebas ARCH por
#       ecuacion. Este es un diagnostico univariado aproximado al bloque
#       multivariado usado por vars::arch.test() en R.


# %%
# Normalidad ----

# H0 del Jarque-Bera multivariado: los residuales tienen distribucion normal.
normalidad_enders = VAR_enders.test_normality()
print(normalidad_enders.summary())

# Tambien miramos Jarque-Bera univariado para cada ecuacion.
for variable in variables:
    jb = jarque_bera(residuales_enders[variable])
    print(f"Jarque-Bera {variable} p-value: {jb.pvalue:.6f}")

# Nota: Se cumple el supuesto de normalidad si no se rechaza H0.


# %%
# ===
# 4.4. Pronostico y funciones impulso-respuesta ====
# ===

# Pronostico ----

# Especificaciones del pronostico
horizonte_pronostico = 12
int_conf_pronostico = 0.95
alpha_pronostico = 1 - int_conf_pronostico

# Pronostico modelo VAR
ultimos_valores = VAR_enders.endog[-VAR_enders.k_ar :]
pronostico_puntual, inferior, superior = VAR_enders.forecast_interval(
    y=ultimos_valores,
    steps=horizonte_pronostico,
    alpha=alpha_pronostico,
)

# Como Y termina en 2012T4, el primer pronostico corresponde a 2013T1.
fechas_futuras = pd.Index(
    Y.index[-1] + np.arange(1, horizonte_pronostico + 1) / 4,
    name="tiempo",
)

pronostico_var = pd.DataFrame(
    pronostico_puntual,
    index=fechas_futuras,
    columns=variables,
)
inferior_var = pd.DataFrame(inferior, index=fechas_futuras, columns=variables)
superior_var = pd.DataFrame(superior, index=fechas_futuras, columns=variables)
print(pronostico_var)

for variable in variables:
    print(f"\nVariable: {variable}")
    print(
        pd.DataFrame(
            {
                "pronostico": pronostico_var[variable],
                "inferior": inferior_var[variable],
                "superior": superior_var[variable],
            }
        )
    )

# Graficar pronostico
fig_pronostico, axes_pronostico = graficar_pronostico_var(
    pronostico_var,
    inferior_var,
    superior_var,
)
fig_pronostico.suptitle("Pronostico VAR - ejemplo de Enders", fontsize=11)
fig_pronostico.tight_layout()
mostrar_graficas()

# Version equivalente a un fanchart basico usando statsmodels.
VAR_enders.plot_forecast(horizonte_pronostico)
fig_fanchart = plt.gcf()
fig_fanchart.set_size_inches(15, 10)
for ax in fig_fanchart.axes:
    ax.legend(loc="upper left")
fig_fanchart.tight_layout()
mostrar_graficas()


# %%
# Pronostico por bootstrapping ----

# Pronostico usando bootstrap residual condicional. Esta seccion reproduce la
# idea de VAR.etp::VAR.BPR en R: remuestrear residuales del VAR estimado y
# propagar la dinamica para obtener una distribucion de pronosticos.

# Especificaciones del pronostico usando bootstrap
repeticiones_bootstrap_pronostico = 1000
semilla_bootstrap_pronostico = 202601

# Comando para generar los pronosticos por medio de bootstrap
For_Boot = pronostico_bootstrap_var(
    VAR_enders,
    pasos=horizonte_pronostico,
    nboot=repeticiones_bootstrap_pronostico,
    semilla=semilla_bootstrap_pronostico,
)

# Pronosticos de bootstrap
boots = pd.DataFrame(
    For_Boot["pronostico"],
    index=fechas_futuras,
    columns=variables,
)
print(boots)

# Graficas para el pronostico calculado usando bootstrap
colores_series = {
    "dl.IPI": "lightblue",
    "Unem": "mediumpurple",
    "dl.CPI": "sienna",
}

fig_bootstrap, axes_bootstrap = plt.subplots(1, len(variables), figsize=(15, 4))
axes_bootstrap = np.atleast_1d(axes_bootstrap)

for ax, variable in zip(axes_bootstrap, variables):
    ax.plot(
        boots.index,
        boots[variable].to_numpy(),
        color=colores_series[variable],
        linewidth=0.8,
    )
    ax.set_title(variable, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, color="#e0e0e0", linewidth=0.8)

fig_bootstrap.suptitle("Pronostico puntual con bootstrapping", fontsize=11)
fig_bootstrap.tight_layout()
mostrar_graficas()


# %%
# Funciones de impulso-respuesta no ortogonalizadas ----

# Nota: Recuerde que para poder calcular las IRF de un modelo VAR
#       este debe tener su representacion como VMA(infinito).
#       Es decir, pasamos del VAR(1) --> VMA(infinito).

# IRFs no ortogonalizadas:
print(VAR_enders.irf(10).irfs)

# Graficacion de las IRFs

# Definimos el numero pasos adelante
pasos_adelante = np.arange(0, 25)
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100

# La funcion graficar_grilla_irf() se importa desde el script
# "funciones_auxiliares_graficacion_VAR.py". Calcula el objeto irf() una sola
# vez y luego crea cada panel con programacion funcional.
# IRF de las variables del sistema ante distintos choques exogenos.
irf_no_ortog = graficar_grilla_irf(
    VAR_enders,
    variables,
    pasos_adelante,
    ortog=False,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de IRF: columnas = impulsos; filas = respuestas.
print(irf_no_ortog["objeto_irf"].irfs)
mostrar_graficas()


# %%
# Funciones de impulso-respuesta ortogonalizadas ----

# Cuando ortog = True, statsmodels usa una descomposicion de Cholesky sobre la
# matriz de covarianzas de los residuales. Con el orden definido arriba, dl.IPI
# es la variable contemporaneamente mas exogena, luego Unem y finalmente dl.CPI.
# Este supuesto debe justificarse economicamente antes de interpretar las OIRF
# como choques estructurales.

# IRFs ortogonalizadas:
print(VAR_enders.irf(10).orth_irfs)

# Graficacion de las IRFs

# Usamos los mismos pasos adelante, intervalo de confianza y semilla.
# IRFs ortogonalizadas de las variables del sistema ante distintos choques
# exogenos.
irf_ortog = graficar_grilla_irf(
    VAR_enders,
    variables,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de OIRF: columnas = impulsos; filas = respuestas.
print(irf_ortog["objeto_irf"].orth_irfs)
assert len(irf_no_ortog["graficas"]) == len(variables) ** 2
assert len(irf_ortog["graficas"]) == len(variables) ** 2
print("VALIDACION_IRF_OK")
mostrar_graficas()


# %%
# ===
# 4.5. Descomposicion de varianza del error de pronostico ====
# ===

# La FEVD resume que proporcion de la varianza del error de pronostico de cada
# variable se atribuye a los choques de cada variable del sistema.
horizonte_fevd = 24
fevd_enders = VAR_enders.fevd(periods=horizonte_fevd)
fevd_enders.summary()

colores_fevd = {
    "dl.IPI": "#8B008B",  # magenta4
    "Unem": "#00CDCD",    # cyan3
    "dl.CPI": "#6959CD",  # slateblue3
}

fig_fevd, axes_fevd = graficar_fevd_var(
    fevd_enders,
    colores=colores_fevd,
)
mostrar_graficas()
