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


# %% ===
# 1. Importacion de paquetes, rutas y funciones auxiliares ====
# ===

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Módulos de statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Importar las funciones auxiliares del script auxiliar "funciones_auxiliares_graficacion_VAR"
from funciones_auxiliares_graficacion_VAR import (
    configurar_entorno_graficas,
    graficar_diagnostico_residuales_var,
    graficar_fanchart_var,
    graficar_fevd_var,
    graficar_grilla_irf,
    graficar_pronostico_var,
    graficar_ts,
    imprimir_adf,
    imprimir_matrices_acof,
    imprimir_seleccion_rezagos,
    mostrar_graficas,
    predecir_var,
    pronostico_bootstrap_var,
    prueba_arch_por_ecuacion,
    prueba_normalidad_por_ecuacion,
)

'''
Nota: Parar mirar la documentación de cada una de las funciones, puede usar el comando help(<funcion>)
      desde la terminal interactiva de ipython. E.g. para ver la documentación de la función 
      "graficar_grilla_irf", use el comando help(graficar_grilla_irf)
'''

# Para configurar las características de las gráficas
configurar_entorno_graficas(max_columns=30)

# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[2]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Ruta donde se encuentra base de datos del Enders (con las variables de interes)
ruta_enders = DATA_DIR / "ENDERS.xlsx"

# %% =========================
# 2. Carga y preparacion de los datos ===
# ============================

# La base de datos de Enders contiene series trimestrales de Estados Unidos
# para 1960T1-2012T4:
  # IPI  = indice de produccion industrial
  # CPI  = indice de precios al consumidor
  # Unem = tasa de desempleo
Base = pd.read_excel(ruta_enders)

# Información general de la base de datos
Base.info()

# Primeras observaciones de la base de datos
print(Base.head())

# Nota: Para trabajar con series de tiempo en python, debemos transformar
#       el objeto Base, para que no solo sea un Dataframe de pandas, sino también
#       tenga un índice temporal. 

"""
Nota: Para trabajar con series de tiempo en python se necesitan dos tipos 
de objetos: 

- pandas.core.series.Series: Para series de tiempo univariadas 
- pandas.core.frame.DataFrame: Para series de tiempo multivariadas

Nota: Un pandas.core.series.Series es básicamente un pandas.core.frame.DataFrame
      univariado

En python los objetos de series de tiempo tiene dos components:

serie de tiempo en python = datos + indice temporal

donde, 

- datos: Generalmente provienen de numpy.ndarray
- Indice temporal: Se crea un índice temporal compatible con los Series o 
                  DataFrames de pandas
    Exiten principalmente, los siguientes tipos de índices temporales: 
        - RangeIndex: Se usa donde las etiquetas no importan mucho
            
            e.g. pd.RangeIndex(start = 0, stop = 5000)
        
        - Index: Si se quiere trabajar con "tiempo numérico"
            
            e.g. pd.Index([1900.00, 1900.25, 1900.50])
            
        - PeriodIndex: Para trabajar con series de tiempo de periodicidad fija
                       mensual, trimestral, anual, ...
            
            e.g. pd.period_range("2020Q1", periods = 100, freq = "Q")
            
        - DatetimeIndex: Para trabajar con series de tiempo con datos calendario
                         Generalmente para series financieras
            
            e.g. pd.date_range("2020-01-01", periods=100, freq="D")
                                    
                         
E.g. para crear un objeto de series de tiempo con periodicidad fija
     para usar en python: 

# De donde provienen los datos
Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T) 

# Índice temporal
tiempo = pd.period_range(start="1900Q1", periods=T, freq="Q", name="tiempo")

# Serie de tiempo (pandas dataframe)
Y_t = pd.DataFrame(Y_t, index=tiempo, columns=variables)

"""

# Creación del dataframe con índice temporal, para trabajar series de tiempo
# en python

# Se crea un objeto de tipo PeriodIndex que permite representar periodos
# trimestrales 
tiempo_niveles = pd.period_range(
    start="1960Q1",
    periods=len(Base),
    freq="Q",
    name="tiempo",
)

# Creamos los objeto individuales de series de tiempo de tipo pandas Series 
# con las variables individuales de la base de datos
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

# Nota: Recuerde que Y debe ser un objeto tipo pandas dataframe con índice temporal 
#       para que se pueda trabajar como serie de tiempo en python
type(Y)

# Se nombran las columnas de la matriz, con las series de tiempo
Y.columns = variables

# Algunas caracteristicas de las series de tiempo del modelo VAR
print("Inicio de Y:", Y.index[0]) # Periodo donde inician las series
print("Fin de Y:", Y.index[-1]) # Periodo donde terminan las series
print(Y.head()) # Observaciones iniciales
print(Y.tail()) # Observaciones finales


# %% ===
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


# %% Pruebas ADF en niveles ----

# Nota: Para aplicar un modelo VAR en niveles, todas las variables tienen que ser
#       estacionarias, entonces se verifica que en efecto las variables sean
#       estacionarias. En caso de tener variables no estacionarias, toca tratar
#       las series, ya sea diferenciandolas o haciendo pruebas de cointegracion.

adf_ipi_nivel = adfuller(IPI, maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_ipi_nivel, "IPI en niveles") # No rechazo: La serie no es estacionaria

adf_cpi_nivel = adfuller(CPI, maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_cpi_nivel, "CPI en niveles") # No rechazo: La serie no es estacionaria

adf_unem_nivel = adfuller(UNEM, maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_unem_nivel, "Unem en niveles") # Rechazo: La serie es estacionaria


# %% Pruebas ADF sobre las variables que entran al VAR ----

# En el VAR se usan la tasa de crecimiento del IPI, la tasa de desempleo y la
# inflacion. Se verifica que estas variables sean estacionarias.

adf_dl_ipi = adfuller(Y["dl.IPI"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_dl_ipi, "dl.IPI") # Rechazo: La serie es estacionaria

adf_dl_cpi = adfuller(Y["dl.CPI"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_dl_cpi, "dl.CPI") # Rechazo: La serie es estacionaria

adf_unem = adfuller(Y["Unem"], maxlag=6, autolag="AIC", regression="c")
imprimir_adf(adf_unem, "Unem") # Rechazo: La serie es estacionaria


# %% ===
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


# %% ===
# 4.1. Identificacion ====
# ===

# Para crear el objeto VAR con statsmodels quitamos el indice temporal y
# dejamos solo los datos. El indice trimestral se conserva en Y para
# graficas y tablas.
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
    incluir_rezago_cero=False,
)

# En el ejemplo de Enders se trabaja con p = 3. A la hora de seleccionar el
# numero de rezagos, los criterios de informacion y la inspeccion de los
# residuales deben usarse conjuntamente: un VAR muy corto puede dejar
# autocorrelacion, mientras que un VAR excesivamente largo consume grados de
# libertad.
p_var = 3

# %% ===
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
#       Recuerde que la matriz de compania es la que se usa para transformar un 
#       VAR(p) a un VAR(1), y sobre sus valores propios se puede estudiar la 
#       estabilidad del modelo transformado al VAR(1)
raices_var = pd.DataFrame(
    {
        "raiz": VAR_enders.roots,
        "modulo": np.abs(VAR_enders.roots),
    }
)

print(raices_var)
print("El proceso es estable:", VAR_enders.is_stable(verbose=True))

# Nota: Note que las raíces del polinomio característico al VAR están por fuera
#       del círculo unitario y además los valores propios asociados a la matriz
#       de compañía del modelo VAR representado como un VAR(1) son menores a 1

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


# %% ===
# 4.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ----

# Portmanteau multivariado. En statsmodels usamos test_whiteness().
P_50 = VAR_enders.test_whiteness(nlags=50, adjusted=False)
print(P_50.summary()) # No rechazo

P_30 = VAR_enders.test_whiteness(nlags=30, adjusted=False)
print(P_30.summary()) # No rechazo

P_20 = VAR_enders.test_whiteness(nlags=20, adjusted=False)
print(P_20.summary()) # No rechazo

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

# Nota: Se cumple el supuesto de no autocorrelación serial en los residuales

# Homocedasticidad ----

# statsmodels no tiene un equivalente directo a arch.test() multivariado de
# vars en R. Por tanto, se construye una función que permite hacer un arch.test()
# univariado para cada uno de los residuales de la regresión, uno por cada
# variable del VAR.
arch_24 = prueba_arch_por_ecuacion(residuales_enders, lags=24, variables=variables) # Rechazo, no se cumple el supuesto
arch_12 = prueba_arch_por_ecuacion(residuales_enders, lags=12, variables=variables) # Rechazo, no se cumple el supuesto

# Nota: La decision se toma revisando los p-valores de las pruebas ARCH por
#       ecuacion. Este es un diagnostico univariado aproximado al bloque
#       multivariado usado por vars::arch.test() en R. En este caso rechazamos,
#       por lo que no se cumple el supuesto


# Normalidad ----

# H0 del Jarque-Bera multivariado: los residuales tienen distribucion normal.
normalidad_enders = VAR_enders.test_normality()
print(normalidad_enders.summary()) # Rechazo, no se cumple el supuesto

normalidad_univariada = prueba_normalidad_por_ecuacion(
    residuales_enders,
    variables=variables,
) # Rechazo, no se cumple el supuesto

# Nota: Jo se cumple el supuesto de normalidad si no se rechaza H0.


# %% ===
# 4.4. Pronostico y funciones impulso-respuesta ====
# ===

# Pronostico ----

# Especificaciones del pronostico
horizonte_pronostico = 12
int_conf_pronostico = 0.95

# Función diseñada para parecerse lo más que se pueda a predict que se usa en R
pronostico_var = predecir_var(
    VAR_enders,
    n_ahead=horizonte_pronostico,
    ci=int_conf_pronostico,
    indice=Y.index,
)
print(pronostico_var)

# Graficas pronostico
fig_pronostico, axes_pronostico = graficar_pronostico_var(
    pronostico_var["pronostico"],
    pronostico_var["inferior"],
    pronostico_var["superior"],
)
fig_pronostico.suptitle("Pronostico VAR - ejemplo de Enders", fontsize=11)
fig_pronostico.tight_layout()

mostrar_graficas()

# Version fanchart, similar a fanchart(predict(...)) en R.
fig_fanchart, axes_fanchart = graficar_fanchart_var(Y, pronostico_var)
mostrar_graficas()

# %% Pronostico por bootstrapping ----

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

# Fechas futuras para el pronóstico por bootstrap
fechas_futuras = pronostico_var["pronostico"].index

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
eje_tiempo_bootstrap = boots.index.year + (boots.index.quarter - 1) / 4

for ax, variable in zip(axes_bootstrap, variables):
    ax.plot(
        eje_tiempo_bootstrap,
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


# %% Funciones de impulso-respuesta no ortogonalizadas ----

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


# %% Funciones de impulso-respuesta ortogonalizadas ----

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


# %% ===
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

#%%
