"""
Universidad Nacional de Colombia
Facultad de Ciencias Economicas

Econometria II | Monitoria
Sesion 8: Cointegracion y metodologia Johansen

Semestre: 2026-1
"""

# ===
# Tabla de contenidos
# ===

# 1. Importacion de paquetes, rutas y funciones auxiliares
# 2. Carga y preparacion de los datos
# 3. Introduccion a metodologia Johansen
# 4. Identificacion del orden de integracion de las series
# 5. Modelo VAR en niveles
# 6. Determinacion del rango de la matriz Pi
#  6.1. Test de Johansen - Sin constante en el vector de cointegracion
#  6.2. Test de Johansen - Con constante en el vector de cointegracion
#  6.3. Estimacion del VECM(2) de acuerdo a los resultados del test de Johansen
#  6.4. Test para determinar tendencia lineal en la reparametrización como VAR usando "lttest"
# 7. Validacion de supuestos y usos del modelo
#  7.1. Reparametrizacion del VECM como un VAR en niveles
#  7.2. Validacion de supuestos de VECM como VAR
#  7.3. Pronostico del VECM reparametrizado
#  7.4. Funciones impulso-respuesta para VECM
# 8. Que pasa si se cambia el orden de las variables en el VECM?


# Nota: Tips practicos en Python
## Para limpiar el entorno en IPython/Jupyter se puede correr: "%reset -f"
## Para cerrar todas las graficas actualmente abiertas: "plt.close('all')"
## En VS Code o Spyder, los bloques marcados con "# %%" se ejecutan por celdas.


# %% ===
# 1. Importacion de paquetes, rutas y funciones auxiliares ====
# ===

# Trabajar con rutas relativas en python
from pathlib import Path

# Modulos de numpy y pandas
import numpy as np
import pandas as pd

# Modulos de statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# Importar las funciones auxiliares del script "funciones_auxiliares_VECM"
from funciones_auxiliares_VECM import (
    configurar_entorno_graficas,
    extraer_matrices_vecm,
    graficar_diagnostico_residuales_vecm,
    graficar_fanchart_vecm,
    graficar_grilla_irf,
    graficar_pronostico_vecm,
    graficar_series_vecm,
    imprimir_adf,
    imprimir_matrices_var_reparametrizado,
    imprimir_seleccion_rezagos,
    imprimir_tabla_johansen,
    mostrar_graficas,
    predecir_vecm,
    prueba_arch_por_ecuacion,
    prueba_normalidad_por_ecuacion,
)

"""
Nota: Para mirar la documentacion de cada una de las funciones, puede usar el
      comando help(<funcion>) desde la terminal interactiva de IPython. E.g. para
      ver la documentacion de la funcion "graficar_grilla_irf", use el comando
      help(graficar_grilla_irf).
"""

# Para configurar las caracteristicas de las graficas
configurar_entorno_graficas(max_columns=30)


# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raiz
BASE_DIR = Path(__file__).resolve().parents[2]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Ruta donde se encuentra la base de datos de petroleo
ruta_petroleo = DATA_DIR / "Petróleo.xlsx"


# %% ===
# 2. Carga y preparacion de los datos ====
# ===

# Ejemplo 1: Precio de referencia Brent y WTI

# Vamos a utilizar una serie del precio spot del petroleo de referencia Brent y
# una serie del precio spot del petroleo de referencia WTI. Las series tienen
# frecuencia mensual y comprenden el periodo de enero del 2000 a diciembre de
# 2020.

# Base de datos con las series de petroleo.
Data = pd.read_excel(ruta_petroleo)

# Informacion general de la base de datos.
Data.info()

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
Data = pd.read_excel(ruta_petroleo)

# Índice temporal
tiempo = pd.period_range(
    start="2000-01",
    periods=len(Data),
    freq="M",
    name="tiempo",
)

# Serie de tiempo en niveles (pandas Series)
P_Brent = pd.Series(Data["Brent"].to_numpy(), index=tiempo, name="P.Brent")
P_WTI = pd.Series(Data["WTI"].to_numpy(), index=tiempo, name="P.WTI")

"""

# Creación del dataframe con índice temporal, para trabajar series de tiempo
# en python

# Se crea un objeto de tipo PeriodIndex que permite representar periodos
# trimestrales 
tiempo = pd.period_range(
    start="2000-01",
    periods=len(Data),
    freq="M",
    name="tiempo",
)

# Creamos los objeto individuales de series de tiempo de tipo pandas Series 
# con las variables individuales de la base de datos
P_Brent = pd.Series(Data["Brent"].to_numpy(), index=tiempo, name="P.Brent")
P_WTI = pd.Series(Data["WTI"].to_numpy(), index=tiempo, name="P.WTI")

# Variables que se modelaran mediante el VECM.
variables = ["P.Brent", "P.WTI"]

# Se construye la matriz Y, que contiene las series de tiempo del modelo.
Y = pd.concat([P_Brent, P_WTI], axis=1)
Y.columns = variables

# Algunas caracteristicas de las series de tiempo del modelo.
print("Inicio de Y:", Y.index[0])
print("Fin de Y:", Y.index[-1])
print(Y.head())
print(Y.tail())


# %% Graficas de las series ----

colores_petroleo = {"P.Brent": "lightblue", "P.WTI": "coral"}
etiquetas_petroleo = {"P.Brent": "Brent", "P.WTI": "WTI"}

fig_precios, ax_precios = graficar_series_vecm(
    Y,
    variables=variables,
    colores=colores_petroleo,
    etiquetas=etiquetas_petroleo,
    titulo="Precios spot del petroleo",
    subtitulo="Petroleo Brent y WTI",
)
mostrar_graficas()


# %% ===
# 3. Introduccion a metodologia Johansen ====
# ===

# Aspectos generales de la metodologia de Johansen ----

# Consiste en un procedimiento en 4 etapas:

## Etapa 1: Verificacion preliminar de las variables a trabajar (orden de
#            integracion y graficas) e identificacion del numero de rezagos del
#            VECM mediante criterios de informacion sobre el VAR en niveles y
#            seleccionando el numero de rezagos tal que los errores sean ruido
#            blanco.
## Etapa 2: Determinacion del rango de la matriz Pi, es decir, del numero de
#            relaciones de cointegracion, y estimacion del modelo apropiado
#            dependiendo del rango de la matriz Pi.
## Etapa 3: Analisis de la matriz beta, que contiene el vector de cointegracion,
#            y de la matriz alpha, que contiene los parametros de velocidad de
#            ajuste.
## Etapa 4: Validacion de supuestos y usos del modelo (pronosticos e IRF).


# %% ===
# 4. Identificacion del orden de integracion de las series ====
# ===

# Procedemos a hacer las pruebas de raiz unitaria para identificar el orden de
# integracion de las dos series.

# En statsmodels:
# "ct" -> tendencia + constante
# "c"  -> solo constante
# "n"  -> sin constante


# %% Referencia Brent ----

adf_brent_tendencia = adfuller(
    P_Brent,
    maxlag=12,
    autolag=None,
    regression="ct",
)
imprimir_adf(adf_brent_tendencia, "P.Brent con tendencia")  # Tendencia no significativa

adf_brent_deriva = adfuller(
    P_Brent,
    maxlag=12,
    autolag=None,
    regression="c",
)
imprimir_adf(adf_brent_deriva, "P.Brent con deriva")  # Deriva no significativa

adf_brent_none = adfuller(
    P_Brent,
    maxlag=12,
    autolag=None,
    regression="n",
)
imprimir_adf(adf_brent_none, "P.Brent sin terminos deterministas")  # Serie no estacionaria


# %% Referencia WTI ----

adf_wti_tendencia = adfuller(
    P_WTI,
    maxlag=12,
    autolag=None,
    regression="ct",
)
imprimir_adf(adf_wti_tendencia, "P.WTI con tendencia")  # Tendencia no significativa

adf_wti_deriva = adfuller(
    P_WTI,
    maxlag=12,
    autolag=None,
    regression="c",
)
imprimir_adf(adf_wti_deriva, "P.WTI con deriva")  # Deriva no significativa

adf_wti_none = adfuller(
    P_WTI,
    maxlag=12,
    autolag=None,
    regression="n",
)
imprimir_adf(adf_wti_none, "P.WTI sin terminos deterministas")  # Serie no estacionaria


# %% Aplicamos diferenciacion ----

d_P_Brent = P_Brent.diff().dropna()
d_P_WTI = P_WTI.diff().dropna()

adf_d_brent = adfuller(
    d_P_Brent,
    maxlag=12,
    autolag=None,
    regression="n",
)

# La diferenciación de P.Brent es I(0), por lo que P.Brent en niveles es I(1)
imprimir_adf(adf_d_brent, "Diferencia de P.Brent")  

adf_d_wti = adfuller(
    d_P_WTI,
    maxlag=12,
    autolag=None,
    regression="n",
)

# La diferenciación de P.WTI es I(0), por lo que P.WTI en niveles es I(1)
imprimir_adf(adf_d_wti, "Diferencia de P.WTI")  # P.WTI en niveles es I(1)


# %% ===
# 5. Modelo VAR en niveles ====
# ===

# Posteriormente, estimaremos un VAR en niveles para determinar el numero de
# rezagos del VECM.

# Nota: Se analizaran los criterios de informacion sobre el VAR en niveles.

# Para estimar con statsmodels se conserva una matriz sin indice temporal
# especial. El indice mensual se conserva en Y para graficas y tablas.
Y_modelo = Y.reset_index(drop=True)
modelo_petroleo = VAR(Y_modelo)

# Seleccion de rezagos para un VAR con tendencia e intercepto.
seleccion_rezagos_both = modelo_petroleo.select_order(maxlags=6, trend="ct")
imprimir_seleccion_rezagos(
    seleccion_rezagos_both,
    "Seleccion de rezagos para un VAR con tendencia e intercepto",
)

# Se trabaja con p = 3 porque AIC/FPE sugieren este rezago.
p_var = 3

VAR3_both = modelo_petroleo.fit(p_var, trend="ct")
print(VAR3_both.summary())  # Tendencia no significativa

# Seleccion de rezagos para un VAR con solo intercepto.
seleccion_rezagos_const = modelo_petroleo.select_order(maxlags=6, trend="c")
imprimir_seleccion_rezagos(
    seleccion_rezagos_const,
    "Seleccion de rezagos para un VAR con solo intercepto",
)

VAR3_const = modelo_petroleo.fit(p_var, trend="c")
print(VAR3_const.summary())  # Intercepto significativo

# Seleccion de rezagos para un VAR sin terminos deterministicos.
seleccion_rezagos_none = modelo_petroleo.select_order(maxlags=6, trend="n")
imprimir_seleccion_rezagos(
    seleccion_rezagos_none,
    "Seleccion de rezagos para un VAR sin terminos deterministicos",
    incluir_rezago_cero=False,
)

VAR3_none = modelo_petroleo.fit(p_var, trend="n")
print(VAR3_none.summary())

# Dado que al estimar el VAR con constante, el intercepto en este modelo
# resulto significativo, decidimos estimar un VAR con constante.

# Elegimos VAR(3) en niveles.
VAR3 = VAR3_const

# Note que como se estimo un VAR(3), su reparametrizacion como un VECM sera un
# VECM(2). Ademas, dicha reparametrizacion siempre se podra hacer
# independientemente de si las variables del VAR son I(0) o I(1).

# Nota: Los residuales del modelo VAR en niveles deben ser ruido blanco,
#       independientemente de si las variables del VAR en niveles son I(0) o
#       son I(1). Si se escogio el numero adecuado de rezagos en el VAR, siempre
#       se podra garantizar que esos residuales seran ruido blanco.

# Nota: Tambien recuerde que en teoria, los errores del VAR en niveles deben ser
#       los mismos errores que los del VECM. Nota de la nota: recuerde que los
#       errores son teoricos y los residuales son una aproximacion a los errores,
#       pero no son los errores.


# %% Validacion preliminar de los residuales del VAR ----

# Vamos a analizar el comportamiento de los residuales. Dado que es una serie
# mensual, analicemos su comportamiento en puntos criticos.

# No autocorrelacion serial ===

P_12 = VAR3.test_whiteness(nlags=12, adjusted=False)
print(P_12.summary())  # No rechazo, se cumple el supuesto

P_16 = VAR3.test_whiteness(nlags=16, adjusted=False)
print(P_16.summary())  # No rechazo, se cumple el supuesto

P_20 = VAR3.test_whiteness(nlags=20, adjusted=False)
print(P_20.summary())  # No rechazo, se cumple el supuesto

# Validacion grafica de otros supuestos ===

# Graficamos los residuales, ACF/PACF y ACF/PACF de residuales al cuadrado.
residuales_var = pd.DataFrame(
    np.asarray(VAR3.resid),
    index=Y.index[VAR3.k_ar :],
    columns=variables,
)

figuras_residuales_var = graficar_diagnostico_residuales_vecm(
    residuales_var,
    lags=12,
    incluir_cuadrados=True,
)
mostrar_graficas()

# Nota: Lo mas importante para seguir con el procedimiento de la metodologia de
#       Johansen, es que los residuales no tengan correlacion serial. Puede que
#       no sean exactamente ruido blanco, si e.g. tienen heterocedasticidad, pero
#       lo fundamental es que los residuales no tengan correlacion serial, ese es 
#       el supuesto clave a validar.


# %% ===
# 6. Determinacion del rango de la matriz Pi ====
# ===

"""
Nota: Argumentos de la función coint_johansen de statsmodels para el test de Johansen

La funcion coint_johansen de statsmodels permite realizar el test de Johansen
en Python.

Para revisar todos los argumentos del test de Johansen se puede usar:
help(coint_johansen)

Nota: Existen 3 versiones diferentes del test de cointegración de Johansen 
            En R:               Equivalencia en python   
    1) ecdet = "none"     ->        det_order = -1 (Relación de cointegración sin constante)
    2) ecdet = "const"    ->        det_order = 0 (Relación de cointegración con constante )
    3) ecdet = "trend"    ->        det_order = 1 (Relación de cointegración con tendencia lineal)

E.g. Asi se verían la relación de cointegración del presente ejemplo en cada uno de los casos
     mencionados arriba: 
    1) det_order = -1 (Relación de cointegración sin constante)
        E.g. P.Brent - beta * P.WTI = 0  
             P.Brent = beta * P.WTI
        
    2) det_order = 0 (Relación de cointegración con constante )
        E.g. P.Brent - beta * P.WTI + c = 0 
             P.Brent = beta * P.WTI - c
         
    3) det_order = 1 (Relación de cointegración con tendencia lineal)
        E.g. P.Brent - beta * P.WTI + c + delta * t = 0 
             P.Brent = beta * P.WTI - c - delta * t
             
Donde "beta" es el coeficiente de cointegración que aparece en el vector de cointegración 


# Nota: El argumento k_ar_diff de coint_johansen para el test de Johansen en python, determinar 
#       el orden del VECM, que es un orden menos que el VAR en niveles que se estimó 
#       previamente. E.g., en nuestro caso particular, estimamos un VAR(3), entonces 
#       k_ar_diff = 2 cuando se usa el comando "coint_johansen"       

# I.e. en statsmodels se usa k_ar_diff = p_var - 1.
"""
# En statsmodels se usa k_ar_diff = p - 1.
k_ar_diff = p_var - 1

# Nota: La especificación del modelo VECM que se usa en el comando "coint_johansen"
#       asociado al test de Johansen en statsmodels es la misma que se trabaja teóricamente
#       en el curso, que en el caso del camando ca.jo que se usa en R, equivaldría a la 
#       especificación spec = "transitory".

"""
Nota: El test de cointegración de Johansen en R y python no dan exactamente igual, porque las
      especificaciones cambian entre las dos implementaciones en los dos lenguajes. Por tanto, 
      el comando coint_johansen de statsmodels en python y ca.jo de urca en R, pueden dar 
      resultados diferentes, como lo que ocurre en éste script. Ésto se debe a que los valores 
      críticos y especificaciones que se usan son distintas. 
      
      En general, trabajaremos con la especificación del test " ecdet = "const" " en R y su 
      equivalente " det_order = 0 " en python,. que son 1) la versión del test que más se 
      trabaja en la práctica y 2) son las dos especificaciones entre ambos test más parecidas
      y que dan los resultados y conclusiones más cercanas.
      
      Importante que tengan éste detalle en cuenta! Las especificaciones en los test de
      cointegración de Johansen de R y python no son exactamente iguales. 
"""

# %% ===
# 6.1. Test de Johansen - Sin constante en el vector de cointegracion ====
# ===

# Nota: Estamos en el caso det_order = -1, que corresponde a una relacion de
#       cointegracion sin constante:
#       P.Brent - beta * P.WTI = 0, es decir, P.Brent = beta * P.WTI.

# Nota: Hay dos maneras de hacer el test de Johansen:
#       1. Criterio del valor propio maximo.
#       2. Criterio de la traza.

# Nota: En python, solo debo crear un objeto tipo "statsmodels.tsa.vector_ar.vecm.JohansenTestResult"
#       y ya el almancena los resultados tanto de la prueba del valor propio máximo, 
#       como la prueba de la traza

# Criterio del valor propio maximo ----

# Generalmente es la prueba preferida y la mas robusta.

# Dado que solo hay dos variables, se realiza el siguiente procedimiento
# secuencial:

# H0: r = 0 vs H1: r = 1,
# luego H0: r = 1 vs H1: r = 2.
# Aqui p = 2 variables y k_ar_diff = 2 rezagos, pues se estimo un VECM(2).
johansen_test_none = coint_johansen(
    Y_modelo,
    det_order=-1,
    k_ar_diff=k_ar_diff,
)

# Note que primero no se rechaza la hipótesis nula r = 0, por lo que en éste caso el test
# sugeriría que no hay relación de cointegración. Ésto difiere del resultado de ca.jo de
# urca donde si se rechaza y se sugiere una relación de cointegración. 
# En éste caso los resultados del test de cointegración entre R y python difieren
eigen_table_none = imprimir_tabla_johansen(
    johansen_test_none,
    tipo="eigen",
    titulo="Johansen sin constante - criterio del valor propio maximo",
)

# Nota: Note que el orden del VECM no determina cuantas pruebas secuenciales se
#       deben realizar en el test de Johansen. Eso solo esta determinado por el
#       numero de variables que tengo en el VECM.


# Criterio de la traza ----

# Al tener el VECM solo dos variables, el procedimiento secuencial a realizar es:

# H0: r = 0 vs H1: r = 1,
# luego H0: r = 1 vs H1: r = 2.
# Aqui p = 2 variables y k_ar_diff = 2 rezagos, pues se estimo un VECM(2).

# Note que primero no se rechaza la hipótesis nula r = 0, por lo que en éste caso el test
# sugeriría que no hay relación de cointegración. Ésto difiere del resultado de ca.jo de
# urca donde si se rechaza y se sugiere una relación de cointegración. 
# En éste caso los resultados del test de cointegración entre R y python difieren
trace_table_none = imprimir_tabla_johansen(
    johansen_test_none,
    tipo="trace",
    titulo="Johansen sin constante - criterio de la traza",
)


# %% ===
# 6.2. Test de Johansen - Con constante en el vector de cointegracion ====
# ===

# Nota: Estamos en el caso det_order = 0, que corresponde a una relacion de
#       cointegracion con constante:
#       P.Brent - beta * P.WTI + c = 0, es decir,
#       P.Brent = beta * P.WTI - c.

# ===
# Nota: Este es el caso mas comun, entonces por lo general trabajaremos con
#       det_order = 0 en coint_johansen y con deterministic = "ci" en VECM; es
#       decir, diremos que la relacion de cointegracion incluye una constante!
# ===

# Criterio del valor propio maximo ----

johansen_test_const = coint_johansen(
    Y_modelo,
    det_order=0,
    k_ar_diff=k_ar_diff,
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. (No se rechaza a un nivel
#  de significancia del 1 % y es muy cercano el estadístico al valor crítico
# asociado al 5%). Recuerde que un test estadístico no se tiene analizar 
# de manera estricta, porque hay componentes aleatorios que pueden afectar
# el valor del estadístico, por lo que el análisis de arriba es adecuado.
eigen_table_const = imprimir_tabla_johansen(
    johansen_test_const,
    tipo="eigen",
    titulo="Johansen con constante - criterio del valor propio maximo",
)

# Nota: El comando ca.jo del paquete urca del test de Johansen si rechaza al 5 %
#       Razón que también justifica el análisis hecho arriba

# Criterio de la traza ----

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. (No se rechaza a un nivel
#  de significancia del 1 % y es muy cercano el estadístico al valor crítico
# asociado al 5%). Recuerde que un test estadístico no se tiene analizar 
# de manera estricta, porque hay componentes aleatorios que pueden afectar
# el valor del estadístico, por lo que el análisis de arriba es adecuado.
trace_table_const = imprimir_tabla_johansen(
    johansen_test_const,
    tipo="trace",
    titulo="Johansen con constante - criterio de la traza",
)

# Nota: El comando ca.jo del paquete urca del test de Johansen si rechaza al 5 %
#       Razón que también justifica el análisis hecho arriba

"""
Nota: Nos quedaremos con el resultado del test de Johansen que dice que existe una relación de
      cointegración, es decir con los resultados del test de cointegración cuando 
      " det_order = 0 "
"""


# %% ===
# 6.3. Estimacion del VECM(2) de acuerdo a los resultados del test de Johansen ====
# ===

# Especificaciones deterministicas en la estimación del modelo VEC
# empleando el comando VECM de statsmodels:
# "n"    - no deterministic terms
# "co"   - constant outside the cointegration relation
# "ci"   - constant within the cointegration relation
# "lo"   - linear trend outside the cointegration relation
# "li"   - linear trend within the cointegration relation


# %% Sin intercepto en la relación de cointegración ----

# En statsmodels la clase VECM permite estimar directamente el modelo VEC.
# Usamos coint_rank=1 para indicar que hay una relacion de cointegracion.

# Creación del objeto de tipo "statsmodels.tsa.vector_ar.vecm.VECM"
VEC_none = VECM(
    Y_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="co",
)

# Estimación del modelo VEC
VEC_none_fit = VEC_none.fit()
print(VEC_none_fit.summary())

# Con esta funcion auxiliar obtenemos el vector de cointegracion normalizado y
# los coeficientes de velocidad de ajuste.
matrices_vecm_none = extraer_matrices_vecm(VEC_none_fit, variables=variables)

print("\nVector de cointegracion normalizado (beta):")
print(matrices_vecm_none["beta"])

print("\nVelocidades de ajuste (alpha):")
print(matrices_vecm_none["alpha"])


# %% Con intercepto en la relación de cointegración ----

# Estimamos ahora el VECM con constante en la relacion de cointegracion.

# Creación del objeto de tipo "statsmodels.tsa.vector_ar.vecm.VECM"
VEC_const = VECM(
    Y_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="ci",
)

# Estimación del modelo VEC
VEC_const_fit = VEC_const.fit()
print(VEC_const_fit.summary())

# Vector de cointegracion (beta), constante de cointegracion y velocidad de
# ajuste (alpha).
matrices_vecm_const = extraer_matrices_vecm(VEC_const_fit, variables=variables)

print("\nVector de cointegracion normalizado (beta):")
print(matrices_vecm_const["beta"])

print("\nConstante de cointegracion:")
print(matrices_vecm_const["constante_cointegracion"])

print("\nVelocidades de ajuste (alpha):")
print(matrices_vecm_const["alpha"])


# %% ===
# 6.4. Test para determinar tendencia lineal en la reparametrización como VAR usando "lttest" ====
# ===

# La funcion lttest del paquete urca permite determinar en R la existencia de
# una tendencia lineal deterministica en el VAR en niveles asociado a la
# reparametrizacion del VECM.

# En Python no hay un equivalente directo de lttest en statsmodels. Por tanto,
# aqui se estima una especificacion con tendencia lineal en el VECM (reparametrización del VAR)
# como revision practica.

# En terminos de la documentacion de lttest, la hipotesis seria:

# H0: No existencia de tendencia lineal en el VAR en niveles asociado a la
#     reparametrizacion del VECM.
# H1: Existencia de tendencia lineal en el VAR en niveles asociado a la
#     reparametrizacion del VECM.

VEC_tendencia = VECM(
    Y_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="cilo",
)
VEC_tendencia_fit = VEC_tendencia.fit()
print(VEC_tendencia_fit.summary())

# No se encuentra evidencia practica para incluir tendencia lineal, por lo que
# no se incluye tendencia lineal en el VAR en niveles asociado a la
# reparametrizacion del VECM.


# %% ===
# 7. Validacion de supuestos y usos del modelo ====
# ===


# %% ===
# 7.1. Reparametrizacion del VECM como un VAR en niveles ====
# ===

# Nota: Dados los resultados anteriores, se usara el modelo VEC con constante en
# el vector de cointegracion.

# Nota: Luego de estimar el modelo VECM, en R se puede reparametrizar el modelo
#       como un VAR en niveles usando la funcion vec2var del paquete vars.
#       En Python, statsmodels guarda esta representacion directamente dentro
#       del resultado estimado.

# statsmodels guarda la reparametrizacion del VECM como VAR en niveles en
# var_rep. Cada elemento corresponde a una matriz A_i del VAR(p).

# Matrices de la reparametrización del VECM como un VAR en niveles
matrices_var_reparametrizado = imprimir_matrices_var_reparametrizado(
    VEC_const_fit,
    variables=variables,
)

# Esto es importante dado que se necesita el modelo VEC reparametrizado como un
# VAR en niveles para poder validar los supuestos y hacer uso del modelo.


# %% ===
# 7.2. Validacion de supuestos de VECM como VAR ====
# ===

# Autocorrelacion ----

# En statsmodels, test_whiteness aplica una prueba Portmanteau. La version
# asintotica se obtiene con adjusted=False, mientras que adjusted=True aplica
# una correccion para muestra pequena cuando esta disponible.

P_12_V = VEC_const_fit.test_whiteness(nlags=12)
print(P_12_V.summary())  # No rechaza H0

P_24_V = VEC_const_fit.test_whiteness(nlags=24)
print(P_24_V.summary())  # No rechaza H0

P_36_V = VEC_const_fit.test_whiteness(nlags=36)
print(P_36_V.summary())  # No rechaza H0

# Nota: Se cumple el supuesto de no correlacion serial en los residuales.


# Homocedasticidad ----

# statsmodels no tiene un equivalente directo a arch.test() multivariado de vars
# en R. Por tanto, se construye una funcion que permite hacer un arch.test()
# univariado para cada uno de los residuales de la regresion, uno por cada
# variable del VECM.
residuales_vecm = pd.DataFrame(
    np.asarray(VEC_const_fit.resid),
    index=Y.index[VEC_const_fit.k_ar :],
    columns=variables,
)

arch_vecm_24 = prueba_arch_por_ecuacion(
    residuales_vecm,
    lags=24,
    variables=variables,
)
arch_vecm_12 = prueba_arch_por_ecuacion(
    residuales_vecm,
    lags=12,
    variables=variables,
)

# Nota: No se cumple el supuesto de homocedasticidad en los residuales.


# Normalidad ----

# H0 del Jarque-Bera multivariado: los residuales tienen distribucion normal.
normalidad_vecm = VEC_const_fit.test_normality()
print(normalidad_vecm.summary())

normalidad_univariada_vecm = prueba_normalidad_por_ecuacion(
    residuales_vecm,
    variables=variables,
)

# Nota: No se cumple el supuesto de normalidad en los residuales.

# Nota: Se cumple el supuesto mas importante, que es el de no correlacion serial
#       en los residuales del modelo.

# Nota: Como se violan los supuestos de heterocedasticidad y normalidad, hay que
#       calcular los intervalos de confianza mediante bootstrapping para poder
#       hacer inferencia estadistica correcta, tanto en los pronosticos como en
#       las OIRF.


# %% ===
# 7.3. Pronostico del VECM reparametrizado ====
# ===

# Recuerden que debido al incumplimiento de normalidad, los intervalos de
# confianza deben computarse por bootstrapping cuando se quiera hacer inferencia
# estadistica mas robusta.

# Especificaciones del pronostico.
horizonte_pronostico = 12
int_conf_pronostico = 0.95

# Pronostico del modelo VEC.
pronostico_VECM = predecir_vecm(
    VEC_const_fit,
    n_ahead=horizonte_pronostico,
    ci=int_conf_pronostico,
    indice=Y.index,
    variables=variables,
)
print(pronostico_VECM)

# Grafica del pronostico del modelo VEC.
fig_pronostico, axes_pronostico = graficar_pronostico_vecm(
    pronostico_VECM["pronostico"],
    pronostico_VECM["inferior"],
    pronostico_VECM["superior"],
)
fig_pronostico.suptitle("Pronostico VECM", fontsize=11)
fig_pronostico.tight_layout()
mostrar_graficas()

# Version fanchart, similar a fanchart(predict(...)) en R.
fig_fanchart, axes_fanchart = graficar_fanchart_vecm(Y, pronostico_VECM)
mostrar_graficas()


# %% ===
# 7.4. Funciones impulso-respuesta para VECM ====
# ===

# Parametros de las graficas de las IRFs.
pasos_adelante = np.arange(0, 19)
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100 # Bootstrappings empleados para construir los IC de las IRFs

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez y luego
# crea cada panel usando funciones auxiliares.

# IRF de las variables del sistema ante distintos choques exogenos.
irf_ortog_vecm = graficar_grilla_irf(
    VEC_const_fit,
    variables,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de OIRF: columnas = impulsos; filas = respuestas.
print(irf_ortog_vecm["objeto_irf"].orth_irfs)
mostrar_graficas()

"""
Nota: Interpretación de las OIRFs

    Dado que por el orden de las variables que escogimos, primero Brent y
    luego WTI, la variable mas exogena es Brent y la mas endogena es WTI.
    En este orden, luego de hacer la descomposicion de Cholesky y trabajar
    con IRFs ortogonalizadas, un choque estructural en el precio del Brent
    afecta tanto al precio del Brent como al precio del WTI, mientras que un
    choque estructural en el precio del WTI no tiene ningun efecto en el
    tiempo sobre el precio del Brent y su efecto sobre si mismo se disipa
    hacia cero en el largo plazo. Note que se llegan a estas conclusiones
    por el orden escogido de las variables, donde se asume que la variable
    mas exogena es el precio del Brent y la mas endogena es el precio del
    WTI.
"""


# %% ===
# 8. Que pasa si se cambia el orden de las variables en el VECM? ====
# ===

"""
Nota: Importancia del orden de las variables a la hora de hacer al descomposición de Cholesky
      e interpretar los choques estructurales de las IRFs ortogonalizadas
      
    Recuerde que el orden de las variables que se usa para construir la
    matriz de series de tiempo Y importa, dado que la primera columna de la
    matriz esta asociada a la variable mas exogena, mientras que la ultima
    columna esta asociada a la variable mas endogena. Esto es importante a
    la hora de realizar la descomposicion de Cholesky, dado que la
    descomposicion de Cholesky tiene en cuenta ese orden, y por ende las
    funciones impulso-respuesta ortogonalizadas dependen fundamentalmente del
    orden que se escoja para las variables.      
"""

# En esta seccion veremos que pasa si se cambia el orden de las columnas de la
# matriz de series de tiempo. En este caso, la llamaremos Y_alt.

variables_alt = ["P.WTI", "P.Brent"]
Y_alt = Y[variables_alt]

# Pandas Dataframe con el que realizaremos la metodología de Johansen
Y_alt_modelo = Y_alt.reset_index(drop=True)

# Se realizara de nuevo la metodologia de Johansen completa que se realizo
# previamente, pero esta vez con las variables intercambiadas.

# Se estima un VAR(3), pero con las variables intercambiadas de orden.
modelo_petroleo_alt = VAR(Y_alt_modelo)
VAR3_alt = modelo_petroleo_alt.fit(p_var, trend="c")
print(VAR3_alt.summary())

# Test de Ljung-Box / Portmanteau

# No autocorrelacion serial ===

P_12_alt = VAR3_alt.test_whiteness(nlags=12, adjusted=False)
print(P_12_alt.summary())  # No rechazo

P_16_alt = VAR3_alt.test_whiteness(nlags=16, adjusted=False)
print(P_16_alt.summary())  # No rechazo

P_20_alt = VAR3_alt.test_whiteness(nlags=20, adjusted=False)
print(P_20_alt.summary())  # No rechazo

# Nota: El supuesto de no correlacion serial en los residuales, el mas
#       importante, se sigue cumpliendo!


# %% Prueba de Johansen con orden alternativo (comando coint_johansen) ----

# Se usa de nuevo coint_johansen para realizar la prueba de Johansen y
# determinar el rango de la matriz Pi.

# Para ello usaremos las siguientes especificaciones del test:
#  - tipo = "eigen": Criterio del valor propio maximo.
#  - det_order = 0: Constante en el vector de cointegracion.

# Al tener el VECM solo dos variables, el procedimiento secuencial a realizar es:

# H0: r = 0 vs H1: r = 1,
# luego H0: r = 1 vs H1: r = 2.
# Aqui p = 2 variables y k_ar_diff = 2 rezagos, pues se estimo un VECM(2).

# Criterio del valor propio maximo y constante.
johansen_test_const_alt = coint_johansen(
    Y_alt_modelo,
    det_order=0,
    k_ar_diff=k_ar_diff,
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. (No se rechaza a un nivel
#  de significancia del 1 % y es muy cercano el estadístico al valor crítico
# asociado al 5%). Recuerde que un test estadístico no se tiene analizar 
# de manera estricta, porque hay componentes aleatorios que pueden afectar
# el valor del estadístico, por lo que el análisis de arriba es adecuado.
eigen_table_const_alt = imprimir_tabla_johansen(
    johansen_test_const_alt,
    tipo="eigen",
    titulo="Johansen con constante y orden alternativo",
)  # Se mantiene la conclusion

# %% Estimacion del modelo VEC con orden alternativo (comando VECM) ----

# La clase VECM permite estimar el modelo VEC en Python.

# Creación del objeto de tipo "statsmodels.tsa.vector_ar.vecm.VECM"
VEC_const_alt = VECM(
    Y_alt_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="ci",
)

# Estimacíón del modelo VEC
VEC_const_alt_fit = VEC_const_alt.fit()
print(VEC_const_alt_fit.summary())

# Con esta funcion auxiliar obtenemos el vector de cointegracion normalizado.
matrices_vecm_const_alt = extraer_matrices_vecm(
    VEC_const_alt_fit,
    variables=variables_alt,
)

print("\nVector de cointegracion normalizado (beta), orden alternativo:")
print(matrices_vecm_const_alt["beta"])

print("\nConstante de cointegracion:")
print(matrices_vecm_const_alt["constante_cointegracion"])

# Con esta funcion auxiliar obtenemos los coeficientes de velocidad de ajuste.
print("\nVelocidades de ajuste (alpha), orden alternativo:")
print(matrices_vecm_const_alt["alpha"])


# %% Reparametrizacion del modelo VEC en un modelo VAR ----

# En Python no se crea un objeto tipo vec2var como en R; la reparametrizacion en
# niveles queda disponible en VEC_const_alt_fit.var_rep y los diagnosticos se
# aplican directamente sobre el resultado del VECM.

P_12_V_alt = VEC_const_alt_fit.test_whiteness(nlags=12)
print(P_12_V_alt.summary())

P_24_V_alt = VEC_const_alt_fit.test_whiteness(nlags=24)
print(P_24_V_alt.summary())

P_36_V_alt = VEC_const_alt_fit.test_whiteness(nlags=36)
print(P_36_V_alt.summary())

# Nota: Se cumple el supuesto de no correlacion serial en los residuales, en el
#       VAR reparametrizado, que es el supuesto mas importante.


# %% Funciones impulso-respuesta ortogonalizadas con orden alternativo ----

# Veamos que ocurre con las OIRF al cambiar el orden de las variables del modelo
# VEC.

irf_ortog_vecm_alt = graficar_grilla_irf(
    VEC_const_alt_fit,
    variables_alt,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de OIRF: columnas = impulsos; filas = respuestas.
mostrar_graficas()

"""
Nota: Interpretación de las IRF ortogonalizadas con el cambio en el orden de las variables 
      del modelo

    Las OIRF cambian sustancialmente. Note que, si asume que P.WTI ahora es la
    variable más exógena, entonces ahora, a diferencia de lo que ocurria anteriormente,
    los choques estructurales del P.WTI ahora sí son significativos y persistentes, y además
    los choques estructurales del P.Brent continuan siendo significativos y persistentes.
    Note que solo con cambiar el orden de las variables, el choque estructural del P.WTI
    paso de no ser casi significativo y disiparse, ahora a ser significativo y persistente
    por lo que se concluye que el orden en que se escogan las variables en el modelo VEC
    , al igual que lo que pasa en el modelo VAR, no es trivial, es fundamental saber escoger
    dicho orden de exógenidad por teoría económica o test estadísticos, por que dada la
    lógica de la descomposición de Cholesky, al escoger un orden diferente de las variables 
    del modelo, se pueden llegar a conclusiones muy distintas!
    Bienvenido a la Economía, donde todo es posible :D !!!!!
"""

"""
Nota: Limitaciones de la identificación de choques estructurales por descomposición de Cholesky
    
    Lo anterior muestra algunas de las limitaciones de la descomposición de Cholesky, como
    estrategía de identificación de choques estructurales. Claramente, el orden en que 
    se escogan las variables afecta la interpretación económico y sobre todo los restultados
    de política, pero en muchos casos el orden de exogenidad entre las variables o 1) no es 
    claro o 2) simplemente no existe. Para superar ese problema en la identificación de 
    choques estructurales usando descomposición de Cholesky, existen otras estrategias de
    identificación como lo puede ser un S-VECM (Structural VECM), donde a partir de una matriz
    S se pueden imponer restricciones más sensibles para la identificación de choque 
    estructurales, en la práctica, se usa mucho más este tipo de identificación que usar
    descomposición de Chokesky a lo maldita sea xD
"""

# %%
