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
#    1.2.1 Construccion de los errores " u_t " usando normal multivariada
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


# %% Importación de paquetes ============================

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Módulos de statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Importación de las funciones que usaremos del script auxiliar "funciones_auxiliares_graficacion_VAR"
from funciones_auxiliares_graficacion_VAR import (
    configurar_entorno_graficas,
    graficar_diagnostico_errores,
    graficar_diagnostico_residuales_var,
    graficar_fanchart_var,
    graficar_fevd_var,
    graficar_grilla_irf,
    graficar_pronostico_var,
    graficar_ts,
    imprimir_adf,
    mostrar_graficas,
    predecir_var,
    prueba_arch_por_ecuacion,
    prueba_normalidad_por_ecuacion,
)

'''
Nota: Parar mirar la documentación de cada una de las funciones, puede usar el comando help(<funcion>)
      desde la terminal interactiva de ipython. E.g. para ver la documentación de la función 
      "graficar_grilla_irf", use el comando help(graficar_grilla_irf)
'''

# Para configurar las características de las gráficas
configurar_entorno_graficas(max_columns=20)


# %% =========================
# 1. Simulacion de un proceso VAR(1) con 3 variables 
# ============================

# 1.1 Especificacion de las condiciones de la simulacion ----

# Fijamos la semilla para que siempre de el mismo resultado
semilla_simulacion = 82901

# Se crea un generador de números aleatorios de Numpy
generador = np.random.default_rng(semilla_simulacion)

"""
Nota: generador, es básicamente el objeto que se usa
      para simular valores aleatorios en Numpy
      
Note algo muy importante: 

Mientras que en R requiero de varias funciones distintas para simular de
distribuciones diferentes, e.g.:

n = 100

rnorm(n)
rt(n, df = 5)
rbinom(n, size = 1, prob = 0.5)
runif(n)

En python un solo objeto generador de tipo "numpy.random._generator.Generator"
permite simular cualquier distribución. Es decir, en python, en lugar de tener 
funciones diferentes para distribuciones diferentes, se crea un solo objeto
generador de números aleatorios que funciona para culquier distribución. E.g.: 

# Objeto "generador" instanciado, que permite generar diferentes números aleatorios
generador = np.random.default_rng(<semilla_simulacion>)

# Métodos del objeto "generador" para generar números aleatorios de dierentes distribuciones 
generador.normal(loc=0, scale=1, size=100)
generador.standard_t(df=5, size=100)
generador.binomial(n=1, p=0.5, size=100)
generador.uniform(low=0, high=1, size=100)
generador.multivariate_normal(mean=media_u, cov=Sigma_u, size=100)

Note que en python si es posible simular en numpy una distribución normal multivariada, mientras
que en R se requiere de paquetes como MASS y mvtnorm y funciones como MASS::mvrnorm() y 
mvtnorm::rmvnorm()
"""


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

"""
Nota: Recuerde que en numpy, las matrices son "numpy.ndarray"
"""

# Para saber el tipo de objeto que es la matriz Y_t
type(Y_t)

# Para saber las dimensiones de la matriz Y_t
Y_t.shape # La matriz tiene 5000 observaciones y 3 variables

# Para imprimir las primeras 5 filas de la matriz Y_t, se transforma primero el numpy.ndarray
# a un pandas.core.frame.DataFrame y luego se usa el método head() del objeto DataFrame para 
# ver las 5 primeras filas
print(pd.DataFrame(Y_t, columns=variables).head())


# %% 1.2 Simulacion de los errores en forma reducida " u_t " ----

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
#   Sigma_u = P_chol * P_chol^{'} ; Donde P_chol es la matriz de la descomposicion de Cholesky
#
# Esta construccion es coherente con una identificacion recursiva tipo Cholesky:
# y_1 es contemporaneamente mas exogena que y_2 y y_3, mientras que y_2 es mas
# exogena que y_3. Los errores reducidos u_t estaran correlacionados, pero los
# errores estructurales e_t que los generan son ortogonales.

# Acá en la simulación partimos al revés de la metodología de Box Jenkins. 
# Inicialmente, definimos la matriz de la descomposición de Cholesky "P_chol" 
# porque ella cumple dos roles importantes: 
  # 1. Determina el orden de exogenidad de las variables: y1, y2 y y3. 
  # 2. Determina la estructura de correlación de la matriz de varianzas y covarianzas
  #    de los errores (i.e. de la matriz Sigma_u_teorica, definida abajo.)
  
# Acá en la simulación partimos al revés de la metodología de Box Jenkins. 
# Inicialmente, definimos la matriz de la descomposición de Cholesky "P_chol" 
# porque ella cumple dos roles importantes: 
  # 1. Determina el orden de exogenidad de las variables: y1, y2 y y3. 
  # 2. Determina la estructura de correlación de la matriz de varianzas y covarianzas
  #    de los errores (i.e. de la matriz Sigma_u_teorica, definida abajo.)

# 1.2.1 Construccion de los errores " u_t " usando normal multivariada ----

# Construcción manual de la matriz P de la descomposición de Cholesky
P_chol = np.array(
    [
        [0.70, 0.00, 0.00],
        [0.35, 1.10, 0.00],
        [0.25, 0.55, 1.60],
    ]
)

# Para imprimir un numpy.ndarray se pasa primero a un pandas dataframe
P_chol_df = pd.DataFrame(P_chol, index=errores, columns=errores)
print(P_chol_df)

# Matriz de varianzas-covarianzas teorica de la distribucion normal multivariada
Sigma_u_teorica = P_chol @ P_chol.T # @ es para realizar multiplicación matrical entre dos numpy.ndarray

# Nota: Hacer P_chol @ P_chol.T es equivalente a hacer np.matmul(P_chol, P_chol.T)

# Para imprimir un numpy.ndarray se pasa primero a un pandas dataframe
Sigma_u_teorica_df = pd.DataFrame(Sigma_u_teorica, index=errores, columns=errores)
print(Sigma_u_teorica_df)

# Matriz de correlaciones teorica

# Para encontrar la matriz de correlaciones teóricas: 
# 1. np.diag(): Extrae los elementos diagonales de la matriz
#               El resultado es un vector 
# 2. np.sqrt(): Aplica raices cuadradas para trabajar con desviaciones estándar
# 3. np.outer(): Producto externo entre dos vectores que resulta en una matriz 
# 4. A/B: División elemento por elemento entre dos numpy.ndarray

desv_u_teoricas = np.sqrt(np.diag(Sigma_u_teorica))
cor_u_teorica = Sigma_u_teorica / np.outer(desv_u_teoricas, desv_u_teoricas)

# Para imprimir un numpy.ndarray se pasa primero a un pandas dataframe
cor_u_teorica_df = pd.DataFrame(cor_u_teorica, index=errores, columns=errores)
print(cor_u_teorica_df)

# Desviaciones estandar teoricas de los errores en forma reducida

# Transformo el vector de desviaciones estándar (un numpy.ndarray) a un 
# pandas.core.series.Series y lo imprime
desv_u_teoricas_s = pd.Series(desv_u_teoricas, index=errores, name="desv_teorica")
print(desv_u_teoricas_s)

# La media de los errores en forma reducida sera el vector de ceros
media_u = np.zeros(len(errores))

# Errores en forma reducida " u_t " simulados directamente de una normal
# trivariada. Esto es equivalente a usar mvtnorm::rmvnorm() en R:
#
#   u_t ~ N_3(media_u, Sigma_u_teorica)
#

# Nota: Se está utilizando el método "multivariate_normal" del objeto 
#       generador de tipo "numpy.random._generator.Generator"
u_t = generador.multivariate_normal(
    mean=media_u,
    cov=Sigma_u_teorica,
    size=T,
    method="cholesky",
)

# Los errores se tranforman de un numpy.ndarray a un pandas Dataframe
u_t = pd.DataFrame(u_t, columns=errores)


# %% 1.2.2 Propiedades de los errores en forma reducida " u_t " ----

# Resumen de algunos de los momentos de los errores (e.g. media y varianza)
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
    correlacion_muestral=cor_u_muestral,
)
mostrar_graficas()


# %% 1.3 Simulacion del VAR(1) de 3 variables ----

# Nota: Recuerde que se va a simular un modelo VAR(1) cuya ecuacion esta dada por:
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Definimos el vector constante A_0
A_0 = np.array([0.5, 0.2, -0.1])

# Para imprimir un numpy.ndarray se pasa primero a un pandas series
print(pd.Series(A_0, index=variables, name="A_0"))

# Definimos la matriz de coeficientes autorregresivos.
A_1 = np.array(
    [
        [0.35, 0.08, 0.04],
        [0.25, 0.30, 0.06],
        [0.15, 0.20, 0.25],
    ]
)

# Para imprimir un numpy.ndarray se pasa primero a un pandas dataframe
A_1_df = pd.DataFrame(A_1, index=variables, columns=[f"L1.{v}" for v in variables])
print(A_1_df)

# La matriz A_1 no es triangular inferior. Por tanto, la simulacion permite
# efectos rezagados cruzados entre las tres variables. Esto separa claramente
# la dinamica del VAR de la identificacion contemporanea de Cholesky: el orden
# recursivo y_1, y_2, y_3 se mantiene por el orden de las columnas de Y_t y por
# la estructura triangular de P_chol, no porque A_1 sea triangular.

# Función que permite simular el VAR(1) de manera recursiva mediante un loop: 
def sim_VAR1(Y_t, A_0, A_1, u_t, T):
    for i in range(1, T):
        # Se usa la formula de un VAR(1): Y_t = A_0 + A_1 Y_{t-1} + u_t
        # Para llenar cada una de las filas de Y_t
        Y_t[i, :] = A_0 + A_1 @ Y_t[i - 1, :] + u_t.iloc[i, :].to_numpy()
    return Y_t


# Nota: La función sim_VAR1 lo que busca es llenar mediante un ciclo, cada una 
#       de las filas (iteración por iteración) de Y_t. La matriz Y_t pasa de ser una
#       matriz de ceros, a una matriz que contendrás los valores de las series 
#       simuladas. Note que, como se construyo la matriz de varianzas y covarianzas
#       "Sigma_u_teorica = P_chol %*% t(P_chol)" usando descomposición de Cholesky, 
#       existe un orden natural de las variables a simular, a saber: y1 es la variable 
#       más exogena, luego le sigue y_2 y por último la variable menos exógena 
#       (o más endógena) es y_3
Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T) 

# El objeto que resulta es un numpy.ndarray
type(Y_t)

# Nota: Para trabajar con series de tiempo en python, debemos transformar
#       el objeto Y_t

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
        - Index: Si se quiere trabajar con "tiempo numérico"
        - PeriodIndex: Para trabajar con series de tiempo de periodicidad fija
                       mensual, trimestral, anual, ...
        - DatetimeIndex: Para trabajar con series de tiempo con datos calendario
                         Generalmente para series financieras
                         
                         
E.g. para crear un objeto de series de tiempo para usar en python: 

# De donde provienen los datos
Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T) 

# Índice temporal
tiempo = pd.period_range(start="1900Q1", periods=T, freq="Q", name="tiempo")

# Serie de tiempo (pandas dataframe)
Y_t = pd.DataFrame(Y_t, index=tiempo, columns=variables)

"""

# Creamos el objeto de serie de tiempo de tipo pandas DataFrame

# Se crea un objeto de tipo PeriodIndex qeu permite representar periodos
# trimestrales 
tiempo = pd.period_range(start="1900Q1", periods=T, freq="Q", name="tiempo")

# Convertimos la serie en un DataFrame con indice trimestral.
Y_t = pd.DataFrame(Y_t, index=tiempo, columns=variables)

# El objeto Y_t ahora es un pandas dataframe que ya se puede utilizar
# para hacer análisis en series de tiempo en python! 
type(Y_t)

# Graficas de las series simuladas usando matplotlib y seaborn

fig_series, axes_series = plt.subplots(1, 3, figsize=(15, 4))

graficar_ts(Y_t["y_1"], titulo="Variable y_1", color="lightblue", ax=axes_series[0])
graficar_ts(Y_t["y_2"], titulo="Variable y_2", color="royalblue", ax=axes_series[1])
graficar_ts(Y_t["y_3"], titulo="Variable y_3", color="darkorange", ax=axes_series[2])

fig_series.tight_layout()
mostrar_graficas()


# %% Nota: Recuerden que los modelos VAR requieren de series estacionarias. Por tanto,
#          empleamos Test ADF para verificar la estacionariedad de las series.


adf1 = adfuller(Y_t["y_1"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf1, "y_1") # Rechazo H0, la serie es I(0)

adf2 = adfuller(Y_t["y_2"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf2, "y_2") # Rechazo H0, la serie es I(0)

adf3 = adfuller(Y_t["y_3"], maxlag=3, regression="n", autolag="AIC")
imprimir_adf(adf3, "y_3") # Rechazo H0, la serie es I(0)


# %% ===
# 2. Metodologia Box-Jenkins para series multivariadas ====
# ===

# ===
# 2.1. Identificacion ====
# ===

# Ya tenemos las series simuladas en la matriz Y_t, por lo que ya es posible
# aplicar la metodologia Box-Jenkins en las series simuladas que se encuentran en Y_t. 

# Para crear el objeto VAR con statsmodels quitamos el indice temporal y
# dejamos solo los datos. El indice trimestral se conserva en Y_t para
# graficas y tablas.
Y_t_modelo = Y_t.reset_index(drop=True)
modelo = VAR(Y_t_modelo) # objeto de tipo "statsmodels.tsa.vector_ar.var_model.VAR"

# Nota: Teniendo el objeto VAR, podemos realizar toda la metodología Box Jenkins
#       para modelos VAR en python

# Veamos que rezago recomienda statsmodels, de forma analoga a VARselect() en R.

# Seleccion de rezagos para un VAR con tendencia e intercepto.
# Equivalente en R: VARselect(Y_t, lag.max = 6, type = "both")
lag_order_ct = modelo.select_order(maxlags=6, trend="ct")
print(lag_order_ct.summary())

# Seleccion de rezagos para un VAR con solo intercepto.
# Equivalente en R: VARselect(Y_t, lag.max = 6, type = "const")
lag_order_c = modelo.select_order(maxlags=6, trend="c")
print(lag_order_c.summary())

# Seleccion de rezagos para un VAR sin terminos deterministicos.
# Equivalente en R: VARselect(Y_t, lag.max = 6, type = "none")
lag_order_n = modelo.select_order(maxlags=6, trend="n")
print("\nSeleccion de rezagos para un VAR sin terminos deterministicos")
print("Rezagos recomendados:")
print(pd.Series(lag_order_n.selected_orders))

# Como el proceso generador de datos es un VAR(1), esperamos que los criterios
# de informacion favorezcan rezagos bajos, especialmente p = 1.

# Note: Que la inclusión de términos determínisticos como tendencia determinística
#       o constante, puede influir en la decisión de cuántos rezagos se deberían incluir
#       en el modelo VAR

# %% ===
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
# significativa.

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

# Nota: Note que las raíces del polinomio característico al VAR están por fuera
#       del círculo unitario y además los valores propios asociados a la matriz
#       A_1 del VAR son menores a 1

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


# %% ===
# 2.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ===

# Portmanteau multivariado. En statsmodels usamos test_whiteness().
P_75 = V_dr.test_whiteness(nlags=75, adjusted=False)
print(P_75.summary()) # No rechazo

P_30 = V_dr.test_whiteness(nlags=30, adjusted=False)
print(P_30.summary()) # No rechazo

P_20 = V_dr.test_whiteness(nlags=20, adjusted=False)
print(P_20.summary()) # No rechazo

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

# Nota: Se cumple el supuesto de no autocorrelación serial en los residuales

# Homocedasticidad ===

# statsmodels no tiene un equivalente directo a arch.test() multivariado de
# vars. Por tanto, se construye una función que permite hacer un arch.test()
# univariado para cada uno de los residuales de la regresión, uno por cada
# variable del VAR.
arch_24 = prueba_arch_por_ecuacion(residuales, lags=24, variables=variables) # No rechazo
arch_12 = prueba_arch_por_ecuacion(residuales, lags=12, variables=variables) # No rechazo

# Nota: Como no rechazo en ninguna ecuación (residual), 
# Se cumple el supuesto de homocedasticidad en los residuales

# Normalidad ===

# Jarque-Bera multivariado de statsmodels para los residuales del VAR.
normalidad = V_dr.test_normality()
print(normalidad.summary())  # No rechazo, se cumple el supuesto.

# Se aplica test Jarque-Bera de normalidad univariados por cada residual individual
normalidad_univariada = prueba_normalidad_por_ecuacion(
    residuales,
    variables=variables,
)
    
# Nota: Se cumple el supuesto de normalidad en los residuales

# %% ===
# 2.4. Uso del modelo: pronostico y funciones Impulso respuesta (IRF) ====
# ===

# Pronostico ===

# Función diseñada para parecerse lo más que se pueda a predict(V.dr, n.ahead = 12, ci = 0.95).
pronostico_var = predecir_var(V_dr, n_ahead=12, ci=0.95, indice=Y_t.index)
print(pronostico_var)

# Graficas pronóstico
fig_pronostico, axes_pronostico = graficar_pronostico_var(
    pronostico_var["pronostico"],
    pronostico_var["inferior"],
    pronostico_var["superior"],
)

# Version fanchart, similar a fanchart(predict(...)) en R.
fig_fanchart, axes_fanchart = graficar_fanchart_var(Y_t, pronostico_var)
mostrar_graficas()


# %% Funciones impulso respuesta no ortogonalizadas ===

# Nota: Recuerde que para poder calcular las IRF de un modelo VAR
#       este debe tener su representacion como VMA(infinito).
#       Es decir, pasamos del VAR(1) --> VMA(infinito)

# IRFs no ortogonalizadas:
print(V_dr.irf(10).irfs)

# Parámetros de las gráficas de las IRFs
pasos_adelante = np.arange(0, 19)
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_montecarlo_irf = 100 # Bootstrappings empleados para construir los IC de las IRFs

# Nota: La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez
# y luego crea cada panel usando funciones auxiliares.

# IRF de las variables del sistema ante distintos choques exogenos.
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

mostrar_graficas()


# %% Funciones de impulso respuesta ortogonalizadas ===

# Cuando ortog = True, statsmodels usa una descomposicion de Cholesky de la
# matriz de varianzas y covarianzas de los residuales. En este script el orden
# de las variables es y_1, y_2, y_3; por tanto, la identificacion recursiva
# interpreta a y_1 como la variable contemporaneamente mas exogena, luego y_2 y
# finalmente y_3 como la mas endogena. 

# IRFs ortogonalizadas:
print(V_dr.irf(10).orth_irfs)

# Graficación de las IRFs

# Usamos los mismos pasos adelante, intervalo de confianza y semilla.
# IRFs ortogonalizadas de las variables del sistema ante distintos choques exogenos.
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

mostrar_graficas()


# %% Descomposicion de varianza del error de pronostico (FEVD) ===

# La FEVD resume que proporcion de la varianza del error de pronostico de cada
# variable se atribuye a los choques de cada variable del sistema.

# Cálculo de la FEVD
fevd_var = V_dr.fevd(periods=18)
fevd_var.summary()

# Gráfica de la FEVD
fig_fevd, axes_fevd = graficar_fevd_var(fevd_var)
mostrar_graficas()
