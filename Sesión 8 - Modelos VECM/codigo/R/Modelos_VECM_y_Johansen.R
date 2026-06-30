#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 8: Cointegracion y metodologia Johansen
#'
#' Semestre: 2026-1


# ===
# Tabla de contenidos ===
# ===

#' 1. Importacion de paquetes, rutas y funciones auxiliares
#' 2. Carga y preparacion de los datos
#' 3. Introduccion a metodologia Johansen
#' 4. Identificacion del orden de integracion de las series
#' 5. Modelo VAR en niveles
#' 6. Determinacion del rango de la matriz Pi
#'  6.1. Test de Johansen - Sin constante en el vector de cointegración
#'  6.2. Test de Johansen - Con constante en el vector de cointegración
#'  6.3. Estimación del VECM(2) de acuerdo a los resultados del test de Johansen
#'  6.4. Test para determinar tendencia lineal en la reparametrización como VAR usando "lttest"
#' 7. Validacion de supuestos y usos del modelo
#'  7.1. Reparametrizacion del VECM como un VAR en niveles
#'  7.2. Validacion de supuestos de VECM como VAR
#'  7.3. Pronostico del VECM reparametrizado
#'  7.4. Funciones impulso-respuesta para VECM
#' 8. Qué pasa si se cambia el orden de las variables en el VECM?

# Nota: Tips practicos en R
## Para limpiar el entorno de trabajo se puede correr: rm(list = ls())
## Para cerrar todas las graficas actualmente abiertas se puede correr: dev.off()
## Para resetear R se puede usar las teclas: Ctrl + Shift + F10


# ===
# 1. Importacion de paquetes, rutas y funciones auxiliares ====
# ===

# Paquetes del tidyverse (Para el manejo, manipulacion y graficacion de datos)
library(tidyverse)

# Paquete principal para estimar modelos VAR y trabajar con objetos vec2var.
library(vars)

# Paquete para pruebas de raiz unitaria y cointegracion.
library(urca)

# Paquete para extraer matrices alpha y beta del VECM.
library(tsDyn)

# Paquete para organizar varias graficas en una misma ventana.
library(gridExtra)

# Paquete para leer archivos .xlsx.
library(readxl)

# Paquetes para manejar rutas relativas en R.
library(here)
library(fs)


# Cargar bases de datos en R usando rutas relativas ----

# Fijar la ruta del archivo actual como referencia para here().
here::i_am("Sesión 8 - Modelos VECM/codigo/R/Modelos_VECM_y_Johansen.R")

# Directorios principales del proyecto.
directorio_sesion_vecm = fs::path(here::here("Sesión 8 - Modelos VECM"))
directorio_datos = fs::path(directorio_sesion_vecm, "datos")
directorio_codigo_R = fs::path(directorio_sesion_vecm, "codigo", "R")

# Ruta donde se encuentra la base de datos de petroleo.
ruta_petroleo = fs::path(directorio_datos, "Petróleo.xlsx")


# Importacion de funciones auxiliares de graficacion

# Ruta con las funciones auxiliares de VECM.
ruta_funciones_auxiliares_vecm = fs::path(
  directorio_codigo_R,
  "funciones_auxiliares_VECM.R"
)

# script auxiliar "funciones_auxiliares_VECM.R"
source(ruta_funciones_auxiliares_vecm, encoding = "UTF-8")


# ===
# 2. Carga y preparacion de los datos ====
# ===

# Ejemplo 1: Precio de referencia Brent y WTI

# Vamos a utilizar una serie del precio spot del petroleo de referencia Brent y
# una serie del precio spot del petroleo de referencia WTI. Las series tienen
# frecuencia mensual y comprenden el periodo de enero del 2000 a diciembre de 2020.

# Base de datos con las series de petroleo.
Data = readxl::read_excel(ruta_petroleo)

# Informacion general de la base de datos.
glimpse(Data)

# Series en niveles.
P.Brent = ts(Data$Brent, start = c(2000, 1), frequency = 12)
P.WTI = ts(Data$WTI, start = c(2000, 1), frequency = 12)

# Variables que se modelaran mediante el VECM.
variables = c("P.Brent", "P.WTI")

# Se construye la matriz Y, que contiene las series de tiempo del modelo.
Y = ts.intersect(P.Brent = P.Brent, P.WTI = P.WTI)

# Algunas caracteristicas de las series de tiempo del modelo.
start(Y) # Periodo donde inician las series
end(Y) # Periodo donde terminan las series
head(Y) # Observaciones iniciales
tail(Y) # Observaciones finales


# Graficas de las series ----

colores_petroleo = c("P.Brent" = "lightblue", "P.WTI" = "coral")
etiquetas_petroleo = c("P.Brent" = "Brent", "P.WTI" = "WTI")

g_precios_petroleo = graficar_series_vecm(
  Y,
  titulo = "Precios spot del petroleo",
  subtitulo = "Petroleo Brent y WTI",
  colores = colores_petroleo,
  etiquetas = etiquetas_petroleo,
  etiqueta_color = "Petroleos"
)

x11(width = 8, height = 4)
print(g_precios_petroleo)


# ===
# 3. Introduccion a metodologia Johansen ====
# ===

# Aspectos generales de la metodologia de Johansen ----

# Consiste en un procedimiento en 4 etapas:

## Etapa 1: Verificacion preliminar de las variables a trabajar (orden de
#            integracion y graficas) e identificacion del numero de rezagos del
#            VECM mediante criterios de informacion sobre el VAR en niveles y 
#            seleccionando el número de reazagos tal que los errores sean ruido blanco. 
## Etapa 2: Determinacion del rango de la matriz Pi, es decir, del numero de
#            relaciones de cointegracion, y estimacion del modelo apropiado
#            dependiendo del rango de la matriz Pi.
## Etapa 3: Analisis de la matriz beta, que contiene el vector de cointegracion,
#            y de la matriz alpha, que contiene los parametros de velocidad de
#            ajuste.
## Etapa 4: Validacion de supuestos y usos del modelo (pronósticos e IRF).


# ===
# 4. Identificacion del orden de integracion de las series ====
# ===

# Procedemos a hacer las pruebas de raiz unitaria para identificar el orden de
# integracion de las dos series.

# Referencia Brent ----

adf_brent_tendencia = ur.df(P.Brent, lags = 12, type = "trend")
summary(adf_brent_tendencia) # Tendencia no significativa

adf_brent_deriva = ur.df(P.Brent, lags = 12, type = "drift")
summary(adf_brent_deriva) # Deriva no significativa

adf_brent_none = ur.df(P.Brent, lags = 12, type = "none")
summary(adf_brent_none) # Serie no estacionaria


# Referencia WTI ----

adf_wti_tendencia = ur.df(P.WTI, lags = 12, type = "trend")
summary(adf_wti_tendencia) # Tendencia no significativa

adf_wti_deriva = ur.df(P.WTI, lags = 12, type = "drift")
summary(adf_wti_deriva) # Deriva no significativa

adf_wti_none = ur.df(P.WTI, lags = 12, type = "none")
summary(adf_wti_none) # Serie no estacionaria


# Aplicamos diferenciacion ----

adf_d_brent = ur.df(diff(P.Brent), lags = 12, type = "none")
summary(adf_d_brent) # La diferenciación de P.Brent es I(0), por lo que P.Brent en niveles es I(1)

adf_d_wti = ur.df(diff(P.WTI), lags = 12, type = "none")
summary(adf_d_wti) # La diferenciación de P.WTI es I(0), por lo que P.WTI en niveles es I(1)


# ===
# 5. Modelo VAR en niveles ====
# ===

# Posteriormente, estimaremos un VAR en niveles para determinar el numero de
# rezagos del VECM.

# Nota: Se analizaran los criterios de informacion sobre el VAR en niveles.

# Seleccion de rezagos para un VAR con tendencia e intercepto.
seleccion_rezagos_both = VARselect(
  Y, lag.max = 6, type = "both", season = NULL
)
seleccion_rezagos_both

# Se trabaja con p = 3 porque AIC/FPE sugieren este rezago 
p_var = 3

VAR3_both = vars::VAR(Y, p = p_var, type = "both", season = NULL)
summary(VAR3_both) # Tendencia no significativa

# Seleccion de rezagos para un VAR con solo intercepto.
seleccion_rezagos_const = VARselect(
  Y, lag.max = 6, type = "const", season = NULL
)
seleccion_rezagos_const

VAR3_const = vars::VAR(Y, p = p_var, type = "const", season = NULL)
summary(VAR3_const) # Intercepto significativo

# Dado que al estimar el VAR con constante,el intercepto en éste modelo 
# resultó significativo, decidimos estimar un VAR con constante.

# Elegimos VAR(3) en niveles.
VAR3 = VAR3_const

# Note que como se estimo un VAR(3), su reparamterización como un VECM será un 
# VECM(2). Además, dicha reparametrización siempre se podrá hacer independidente de
# si las variables del VAR son I(0) o I(1). 

# Nota: Los residuales del modelo VAR en niveles deben ser ruido blanco, independientemente
#       de si las variables del VAR en niveles son I(0) o son I(1). Si se escogió
#       el número adecuado de rezagos en el VAR, siempre se podrá garantizar que esos
#       residuales serán ruido blanco. 

# Nota: También recuerde que en teoría, los errores del VAR en niveles deben ser los
#       mismos errores que los del VECM (Nota de la nota: recuerde que los errores son teóricos
#       y los residuales son una aproximación a los errores, pero no son los errores). 

# Vamos a analizar el comportamiento de los residuales. Dado que es una serie
# mensual, analicemos su comportamiento en puntos criticos.

# No autocorrelación serial ===

P.12 = vars::serial.test(VAR3, lags.pt = 12, type = "PT.asymptotic"); P.12
# No rechazo, se cumple el supuesto

P.16 = vars::serial.test(VAR3, lags.pt = 16, type = "PT.asymptotic"); P.16
# No rechazo, se cumple el supuesto

P.20 = vars::serial.test(VAR3, lags.pt = 20, type = "PT.asymptotic"); P.20
# No rechazo, se cumple el supuesto

# Validación grafica de otros supuestos ===

# Graficamos los residuales para 12 lags:

x11()
plot(P.12, names = "P.Brent") # Bien comportados salvo por heterocedasticidad

x11()
plot(P.12, names = "P.WTI") # Bien comportados salvo por heterocedasticidad


# Nota: Lo más importante para seguir con el procedimiento de la metodología de Johansen, es que
#       los residuales no tengan correlación serial. Puede que no sean exactamente ruido blanco, 
#       si e.g. tienen heterocedasticidad, pero lo fundamental es que los residuales no tengan
#       correlación serial, ese es el supuesto clave a validar.

# ===
# 6. Determinacion del rango de la matriz Pi ====
# ===

# La funcion ca.jo del paquete urca permite realizar el test de Johansen en R.

# Argumentos del comando ca.jo para realizar el test de Johansen: 

# Para revisar todos los argumentos del test de Johansen se puede usar: ?ca.jo
args(urca::ca.jo)

# Nota: Existen 3 versiones diferentes del test de cointegración de Johansen 
#       1) ecdet = "none": Relación de cointegración sin constante
#                           P.Brent - beta * P.WTI = 0 (Cómo se vería la relación de cointetración 
#                           P.Brent = beta * P.WTI     En el actual ejemplo)
#
#       2) ecdet = "const": Relación de cointegración con constante 
#                           P.Brent - beta * P.WTI + c = 0 (Cómo se vería la relación de cointetración 
#                           P.Brent = beta * P.WTI - c     En el actual ejemplo)
#
#       3) ecdet = "trend": Relación de cointegración con tendencia lineal
#                           P.Brent - beta * P.WTI + c + delta * t = 0 (Cómo se vería la relación de cointetración   
#                           P.Brent = beta * P.WTI - c - delta * t     En el actual ejemplo)
#

# Donde "beta" es el coeficiente de cointegración que aparece en el vector de cointegración

# Nota: Para el curso, siempre trabajaremos con la especificación "spec = transitory"
#       en el comando ca.jo, que genera un representación del modelo VEC equivalente a 
#       la que vemos teóricamente en el curso.
#       La especificación "spec = longrun" es otra manera de representar el modelo VAR, 
#       pero esa representación no la trabajamos en el curso 

# Nota: El argumento K del test de Johansen, determinar el orden del VAR en niveles que estimo
#       previamente. E.g., en nuestro caso particular, estimamos un VAR(3), entonces K = 3 cuando
#       se usa el comando ca.jo, a pesar de que la reparametrización de ese VAR(3) es un VECM(2)

# ===
# 6.1. Test de Johansen - Sin constante en el vector de cointegración ====
# ===

# Nota: Estamos en el caso: ecdet = "none": Relación de cointegración sin constante
#                           (P.Brent - beta * P.WTI = 0, es decir, 
#                            P.Brent = beta * P.WTI)  

# Nota: Hay dos manera de hacer el test de Johansen: 
#       1. Criterio del valor propio máximo
#       2. Criterio de la traza

# Criterio del valor propio maximo ----

# Generalmente es la prueba preferida y la mas robusta.

# Dado que solo hay dos variables, se realiza el siguiente procedimiento secuencial
# el procedimiento que se analiza es:

# H0: r = 0 vs H1: r >= 1, 
# luego H0: r = 1 vs H1: r = 2. 
# Aquí p = 2 variables y K = 3 rezagos (se estimo un VAR(3)).

eigen_none = urca::ca.jo(
  Y, ecdet = "none", type = "eigen", K = p_var, spec = "transitory"
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. 
summary(eigen_none) 

# Nota: Note que el orden del VECM no me determinará cuántas pruebas secuenciales
#       debo realizar en el test de Johansen, eso solo va a estar determinado por
#       el número de variables que tengo el VECM

# Criterio de la traza ----

# Al tener el VECM solo dos variables, el procedimiento secuencial a realizar es: 

# H0: r = 0 vs H1: r >= 1, 
# H0: r = 1 vs H1: r = 1, y asi
# Aquí p = 2 variables y K = 3 rezagos (se estimo un VAR(3)).

trace_none = urca::ca.jo(
  Y, ecdet = "none", type = "trace", K = p_var, spec = "transitory"
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. 
summary(trace_none) 


# ===
# 6.2. Test de Johansen - Con constante en el vector de cointegración ====
# ===

# Nota: Estamos en el caso: ecdet = "const": Relación de cointegración con constante 
#                           (P.Brent - beta * P.WTI + c = 0, es decir, 
#                            P.Brent = beta * P.WTI - c)  

# ===
# Nota: Este es el el caso más común, entonces por lo general trabajaremos con 
#       ecdet = "const", y diremos que la relación de cointegración incluye una constante! 
# ===

# Criterio del valor propio maximo ----

eigen_const = urca::ca.jo(
  Y, ecdet = "const", type = "eigen", K = p_var, spec = "transitory"
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas.
summary(eigen_const) 


# Criterio de la traza ----

trace_const = urca::ca.jo(
  Y, ecdet = "const", type = "trace", K = p_var, spec = "transitory"
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas.
summary(trace_const) 


# ===
# 6.3. Estimación del VECM(2) de acuerdo a los resultados del test de Johansen ====
# ===

# Sin intercepto en la relación de cointegración ----

# La funcion cajorls del paquete urca permite estimar el modelo VEC.
VEC_none = urca::cajorls(eigen_none, r = 1) # r=1 para indicar que hay una relación de cointegración
VEC_none

# Con esta funcion obtenemos el vector de cointegracion normalizado.
coefB(VEC_none)

# Con esta funcion obtenemos los coeficientes de velocidad de ajuste.
coefA(VEC_none)


# Con intercepto en la relación de cointegración ----

# La funcion cajorls permite estimar el modelo VEC.
VEC_const = urca::cajorls(eigen_const, r = 1)
VEC_const

# Con esta funcion obtenemos el vector de cointegracion normalizado.
coefB(VEC_const)

# Con esta funcion obtenemos los coeficientes de velocidad de ajuste.
coefA(VEC_const)


# ===
# 6.4. Test para determinar tendencia lineal en la reparametrización como VAR usando "lttest" ====
# ===

# La funcion lttest del paquete urca permite determinar la existencia de una tendencia 
# lineal determinística en el VAR en niveles asociado a la reparametrización del VECM.

# Para revisar que hace la función se puede usar el siguiente comando:
# ?lttest

# De la documentación lttest indica que:

# H0: No existencia de tendencia lineal en el VAR en niveles asociado a la reparametrización del VECM.
# H1: Existencia de tendencia lineal en el VAR en niveles asociado a la reparametrización del VECM.

urca::lttest(eigen_const, r = 1)

# No rechazo la hipotesis nula, por lo que no se debe incluir tendencia lineal
# el VAR en niveles asociado a la reparametrización del VECM.


# ===
# 7. Validacion de supuestos y usos del modelo ====
# ===


# ===
# 7.1. Reparametrizacion del VECM como un VAR en niveles ====
# ===

# Nota: Dados los resultados anteriores, se usara el modelo VEC con constante en
# el vector de cointegracion.

# Nota: Luego de estimar el modelo VECM usando la función cajorls del paquete
#       urca, se puede reparatremizar el modelo VECM de nuevo como un modelo VAR
#       usando la funcíón vec2var del paquete vars

VAR.oil = vars::vec2var(eigen_const, r = 1)

# VAR.oil va a ser la reparametrizacion del modelo VEC anterior como VAR en
# niveles.

VAR.oil
class(VAR.oil) # Notar que la clase del objeto ahora es vec2var

# Esto es importante dado que se necesita el modelo VEC reparametrizado como un
# VAR en niveles para poder validar los supuestos y hacer uso del modelo.


# ===
# 7.2. Validacion de supuestos de VECM como VAR ====
# ===

# Autocorrelacion ----

# PT.asymptotic es para muestra grande y "PT.adjusted" es correccion para
# muestra pequena.

P.12_V = vars::serial.test(VAR.oil, lags.pt = 12, type = "PT.asymptotic"); P.12_V # No rechaza H0
P.24_V = vars::serial.test(VAR.oil, lags.pt = 24, type = "PT.asymptotic"); P.24_V # No rechaza H0
P.36_V = vars::serial.test(VAR.oil, lags.pt = 36, type = "PT.asymptotic"); P.36_V # No rechaza H0

# Nota: Se cumple el supuesto de no correlación serial en los residuales 

# Homocedasticidad ----

# Test tipo ARCH multivariado.
vars::arch.test(VAR.oil, lags.multi = 24, multivariate.only = TRUE) # Rechaza H0
vars::arch.test(VAR.oil, lags.multi = 12, multivariate.only = TRUE) # Rechaza H0

# Nota: No se cumple el supuesto de heterocedasticidad en los residuales 

# Normalidad ----

# Test Jarque-Bera multivariado.
vars::normality.test(VAR.oil) # Rechaza H0 

# Nota: No se cumple el supuesto de normalidad en los residuales 

# Nota: Se cumple el supuesto más importante, que es el de no correlación serial
#       en los residuales del modelo 

# Nota: Como se violan los supuestos de heterocedasticidad y normalidad, hay que
#       calcular los intervalos de confianza mediante bootstrapping para poder
#       hacer inferencia estadistica correcta, tanto en los pronosticos como en
#       las OIRF.


# ===
# 7.3. Pronostico del VECM reparametrizado ====
# ===

# Recuerden que debido al incumplimiento de normalidad, los intervalos de
# confianza deben computarse por bootstrapping.

# Especificaciones del pronóstico
horizonte_pronostico = 12
int_conf_pronostico = 0.95

# Pronóstico del modelo VEC
prono_VECM = predict(
  VAR.oil,
  n.ahead = horizonte_pronostico,
  ci = int_conf_pronostico
)
prono_VECM

# Gráfica del pronóstico del modelo VEC
g_pronostico_vecm = graficar_pronostico_vecm(prono_VECM) +
  ggtitle("Pronostico VECM reparametrizado") +
  labs(subtitle = "Horizonte: 12 meses")

x11()
print(g_pronostico_vecm)

# Version fanchart, similar a fanchart(predict(...)) en R.
g_fanchart_vecm = graficar_fanchart_vecm(Y, prono_VECM)

x11(width = 12, height = 6)
print(g_fanchart_vecm)


# ===
# 7.4. Funciones impulso-respuesta para VECM ====
# ===

# Parametros de las graficas de las IRFs.
pasos_adelante = 0:18
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100 # Bootstrappings empleados para construir los IC de las IRFs

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez y luego
# crea cada panel con programacion funcional.

# IRF de las variables del sistema ante distintos choques exogenos.
irf_ortog_vecm = graficar_grilla_irf(
  VAR.oil,
  variables,
  pasos_adelante,
  ortog = TRUE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso ortogonal",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

x11()
# Grilla de OIRF: columnas = impulsos; filas = respuestas.
grid.arrange(grobs = irf_ortog_vecm$graficas,
             layout_matrix = matrix(seq_along(irf_ortog_vecm$graficas),
                                    nrow = length(variables), byrow = TRUE))

# Nota: Dado que por el orden de las variables que escogimos, primero Brent y luego WTI, 
#       la variable más exógena es Brent y la más endógena es WTI. Vemos que en éste orden, 
#       luego de hacer la descomposción de Cholesky y trabajar con IRFs ortogonalizadas, 
#       un choque estructural en el precio el Brent afecta tanto al precio del Brent como al
#       precio del WTI, mientras que un choque estructural en precio del WTI no tiene ningún
#       efecto en el tiempo sobre el precio del Brent y que el efecto de un choque estructural
#       en el precio del WTI sobre si mismo se disipa hacia cero en el largo plazo. 
#       Note que se llegan a éstas conclusiones
#       por el orden escogido de las variables, donde se asume que la variable más exógena es
#       el precio del Brent y la más endógena es el precio del WTI.


# ===
# 8. Qué pasa si se cambia el orden de las variables en el VECM? ====
# ===

# Nota: Recuerde que el orden de las variables que se usa para construir la matriz 
#       de series de tiempo Y importa, dada que la primera columna de la matriz 
#       está asociada a la variable más exógena, mientras que la última columna 
#       está asociada a la variable más endógena. Ésto es importante a la hora de
#       realizar la descomposición de Cholesky, dada que la descomposición de Cholesky
#       tiene en cuenta ese orden, y por ende las funciones impulso respuseta ortgonalizadas
#       dependen fundamentalmente del orden que se escoga de las variables. 

# En ésta sección veremos que pasa si se cambia el orden de las columnas de la matriz de
# series de tiempo, en éste caso la llamaremos: Y_alt

variables_alt = c("P.WTI", "P.Brent")
Y_alt = ts.intersect(P.WTI = P.WTI, P.Brent = P.Brent)

# Se realizará de nuevo la metodología de Johansen de manera completa que se realizó previamente, 
# pero está vez con las variables intercambiadas

# Se estima un VAR(3), pero con las variables intercambiadas de orden 
VAR3_alt = vars::VAR(Y_alt, p = p_var, type = "const", season = NULL); summary(VAR3_alt)

# Test de ljung-box

# No autocorrelación serial ===

P.12_alt = vars::serial.test(VAR3_alt, lags.pt = 12, type = "PT.asymptotic"); P.12_alt # No rechazo
P.16_alt = vars::serial.test(VAR3_alt, lags.pt = 16, type = "PT.asymptotic"); P.16_alt # No rechazo
P.20_alt = vars::serial.test(VAR3_alt, lags.pt = 20, type = "PT.asymptotic"); P.20_alt # No rechazo

# Nota: El supuesto de no correlación serial en los residuales, el más importante, 
#       se sigue cumpliendo! 

# Prueba de Johansen (comando ca.jo) ----

# Se usa de nuevo el comando ca.jo para realizar la prueba de Johansen y determinar
# el rango de la matriz Pi

# Para ello usaremos las siguientes especificaciones del test: 
#  - type = "eigen": Criterio del valor propio máximo 
#  - ecdet = "const":  Constante en el vector de cointegración

# Al tener el VECM solo dos variables, el procedimiento secuencial a realizar es: 

# H0: r = 0 vs H1: r >= 1, 
# H0: r = 1 vs H1: r = 1, y asi
# Aquí p = 2 variables y K = 3 rezagos (se estimo un VAR(3)).

eigen_const_alt = urca::ca.jo(
  Y_alt, ecdet = "const", type = "eigen", K = p_var, spec = "transitory"
)

# Note que primero se rechaza la hipótesis nula r = 0, pero luego no se rechaza
# la hipotesis nula r = 1, por lo que se concluye que existe una relación de
# cointegración y las series están cointegradas. 
summary(eigen_const_alt) # Se mantiene la conclusion

# Estimacion del modelo VEC (comando cajorls) ----

# La funcion cajorls permite estimar el modelo VEC.
VEC_const_alt = urca::cajorls(eigen_const_alt, r = 1)
VEC_const_alt

# Con esta funcion obtenemos el vector de cointegracion normalizado.
coefB(VEC_const_alt)

# Con esta funcion obtenemos los coeficientes de velocidad de ajuste.
coefA(VEC_const_alt)

# Reparametrizacion del modelo VEC en un modelo VAR (comando vec2var) ----

VAR.oil_alt = vars::vec2var(eigen_const_alt, r = 1)

P.12_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 12,
                               type = "PT.asymptotic"); P.12_V_alt
P.24_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 24,
                               type = "PT.asymptotic"); P.24_V_alt
P.36_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 36,
                               type = "PT.asymptotic"); P.36_V_alt

# Nota: Se cumple el supuesto de no correlación serial en los residaules, en el 
# VAR reparametrizado, que es el supuesto más importante

# Funcioens Impulso-respuesta ortogonalizadas ----

# Veamos que ocurre con las OIRF al cambiar el orden de las variables del modelo VEC.

irf_ortog_vecm_alt = graficar_grilla_irf(
  VAR.oil_alt,
  variables_alt,
  pasos_adelante,
  ortog = TRUE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso ortogonal",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

x11()
# Grilla de OIRF: columnas = impulsos; filas = respuestas.
grid.arrange(grobs = irf_ortog_vecm_alt$graficas,
             layout_matrix = matrix(seq_along(irf_ortog_vecm_alt$graficas),
                                    nrow = length(variables_alt), byrow = TRUE))

# Nota: Las OIRF cambian sustancialmente. Note que, si asume que P.WTI ahora es la
#       variable más exógena, entonces ahora, a diferencia de lo que ocurria anteriormente,
#       los choques estructurales del P.WTI ahora sí son significativos y persistentes, y además
#       los choques estructurales del P.Brent continuan siendo significativos y persistentes.
#       Note que solo con cambiar el orden de las variables, el choque estructural del P.WTI
#       paso de no ser casi significativo y disiparse, ahora a ser significativo y persistente
#       por lo que se concluye que el orden en que se escogan las variables en el modelo VEC
#       , al igual que lo que pasa en el modelo VAR, no es trivial, es fundamental saber escoger
#       dicho orden de exógenidad por teoría económica o test estadísticos, por que dada la
#       lógica de la descomposición de Cholesky, al escoger un orden diferente de las variables 
#       del modelo, se pueden llegar a conclusiones muy distintas!
#       Bienvenido a la Economía, donde todo es posible :D !!!!!

# Nota: Lo anterior muestra algunas de las limitaciones de la descomposición de Cholesky, como
#       estrategía de identificación de choques estructurales. Claramente, el orden en que 
#       se escogan las variables afecta la interpretación económico y sobre todo los restultados
#       de política, pero en muchos casos el orden de exogenidad entre las variables o 1) no es 
#       claro o 2) simplemente no existe. Para superar ese problema en la identificación de 
#       choques estructurales usando descomposición de Cholesky, existen otras estrategias de
#       identificación como lo puede ser un S-VECM (Structural VECM), donde a partir de una matriz
#       S se pueden imponer restricciones más sensibles para la identificación de choque 
#       estructurales, en la práctica, se usa mucho más este tipo de identificación que usar
#       descomposición de Chokesky a lo maldita sea xD
