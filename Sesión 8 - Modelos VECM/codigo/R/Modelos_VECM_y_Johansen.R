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
#'  6.1. Test de Johansen - Sin intercepto
#'  6.2. Test de Johansen - Con intercepto
#'  6.3. Estimaciones de los modelos vistos
#'  6.4. Test para determinar tendencia lineal en el modelo
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
#            VECM mediante criterios de informacion sobre el VAR en niveles.
## Etapa 2: Determinacion del rango de la matriz Pi, es decir, del numero de
#            relaciones de cointegracion, y estimacion del VECM.
## Etapa 3: Analisis de la matriz beta, que contiene el vector de cointegracion,
#            y de la matriz alpha, que contiene los parametros de velocidad de
#            ajuste.
## Etapa 4: Validacion de supuestos y usos del modelo.


# ===
# 4. Identificacion del orden de integracion de las series ====
# ===

# Procedemos a hacer las pruebas de raiz unitaria para identificar el orden de
# integracion de las dos series.

# Referencia Brent ----

adf_brent_tendencia = urca::ur.df(P.Brent, lags = 12, type = "trend")
summary(adf_brent_tendencia) # Tendencia no significativa

adf_brent_deriva = urca::ur.df(P.Brent, lags = 12, type = "drift")
summary(adf_brent_deriva) # Deriva no significativa

adf_brent_none = urca::ur.df(P.Brent, lags = 12, type = "none")
summary(adf_brent_none) # Serie no estacionaria


# Referencia WTI ----

adf_wti_tendencia = urca::ur.df(P.WTI, lags = 12, type = "trend")
summary(adf_wti_tendencia) # Tendencia no significativa

adf_wti_deriva = urca::ur.df(P.WTI, lags = 12, type = "drift")
summary(adf_wti_deriva) # Deriva no significativa

adf_wti_none = urca::ur.df(P.WTI, lags = 12, type = "none")
summary(adf_wti_none) # Serie no estacionaria


# Aplicamos diferenciacion ----

adf_d_brent = urca::ur.df(diff(P.Brent), lags = 12, type = "none")
summary(adf_d_brent) # I(1)

adf_d_wti = urca::ur.df(diff(P.WTI), lags = 12, type = "none")
summary(adf_d_wti) # I(1)


# ===
# 5. Modelo VAR en niveles ====
# ===

# Posteriormente, estimaremos un VAR en niveles para determinar el numero de
# rezagos del VECM.

# Ojo: se analizaran los criterios de informacion sobre el VAR en niveles.

# Seleccion de rezagos para un VAR con tendencia e intercepto.
seleccion_rezagos_both = vars::VARselect(
  Y, lag.max = 6, type = "both", season = NULL
)
seleccion_rezagos_both

VAR2_both = vars::VAR(Y, p = 2, type = "both", season = NULL)
summary(VAR2_both) # Tendencia no significativa

# Seleccion de rezagos para un VAR con solo intercepto.
seleccion_rezagos_const = vars::VARselect(
  Y, lag.max = 6, type = "const", season = NULL
)
seleccion_rezagos_const

VAR2_const = vars::VAR(Y, p = 2, type = "const", season = NULL)
summary(VAR2_const) # Intercepto significativo

# No tiene sentido analizar sin constante ya que este ultimo modelo resulta no
# significativo.

# Elegimos VAR(2) en niveles.
VAR2 = VAR2_const

# Vamos a analizar el comportamiento de los residuales. Dado que es una serie
# mensual, analicemos su comportamiento en puntos criticos.

P.12 = vars::serial.test(VAR2, lags.pt = 12, type = "PT.asymptotic"); P.12
# Rechazo, se viola el supuesto

P.24 = vars::serial.test(VAR2, lags.pt = 24, type = "PT.asymptotic"); P.24
# No rechazo, se cumple el supuesto

P.36 = vars::serial.test(VAR2, lags.pt = 36, type = "PT.asymptotic"); P.36
# No rechazo, se cumple el supuesto

# A medida que se alejan los periodos, se cumple el supuesto.
# Efecto desvanecimiento. Es normal que ocurra esto, por lo que en general,
# validaremos el cumplimiento del supuesto.

# Graficamos los residuales para 20 lags:

x11()
plot(P.12, names = "P.Brent") # Bien comportados salvo por heterocedasticidad

x11()
plot(P.12, names = "P.WTI") # Bien comportados salvo por heterocedasticidad


# ===
# 6. Determinacion del rango de la matriz Pi ====
# ===

# La funcion ca.jo nos permitira analizar el test de Johansen.

# Argumentos del test de Johansen

# Para revisar todos los argumentos del test de Johansen se puede usar: ?ca.jo
args(urca::ca.jo)


# ===
# 6.1. Test de Johansen - Sin intercepto ====
# ===

# Criterio del valor propio maximo ----

# Generalmente es la prueba preferida y la mas robusta.
# El procedimiento que se analiza es:

# H0: r = 0 vs H1: r = 1, luego H0: r = 1 vs H1: r = 2, y asi
# sucesivamente. Aqui k = 2.

eigen_none = urca::ca.jo(
  Y, ecdet = "none", type = "eigen", K = 2, spec = "transitory"
)
summary(eigen_none) # Al 5% de confianza las series estan cointegradas


# Criterio de la traza ----

# Es un procedimiento secuencial en donde se contrasta:
# H0: r = 0 vs H1: r >= 1, luego H0: r <= 1 vs H1: r > 1, y asi
# sucesivamente. Aqui k = 2.

trace_none = urca::ca.jo(
  Y, ecdet = "none", type = "trace", K = 2, spec = "transitory"
)
summary(trace_none) # Al 5% de confianza las series estan cointegradas


# ===
# 6.2. Test de Johansen - Con intercepto ====
# ===

# Criterio del valor propio maximo ----

eigen_const = urca::ca.jo(
  Y, ecdet = "const", type = "eigen", K = 2, spec = "transitory"
)
summary(eigen_const) # Al 5% de confianza las series estan cointegradas


# Criterio de la traza ----

trace_const = urca::ca.jo(
  Y, ecdet = "const", type = "trace", K = 2, spec = "transitory"
)
summary(trace_const) # Al 5% de confianza las series estan cointegradas


# ===
# 6.3. Estimaciones de los modelos vistos ====
# ===

# Sin constante ----

# La funcion cajorls permite estimar el modelo VEC.
VEC_none = urca::cajorls(eigen_none, r = 1)
VEC_none

# Con esta funcion obtenemos el vector de cointegracion normalizado.
coefB(VEC_none)

# Con esta funcion obtenemos los coeficientes de velocidad de ajuste.
coefA(VEC_none)


# Con constante ----

# La funcion cajorls permite estimar el modelo VEC.
VEC_const = urca::cajorls(eigen_const, r = 1)
VEC_const

# Con esta funcion obtenemos el vector de cointegracion normalizado.
coefB(VEC_const)

# Con esta funcion obtenemos los coeficientes de velocidad de ajuste.
coefA(VEC_const)


# ===
# 6.4. Test para determinar tendencia lineal en el modelo ====
# ===

# Tenemos la funcion lttest del paquete urca. Para revisar que hace se puede usar:
# ?lttest

# Es decir:

# H0: No existencia de tendencia lineal.
# H1: Existencia de tendencia lineal.

urca::lttest(eigen_const, r = 1)

# No rechazo la hipotesis nula, por lo que no se debe incluir tendencia lineal
# en el modelo.


# ===
# 7. Validacion de supuestos y usos del modelo ====
# ===


# ===
# 7.1. Reparametrizacion del VECM como un VAR en niveles ====
# ===

# Nota: Dados los resultados de la prueba lttest, se usara el modelo VEC con
# constante en el vector de cointegracion.

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

P.12_V = vars::serial.test(VAR.oil, lags.pt = 12, type = "PT.asymptotic"); P.12_V
P.24_V = vars::serial.test(VAR.oil, lags.pt = 24, type = "PT.asymptotic"); P.24_V
P.36_V = vars::serial.test(VAR.oil, lags.pt = 36, type = "PT.asymptotic"); P.36_V


# Homocedasticidad ----

# Test tipo ARCH multivariado.
vars::arch.test(VAR.oil, lags.multi = 24, multivariate.only = TRUE)
vars::arch.test(VAR.oil, lags.multi = 12, multivariate.only = TRUE)


# Normalidad ----

# Test Jarque-Bera multivariado.
vars::normality.test(VAR.oil)


# Nota: Como se violan los supuestos de heterocedasticidad y normalidad, hay que
#       calcular los intervalos de confianza mediante bootstrapping para poder
#       hacer inferencia estadistica correcta, tanto en los pronosticos como en
#       las OIRF.


# ===
# 7.3. Pronostico del VECM reparametrizado ====
# ===

# Recuerden que debido al incumplimiento de normalidad, los intervalos de
# confianza deben computarse por bootstrapping.

horizonte_pronostico = 12
int_conf_pronostico = 0.95

prono_VECM = predict(
  VAR.oil,
  n.ahead = horizonte_pronostico,
  ci = int_conf_pronostico
)
prono_VECM

g_pronostico_vecm = graficar_pronostico_vecm(prono_VECM) +
  ggtitle("Pronostico VECM reparametrizado") +
  labs(subtitle = "Horizonte: 12 meses")

x11()
print(g_pronostico_vecm)


# ===
# 7.4. Funciones impulso-respuesta para VECM ====
# ===

# Parametros de las graficas de las IRFs.
pasos_adelante = 0:18
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100

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

# Segun esto, los impulsos de WTI no son significativos en las respuestas de
# Brent ni el mismo WTI.


# ===
# 8. Qué pasa si se cambia el orden de las variables en el VECM? ====
# ===

# Cuando creamos el vector de VAR en niveles, fijamos en la primera variable el
# precio del petroleo Brent. Veamos que sucede si hacemos un modesto cambio.

variables_alt = c("P.WTI", "P.Brent")
Y_alt = ts.intersect(P.WTI = P.WTI, P.Brent = P.Brent)

# Estimemos de misma forma todo el modelo y veamos hasta donde llegamos con los
# impulso-respuesta.

VAR2_alt = vars::VAR(Y_alt, p = 2, type = "const", season = NULL)

P.12_alt = vars::serial.test(VAR2_alt, lags.pt = 12, type = "PT.asymptotic"); P.12_alt
P.24_alt = vars::serial.test(VAR2_alt, lags.pt = 24, type = "PT.asymptotic"); P.24_alt
P.36_alt = vars::serial.test(VAR2_alt, lags.pt = 36, type = "PT.asymptotic"); P.36_alt

# Los supuestos no se alteran significativamente.


# Estimacion VECM ----

# Criterio del valor propio maximo y constante.
eigen_const_alt = urca::ca.jo(
  Y_alt, ecdet = "const", type = "eigen", K = 2, spec = "transitory"
)
summary(eigen_const_alt) # Se mantiene la conclusion


# Reparametrizacion ----

VAR.oil_alt = vars::vec2var(eigen_const_alt, r = 1)

P.12_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 12,
                               type = "PT.asymptotic"); P.12_V_alt
P.24_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 24,
                               type = "PT.asymptotic"); P.24_V_alt
P.36_V_alt = vars::serial.test(VAR.oil_alt, lags.pt = 36,
                               type = "PT.asymptotic"); P.36_V_alt

# Los supuestos del VAR reparametrizado se mantienen similares.


# Impulso-respuesta ----

# Ahora, tenemos el momento decisivo. Veamos que ocurre con las OIRF.

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

# Los impulso-respuesta cambian significativamente. ¿La nocion de esto? Depende
# del contexto economico.

# Lo que muestra la conclusion teorica es que la variable que se coloca primero
# es la mas exogena de las dos. Por lo que, cuando construyan su VAR, coloquen
# la mas exogena arriba.
