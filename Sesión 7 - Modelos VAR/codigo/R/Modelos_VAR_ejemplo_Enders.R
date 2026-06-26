#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 7: Modelos de vectores autorregresivos - Ejemplo de Enders
#'
#' Semestre: 2026-1


# ===
# Tabla de contenidos ===
# ===

#' 1. Importacion de paquetes, rutas y funciones auxiliares
#' 2. Carga y preparacion de los datos
#' 3. Analisis grafico y pruebas de estacionariedad
#' 4. Metodologia Box-Jenkins para series multivariadas
#'  4.1. Identificacion
#'  4.2. Estimacion
#'  4.3. Validacion de supuestos
#'  4.4. Pronostico y funciones impulso-respuesta
#'  4.5. Descomposicion de varianza del error de pronostico


# Nota: Tips practicos en R
## Para limpiar el entorno de trabajo se puede correr: rm(list = ls())
## Para cerrar todas las graficas actualmente abiertas se puede correr: dev.off()
## Para resetear R se puede usar las teclas: Ctrl + Shift + F10


# ===
# 1. Importacion de paquetes, rutas y funciones auxiliares ====
# ===

# Paquetes del tidyverse (Para el manejo, manipulación y graficación de datos)
library(tidyverse)

# Paquete principal para estimar modelos VAR y VECM.
library(vars)

# Paquete para realizar pronosticos en un VAR usando bootstrap.
library(VAR.etp)

# Paquete para pruebas de raiz unitaria y cointegracion.
library(urca)

# Paquete para organizar varias graficas en una misma ventana.
library(gridExtra)

# Paquete para leer archivos .xlsx.
library(readxl)

# Paquetes para manejar rutas relativas en R
library(here)
library(fs)


# Cargar bases de datos en R usando rutas relativas ----

# Fijar la ruta del archivo actual como referencia para here().
here::i_am("Sesión 7 - Modelos VAR/codigo/R/Modelos_VAR_ejemplo_Enders.R")

# Directorios principales del proyecto.
directorio_sesion_var = fs::path(here::here("Sesión 7 - Modelos VAR"))
directorio_datos = fs::path(directorio_sesion_var, "datos")
directorio_codigo_R = fs::path(directorio_sesion_var, "codigo", "R")

# Ruta donde se encuentra base de datos del Enders (con las variables de interés)
ruta_enders = fs::path(directorio_datos, "ENDERS.xlsx")


# Importación de funciones auxiliares de graficación

# Ruta con las funciones auxiliares de graficación 
ruta_funciones_auxiliares_var = fs::path(
  directorio_codigo_R,
  "funciones_auxiliares_graficacion_VAR.R"
)

# script auxiliar "funciones_auxiliares_graficacion_VAR.R"
source(ruta_funciones_auxiliares_var, encoding = "UTF-8")


# ===
# 2. Carga y preparacion de los datos ====
# ===

# La base de datos de Enders contiene series trimestrales de Estados Unidos
# para 1960T1-2012T4:
  # IPI  = indice de produccion industrial
  # CPI  = indice de precios al consumidor
  # Unem = tasa de desempleo
Base = readxl::read_excel(ruta_enders)
glimpse(Base)

# Series en niveles.
IPI = ts(Base$IPI, start = c(1960, 1), frequency = 4)
CPI = ts(Base$CPI, start = c(1960, 1), frequency = 4)
UNEM = ts(Base$Unem, start = c(1960, 1), frequency = 4)

# Transformaciones usadas por Enders:
  # dl.IPI: aproxima la tasa de crecimiento del indice de produccion industrial.
  # dl.CPI: aproxima la inflacion trimestral.
  # Unem: se conserva en niveles porque es una tasa.
dl_IPI = diff(log(IPI))
dl_CPI = diff(log(CPI))

# Al tomar diferencias logaritmicas se pierde la primera observacion. Por ello,
# el desempleo se alinea desde 1960T2 hasta 2012T4, que es el periodo comun de
# las tres variables transformadas.
Unem = window(UNEM, start = start(dl_IPI), end = end(dl_IPI))

# Nota: Se ordenan las variables del modelo VAR. Este orden es importante, porque 
# determina el orden de las exogenidad de las variables, que será muy importante
# a la hora de graficar de construir las IRF ortogonalizadas, porque la 
# identificacion de Cholesky usa el orden de las columnas de la Matriz Y, 
# para determinar cuáles son las variables más exógenas. La variable en la primera
# columna de Y será la más exógena, mientras que la variable en la última columna
# será la más endógena. 

# Variables que se modelaran mediante el VAR
variables = c("dl.IPI", "Unem", "dl.CPI")

# Se construye la matriz Y, que contiene las series de tiempo del modelo VAR
Y = ts.intersect(dl.IPI = dl_IPI, Unem = Unem, dl.CPI = dl_CPI)

# Se nombras las columnas de las matriz, con las series de tiempo
colnames(Y) = variables

# Algunas característica de las series de tiempo del modelO VAR
start(Y) # Periodo donde inician las series
end(Y) # Periodo donde terminan las series
head(Y) # Observaciones iniciales
tail(Y) # Observaciones finales


# ===
# 3. Analisis grafico y pruebas de estacionariedad ====
# ===

# Graficas de las series transformadas que entraran al VAR.
g_dl_ipi = graficar_ts(
  Y[, "dl.IPI"],
  titulo = "Crecimiento logaritmico del IPI",
  color = "lightblue"
)

g_unem = graficar_ts(
  Y[, "Unem"],
  titulo = "Tasa de desempleo",
  color = "mediumpurple2"
)

g_dl_cpi = graficar_ts(
  Y[, "dl.CPI"],
  titulo = "Inflacion logaritmica del CPI",
  color = "sienna1"
)

x11()
grid.arrange(g_dl_ipi, g_unem, g_dl_cpi, ncol = 3)


# Pruebas ADF en niveles ----

# Nota: Para aplicar un modelo VAR en niveles, todas las variables tiene que ser
#       estacionarias, entonces se verifica que en efecto las variables sean
#       estacionarias! En caso de tener variables no estacionarias, toca tratar
#       las series, ya sea diferenciadolas o haciendo pruebas de cointegración

adf_ipi_nivel = urca::ur.df(IPI, lags = 6, selectlags = "AIC", type = "trend")
summary(adf_ipi_nivel) # No rechazo: La serie no es estacionaria

adf_cpi_nivel = urca::ur.df(CPI, lags = 6, selectlags = "AIC", type = "trend")
summary(adf_cpi_nivel) # No rechazo: La serie no es estacionaria

adf_unem_nivel = urca::ur.df(UNEM, lags = 6, selectlags = "AIC", type = "drift")
summary(adf_unem_nivel) # Rechazo: La serie no es estacionaria


# Pruebas ADF sobre las variables que entran al VAR ----

# En el VAR se usan la tasa de crecimiento del IPI, la tasa de desempleo y la inflacion. 
# Se verifican que estas variables sean estacionarias

adf_dl_ipi = urca::ur.df(Y[, "dl.IPI"], lags = 6, selectlags = "AIC",
                         type = "drift")
summary(adf_dl_ipi) # Rechazo: La serie no es estacionaria

adf_dl_cpi = urca::ur.df(Y[, "dl.CPI"], lags = 6, selectlags = "AIC",
                         type = "drift")
summary(adf_dl_cpi) # Rechazo: La serie no es estacionaria

adf_unem = urca::ur.df(Y[, "Unem"], lags = 6, selectlags = "AIC",
                       type = "drift")
summary(adf_unem) # Rechazo: La serie no es estacionaria


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


# ===
# 4.1. Identificacion ====
# ===

# Seleccion de rezagos para un VAR con tendencia e intercepto.
seleccion_rezagos_both = vars::VARselect(
  Y, lag.max = 6, type = "both", season = NULL
)
seleccion_rezagos_both

# Seleccion de rezagos para un VAR con solo intercepto.
seleccion_rezagos_const = vars::VARselect(
  Y, lag.max = 6, type = "const", season = NULL
)
seleccion_rezagos_const

# Seleccion de rezagos para un VAR sin terminos deterministas.
seleccion_rezagos_none = vars::VARselect(
  Y, lag.max = 6, type = "none", season = NULL
)
seleccion_rezagos_none

# En el ejemplo de Enders se trabaja con p = 3. A la hora de seleccionar el número de rezagos,
# Los criterios de informacion y la inspeccion de los residuales deben usarse conjuntamente: 
# un VAR muy corto puede dejar autocorrelacion, mientras que un VAR excesivamente 
# largo consume grados de libertad.
p_var = 3

# ===
# 4.2. Estimacion ====
# ===

# VAR con tendencia e intercepto.
V.tr.1 = vars::VAR(Y, p = p_var, type = "both", season = NULL)
summary(V.tr.1)

# VAR con intercepto.
V.dr.1 = vars::VAR(Y, p = p_var, type = "const", season = NULL)
summary(V.dr.1)

# VAR sin terminos deterministas.
V.no.1 = vars::VAR(Y, p = p_var, type = "none", season = NULL)
summary(V.no.1)

# Se trabajará con un VAR(3) con constante. La constante es razonable porque las
# variables transformadas pueden tener medias distintas de cero.
VAR_enders = V.dr.1

# Estabilidad del VAR(3):

# Nota: En la práctica R transforma un VAR(3) en un VAR(1) usando la matriz de
#       compañía. Luego mira que los valores propios de la matriz A_1 del VAR(1)
# resultante sean en valor absoluto menores a 1 para que el VAR(1) sea estacionario,
# con ello se garantiza que sus inversos múltiplicativos, que son las raíces del 
# polinomio característico asociado al proceso VAR, sean en valor absoluto mayores 
# a 1 y se pueda garantizar la estabilidad del proceso.
roots(VAR_enders) # El proceso es estable

# Coeficientes estimados

# Coeficientes estimados por ecuacion. Acoef() separa las matrices A_1, A_2 y A_3.
coeficientes_var = Acoef(VAR_enders); coeficientes_var

# Matriz de varianzas y covarianzas

# Matriz de varianzas y covarianzas estimada de los residuales en forma reducida.
Sigma.e = summary(VAR_enders)$covres
Sigma.e


# Análisis de todos el modelo VAR(3) estimado
summary(VAR_enders)

# ===
# 4.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ----

# PT.asymptotic es para muestra grande y "PT.adjusted" para muestra pequeña.
P.50=serial.test(VAR_enders, lags.pt = 50, type = "PT.asymptotic"); P.50 # No rechazo
P.30=serial.test(VAR_enders, lags.pt = 30, type = "PT.asymptotic"); P.30 # No rechazo
P.20=serial.test(VAR_enders, lags.pt = 20, type = "PT.asymptotic"); P.20 # No rechazo
P.10=serial.test(VAR_enders, lags.pt = 10, type = "PT.asymptotic"); P.10 # No rechazo

# Graficamos los resultados del test usando 20 residuos: Se grafican los residuales, 
# su distribución, la ACF y PACF de los residuales y a ACF y PACF de los 
# residuales al cuadrado (proxy para heterocedasticidad)
x11()
plot(P.20, names = "dl.IPI")

x11()
plot(P.20, names = "Unem")

x11()
plot(P.20, names = "dl.CPI")

# Nota: Se cumple el supuesto de no autocorrelación serial en los residuales

# Homocedasticidad ----

# Test tipo ARCH multivariado
arch.test(VAR_enders, lags.multi = 24, multivariate.only = TRUE); arch_24 # No rechazo
arch.test(VAR_enders, lags.multi = 12, multivariate.only = TRUE); arch_12 # No rechazo

# Nota: Se cumple el supuesto de homocedasticidad

# Normalidad ----

# H0 del Jarque-Bera multivariado: los residuales tienen distribucion normal.
normality.test(VAR_enders) # No rechazo, se cumple el supuesto.

# Nota: Se cumple el supuesto de normalidad

# ===
# 4.4. Pronostico y funciones impulso-respuesta ====
# ===

# Pronostico ----

# Especificaciones del pronóstico
horizonte_pronostico = 12
int_conf_pronostico = 0.95

# Pronóstico modelo VAR
pronostico_var = predict(
  VAR_enders,
  n.ahead = horizonte_pronostico,
  ci = int_conf_pronostico
)
pronostico_var

# Gráficar pronóstico
g_pronostico_var = graficar_pronostico_var(pronostico_var) +
  ggtitle("Pronostico VAR - ejemplo de Enders") +
  labs(subtitle = "Horizonte: 12 trimestres")

x11()
print(g_pronostico_var)

# Version fanchart de vars.
x11()
vars::fanchart(
  predict(VAR_enders, n.ahead = horizonte_pronostico),
  colors = c("blue", "lightblue")
)


# Pronostico por bootstrapping ----

# Pronóstico usando el comando VAR.etp para construir pronosticos usando bootstrap. 

# Especificaciones del pronóstico usando bootstrap
repeticiones_bootstrap_pronostico = 1000
set.seed(202601)

# Comando para generar los pronósticos por medio de bootstrap
For.Boot = VAR.etp::VAR.BPR(
  Y,
  p_var,
  horizonte_pronostico,
  nboot = repeticiones_bootstrap_pronostico,
  type = "const",
  alpha = int_conf_pronostico
); For.Boot # Objeto que contiene los pronósticos usando bootstrap! 

# Pronósticos de bootstrap
boots = For.Boot$Forecast
boots

# Como Y termina en 2012T4, el primer pronostico corresponde a 2013T1.
if (is.null(colnames(boots))) {
  colnames(boots) = variables
}

boots_forecast = ts(boots, start = c(2013, 1), frequency = 4)

boots_forecast_df = as.data.frame(boots_forecast) %>%
  mutate(tiempo = as.numeric(time(boots_forecast))) %>%
  pivot_longer(cols = -tiempo, names_to = "variable", values_to = "valor")

# Graficas para el pronóstico calculado usando bootstrap
g_bootstrap = boots_forecast_df %>%
  ggplot(aes(x = tiempo, y = valor, color = variable)) +
  geom_linea_actual(ancho = 0.8) +
  facet_wrap(~ variable, scales = "free_y") +
  scale_color_manual(values = c("dl.IPI" = "lightblue",
                                "Unem" = "mediumpurple2",
                                "dl.CPI" = "sienna1")) +
  theme_light() +
  ggtitle("Pronostico puntual con bootstrapping") +
  labs(subtitle = "Horizonte: 2013T1-2015T4")

x11()
print(g_bootstrap)


# Funciones de impulso-respuesta no ortogonalizadas ----

# Para calcular IRF, el VAR debe admitir una representacion VMA(infinito).
# En la practica esto se revisa con la estabilidad del VAR estimado.
Phi(VAR_enders, nstep = 10)

pasos_adelante = 0:24
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez y luego
# genera automaticamente las 9 graficas: columnas = impulsos; filas = respuestas.
irf_no_ortog = graficar_grilla_irf(
  VAR_enders,
  variables,
  pasos_adelante,
  ortog = FALSE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

x11()
grid.arrange(grobs = irf_no_ortog$graficas,
             layout_matrix = matrix(seq_along(irf_no_ortog$graficas),
                                    nrow = length(variables), byrow = TRUE))


# Funciones de impulso-respuesta ortogonalizadas ----

# Cuando ortog = TRUE, irf() usa Cholesky sobre la matriz de covarianzas de los
# residuales. Con el orden definido arriba, dl.IPI es la variable
# contemporaneamente mas exogena, luego Unem y finalmente dl.CPI. Este supuesto
# debe justificarse economicamente antes de interpretar las OIRF como choques
# estructurales.
Psi(VAR_enders, nstep = 10)

irf_ortog = graficar_grilla_irf(
  VAR_enders,
  variables,
  pasos_adelante,
  ortog = TRUE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso ortogonal",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

x11()
grid.arrange(grobs = irf_ortog$graficas,
             layout_matrix = matrix(seq_along(irf_ortog$graficas),
                                    nrow = length(variables), byrow = TRUE))


# ===
# 4.5. Descomposicion de varianza del error de pronostico ====
# ===

# La FEVD resume que proporcion de la varianza del error de pronostico de cada
# variable se atribuye a los choques de cada variable del sistema.
horizonte_fevd = 24
fevd_enders = vars::fevd(VAR_enders, n.ahead = horizonte_fevd)
fevd_enders

x11()
plot(fevd_enders, col = c("magenta4", "cyan3", "slateblue3"))
