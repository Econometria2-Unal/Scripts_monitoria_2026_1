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

# Paquetes del tidyverse para manipulacion y graficacion de datos.
library(tidyverse)

# Paquete principal para estimar modelos VAR y VECM.
library(vars)

# Paquete para pruebas de raiz unitaria y cointegracion.
library(urca)

# Paquete para organizar varias graficas en una misma ventana.
library(gridExtra)

# Paquete para leer archivos .xlsx.
library(readxl)

# Paquetes para manejar rutas relativas de manera reproducible.
library(here)
library(fs)

# Paquete para realizar pronosticos en un VAR usando bootstrap.
library(VAR.etp)


# Cargar bases de datos en R usando rutas relativas ----

# Fijar la ruta del archivo actual como referencia para here().
here::i_am("Sesión 7 - Modelos VAR/codigo/R/Modelos_VAR_ejemplo_Enders.R")

# Directorios principales del proyecto.
directorio_datos = fs::path(here::here("datos"))
directorio_codigo_R = fs::path(here::here("codigo", "R"))

# Rutas de insumos usados por el script.
ruta_enders = fs::path(directorio_datos, "ENDERS.xlsx")

ruta_funciones_auxiliares_var = fs::path(
  directorio_codigo_R,
  "funciones_auxiliares_graficacion_VAR.R"
)

verificar_archivo = function(ruta, descripcion){
  if (!fs::file_exists(ruta)) {
    stop(paste("No se encontro", descripcion, "en la ruta:", ruta),
         call. = FALSE)
  }
  invisible(ruta)
}

verificar_archivo(ruta_enders, "la base ENDERS.xlsx")
verificar_archivo(ruta_funciones_auxiliares_var,
                  "el script de funciones auxiliares")

# Importacion de funciones auxiliares de graficacion del script auxiliar.
source(ruta_funciones_auxiliares_var, encoding = "UTF-8")


# Funciones auxiliares locales ----

abrir_ventana_grafica = function(){
  if (interactive()) {
    grDevices::x11()
  }
}

mostrar_grafico = function(grafico){
  if (interactive()) {
    abrir_ventana_grafica()
    print(grafico)
  }
  invisible(grafico)
}

mostrar_grilla = function(grobs, ncol = NULL, layout_matrix = NULL){
  if (interactive()) {
    abrir_ventana_grafica()
    argumentos = list(grobs = grobs)
    
    if (!is.null(ncol)) {
      argumentos$ncol = ncol
    }
    
    if (!is.null(layout_matrix)) {
      argumentos$layout_matrix = layout_matrix
    }
    
    do.call(gridExtra::grid.arrange, argumentos)
  }
  
  invisible(grobs)
}

probar_raiz_unitaria_adf = function(serie, nombre, lags = 6,
                                    selectlags = "AIC", type = "none"){
  cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Prueba ADF:", nombre, "| especificacion:", type, "\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")
  
  prueba = urca::ur.df(serie, lags = lags, selectlags = selectlags, type = type)
  print(summary(prueba))
  
  invisible(prueba)
}

graficar_diagnostico_serial = function(prueba_serial, variables){
  if (interactive()) {
    purrr::walk(variables, function(variable){
      abrir_ventana_grafica()
      plot(prueba_serial, names = variable)
    })
  }
  
  invisible(prueba_serial)
}

graficar_mts = function(series, titulo, colores = NULL){
  series_df = as.data.frame(series) %>%
    mutate(tiempo = as.numeric(time(series))) %>%
    pivot_longer(cols = -tiempo, names_to = "variable", values_to = "valor")
  
  grafico = series_df %>%
    ggplot(aes(x = tiempo, y = valor, color = variable)) +
    geom_linea_actual(ancho = 0.8) +
    facet_wrap(~ variable, scales = "free_y") +
    theme_light() +
    ggtitle(titulo) +
    xlab("") +
    ylab("") +
    theme(plot.title = element_text(size = 11, hjust = 0.5),
          legend.position = "none")
  
  if (!is.null(colores)) {
    grafico = grafico + scale_color_manual(values = colores)
  }
  
  grafico
}


# ===
# 2. Carga y preparacion de los datos ====
# ===

# La base contiene series trimestrales de Estados Unidos para 1960T1-2012T4:
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
# - dl.IPI aproxima la tasa de crecimiento del indice de produccion industrial.
# - dl.CPI aproxima la inflacion trimestral.
# - Unem se conserva en niveles porque es una tasa.
dl_IPI = diff(log(IPI))
dl_CPI = diff(log(CPI))

# Al tomar diferencias logaritmicas se pierde la primera observacion. Por ello,
# el desempleo se alinea desde 1960T2 hasta 2012T4, que es el periodo comun de
# las tres variables transformadas.
Unem = window(UNEM, start = start(dl_IPI), end = end(dl_IPI))

# Ordenamiento del sistema VAR. Este orden tambien importa para las IRF
# ortogonalizadas, porque la identificacion de Cholesky usa el orden de columnas.
variables = c("dl.IPI", "Unem", "dl.CPI")
Y = ts.intersect(dl.IPI = dl_IPI, Unem = Unem, dl.CPI = dl_CPI)
colnames(Y) = variables

start(Y)
end(Y)
head(Y)
tail(Y)


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

mostrar_grilla(list(g_dl_ipi, g_unem, g_dl_cpi), ncol = 3)


# Pruebas ADF en niveles ----

# Las pruebas en niveles se conservan como referencia diagnostica. La decision
# se toma comparando el estadistico tau con los valores criticos reportados por
# summary(). Para modelos VAR en niveles se requiere estacionariedad conjunta; si
# alguna serie no es estacionaria, conviene transformar o revisar cointegracion.
adf_ipi_nivel = probar_raiz_unitaria_adf(
  IPI, "IPI en nivel", lags = 6, selectlags = "AIC", type = "trend"
)

adf_cpi_nivel = probar_raiz_unitaria_adf(
  CPI, "CPI en nivel", lags = 6, selectlags = "AIC", type = "trend"
)

adf_unem_nivel = probar_raiz_unitaria_adf(
  UNEM, "desempleo en nivel", lags = 6, selectlags = "AIC", type = "drift"
)


# Pruebas ADF sobre las variables que entran al VAR ----

# En el sistema final se usan crecimiento del IPI, desempleo e inflacion. Esta
# es la version que debe pasar el chequeo de estacionariedad antes de estimar el
# VAR en forma reducida.
adf_dl_ipi = probar_raiz_unitaria_adf(
  Y[, "dl.IPI"], "crecimiento logaritmico del IPI",
  lags = 6, selectlags = "AIC", type = "none"
)

adf_dl_cpi = probar_raiz_unitaria_adf(
  Y[, "dl.CPI"], "inflacion logaritmica del CPI",
  lags = 6, selectlags = "AIC", type = "trend"
)

adf_unem = probar_raiz_unitaria_adf(
  Y[, "Unem"], "tasa de desempleo",
  lags = 6, selectlags = "AIC", type = "drift"
)


# ===
# 4. Metodologia Box-Jenkins para series multivariadas ====
# ===

# El sistema a estimar es un VAR(p) sobre:
#   Y_t = (dl.IPI_t, Unem_t, dl.CPI_t)'
#
# La metodologia sigue cuatro pasos: identificacion, estimacion, validacion y
# uso del modelo para pronostico e impulso-respuesta.


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

# En el ejemplo de Enders se trabaja con p = 3. Los criterios de informacion y
# la inspeccion de los residuales deben usarse conjuntamente: un VAR muy corto
# puede dejar autocorrelacion, mientras que un VAR excesivamente largo consume
# grados de libertad.
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

# Seleccion del modelo de trabajo. Se conserva la eleccion del ejemplo original:
# VAR(3) con constante. La constante es razonable porque las variables
# transformadas pueden tener medias distintas de cero.
modelo_var_enders = V.dr.1

# Estabilidad del VAR: las raices reportadas por roots() deben estar dentro del
# circulo unitario para que el VAR sea estable.
raices_modelo = vars::roots(modelo_var_enders)
raices_modelo

# Coeficientes estimados por ecuacion. Acoef() separa las matrices A_1, A_2 y A_3.
coeficientes_var = vars::Acoef(modelo_var_enders)
coeficientes_var

# Matriz de varianzas y covarianzas estimada de los residuales en forma reducida.
Sigma.e = summary(modelo_var_enders)$covres
Sigma.e


# ===
# 4.3. Validacion de supuestos ====
# ===

# No autocorrelacion serial ----

# PT.asymptotic se usa para muestras grandes. Si la muestra es pequena, tambien
# puede revisarse type = "PT.adjusted".
P.50.1 = vars::serial.test(
  modelo_var_enders, lags.pt = 50, type = "PT.asymptotic"
)
P.50.1

P.30.1 = vars::serial.test(
  modelo_var_enders, lags.pt = 30, type = "PT.asymptotic"
)
P.30.1

P.20.1 = vars::serial.test(
  modelo_var_enders, lags.pt = 20, type = "PT.asymptotic"
)
P.20.1

# Graficas de diagnostico de residuales por variable: residuales, distribucion,
# ACF/PACF y ACF/PACF de residuales al cuadrado.
graficar_diagnostico_serial(P.20.1, variables)


# Homocedasticidad ----

# H0 del test ARCH multivariado: no hay efectos ARCH en los residuales.
arch_24 = vars::arch.test(
  modelo_var_enders, lags.multi = 24, multivariate.only = TRUE
)
arch_24

arch_12 = vars::arch.test(
  modelo_var_enders, lags.multi = 12, multivariate.only = TRUE
)
arch_12


# Normalidad ----

# H0 del Jarque-Bera multivariado: los residuales tienen distribucion normal.
normalidad_residuos = vars::normality.test(modelo_var_enders)
normalidad_residuos

# Nota didactica: en aplicaciones macroeconomicas es frecuente encontrar
# rechazo de normalidad u homocedasticidad. Esto no invalida automaticamente el
# VAR como herramienta descriptiva, pero si debe tenerse presente al interpretar
# intervalos y pruebas.


# ===
# 4.4. Pronostico y funciones impulso-respuesta ====
# ===

# Pronostico ----

horizonte_pronostico = 12
int_conf_pronostico = 0.95

pronostico_var = predict(
  modelo_var_enders,
  n.ahead = horizonte_pronostico,
  ci = int_conf_pronostico
)
pronostico_var

# Funcion auxiliar importada desde funciones_auxiliares_graficacion_VAR.R.
g_pronostico_var = graficar_pronostico_var(pronostico_var) +
  ggtitle("Pronostico VAR - ejemplo de Enders") +
  labs(subtitle = "Horizonte: 12 trimestres")

mostrar_grafico(g_pronostico_var)

# Version fanchart de vars.
if (interactive()) {
  abrir_ventana_grafica()
  vars::fanchart(
    predict(modelo_var_enders, n.ahead = horizonte_pronostico),
    colors = c("blue", "lightblue")
  )
}


# Pronostico por bootstrapping ----

# El script original usa VAR.etp para construir pronosticos bootstrap. Se fija
# semilla para que el resultado sea reproducible en clase.
semilla_bootstrap_pronostico = 202601
repeticiones_bootstrap_pronostico = 1000
set.seed(semilla_bootstrap_pronostico)

For.Boot = VAR.etp::VAR.BPR(
  Y,
  p_var,
  horizonte_pronostico,
  nboot = repeticiones_bootstrap_pronostico,
  type = "const",
  alpha = int_conf_pronostico
)
For.Boot

boots = For.Boot$Forecast
boots

# Como Y termina en 2012T4, el primer pronostico corresponde a 2013T1.
if (is.null(colnames(boots))) {
  colnames(boots) = variables
}

boots_forecast = ts(boots, start = c(2013, 1), frequency = 4)

g_bootstrap = graficar_mts(
  boots_forecast,
  titulo = "Pronostico puntual con bootstrapping",
  colores = c("dl.IPI" = "lightblue",
              "Unem" = "mediumpurple2",
              "dl.CPI" = "sienna1")
) +
  labs(subtitle = "Horizonte: 2013T1-2015T4")

mostrar_grafico(g_bootstrap)


# Funciones de impulso-respuesta no ortogonalizadas ----

# Para calcular IRF, el VAR debe admitir una representacion VMA(infinito).
# En la practica esto se revisa con la estabilidad del VAR estimado.
Phi(modelo_var_enders, nstep = 10)

pasos_adelante = 0:24
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez y luego
# genera automaticamente las 9 graficas: columnas = impulsos; filas = respuestas.
irf_no_ortog = graficar_grilla_irf(
  modelo_var_enders,
  variables,
  pasos_adelante,
  ortog = FALSE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

mostrar_grilla(
  irf_no_ortog$graficas,
  layout_matrix = matrix(seq_along(irf_no_ortog$graficas),
                         nrow = length(variables), byrow = TRUE)
)


# Funciones de impulso-respuesta ortogonalizadas ----

# Cuando ortog = TRUE, irf() usa Cholesky sobre la matriz de covarianzas de los
# residuales. Con el orden definido arriba, dl.IPI es la variable
# contemporaneamente mas exogena, luego Unem y finalmente dl.CPI. Este supuesto
# debe justificarse economicamente antes de interpretar las OIRF como choques
# estructurales.
Psi(modelo_var_enders, nstep = 10)

irf_ortog = graficar_grilla_irf(
  modelo_var_enders,
  variables,
  pasos_adelante,
  ortog = TRUE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso ortogonal",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)

mostrar_grilla(
  irf_ortog$graficas,
  layout_matrix = matrix(seq_along(irf_ortog$graficas),
                         nrow = length(variables), byrow = TRUE)
)


# ===
# 4.5. Descomposicion de varianza del error de pronostico ====
# ===

# La FEVD resume que proporcion de la varianza del error de pronostico de cada
# variable se atribuye a los choques de cada variable del sistema.
horizonte_fevd = 24
fevd_enders = vars::fevd(modelo_var_enders, n.ahead = horizonte_fevd)
fevd_enders

if (interactive()) {
  abrir_ventana_grafica()
  plot(fevd_enders, col = c("magenta4", "cyan3", "slateblue3"))
}
