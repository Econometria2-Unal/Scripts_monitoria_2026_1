# Importación de paquetes ---

# Trabajar con rutas relativas en R
library(fs)
library(here)

# Paquetes del tidyverse (Para el manejo, manipulación y graficación de datos)
library(readr)
library(dplyr)
library(ggplot2)
library(ggtime)

# Paquetes del tidyverts (Para un manejo moderno de series de tiempo en R)
library(tsibble)
library(feasts)
library(fable)

# Paquetes adicionales para trabajar con series de tiempo en R
library(tseries)
library(FinTS)
library(lmtest)
library(urca) # Test de raíz unitaria

# Para que la función ARIMA que se use por defecto sea la de fable.
ARIMA <- fable::ARIMA

# Cargar bases de datos en R usando rutas relativas ---

# Fijar la ruta del archivo actual como referencia para here()
here::i_am("Sesión 3 - Metodología Box Jenkins/codigo/R/Box_jenkins_ej1_expo_tradicionales.R")

# Obtener la ruta del directorio con los datos
directorio <- fs::path(here::here("Sesión 3 - Metodología Box Jenkins", "datos"))

# Rutas de las bases de datos
ruta_exp <- fs::path(directorio, "Expotradicionales1990-2017.csv") # Base de datos de exportaciones

# Funciones auxiliares ---

# Función auxiliar para mostrar gráficos en una grilla m x n
grilla <- function(..., nrow, ncol) {
  graficos <- list(...)

  if (length(graficos) > nrow * ncol) {
    stop("La cantidad de gráficos supera el tamaño de la grilla.")
  }

  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(nrow = nrow, ncol = ncol)))

  for (i in seq_along(graficos)) {
    fila <- ceiling(i / ncol)
    columna <- ((i - 1) %% ncol) + 1

    print(
      graficos[[i]],
      vp = grid::viewport(layout.pos.row = fila, layout.pos.col = columna)
    )
  }

  grid::popViewport()
}

# ===
# PRIMERA SERIE: EXPORTACIONES TRADICIONALES ====
# ===

# Base de datos con la serie importada a R
expo_base <- read_csv(
  ruta_exp,
  col_names = "expo_tradicionales",
  show_col_types = FALSE
)

# Ver el tipo de objeto de la base de datos (tibble/data.frame)
print(class(expo_base))

# Ver primeras y últimas observaciones de la base de datos
print(head(expo_base)) # Primeras observaciones
print(tail(expo_base)) # Últimas observaciones

# Creación de la serie de tiempo de "exportaciones" ---

# Creación del índice temporal de la serie de tiempo ---
fechas_expo_base <- yearmonth(seq(
  from = as.Date("1990-01-01"),
  by = "month",
  length.out = nrow(expo_base)
))

# Agregar el índice temporal a la base de datos de exportaciones tradicionales
expo_serie <- expo_base |>
  mutate(fecha = fechas_expo_base) |>
  as_tsibble(index = fecha)

# El tipo de objeto de la base de datos ahora es un tsibble
# Los objetos tsibble son los que funcionan con el paquete fable 
print(class(expo_serie))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(head(expo_serie)) # Primeras observaciones
print(tail(expo_serie)) # Últimas observaciones

# La nueva serie de tiempo se va a llamar "expo_serie" y va a tener valores numéricos
expo_serie <- expo_serie |>
  mutate(expo_tradicionales = as.numeric(expo_tradicionales)) |>
  filter(!is.na(expo_tradicionales)) # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(head(expo_serie))
print(tail(expo_serie))

# El tipo de objeto de la base de datos ahora es un tsibble con la serie limpia
print(class(expo_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo
summary(expo_serie$expo_tradicionales)

# ===
# Paso 1: Identificación ====
# ===

# Gráfica de la serie de tiempo "exportaciones tradicionales"
print(
  ggtime::autoplot(expo_serie, expo_tradicionales, linewidth = 0.7) +
    ggtitle("Exportaciones tradicionales, 1990-2017") +
    xlab("Fecha") +
    ylab("Valor") +
    theme_light()
)


# FAC y FACP de la serie original ---

grafico_fac_original <- expo_serie |>
  ACF(expo_tradicionales, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FAC exportaciones tradicionales") +
  ylim(-1, 1) +
  theme_light()

grafico_facp_original <- expo_serie |>
  PACF(expo_tradicionales, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FACP de la serie original") +
  ylim(-1, 1) +
  theme_light()

grilla(grafico_fac_original, grafico_facp_original, nrow = 1, ncol = 2)


# Tests de Raíz Unitaria ---

# Test de Augmented Dickey Fuller (ADF)
adf_result <- ur.df(
  expo_serie$expo_tradicionales,
  type = "drift",
  selectlags = "AIC"
)

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria.
#       Con urca, se rechaza H0 si el estadístico es menor que el valor crítico.

cat("=== Test ADF ===\n")
print(summary(adf_result))

adf_stat <- adf_result@teststat[1, "tau2"]
adf_critico_5 <- adf_result@cval["tau2", "5pct"]

if (adf_stat < adf_critico_5) {
  cat("ADF: Rechazamos H0. Según el test, la serie es estacionaria.\n")
} else {
  cat("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.\n")
}

# Test KPSS
kpss_result <- ur.kpss(
  expo_serie$expo_tradicionales,
  type = "mu",
  lags = "short"
)

# Nota: La prueba KPSS se interpreta al contrario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria.
#       Con urca, se rechaza H0 si el estadístico es mayor que el valor crítico.

cat("\n=== Test KPSS ===\n")
print(summary(kpss_result))

kpss_stat <- kpss_result@teststat[1]
kpss_critico_5 <- kpss_result@cval["critical values", "5pct"]

if (kpss_stat > kpss_critico_5) {
  cat("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.\n")
} else {
  cat("KPSS: no rechazamos H0. Según el test, la serie es estacionaria.\n")
}

# Nota: Según los resultados de la prueba ADF y KPSS, hay que
#       diferenciar la serie.


# Serie diferenciada ---

expo_serie <- expo_serie |>
  mutate(diff_expo = difference(expo_tradicionales))

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
print(
  expo_serie |>
    filter(!is.na(diff_expo)) |>
    ggtime::autoplot(diff_expo) +
    ggtitle("Serie diferenciada") +
    xlab("Fecha") +
    ylab("Valor") + 
    theme_light()
)

# Diferencia del logaritmo ---

expo_serie <- expo_serie |>
  mutate(
    log_expo = log(expo_tradicionales),
    log_diff = difference(log_expo)
  )

# Gráfica de la serie de tiempo de la "diferencia del logaritmo de exportaciones tradicionales"
print(
  expo_serie |>
    filter(!is.na(log_diff)) |>
    ggtime::autoplot(log_diff) +
    ggtitle("Diferencia del logaritmo de la serie original") +
    xlab("Fecha") +
    ylab("Valor") + 
    theme_light()
)


# %% FAC y FACP de la diferencia del logaritmo ---

grafico_fac_log_diff <- expo_serie |>
  filter(!is.na(log_diff)) |>
  ACF(log_diff, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FAC de la diferencia del logaritmo") +
  ylim(-1, 1) +
  theme_light()

grafico_facp_log_diff <- expo_serie |>
  filter(!is.na(log_diff)) |>
  PACF(log_diff, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FACP de la diferencia del logaritmo") +
  ylim(-1, 1) +
  theme_light()

grilla(grafico_fac_log_diff, grafico_facp_log_diff, nrow = 1, ncol = 2)

# Identificación del modelo usando FAC y FACP ---

# La FAC muestra que solo la primera autocorrelación es significativa.
# La FACP decae rápidamente.
# Por tanto, se propone inicialmente un modelo MA(1) sobre la serie
# transformada, equivalente a un ARIMA(0, 1, 1) sobre log(expo_serie).

# P.d. También se pueden usar criterios de información para la
#      Selección de los órdenes p y q del modelo ARIMA.
#      Para acá se usó el criterio de la FAC y la FACP.

# ===
# Paso 2: Estimación =========================
# ===

# En fable la especificación del modelo se hace dentro de model().
# Acá se estima un ARIMA(0, 1, 1) sin constante sobre log(expo_tradicionales),
# equivalente a un MA(1) sobre la diferencia del logaritmo.

# En fable es muy importante tener encuenta lo siguiente 
# 1. Hay que especificar el orden (p,d,q) del modelo, o sino el lo selecciona
#    automáticamente (lo cuál no es válido en el taller).
# 2. Hay que especificar PDQ(0,0,0), o sino de lo contrario, el estimará un modelo
#    SARIMA de manera automática. Con PDQ(0,0,0) estima un modelo ARIMA puro. 
#    Si quieren estimar un modelo SARIMA, deben espeficiar los ordenes del PDQ
#    pero deben justificar porque eso ordenes.

fit_ma1 <- expo_serie |>
  select(fecha, expo_tradicionales) |>
  model(ma1 = fable::ARIMA(log(expo_tradicionales) ~ 0 + pdq(0, 1, 1) + PDQ(0, 0, 0)))

# Nota: Como la transformación log se realizó dentro del comando ARIMA, ya no es
#       necesario hacer la transformación inversa de la exponencial a la hora de
#       hacer el pronóstico, el comando fable hará automáticamente la transforamción 
#       inversa (de la exponencial) cuando haga el pronóstico. 

# Objeto tipo "mdl_df" del paquete fable
print(class(fit_ma1))

# Nota: El método de estimación es máxima verosimilitud
#       sobre la representación del modelo en un espacio de estados.

# Se imprimen los resultados principales de la estimación
# E.g. Coeficientes y significancia de los coeficientes
report(fit_ma1)

# ===
# Paso 3: Validación de supuestos =========================
# ===

# Residuales del modelo
residuales_tbl <- fit_ma1 |>
  residuals() |>
  filter(!is.na(.resid))

residuales <- residuales_tbl$.resid

# Los residuales son un objeto tipo numérico
print(class(residuales))

# Gráfica de los residuales (deberían comportarse como ruido blanco)
print(
  residuales_tbl |>
    ggtime::autoplot(.resid) +
    ggtitle("Residuales del modelo MA(1)") +
    xlab("Fecha") +
    ylab("Valor") +
    theme_light()
)

# Descripción de los residuales
summary(residuales)


# FAC y FACP de los residuales ---

grafico_fac_residuales <- residuales_tbl |>
  ACF(.resid, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FAC de los residuales") +
  ylim(-1, 1) +
  theme_light()

grafico_facp_residuales <- residuales_tbl |>
  PACF(.resid, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FACP de los residuales") +
  ylim(-1, 1) +
  theme_light()

grilla(grafico_fac_residuales, grafico_facp_residuales, nrow = 1, ncol = 2)

# Prueba Ljung-Box ---

cat("Prueba Ljung-Box\n")

rezagos_ljung_box <- c(6, 12, 18, 24)

prueba_ljung_box <- lapply(rezagos_ljung_box, function(lag) {
  prueba <- Box.test(residuales, lag = lag, type = "Ljung-Box", fitdf = 1)

  data.frame(
    lb_stat = as.numeric(prueba$statistic),
    lb_pvalue = prueba$p.value
  )
}) |>
  bind_rows()

rownames(prueba_ljung_box) <- rezagos_ljung_box

print(round(prueba_ljung_box, 6))

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.

# FAC y FACP de los residuales al cuadrado ---

residuales_tbl <- residuales_tbl |>
  mutate(residuo_cuadrado = .resid^2)

grafico_fac_residuales_cuadrado <- residuales_tbl |>
  ACF(residuo_cuadrado, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FAC de los residuales al cuadrado") +
  ylim(-1, 1) +
  theme_light()

grafico_facp_residuales_cuadrado <- residuales_tbl |>
  PACF(residuo_cuadrado, lag_max = 24) |>
  ggtime::autoplot() +
  ggtitle("FACP de los residuales al cuadrado") +
  ylim(-1, 1) +
  theme_light()

grilla(grafico_fac_residuales_cuadrado, grafico_facp_residuales_cuadrado, nrow = 1, ncol = 2)

# Prueba ARCH de heterocedasticidad ---

arch_test <- ArchTest(residuales, lags = 12)

cat("Prueba ARCH\n")
print(arch_test)

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.

# Q-Q plot de los residuos ---

# Función qqplot del paquete car
car::qqPlot(
  residuales,
  main = "Q-Q plot de los residuos",
  xlab = "Cuantiles teóricos",
  ylab = "Residuales"
)

# Prueba de normalidad Jarque-Bera ---

jb_test <- jarque.bera.test(residuales)

cat("Prueba Jarque-Bera\n")
print(jb_test)

# H0: los residuos no siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.

# ===
# PASO 4: Pronóstico =========================
# ===

# A partir de la estimación del MA(1) se realiza el pronóstico

# Pronóstico 12 pasos adelante
pronostico <- fit_ma1 |>
  forecast(h = 12)

# Pronóstico puntual e intervalos de predicción en niveles.
# fable revierte automáticamente la transformación log() en forecast().
tabla_pronostico <- pronostico |>
  as_tibble() |>
  select(fecha, .mean) |>
  rename(pronostico = .mean)

intervalos <- pronostico |>
  hilo(level = 95) |>
  fabletools::unpack_hilo(`95%`) |>
  as_tibble() |>
  select(fecha, `95%_lower`, `95%_upper`) |>
  rename(limite_inferior = `95%_lower`, limite_superior = `95%_upper`)

tabla_pronostico <- left_join(tabla_pronostico, intervalos, by = "fecha")

print(tabla_pronostico)

# Gráfica del pronóstico
print(
  ggtime::autoplot(expo_serie, expo_tradicionales) +
    ggtime::autolayer(pronostico, level = 95) +
    ggtitle("Pronóstico de exportaciones") +
    xlab("Fecha") +
    ylab("Valor") +
    theme_light()
)

# Note que para un MA(1), los pronósticos puntuales se vuelven
# constantes después del primer paso adelante.
