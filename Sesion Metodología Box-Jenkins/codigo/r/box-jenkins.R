# ============================================
# IMPORTAR PAQUETES
# ============================================

library(fs)
library(here)

library(readr)
library(ggplot2)
library(ggtime)
library(tsibble)
library(feasts)
library(fable)
library(dplyr)
library(tseries)
library(FinTS)
library(lmtest)

# Evita que FinTS::ARIMA enmascare el estimador ARIMA de fable.
ARIMA <- fable::ARIMA

# ============================================
# RUTAS
# ============================================

here::i_am("Sesion Metodología Box-Jenkins/codigo/r/box-jenkins.R")

directorio <- fs::path(here::here("Sesion Metodología Box-Jenkins", "datos"))

ruta_exp <- fs::path(directorio, "Expotradicionales1990-2017.csv")
ruta_te <- fs::path(directorio, "PTEAUSDM2005-202506.csv")

# ============================================
# CARGAR DATOS
# ============================================

expotradicionales <- read_csv(
  ruta_exp,
  col_names = "expo_tradicionales",
  show_col_types = FALSE
)

head(expotradicionales)

# ============================================
# CREAR TSIBBLE (Serie de tiempo)
# ============================================

y_tbl <- expotradicionales |>
  mutate(
    fecha = yearmonth(seq(
      from = as.Date("1990-01-01"),
      by   = "month",
      length.out = n()
    ))
  ) |>
  as_tsibble(index = fecha)

# Columna en log para trabajar la diferencia
y_tbl <- y_tbl |>
  mutate(
    log_expo = log(expo_tradicionales)
  )

# ============================================
# PASO 1: IDENTIFICACIÓN
# ============================================

print(
  ggtime::autoplot(y_tbl, expo_tradicionales) +
    ggtitle("Exportaciones tradicionales, 1990-2017") +
    xlab("Fecha") +
    ylab("Valor")
)

# ============================================
# FAC DE LA SERIE ORIGINAL
# ============================================

print(
  y_tbl |>
    ACF(expo_tradicionales, lag_max = 24) |>
    ggtime::autoplot() +
    ggtitle("Función de autocorrelación (FAC)")
)

# ============================================
# SERIE DIFERENCIADA
# ============================================

print(
  y_tbl |>
    mutate(diff_expo = difference(expo_tradicionales)) |>
    filter(!is.na(diff_expo)) |>
    ggtime::autoplot(diff_expo) +
    ggtitle("Serie diferenciada") +
    xlab("Fecha") +
    ylab("Valor")
)

# ============================================
# DIFERENCIA DEL LOGARITMO
# ============================================

y_tbl <- y_tbl |>
  mutate(log_diff = difference(log_expo))

print(
  y_tbl |>
    filter(!is.na(log_diff)) |>
    ggtime::autoplot(log_diff) +
    ggtitle("Diferencia del logaritmo de la serie original") +
    xlab("Fecha") +
    ylab("Valor")
)

# ============================================
# FAC Y FACP
# ============================================

print(
  y_tbl |>
    filter(!is.na(log_diff)) |>
    ACF(log_diff, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle("FAC de la diferencia del logaritmo")
)

print(
  y_tbl |>
    filter(!is.na(log_diff)) |>
    PACF(log_diff, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle("FACP de la diferencia del logaritmo")
)

# ============================================
# INTERPRETACIÓN PRELIMINAR
# ============================================

# La FAC muestra que solo la primera autocorrelación
# es significativa.
# La FACP decae rápidamente.
# Se propone inicialmente un modelo MA(1)
# equivalente a un ARIMA(0,1,1) sobre log(y).

# ============================================
# PASO 2: ESTIMACIÓN
# ============================================

# fable estima directamente sobre log_expo con
# la diferenciación integrada en el operador ARIMA.
# ARIMA(0,1,1) sin constante = MA(1) sobre la diferencia.

y_tbl2 <- y_tbl |>
  select(fecha, expo_tradicionales) |>
  filter(!is.na(expo_tradicionales))

str(y_tbl2)

fit_ma1 <- y_tbl2 |>
  model(ma1 = fable::ARIMA(log(expo_tradicionales) ~ 0 + pdq(0, 1, 1)))

report(fit_ma1)

# ============================================
# PASO 3: VALIDACIÓN
# ============================================

residuos <- residuals(fit_ma1)$.resid

summary(residuos)

# ============================================
# FAC DE LOS RESIDUOS
# ============================================

print(
  fit_ma1 |>
    augment() |>
    ACF(.resid, lag_max = 24) |>
    ggtime::autoplot() +
    ggtitle("FAC de los residuos")
)

# ============================================
# PRUEBA LJUNG-BOX
# ============================================



for (lag in c(6, 12, 18, 24)) {
  cat("\nLjung-Box, lag =", lag, "\n")
  print(Box.test(residuos, lag = lag, type = "Ljung-Box"))
}

# H0: no hay autocorrelación

# ============================================
# PRUEBA ARCH
# ============================================

ArchTest(residuos, lags = 12)

# H0: no hay efectos ARCH

# ============================================
# PRUEBA JARQUE-BERA
# ============================================

jarque.bera.test(residuos)

# H0: residuos normales

# ============================================
# Q-Q PLOT
# ============================================

qqnorm(residuos)
qqline(residuos)

# ============================================
# PASO 4: PRONÓSTICO
# ============================================

pronostico <- fit_ma1 |>
  forecast(h = 12)

# Tabla en niveles (exp ya aplicado internamente)
tabla_pronostico <- pronostico |>
  as_tibble() |>
  select(fecha, .mean) |>
  rename(pronostico = .mean)

# Intervalos de confianza al 95 %
intervalos <- pronostico |>
  hilo(level = 95) |>
  fabletools::unpack_hilo(`95%`) |>
  as_tibble() |>
  select(fecha, `95%_lower`, `95%_upper`) |>
  rename(limite_inferior = `95%_lower`, limite_superior = `95%_upper`)

tabla_pronostico <- left_join(tabla_pronostico, intervalos, by = "fecha")

# fable revierte automaticamente la transformacion log()
# en forecast(), por lo que estos valores ya estan en niveles.

print(tabla_pronostico)

# ============================================
# GRÁFICA DEL PRONÓSTICO
# ============================================

print(
  ggtime::autoplot(y_tbl, expo_tradicionales) +
    ggtime::autolayer(pronostico, level = 95) +
    ggtitle("Pronóstico de exportaciones") +
    xlab("Fecha") +
    ylab("Valor")
)

# ============================================
# SEGUNDA SERIE:
# PRECIO INTERNACIONAL DEL TÉ
# ============================================

te <- read_delim(
  ruta_te,
  delim   = "\t",
  col_names = "precio_te",
  locale  = locale(decimal_mark = ","),
  show_col_types = FALSE
)

head(te)

# ============================================
# CREAR TSIBBLE
# ============================================

te_tbl <- te |>
  mutate(
    fecha = yearmonth(seq(
      from = as.Date("2005-01-01"),
      by   = "month",
      length.out = n()
    ))
  ) |>
  as_tsibble(index = fecha)

# ============================================
# IDENTIFICACIÓN
# ============================================

print(
  ggtime::autoplot(te_tbl, precio_te) +
    ggtitle("Precio internacional del té, 2005-2025") +
    xlab("Fecha") +
    ylab("Precio té (USD)")
)

# ============================================
# FAC Y FACP
# ============================================

print(
  te_tbl |>
    ACF(precio_te, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle("FAC del precio del té")
)

print(
  te_tbl |>
    PACF(precio_te, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle("FACP del precio del té")
)

# ============================================
# LOGARITMO
# ============================================

print(
  te_tbl |>
    mutate(log_te = log(precio_te)) |>
    ggtime::autoplot(log_te) +
    ggtitle("Logaritmo del precio internacional del té")
)

# ============================================
# ESTIMACIÓN
# ============================================

fit_te <- te_tbl |>
  model(
    ar1     = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 0)),
    ar2     = fable::ARIMA(precio_te ~ 1 + pdq(2, 0, 0)),
    arma11  = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 1))
  )

# ============================================
# TABLA RESUMEN AIC / BIC
# ============================================

tabla_modelos <- glance(fit_te) |>
  select(.model, AIC, BIC) |>
  rename(Modelo = .model) |>
  mutate(across(c(AIC, BIC), ~ round(.x, 3)))

print(tabla_modelos)

# ============================================
# GRÁFICAS DE RESIDUOS (FAC y FAC²)
# ============================================

nombres_te <- c("ar1", "ar2", "arma11")

for (nom in nombres_te) {
  
  res_vec <- residuals(fit_te |> select(all_of(nom)))$.resid
  
  # Residuos en el tiempo
  plot(res_vec, main = paste("Residuos", nom), type = "l")
  
  # FAC residuos
  print(
    fit_te |>
      select(all_of(nom)) |>
      augment() |>
      ACF(.resid, lag_max = 15) |>
      ggtime::autoplot() +
      ggtitle(paste("FAC residuos", nom))
  )
  
  # FAC residuos²
  print(
    fit_te |>
      select(all_of(nom)) |>
      augment() |>
      mutate(resid2 = .resid^2) |>
      ACF(resid2, lag_max = 15) |>
      ggtime::autoplot() +
      ggtitle(paste("FAC residuos\u00b2", nom))
  )
}

# ============================================
# VALIDACIÓN DE SUPUESTOS
# ============================================

for (nom in nombres_te) {
  
  res_vec <- residuals(fit_te |> select(all_of(nom)))$.resid
  
  cat("\n========================\n")
  cat(nom, "\n")
  cat("========================\n")
  
  cat("\nJarque-Bera\n")
  print(jarque.bera.test(res_vec))
  
  cat("\nARCH\n")
  print(ArchTest(res_vec, lags = 5))
  
  cat("\nLjung-Box\n")
  for (lag in c(5, 10, 20)) {
    print(Box.test(res_vec, lag = lag, type = "Ljung-Box"))
  }
}
