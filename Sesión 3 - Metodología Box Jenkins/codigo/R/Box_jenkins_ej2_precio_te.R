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
here::i_am("Sesión 3 - Metodología Box Jenkins/codigo/R/Box_jenkins_ej2_precio_te.R")

# Obtener la ruta del directorio con los datos
directorio <- fs::path(here::here("Sesión 3 - Metodología Box Jenkins", "datos"))

# Rutas de las bases de datos
ruta_te <- fs::path(directorio, "PTEAUSDM2005-202506.csv") # Base de datos de exportaciones

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
# SEGUNDA SERIE: PRECIO INTERNACIONAL DEL TÉ ====
# ===

#  -------------------------------------------------------------------

te <- read_delim(
  ruta_te,
  delim   = "\t",
  col_names = "precio_te",
  locale  = locale(decimal_mark = ","),
  show_col_types = FALSE
)

head(te)

# Crear tsibble -------------------------------------------------------------------

te_tbl <- te |>
  mutate(
    fecha = yearmonth(seq(
      from = as.Date("2005-01-01"),
      by   = "month",
      length.out = n()
    ))
  ) |>
  as_tsibble(index = fecha)

# Identificación -------------------------------------------------------------------

print(
  ggtime::autoplot(te_tbl, precio_te) +
    ggtitle("Precio internacional del té, 2005-2025") +
    xlab("Fecha") +
    ylab("Precio té (USD)")
)

# Fac y facp -------------------------------------------------------------------

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

# Logaritmo -------------------------------------------------------------------

print(
  te_tbl |>
    mutate(log_te = log(precio_te)) |>
    ggtime::autoplot(log_te) +
    ggtitle("Logaritmo del precio internacional del té")
)

# Estimación -------------------------------------------------------------------

fit_te <- te_tbl |>
  model(
    ar1     = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 0)),
    ar2     = fable::ARIMA(precio_te ~ 1 + pdq(2, 0, 0)),
    arma11  = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 1))
  )

# Tabla resumen aic / bic -------------------------------------------------------------------

tabla_modelos <- glance(fit_te) |>
  select(.model, AIC, BIC) |>
  rename(Modelo = .model) |>
  mutate(across(c(AIC, BIC), ~ round(.x, 3)))

print(tabla_modelos)

# Gráficas de residuos (fac y fac²) -------------------------------------------------------------------

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

# Validación de supuestos -------------------------------------------------------------------

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
