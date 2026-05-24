# %% Importacion de paquetes =========================

# Trabajar con rutas relativas en R
library(fs)
library(here)

# Paquetes del tidyverse (Para el manejo, manipulacion y graficacion de datos)
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
library(urca) # Test de raiz unitaria

# Para que la funcion ARIMA que se use por defecto sea la de fable.
ARIMA <- fable::ARIMA

# TODO: Tal vez todo esto se pueda hacer mejor con programacion funcional!
#       Explorar en el futuro.

# %% Cargar bases de datos en R usando rutas relativas =========================

# Fijar la ruta del archivo actual como referencia para here()
here::i_am(
  "Sesi\u00f3n 3 - Metodolog\u00eda Box Jenkins/codigo/R/Box_jenkins_ej2_precio_te.R"
)

# Obtener la ruta del directorio raiz
BASE_DIR <- here::here("Sesi\u00f3n 3 - Metodolog\u00eda Box Jenkins")

# Obtener la ruta del directorio con los datos
DATA_DIR <- fs::path(BASE_DIR, "datos")

# Rutas de las bases de datos
ruta_te <- fs::path(DATA_DIR, "PTEAUSDM2005-202506.csv") # Base de datos del precio del te

# %% Funciones auxiliares =========================

# Funcion auxiliar para mostrar graficos en una grilla m x n
grilla <- function(..., nrow, ncol) {
  graficos <- list(...)

  if (length(graficos) > nrow * ncol) {
    stop("La cantidad de graficos supera el tamano de la grilla.")
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

obtener_parametro <- function(tabla_coeficientes, modelo, termino, columna) {
  valor <- tabla_coeficientes |>
    filter(.model == modelo, term == termino) |>
    pull({{ columna }})

  if (length(valor) == 0) {
    return(NA_real_)
  }

  valor[[1]]
}

formato_estimacion <- function(valor, decimales = 3) {
  if (is.na(valor)) {
    return("")
  }

  sprintf(paste0("%.", decimales, "f"), valor)
}

formato_error_estandar <- function(valor, decimales = 3) {
  if (is.na(valor)) {
    return("")
  }

  paste0("(", sprintf(paste0("%.", decimales, "f"), valor), ")")
}

media_incondicional_fable <- function(modelo_estimado) {
  parametros <- stats::coef(modelo_estimado)

  if ("constant" %in% names(parametros)) {
    return(unname(parametros[["constant"]]))
  }

  NA_real_
}

se_media_incondicional_fable <- function(modelo_estimado) {
  matriz_varianzas <- modelo_estimado$var.coef

  if (is.null(matriz_varianzas) || !("constant" %in% rownames(matriz_varianzas))) {
    return(NA_real_)
  }

  sqrt(matriz_varianzas["constant", "constant"])
}

orden_modelo <- function(nombre_modelo) {
  switch(
    nombre_modelo,
    "ARMA(1,0)" = c(p = 1, q = 0),
    "ARMA(2,0)" = c(p = 2, q = 0),
    "ARMA(1,1)" = c(p = 1, q = 1)
  )
}

residuos_modelo <- function(fit, nombre_modelo) {
  orden <- orden_modelo(nombre_modelo)
  n_inicial <- max(orden[["p"]], orden[["q"]], 1)

  fit |>
    select(all_of(nombre_modelo)) |>
    augment() |>
    as_tibble() |>
    slice((n_inicial + 1):n()) |>
    select(fecha, .resid)
}

# %% =========================
# SEGUNDA SERIE: PRECIO INTERNACIONAL DEL TE
# ============================

# Base de datos con la serie importada a R
te_base <- read_delim(
  ruta_te,
  delim = "\t",
  col_names = "precio_te",
  locale = locale(decimal_mark = ","),
  show_col_types = FALSE
)

# Ver el tipo de objeto de la base de datos
print(class(te_base))

# Ver primeras y ultimas observaciones de la base de datos
print(head(te_base)) # Primeras observaciones
print(tail(te_base)) # Ultimas observaciones

# %% Creacion del indice temporal de las series de tiempo

# Creacion del indice temporal
fechas_te_base <- yearmonth(seq(
  from = as.Date("2005-01-01"),
  by = "month",
  length.out = nrow(te_base)
))

# Agregar el indice temporal a la base de datos del precio del te
te_tbl <- te_base |>
  mutate(fecha = fechas_te_base) |>
  as_tsibble(index = fecha)

# El tipo de objeto de la base de datos ahora es un tsibble
print(class(te_tbl))

# Ver primeras y ultimas observaciones de la base de datos, ahora con indice temporal
print(head(te_tbl)) # Primeras observaciones
print(tail(te_tbl)) # Ultimas observaciones

# %% Creacion de la serie de tiempo del "precio del te"

te_serie <- te_tbl$precio_te
te_serie <- as.numeric(te_serie)
te_serie <- te_serie[!is.na(te_serie)]

# Ver el principio y final de la serie de tiempo
print(head(te_serie))
print(tail(te_serie))

# El tipo de objeto de la serie en R es numeric
print(class(te_serie))

# Ver algunas estadisticas descriptivas de la serie de tiempo
print(summary(te_serie))
cat(sprintf("Media muestral precio del te: %.3f\n", mean(te_serie)))

# %% =========================
# Paso 1: Identificacion
# ============================

# Grafica de la serie de tiempo "precio del te"
print(
  ggtime::autoplot(te_tbl, precio_te) +
    ggtitle("Precio internacional del te, 2005-2025") +
    xlab("Fecha") +
    ylab("Precio te (USD)") +
    theme_minimal()
)

# %% FAC y FACP del precio del te (serie original)

grafico_fac_te <- te_tbl |>
  ACF(precio_te, lag_max = 15) |>
  ggtime::autoplot() +
  ggtitle("FAC del precio del te") +
  ylim(-1, 1) +
  xlab("Rezago") +
  ylab("ACF")

grafico_facp_te <- te_tbl |>
  PACF(precio_te, lag_max = 15) |>
  ggtime::autoplot() +
  ggtitle("FACP del precio del te") +
  ylim(-1, 1) +
  xlab("Rezago") +
  ylab("PACF")

grilla(grafico_fac_te, grafico_facp_te, nrow = 1, ncol = 2)

# %% Tests de Raiz Unitaria

# Test de Augmented Dickey Fuller (ADF)
adf_result <- urca::ur.df(te_serie, type = "drift", selectlags = "AIC")
adf_pvalor <- tseries::adf.test(te_serie, alternative = "stationary")

# Nota: En el test ADF, si no rechazo H0 la serie no es estacionaria
#       y si rechazo H0 la serie es estacionaria.

cat("\n=== Test ADF ===\n")
cat("Estadistico ADF:", adf_result@teststat[1, "tau2"], "\n")
cat("p-valor aproximado:", adf_pvalor$p.value, "\n")
cat("Rezagos usados:", adf_result@lags, "\n")
cat("Observaciones:", length(te_serie), "\n")
cat("Valores criticos:\n")
print(adf_result@cval["tau2", ])

if (adf_pvalor$p.value < 0.05) {
  cat("ADF: Rechazamos H0. Segun el test, la serie es estacionaria.\n")
} else {
  cat("ADF: No rechazamos H0. Segun el test, la serie no es estacionaria.\n")
}

# Test KPSS
kpss_result <- tseries::kpss.test(te_serie, null = "Level")

# Nota: En prueba KPSS se interpreta al contrario que una prueba ADF.
#       Si no rechazo H0 la serie es estacionaria
#       y si rechazo H0 la serie es no estacionaria.

cat("\n=== Test KPSS ===\n")
cat("Estadistico KPSS:", unname(kpss_result$statistic), "\n")
cat("p-valor:", kpss_result$p.value, "\n")
cat("Rezagos usados:", unname(kpss_result$parameter), "\n")
cat("Valores criticos: revise la salida completa de kpss.test() para detalles.\n")
print(kpss_result)

if (kpss_result$p.value < 0.05) {
  cat("KPSS: rechazamos H0. Segun el test, la serie es no estacionaria.\n")
} else {
  cat("KPSS: no rechazamos H0. Segun el test, la serie es estacionaria.\n")
}

# Nota: Segun los resultados de la prueba ADF y KPSS, hay que
#       considerar diferenciar la serie.

# %% Logaritmo del precio del te (en niveles)

te_tbl <- te_tbl |>
  mutate(log_te = log(precio_te))

print(
  ggtime::autoplot(te_tbl, log_te) +
    ggtitle("Logaritmo del precio internacional del te") +
    xlab("Fecha") +
    ylab("Log(Precio te)") +
    theme_minimal()
)

# %% FAC y FACP del precio del te (serie en logaritmos)

grafico_fac_log_te <- te_tbl |>
  ACF(log_te, lag_max = 15) |>
  ggtime::autoplot() +
  ggtitle("FAC del logaritmo precio del te") +
  ylim(-1, 1) +
  xlab("Rezago") +
  ylab("ACF")

grafico_facp_log_te <- te_tbl |>
  PACF(log_te, lag_max = 15) |>
  ggtime::autoplot() +
  ggtitle("FACP del logaritmo precio del te") +
  ylim(-1, 1) +
  xlab("Rezago") +
  ylab("PACF")

grilla(grafico_fac_log_te, grafico_facp_log_te, nrow = 1, ncol = 2)

# %% Identificacion del modelo usando FAC y FACP

# En este caso no es tan sencillo determinar el orden p y q del modelo ARIMA de la
# FAC y la FACP.
# Algunos modelos sugeridos por la FAC y la FACP son: ARMA(1,0), ARMA(2,0) y ARMA(1,1)

# La FACP no decae tan rapidamente, pero si esta decayendo.

# P.d. Tambien se usaran criterios de informacion para la
#      seleccion de los ordenes p y q del modelo ARIMA.

# %% =========================
# Paso 2: Estimacion
# ============================

# Se estimaran 3 modelos en este caso, un ARIMA(1,0,0), un ARIMA(2,0,0)
# y un ARIMA(1,0,1).

# Se crea un vector con los nombres de los modelos que se estimaran.
nombres_modelos <- c("ARMA(1,0)", "ARMA(2,0)", "ARMA(1,1)")

# Nota: Se especifican los modelos en niveles, para que la estructura sea
#       comparable con el script de Python.
#       PDQ(0,0,0) evita que fable agregue terminos estacionales automaticos.
fit_te <- te_tbl |>
  model(
    "ARMA(1,0)" = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 0) + PDQ(0, 0, 0)),
    "ARMA(2,0)" = fable::ARIMA(precio_te ~ 1 + pdq(2, 0, 0) + PDQ(0, 0, 0)),
    "ARMA(1,1)" = fable::ARIMA(precio_te ~ 1 + pdq(1, 0, 1) + PDQ(0, 0, 0))
  )

# Nota: El metodo de estimacion es maxima verosimilitud usando stats::arima()
#       a traves de fable::ARIMA().
cat("\nResumen de modelos estimados\n")
for (nombre in nombres_modelos) {
  cat("\n", nombre, "\n", sep = "")
  print(report(fit_te |> select(all_of(nombre))))
}

# %% Tabla resumen de modelos estimados

coeficientes_te <- coef(fit_te)
ajuste_te <- glance(fit_te)

# Inicialmente la tabla de modelos es una lista de data frames
tabla_modelos <- list()

# Se itera sobre cada modelo para extraer sus resultados principales
for (nombre in nombres_modelos) {
  modelo_interno <- fit_te[[nombre]][[1]]$fit$model

  # Estimacion del intercepto del modelo y su error estandar.
  # En fable, el termino "constant" de coef() corresponde a c = mu * (1 - suma AR).
  intercepto_fable <- obtener_parametro(coeficientes_te, nombre, "constant", estimate)
  se_intercepto_fable <- obtener_parametro(coeficientes_te, nombre, "constant", std.error)

  # Parametros de la parte AR y MA
  ar1 <- obtener_parametro(coeficientes_te, nombre, "ar1", estimate)
  ar2 <- obtener_parametro(coeficientes_te, nombre, "ar2", estimate)
  ma1 <- obtener_parametro(coeficientes_te, nombre, "ma1", estimate)

  # Errores estandar de la parte AR y MA
  se_ar1 <- obtener_parametro(coeficientes_te, nombre, "ar1", std.error)
  se_ar2 <- obtener_parametro(coeficientes_te, nombre, "ar2", std.error)
  se_ma1 <- obtener_parametro(coeficientes_te, nombre, "ma1", std.error)

  # Media de largo plazo del modelo ARMA
  mu <- media_incondicional_fable(modelo_interno)
  se_mu <- se_media_incondicional_fable(modelo_interno)

  ajuste_modelo <- ajuste_te |>
    filter(.model == nombre)

  # Se llena una a una la lista con los resultados principales de las estimaciones.
  tabla_modelos[[nombre]] <- data.frame(
    Modelo = nombre,
    intercepto_fable = intercepto_fable,
    se_intercepto_fable = se_intercepto_fable,
    media_incondicional = mu,
    se_media_incondicional = se_mu,
    a1 = ar1,
    se_a1 = se_ar1,
    a2 = ar2,
    se_a2 = se_ar2,
    b1 = ma1,
    se_b1 = se_ma1,
    AIC = ajuste_modelo$AIC,
    BIC = ajuste_modelo$BIC
  )
}

# Transformar la lista en un data frame
tabla_modelos_te <- bind_rows(tabla_modelos)

# Aca se genera la tabla final de publicacion.
filas_tabla_publicacion <- list(
  c("a1", "a1", "se_a1"),
  c("", "se_a1", NA),
  c("a2", "a2", "se_a2"),
  c("", "se_a2", NA),
  c("b1", "b1", "se_b1"),
  c("", "se_b1", NA),
  c("intercepto fable", "intercepto_fable", "se_intercepto_fable"),
  c("", "se_intercepto_fable", NA),
  c("media incondicional", "media_incondicional", "se_media_incondicional"),
  c("", "se_media_incondicional", NA),
  c("AIC", "AIC", NA),
  c("BIC", "BIC", NA)
)

# Primero la tabla final es una lista
tabla_publicacion <- list()

# Loop que llena la lista para la tabla final
for (i in seq_along(filas_tabla_publicacion)) {
  especificacion_fila <- filas_tabla_publicacion[[i]]
  etiqueta <- especificacion_fila[[1]]
  columna_valor <- especificacion_fila[[2]]
  columna_error <- especificacion_fila[[3]]

  fila <- data.frame(Parametro = etiqueta)

  for (nombre in nombres_modelos) {
    modelo <- tabla_modelos_te |>
      filter(Modelo == nombre)

    if (is.na(columna_error) && startsWith(columna_valor, "se_")) {
      fila[[nombre]] <- formato_error_estandar(modelo[[columna_valor]])
    } else if (columna_valor %in% c("AIC", "BIC")) {
      fila[[nombre]] <- formato_estimacion(modelo[[columna_valor]], decimales = 1)
    } else {
      fila[[nombre]] <- formato_estimacion(modelo[[columna_valor]])
    }
  }

  tabla_publicacion[[i]] <- fila
}

# Genera la tabla final como un data frame
tabla_modelos_te_publicacion <- bind_rows(tabla_publicacion)

# Imprimir la tabla final en formato de texto
cat("\nTabla resumen de modelos estimados\n")
print(tabla_modelos_te_publicacion, row.names = FALSE)
cat("\nErrores estandar entre parentesis.\n")

# %% =========================
# Paso 3: Validacion de Supuestos
# ============================

# Grafica de los residuales, FAC de los residuales y FAC de los residuales al cuadrado

graficos_diagnostico <- list()

# Itero en cada uno de los nombres de los modelos
for (nombre in nombres_modelos) {
  # Nota: Cada iteracion se da modelo por modelo, una iteracion por cada modelo.

  residuos_tbl <- residuos_modelo(fit_te, nombre)
  residuos_cuadrado_tbl <- residuos_tbl |>
    mutate(residuos_cuadrado = .resid^2)

  # Grafica de los residuales
  grafico_residuos <- ggplot(residuos_tbl, aes(x = fecha, y = .resid)) +
    geom_line(color = "black", linewidth = 0.4) +
    ggtitle(paste("Residuales", nombre)) +
    xlab("Fecha") +
    ylab("Residuales") +
    theme_minimal()

  # FAC de los residuales
  grafico_fac_residuos <- residuos_tbl |>
    as_tsibble(index = fecha) |>
    ACF(.resid, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle(paste("FAC residuos", nombre)) +
    ylim(-1, 1) +
    xlab("Rezago") +
    ylab("ACF")

  # FAC de los residuales al cuadrado
  grafico_fac_residuos_cuadrado <- residuos_cuadrado_tbl |>
    as_tsibble(index = fecha) |>
    ACF(residuos_cuadrado, lag_max = 15) |>
    ggtime::autoplot() +
    ggtitle(paste("FAC residuos^2", nombre)) +
    ylim(-1, 1) +
    xlab("Rezago") +
    ylab("ACF")

  graficos_diagnostico <- c(
    graficos_diagnostico,
    list(grafico_residuos, grafico_fac_residuos, grafico_fac_residuos_cuadrado)
  )
}

# Genera una grilla 3x3, equivalente a la del script de Python
do.call(grilla, c(graficos_diagnostico, list(nrow = 3, ncol = 3)))

# %% Grafica residuales Q-Q plot

graficos_qq <- list()

for (nombre in nombres_modelos) {
  residuos_tbl <- residuos_modelo(fit_te, nombre)

  grafico_qq <- ggplot(residuos_tbl, aes(sample = .resid)) +
    stat_qq(color = "black", size = 1) +
    stat_qq_line(color = "red", linewidth = 0.6) +
    ggtitle(paste("Q-Q plot residuos", nombre)) +
    xlab("Cuantiles teoricos") +
    ylab("Cuantiles muestrales") +
    theme_minimal()

  graficos_qq <- c(graficos_qq, list(grafico_qq))
}

grilla(graficos_qq[[1]], graficos_qq[[2]], graficos_qq[[3]], nrow = 1, ncol = 3)

# %% Tabla con los principales resultados de las pruebas de validacion de supuestos

# La tabla con los resultados de las pruebas de validacion de supuestos inicialmente sera una lista
tabla_diagnostico <- list()

# Itera a traves de los nombres de los modelos y genera los resultados de cada test
# de validacion por modelo.
for (nombre in nombres_modelos) {
  # Nota: Cada iteracion se da por modelo.

  residuos <- residuos_modelo(fit_te, nombre)$.resid

  # Prueba de Jarque Bera para los residuales
  jb_pvalue <- unname(jarque.bera.test(residuos)$p.value)

  # Prueba ARCH para los residuales
  arch_1 <- unname(FinTS::ArchTest(residuos, lags = 1)$p.value)
  arch_2 <- unname(FinTS::ArchTest(residuos, lags = 2)$p.value)
  arch_5 <- unname(FinTS::ArchTest(residuos, lags = 5)$p.value)

  # Prueba Ljung-Box para los residuales
  lb_5 <- unname(Box.test(residuos, lag = 5, type = "Ljung-Box")$p.value)
  lb_10 <- unname(Box.test(residuos, lag = 10, type = "Ljung-Box")$p.value)
  lb_20 <- unname(Box.test(residuos, lag = 20, type = "Ljung-Box")$p.value)

  tabla_diagnostico[[nombre]] <- data.frame(
    Modelo = nombre,
    JB = jb_pvalue,
    `A(1)` = arch_1,
    `A(2)` = arch_2,
    `A(5)` = arch_5,
    `LB(5)` = lb_5,
    `LB(10)` = lb_10,
    `LB(20)` = lb_20,
    check.names = FALSE
  )
}

# Luego la tabla se transforma en un data frame
tabla_diagnostico <- bind_rows(tabla_diagnostico)

print(
  tabla_diagnostico |>
    mutate(across(where(is.numeric), ~ round(.x, 3)))
)

# %% =========================
# Paso 4: Pronostico
# ============================

# Se calculan pronosticos 12 pasos adelante para cada uno de los modelos estimados
horizonte_pronostico <- 12

# Tabla con los pronosticos que se usaran en la grafica
pronosticos_te <- fit_te |>
  forecast(h = horizonte_pronostico) |>
  hilo(level = 95) |>
  unpack_hilo("95%")

calcular_pronostico_modelo <- function(nombre) {
  # Genera la tabla de pronostico de un modelo estimado.
  pronosticos_te |>
    filter(.model == nombre) |>
    as_tibble() |>
    select(
      Fecha = fecha,
      pronostico = .mean,
      limite_inferior = `95%_lower`,
      limite_superior = `95%_upper`
    )
}

# Tabla 1: Pronostico ARMA(1,0)
tabla_pronostico_arma10 <- calcular_pronostico_modelo("ARMA(1,0)")

# Tabla 2: Pronostico ARMA(2,0)
tabla_pronostico_arma20 <- calcular_pronostico_modelo("ARMA(2,0)")

# Tabla 3: Pronostico ARMA(1,1)
tabla_pronostico_arma11 <- calcular_pronostico_modelo("ARMA(1,1)")

# Imprimir las 3 tablas de pronosticos con los doce pasos adelante
cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("Pronosticos 12 pasos adelante - ARMA(1,0)\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
print(
  tabla_pronostico_arma10 |>
    mutate(across(where(is.numeric), ~ round(.x, 3))),
  row.names = FALSE
)

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("Pronosticos 12 pasos adelante - ARMA(2,0)\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
print(
  tabla_pronostico_arma20 |>
    mutate(across(where(is.numeric), ~ round(.x, 3))),
  row.names = FALSE
)

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("Pronosticos 12 pasos adelante - ARMA(1,1)\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
print(
  tabla_pronostico_arma11 |>
    mutate(across(where(is.numeric), ~ round(.x, 3))),
  row.names = FALSE
)

# Colores para que coincidan con la descripcion de la grafica:
# ARMA(1,0), ARMA(2,0) y ARMA(1,1) en verde, azul y rojo, respectivamente.
colores_pronostico <- c(
  "ARMA(1,0)" = "green",
  "ARMA(2,0)" = "blue",
  "ARMA(1,1)" = "red"
)

# Grafica conjunta del historico y los pronosticos de los 3 modelos
print(
  ggplot() +
    geom_line(
      data = te_tbl,
      aes(x = fecha, y = precio_te),
      color = "black",
      linewidth = 0.4
    ) +
    geom_line(
      data = pronosticos_te,
      aes(x = fecha, y = .mean, color = .model),
      linewidth = 0.8
    ) +
    scale_color_manual(values = colores_pronostico) +
    ggtitle("Pronosticos del precio internacional del te") +
    xlab("Fecha") +
    ylab("Precio del te") +
    labs(color = "Modelo") +
    theme_minimal()
)

# Nota: Revise la validacion de supuestos, la FAC y la FACP de la serie original,
#       y los criterios de informacion para seleccionar el modelo que considere
#       mejor para modelar la serie.
