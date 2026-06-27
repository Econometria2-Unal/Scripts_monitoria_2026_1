#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 7: Modelos de vectores autorregresivos
#'
#' Simulacion de una normal multivariada correlacionada
#' usando descomposicion espectral, SVD y Cholesky.


# ===
# Tabla de contenidos ===
# ===

#' 1. Preparacion del entorno
#' 2. Simulacion de una normal estandar bivariada no correlacionada
#' 3. Matriz de varianzas-covarianzas con alta correlacion
#' 4. Simulacion manual: espectral, SVD y Cholesky
#' 5. Simulacion con mvtnorm: espectral, SVD y Cholesky
#' 6. Comparacion de resultados manuales y con mvtnorm
#' 7. Graficas 2D interactivas con plotly
#' 8. Graficas 3D interactivas de densidad con plotly
#' 9. Exportacion de graficas a archivos HTML


# ===
# 1. Preparacion del entorno ----
# ===

# Paquetes necesarios para este script:
# - mvtnorm: simulacion de normales multivariadas.
# - plotly: graficas interactivas.
# - htmlwidgets: exportacion de graficas plotly a archivos HTML.
paquetes_necesarios = c("mvtnorm", "plotly", "htmlwidgets")

paquetes_faltantes = paquetes_necesarios[
  !vapply(paquetes_necesarios, requireNamespace, logical(1), quietly = TRUE)
]

if (length(paquetes_faltantes) > 0) {
  stop(
    "Faltan paquetes por instalar: ",
    paste(paquetes_faltantes, collapse = ", "),
    "\nInstalalos con install.packages(c(",
    paste(sprintf('\"%s\"', paquetes_faltantes), collapse = ", "),
    "))"
  )
}

library(mvtnorm)
library(plotly)

# Fijamos la semilla para que la simulacion sea reproducible.
semilla_simulacion = 82901
set.seed(semilla_simulacion)

# Numero de observaciones solicitado.
n_observaciones = 100000

# Media de la normal bivariada correlacionada.
media = c(u_1 = 0, u_2 = 0)


# ===
# 2. Simulacion de una normal estandar bivariada no correlacionada ----
# ===

# Primero simulamos dos errores normales estandar univariados independientes:
#
#   z_1 ~ N(0, 1)
#   z_2 ~ N(0, 1)
#
# Al juntarlos en una matriz Z = (z_1, z_2), cada fila es una observacion de
# una normal estandar bivariada no correlacionada: Z ~ N_2(0, I_2).
normales_estandar = rnorm(n_observaciones * 2)
Z = matrix(normales_estandar, ncol = 2, byrow = TRUE)
colnames(Z) = c("z_1", "z_2")

cat("\nCorrelacion muestral de los errores estandar no correlacionados:\n")
print(cor(Z))


# ===
# 3. Matriz de varianzas-covarianzas con alta correlacion ----
# ===

# Usaremos la misma matriz Sigma para las tres descomposiciones. En este
# ejemplo las desviaciones estandar marginales son distintas, pero la
# correlacion teorica entre los errores es alta.
desv_estandar = c(u_1 = 1.0, u_2 = 1.5)
rho = 0.90

Sigma = matrix(
  c(
    desv_estandar["u_1"]^2,
    rho * prod(desv_estandar),
    rho * prod(desv_estandar),
    desv_estandar["u_2"]^2
  ),
  nrow = 2,
  byrow = TRUE
)
colnames(Sigma) = rownames(Sigma) = names(media)

cat("\nMatriz de varianzas-covarianzas teorica Sigma:\n")
print(Sigma)

cat("\nMatriz de correlaciones teorica asociada a Sigma:\n")
print(cov2cor(Sigma))


# ===
# 4. Simulacion manual: espectral, SVD y Cholesky ----
# ===

# Idea de la simulacion manual:
#
# Si Z ~ N_2(0, I_2) y encontramos una matriz R tal que:
#
#   t(R) %*% R = Sigma,
#
# entonces:
#
#   U = Z %*% R + media
#
# tiene distribucion normal bivariada con matriz de covarianzas Sigma.
#
# La funcion siguiente replica los factores usados internamente por
# mvtnorm::rmvnorm() para los metodos "eigen", "svd" y "chol". Asi la
# comparacion manual vs. mvtnorm se puede hacer observacion por observacion.
obtener_factor = function(Sigma, metodo) {
  if (metodo == "espectral") {
    ev = eigen(Sigma, symmetric = TRUE)
    
    if (!all(ev$values >= -sqrt(.Machine$double.eps) * abs(ev$values[1]))) {
      warning("Sigma no es positiva semidefinida numericamente.")
    }
    
    factor = t(ev$vectors %*% (t(ev$vectors) * sqrt(pmax(ev$values, 0))))
    
  } else if (metodo == "svd") {
    sv = svd(Sigma)
    
    if (!all(sv$d >= -sqrt(.Machine$double.eps) * abs(sv$d[1]))) {
      warning("Sigma no es positiva semidefinida numericamente.")
    }
    
    factor = t(sv$v %*% (t(sv$u) * sqrt(pmax(sv$d, 0))))
    
  } else if (metodo == "cholesky") {
    factor_chol = chol(Sigma, pivot = TRUE)
    factor = factor_chol[, order(attr(factor_chol, "pivot"))]
    
  } else {
    stop("Metodo no reconocido. Usa: espectral, svd o cholesky.")
  }
  
  return(factor)
}

simular_manual = function(Z, media, Sigma, metodo) {
  factor = obtener_factor(Sigma, metodo)
  U = Z %*% factor
  U = sweep(U, 2, media, "+")
  colnames(U) = names(media)
  return(U)
}

U_manual_espectral = simular_manual(Z, media, Sigma, "espectral")
U_manual_svd = simular_manual(Z, media, Sigma, "svd")
U_manual_cholesky = simular_manual(Z, media, Sigma, "cholesky")


# ===
# 5. Simulacion con mvtnorm: espectral, SVD y Cholesky ----
# ===

# rmvnorm() normalmente genera sus propias normales estandar. Para comparar
# exactamente con la simulacion manual, aqui le entregamos a rmvnorm() el mismo
# vector de normales estandar que usamos para construir Z.
#
# Importante: rmvnorm(..., pre0.9_9994 = FALSE) arma la matriz de normales por
# filas. Por eso usamos as.vector(t(Z)).
crear_generador_desde_Z = function(Z) {
  normales = as.vector(t(Z))
  
  function(n) {
    if (n != length(normales)) {
      stop("El generador fijo esperaba ", length(normales), " normales.")
    }
    
    return(normales)
  }
}

U_mvtnorm_espectral = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "eigen",
  pre0.9_9994 = FALSE,
  rnorm = crear_generador_desde_Z(Z)
)

U_mvtnorm_svd = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "svd",
  pre0.9_9994 = FALSE,
  rnorm = crear_generador_desde_Z(Z)
)

U_mvtnorm_cholesky = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "chol",
  pre0.9_9994 = FALSE,
  rnorm = crear_generador_desde_Z(Z)
)


# ===
# 6. Comparacion de resultados manuales y con mvtnorm ----
# ===

comparar_manual_mvtnorm = function(metodo, U_manual, U_mvtnorm) {
  diferencia_maxima = max(abs(U_manual - U_mvtnorm))
  son_iguales = isTRUE(
    all.equal(
      U_manual,
      U_mvtnorm,
      tolerance = 1e-12,
      check.attributes = FALSE
    )
  )
  
  resumen = data.frame(
    metodo = metodo,
    iguales_manual_mvtnorm = son_iguales,
    diferencia_maxima = diferencia_maxima,
    media_u_1 = mean(U_manual[, "u_1"]),
    media_u_2 = mean(U_manual[, "u_2"]),
    var_u_1 = var(U_manual[, "u_1"]),
    cov_u_1_u_2 = cov(U_manual[, "u_1"], U_manual[, "u_2"]),
    var_u_2 = var(U_manual[, "u_2"]),
    cor_u_1_u_2 = cor(U_manual[, "u_1"], U_manual[, "u_2"])
  )
  
  return(resumen)
}

tabla_comparacion = rbind(
  comparar_manual_mvtnorm(
    "Descomposicion espectral",
    U_manual_espectral,
    U_mvtnorm_espectral
  ),
  comparar_manual_mvtnorm(
    "SVD",
    U_manual_svd,
    U_mvtnorm_svd
  ),
  comparar_manual_mvtnorm(
    "Cholesky",
    U_manual_cholesky,
    U_mvtnorm_cholesky
  )
)

cat("\nComparacion manual vs. mvtnorm:\n")
print(tabla_comparacion, row.names = FALSE)

cat("\nCovarianza muestral - metodo espectral manual:\n")
print(cov(U_manual_espectral))

cat("\nCovarianza teorica objetivo Sigma:\n")
print(Sigma)


# ===
# 7. Graficas 2D interactivas con plotly ----
# ===

# Con 100000 puntos, una grafica interactiva puede volverse pesada. Por eso
# simulamos todas las observaciones, pero graficamos una submuestra aleatoria
# suficientemente grande para ver la forma de la distribucion.
n_puntos_grafica = 8000
semilla_graficos = 202601

preparar_muestra_grafica = function(U, n_puntos = n_puntos_grafica) {
  set.seed(semilla_graficos)
  indices = sample(seq_len(nrow(U)), size = min(n_puntos, nrow(U)))
  datos = data.frame(
    u_1 = U[indices, 1],
    u_2 = U[indices, 2]
  )
  return(datos)
}

graficar_normal_bivariada = function(U,
                                     titulo,
                                     color_puntos = "#1f77b4",
                                     etiqueta_x = "u_1",
                                     etiqueta_y = "u_2") {
  datos_grafica = preparar_muestra_grafica(U)
  
  grafica = plot_ly()
  
  grafica = add_trace(
    grafica,
    data = datos_grafica,
    x = ~u_1,
    y = ~u_2,
    type = "histogram2dcontour",
    colorscale = "Viridis",
    reversescale = TRUE,
    showscale = FALSE,
    contours = list(
      coloring = "lines",
      showlabels = TRUE
    ),
    hoverinfo = "skip",
    name = "Contornos"
  )
  
  grafica = add_trace(
    grafica,
    data = datos_grafica,
    x = ~u_1,
    y = ~u_2,
    type = "scattergl",
    mode = "markers",
    marker = list(
      size = 4,
      color = color_puntos,
      opacity = 0.35
    ),
    hovertemplate = paste(
      etiqueta_x, ": %{x:.3f}<br>",
      etiqueta_y, ": %{y:.3f}<extra></extra>"
    ),
    name = "Observaciones simuladas"
  )
  
  grafica = layout(
    grafica,
    title = list(text = titulo),
    xaxis = list(title = etiqueta_x),
    yaxis = list(title = etiqueta_y, scaleanchor = "x", scaleratio = 1),
    legend = list(orientation = "h", x = 0, y = -0.15),
    margin = list(l = 60, r = 30, b = 80, t = 70)
  )
  
  return(grafica)
}

# Grafica interactiva de la normal estandar bivariada no correlacionada.
g_normal_estandar = graficar_normal_bivariada(
  Z,
  "Normal estandar bivariada no correlacionada",
  color_puntos = "#2a6fbb",
  etiqueta_x = "z_1",
  etiqueta_y = "z_2"
)

# Graficas interactivas de las normales bivariadas correlacionadas obtenidas
# por cada descomposicion. Se grafica la version manual, porque arriba se
# verifica que coincide con mvtnorm observacion por observacion.
g_espectral = graficar_normal_bivariada(
  U_manual_espectral,
  "Normal bivariada correlacionada - Descomposicion espectral",
  color_puntos = "#c03a2b"
)

g_svd = graficar_normal_bivariada(
  U_manual_svd,
  "Normal bivariada correlacionada - SVD",
  color_puntos = "#2b8c56"
)

g_cholesky = graficar_normal_bivariada(
  U_manual_cholesky,
  "Normal bivariada correlacionada - Cholesky",
  color_puntos = "#6f4bb3"
)


# ===
# 8. Graficas 3D interactivas de densidad con plotly ----
# ===

# Las graficas 3D muestran la funcion de densidad teorica:
#
#   f(u_1, u_2)
#
# evaluada sobre una grilla. Para la normal estandar usamos I_2; para las
# normales correlacionadas usamos la misma matriz Sigma definida arriba. Como
# los tres metodos producen la misma distribucion teorica, la superficie 3D
# objetivo es la misma, pero se guarda con el nombre de cada descomposicion
# para facilitar la explicacion en clase.
crear_grilla_densidad = function(media,
                                 Sigma,
                                 n_grilla = 80,
                                 multiplicador_sd = 3.5) {
  desv = sqrt(diag(Sigma))
  
  eje_x = seq(
    from = media[1] - multiplicador_sd * desv[1],
    to = media[1] + multiplicador_sd * desv[1],
    length.out = n_grilla
  )
  
  eje_y = seq(
    from = media[2] - multiplicador_sd * desv[2],
    to = media[2] + multiplicador_sd * desv[2],
    length.out = n_grilla
  )
  
  densidad = outer(
    eje_x,
    eje_y,
    Vectorize(function(x, y) {
      mvtnorm::dmvnorm(
        x = c(x, y),
        mean = media,
        sigma = Sigma
      )
    })
  )
  
  return(list(
    x = eje_x,
    y = eje_y,
    z = t(densidad)
  ))
}

graficar_densidad_3d = function(media,
                                Sigma,
                                titulo,
                                etiqueta_x = "u_1",
                                etiqueta_y = "u_2",
                                escala_color = "Viridis") {
  grilla = crear_grilla_densidad(media = media, Sigma = Sigma)
  
  grafica = plot_ly(
    x = grilla$x,
    y = grilla$y,
    z = grilla$z,
    type = "surface",
    colorscale = escala_color,
    contours = list(
      z = list(
        show = TRUE,
        usecolormap = TRUE,
        highlightcolor = "#ffffff",
        project = list(z = TRUE)
      )
    ),
    hovertemplate = paste(
      etiqueta_x, ": %{x:.3f}<br>",
      etiqueta_y, ": %{y:.3f}<br>",
      "Densidad: %{z:.5f}<extra></extra>"
    )
  )
  
  grafica = layout(
    grafica,
    title = list(text = titulo),
    scene = list(
      xaxis = list(title = etiqueta_x),
      yaxis = list(title = etiqueta_y),
      zaxis = list(title = "Densidad"),
      aspectmode = "cube",
      camera = list(
        eye = list(x = 1.55, y = -1.65, z = 1.15)
      )
    ),
    margin = list(l = 0, r = 0, b = 0, t = 70)
  )
  
  return(grafica)
}

g_normal_estandar_3d = graficar_densidad_3d(
  media = c(z_1 = 0, z_2 = 0),
  Sigma = diag(2),
  titulo = "Densidad 3D - Normal estandar bivariada",
  etiqueta_x = "z_1",
  etiqueta_y = "z_2",
  escala_color = "Viridis"
)

g_espectral_3d = graficar_densidad_3d(
  media = media,
  Sigma = Sigma,
  titulo = "Densidad 3D - Descomposicion espectral",
  escala_color = "YlOrRd"
)

g_svd_3d = graficar_densidad_3d(
  media = media,
  Sigma = Sigma,
  titulo = "Densidad 3D - SVD",
  escala_color = "Plasma"
)

g_cholesky_3d = graficar_densidad_3d(
  media = media,
  Sigma = Sigma,
  titulo = "Densidad 3D - Cholesky",
  escala_color = "Portland"
)


# ===
# 9. Exportacion de graficas a archivos HTML ----
# ===

# Guardamos todas las graficas como archivos HTML autocontenidos. Para evitar
# problemas de pandoc con rutas largas o con caracteres especiales en Windows,
# primero se crea cada HTML en una carpeta temporal corta y luego se copia a la
# carpeta final del proyecto.
directorio_graficas = if (dir.exists(file.path("codigo", "R"))) {
  file.path("codigo", "R", "graficas_html_normal_multivariada")
} else {
  file.path(getwd(), "graficas_html_normal_multivariada")
}

dir.create(directorio_graficas, recursive = TRUE, showWarnings = FALSE)

guardar_grafica_html = function(grafica, nombre_archivo) {
  ruta_archivo = file.path(directorio_graficas, nombre_archivo)
  ruta_temporal = file.path(tempdir(), nombre_archivo)
  
  if (file.exists(ruta_temporal)) {
    unlink(ruta_temporal)
  }
  
  htmlwidgets::saveWidget(
    widget = grafica,
    file = ruta_temporal,
    selfcontained = TRUE,
    title = tools::file_path_sans_ext(nombre_archivo)
  )
  
  file.copy(
    from = ruta_temporal,
    to = ruta_archivo,
    overwrite = TRUE
  )
  
  cat(
    "Grafica guardada: ",
    normalizePath(ruta_archivo, winslash = "/", mustWork = FALSE),
    "\n",
    sep = ""
  )
}

guardar_grafica_html(
  g_normal_estandar,
  "01_normal_estandar_bivariada_2d.html"
)

guardar_grafica_html(
  g_espectral,
  "02_normal_correlacionada_espectral_2d.html"
)

guardar_grafica_html(
  g_svd,
  "03_normal_correlacionada_svd_2d.html"
)

guardar_grafica_html(
  g_cholesky,
  "04_normal_correlacionada_cholesky_2d.html"
)

guardar_grafica_html(
  g_normal_estandar_3d,
  "05_normal_estandar_bivariada_3d.html"
)

guardar_grafica_html(
  g_espectral_3d,
  "06_normal_correlacionada_espectral_3d.html"
)

guardar_grafica_html(
  g_svd_3d,
  "07_normal_correlacionada_svd_3d.html"
)

guardar_grafica_html(
  g_cholesky_3d,
  "08_normal_correlacionada_cholesky_3d.html"
)

cat(
  "\nDirectorio con graficas HTML:\n",
  normalizePath(directorio_graficas, winslash = "/", mustWork = FALSE),
  "\n",
  sep = ""
)

# Si se ejecuta el script interactivamente, tambien se muestran las graficas
# en el visor de RStudio. Al usar Rscript, se guardan en HTML sin abrir ventanas.
if (interactive()) {
  print(g_normal_estandar)
  print(g_espectral)
  print(g_svd)
  print(g_cholesky)
  print(g_normal_estandar_3d)
  print(g_espectral_3d)
  print(g_svd_3d)
  print(g_cholesky_3d)
}
# )
